"""MECI for pi0.5 (openpi, JAX / Flax NNX): spoken metric corrections injected into the action expert.

Hook points (Physical-Intelligence/openpi @ 215abfb217dbac7d5f1273282331b9b1866c0479,
src/openpi/models/pi0.py):
  * Pi0.embed_suffix (l.140-186), pi05 branch (l.162-169): the action-expert input tokens
    `action_expert_tokens` and the adaRMS conditioning vector `adarms_cond` (flow-matching time MLP)
    are produced here. MECI adds (a) gated cross-attention from the action tokens to the correction
    tokens, (b) a gated additive term on `adarms_cond`, and optionally (c) extra suffix tokens.
  * The action read-out `self.action_out_proj(suffix_out[:, -self.action_horizon:])`
    (compute_loss l.212, sample_actions step l.269): MECI applies a gated cross-attention
    "pre-head" block to the action-position hidden states just before `action_out_proj`
    (the analogue of the OpenVLA-OFT fusion point).
  * adaRMS itself (src/openpi/models/gemma.py RMSNorm l.112-131): `modulation = Dense(3*d,
    kernel_init=zeros)(cond)` -> scale, shift, gate applied to every action-expert layer, so a change
    of `adarms_cond` reaches all 18 expert layers (gemma_300m depth=18, gemma.py l.69-78).

Identity guarantees:
  * every fusion path is multiplied by tanh(g) with g a zero-initialised scalar, and by the per-sample
    correction strength s in [0, 1] (0 when no correction is active). With g = 0 or s = 0 the action
    tokens, adarms_cond and pre-head hidden states are *exactly* the pretrained ones.
  * "suffix_tokens" mode adds K extra suffix tokens whose input_mask is False when no correction is
    active; openpi's make_attn_mask (pi0.py l.19-44) then excludes them, and positions are computed
    with cumsum(input_mask) so the action-token positions are unchanged -> identical outputs when no
    correction is active (but NOT identical at initialisation while a correction is active).
"""

from __future__ import annotations

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp

# Scale used to feed metric delta-p (m, rad) to the networks at O(1): 5 cm and 0.5 rad.
DELTA_SCALE = (0.05, 0.05, 0.05, 0.5, 0.5, 0.5)
NUM_EFFECTORS = 3  # default / left / right (meci_metric.EFFECTORS)
MODES = ("gated", "suffix_tokens", "gated+suffix_tokens")


@dataclasses.dataclass(frozen=True)
class MECIConfig:
    audio_dim: int = 768  # Whisper-small encoder width (d_model=768)
    num_queries: int = 16  # learned-query projector -> K correction tokens
    num_heads: int = 8
    mode: str = "gated"
    use_given_delta: bool = True  # condition on the parsed/teacher delta-p when provided
    aux_weight: float = 0.1  # weight of the auxiliary delta-p / effector loss in compute_loss
    explicit_residual: bool = True  # add obs.correction_residual to sampled actions / subtract from targets

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")


@dataclasses.dataclass
class CorrectionContext:
    tokens: jax.Array  # (b, K, w) correction tokens
    token_mask: jax.Array  # (b, K) bool, True where a correction is active
    pooled: jax.Array  # (b, w)
    strength: jax.Array  # (b,) in [0, 1]; 0 when inactive
    delta_hat: jax.Array  # (b, 6) predicted delta-p in metres / radians
    effector_logits: jax.Array  # (b, 3)


class _CrossAttention(nnx.Module):
    """Multi-head cross-attention x -> ctx with optional zero-initialised output projection."""

    def __init__(self, q_dim: int, kv_dim: int, width: int, num_heads: int, *, zero_out: bool, rngs: nnx.Rngs):
        assert width % num_heads == 0
        self.num_heads = num_heads
        self.q = nnx.Linear(q_dim, width, rngs=rngs)
        self.k = nnx.Linear(kv_dim, width, rngs=rngs)
        self.v = nnx.Linear(kv_dim, width, rngs=rngs)
        init = nnx.initializers.zeros_init() if zero_out else nnx.initializers.lecun_normal()
        self.o = nnx.Linear(width, q_dim, kernel_init=init, rngs=rngs)

    def __call__(self, x: jax.Array, ctx: jax.Array, ctx_mask: jax.Array | None = None) -> jax.Array:
        b, n, _ = x.shape
        m = ctx.shape[1]
        h = self.num_heads
        q = self.q(x).reshape(b, n, h, -1)
        k = self.k(ctx).reshape(b, m, h, -1)
        v = self.v(ctx).reshape(b, m, h, -1)
        logits = jnp.einsum("bnhd,bmhd->bhnm", q, k) / jnp.sqrt(q.shape[-1]).astype(q.dtype)
        if ctx_mask is not None:
            logits = jnp.where(ctx_mask[:, None, None, :], logits, -1e30)
        probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(v.dtype)
        out = jnp.einsum("bhnm,bmhd->bnhd", probs, v).reshape(b, n, -1)
        return self.o(out)


class MECICorrection(nnx.Module):
    """Audio -> correction tokens -> conditioning of the pi0.5 action expert."""

    def __init__(self, width: int, config: MECIConfig, rngs: nnx.Rngs):
        self.config = config
        self.width = width
        self.audio_norm = nnx.LayerNorm(config.audio_dim, rngs=rngs)
        self.queries = nnx.Param(jax.random.normal(rngs.params(), (config.num_queries, width)) * 0.02)
        # learned-query (Perceiver-style) projector over frozen Whisper features
        self.projector = _CrossAttention(width, config.audio_dim, width, config.num_heads, zero_out=False, rngs=rngs)
        self.proj_norm = nnx.LayerNorm(width, rngs=rngs)
        self.proj_mlp_in = nnx.Linear(width, 2 * width, rngs=rngs)
        self.proj_mlp_out = nnx.Linear(2 * width, width, rngs=rngs)
        # displacement head: pooled tokens -> delta-p (6, scaled) + effector logits (3); zero-init output
        self.disp_in = nnx.Linear(width, width, rngs=rngs)
        self.disp_out = nnx.Linear(width, 6 + NUM_EFFECTORS, kernel_init=nnx.initializers.zeros_init(), rngs=rngs)
        # zero-init conditioners: delta-p and effector embeddings added to every correction token
        self.delta_embed = nnx.Linear(6, width, kernel_init=nnx.initializers.zeros_init(), rngs=rngs)
        self.effector_embed = nnx.Embed(NUM_EFFECTORS, width, embedding_init=nnx.initializers.zeros_init(), rngs=rngs)
        # fusion paths, each with a zero-initialised scalar gate
        self.in_xattn = _CrossAttention(width, width, width, config.num_heads, zero_out=False, rngs=rngs)
        self.gate_in = nnx.Param(jnp.zeros(()))
        self.ada_proj = nnx.Linear(width, width, rngs=rngs)
        self.gate_ada = nnx.Param(jnp.zeros(()))
        self.head_xattn = _CrossAttention(width, width, width, config.num_heads, zero_out=False, rngs=rngs)
        self.gate_head = nnx.Param(jnp.zeros(()))

    # ------------------------------------------------------------------ encode
    def encode(self, obs) -> CorrectionContext:
        """Build correction tokens from `obs.correction_*` (see patched openpi Observation)."""
        audio = obs.correction_audio.astype(jnp.float32)
        b = audio.shape[0]
        audio_mask = (
            obs.correction_audio_mask
            if obs.correction_audio_mask is not None
            else jnp.ones(audio.shape[:2], dtype=jnp.bool_)
        )
        active = (
            obs.correction_active.astype(jnp.bool_)
            if obs.correction_active is not None
            else jnp.any(audio_mask, axis=-1)
        )
        strength = obs.correction_strength if obs.correction_strength is not None else jnp.ones((b,))
        strength = jnp.where(active, strength.astype(jnp.float32), 0.0)

        q = jnp.broadcast_to(self.queries.value[None], (b, *self.queries.value.shape))
        tok = q + self.projector(q, self.audio_norm(audio), audio_mask)
        tok = tok + self.proj_mlp_out(nnx.gelu(self.proj_mlp_in(self.proj_norm(tok))))

        pooled = jnp.mean(tok, axis=1)
        head = self.disp_out(nnx.swish(self.disp_in(pooled)))
        scale = jnp.asarray(DELTA_SCALE, dtype=jnp.float32)
        delta_hat = head[:, :6] * scale
        eff_logits = head[:, 6:]

        if self.config.use_given_delta and obs.correction_delta is not None:
            delta_cond = obs.correction_delta.astype(jnp.float32) / scale
        else:
            delta_cond = jax.lax.stop_gradient(delta_hat) / scale
        if self.config.use_given_delta and obs.correction_effector is not None:
            eff_emb = self.effector_embed(obs.correction_effector.astype(jnp.int32))
        else:
            eff_emb = jax.nn.softmax(eff_logits, axis=-1) @ self.effector_embed.embedding.value
        tok = tok + self.delta_embed(delta_cond)[:, None, :] + eff_emb[:, None, :]
        pooled = jnp.mean(tok, axis=1)
        token_mask = jnp.broadcast_to(active[:, None], tok.shape[:2])
        return CorrectionContext(tok, token_mask, pooled, strength, delta_hat, eff_logits)

    # ------------------------------------------------------------------ fusion
    def condition_suffix(self, ctx: CorrectionContext, action_tokens: jax.Array, adarms_cond: jax.Array):
        """Gated conditioning of the action-expert inputs. Identity when gates are 0 or strength is 0."""
        s = ctx.strength.astype(action_tokens.dtype)
        if self.config.mode in ("gated", "gated+suffix_tokens"):
            g_in = jnp.tanh(self.gate_in.value).astype(action_tokens.dtype)
            action_tokens = action_tokens + g_in * s[:, None, None] * self.in_xattn(
                action_tokens, ctx.tokens.astype(action_tokens.dtype), ctx.token_mask
            )
            g_ada = jnp.tanh(self.gate_ada.value).astype(adarms_cond.dtype)
            adarms_cond = adarms_cond + g_ada * s[:, None].astype(adarms_cond.dtype) * self.ada_proj(ctx.pooled).astype(
                adarms_cond.dtype
            )
        return action_tokens, adarms_cond

    def suffix_tokens(self, ctx: CorrectionContext):
        """Extra suffix tokens (mode includes 'suffix_tokens'): (tokens, input_mask, ar_mask)."""
        k = ctx.tokens.shape[1]
        ar = [True] + [False] * (k - 1)  # own block: sees prefix + itself; action tokens see it
        return ctx.tokens, ctx.token_mask, ar

    def pre_head(self, ctx: CorrectionContext, h: jax.Array) -> jax.Array:
        """Gated cross-attention on action-position hidden states right before action_out_proj."""
        if self.config.mode not in ("gated", "gated+suffix_tokens"):
            return h
        g = jnp.tanh(self.gate_head.value).astype(h.dtype)
        s = ctx.strength.astype(h.dtype)[:, None, None]
        return h + g * s * self.head_xattn(h, ctx.tokens.astype(h.dtype), ctx.token_mask)

    def gated_residual(self, ctx: CorrectionContext, residual: jax.Array) -> jax.Array:
        """Explicit residual (normalized units), applied only to samples with an active correction."""
        return jnp.where(ctx.token_mask[:, 0, None, None], residual.astype(jnp.float32), 0.0)

    # ------------------------------------------------------------------ losses
    def aux_loss(self, ctx: CorrectionContext, obs) -> jax.Array:
        """Per-sample auxiliary loss: L1 on scaled delta-p + effector cross-entropy, active samples only."""
        b = ctx.delta_hat.shape[0]
        active = ctx.token_mask[:, 0].astype(jnp.float32)
        loss = jnp.zeros((b,), jnp.float32)
        scale = jnp.asarray(DELTA_SCALE, dtype=jnp.float32)
        if obs.correction_delta is not None:
            loss = loss + jnp.mean(jnp.abs((ctx.delta_hat - obs.correction_delta) / scale), axis=-1)
        if obs.correction_effector is not None:
            logp = jax.nn.log_softmax(ctx.effector_logits, axis=-1)
            loss = loss - jnp.take_along_axis(logp, obs.correction_effector[:, None].astype(jnp.int32), axis=-1)[:, 0]
        return active * loss


# ---------------------------------------------------------------------- RL helpers (flow matching)
def awr_weights(advantages: jax.Array, beta: float = 1.0, w_max: float = 20.0) -> jax.Array:
    """AWR weights exp(A / beta), clipped; multiply openpi's per-sample flow-matching loss by these."""
    return jnp.minimum(jnp.exp(advantages / beta), w_max)


def flow_dpo_loss(
    loss_pos: jax.Array, loss_neg: jax.Array, ref_loss_pos: jax.Array, ref_loss_neg: jax.Array, beta: float = 1.0
) -> jax.Array:
    """Diffusion-DPO-style preference loss on flow-matching errors (Wallace et al., arXiv:2311.12908).

    Each argument is a per-sample flow-matching loss (mean over chunk) evaluated with the SAME noise and
    time for the policy and the frozen reference; the flow-matching error stands in for -log-likelihood.
    pos = during-correction (preferred) chunk, neg = pre-correction chunk from the same state.
    """
    margin = (loss_pos - ref_loss_pos) - (loss_neg - ref_loss_neg)
    return -jax.nn.log_sigmoid(-beta * margin)
