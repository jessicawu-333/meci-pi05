"""MECI for pi0.5 -- PyTorch variant for openpi's PyTorch port (src/openpi/models_pytorch/pi0_pytorch.py).

Mirrors meci_correction.py (JAX / Flax NNX) one-to-one. Hook points in PI0Pytorch
(openpi @ 215abfb217dbac7d5f1273282331b9b1866c0479):
  * embed_suffix (l.238-315), pi05 branch (l.288-298): `action_time_emb = action_emb`, `adarms_cond = time_emb`.
  * forward (l.317-374) and denoise_step (l.422-462): `self.action_out_proj(suffix_out)` read-out.
  * sample_actions (l.376-420): Euler loop; the explicit residual is added to the final x_t.
Same identity guarantees: zero-initialised tanh gates x per-sample strength (0 when inactive).
"""

from __future__ import annotations

import dataclasses

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from openpi.models.meci_correction import DELTA_SCALE
from openpi.models.meci_correction import NUM_EFFECTORS
from openpi.models.meci_correction import MECIConfig


@dataclasses.dataclass
class CorrectionContextTorch:
    tokens: torch.Tensor  # (b, K, w)
    token_mask: torch.Tensor  # (b, K) bool
    pooled: torch.Tensor  # (b, w)
    strength: torch.Tensor  # (b,)
    delta_hat: torch.Tensor  # (b, 6)
    effector_logits: torch.Tensor  # (b, 3)


class _CrossAttention(nn.Module):
    def __init__(self, q_dim: int, kv_dim: int, width: int, num_heads: int, *, zero_out: bool):
        super().__init__()
        assert width % num_heads == 0
        self.num_heads = num_heads
        self.q = nn.Linear(q_dim, width)
        self.k = nn.Linear(kv_dim, width)
        self.v = nn.Linear(kv_dim, width)
        self.o = nn.Linear(width, q_dim)
        if zero_out:
            nn.init.zeros_(self.o.weight)
            nn.init.zeros_(self.o.bias)

    def forward(self, x, ctx, ctx_mask=None):
        b, n, _ = x.shape
        m = ctx.shape[1]
        h = self.num_heads
        q = self.q(x).view(b, n, h, -1).transpose(1, 2)
        k = self.k(ctx).view(b, m, h, -1).transpose(1, 2)
        v = self.v(ctx).view(b, m, h, -1).transpose(1, 2)
        logits = (q @ k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
        if ctx_mask is not None:
            logits = logits.masked_fill(~ctx_mask[:, None, None, :], -1e30)
        probs = torch.softmax(logits.float(), dim=-1).to(v.dtype)
        out = (probs @ v).transpose(1, 2).reshape(b, n, -1)
        return self.o(out)


class MECICorrectionTorch(nn.Module):
    def __init__(self, width: int, config: MECIConfig):
        super().__init__()
        self.config = config
        self.audio_norm = nn.LayerNorm(config.audio_dim)
        self.queries = nn.Parameter(torch.randn(config.num_queries, width) * 0.02)
        self.projector = _CrossAttention(width, config.audio_dim, width, config.num_heads, zero_out=False)
        self.proj_norm = nn.LayerNorm(width)
        self.proj_mlp = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
        self.disp_in = nn.Linear(width, width)
        self.disp_out = nn.Linear(width, 6 + NUM_EFFECTORS)
        nn.init.zeros_(self.disp_out.weight)
        nn.init.zeros_(self.disp_out.bias)
        self.delta_embed = nn.Linear(6, width)
        nn.init.zeros_(self.delta_embed.weight)
        nn.init.zeros_(self.delta_embed.bias)
        self.effector_embed = nn.Embedding(NUM_EFFECTORS, width)
        nn.init.zeros_(self.effector_embed.weight)
        self.in_xattn = _CrossAttention(width, width, width, config.num_heads, zero_out=False)
        self.gate_in = nn.Parameter(torch.zeros(()))
        self.ada_proj = nn.Linear(width, width)
        self.gate_ada = nn.Parameter(torch.zeros(()))
        self.head_xattn = _CrossAttention(width, width, width, config.num_heads, zero_out=False)
        self.gate_head = nn.Parameter(torch.zeros(()))
        self.register_buffer("delta_scale", torch.tensor(DELTA_SCALE, dtype=torch.float32), persistent=False)

    def encode(self, obs) -> CorrectionContextTorch:
        audio = obs.correction_audio.float()
        b = audio.shape[0]
        audio_mask = getattr(obs, "correction_audio_mask", None)
        if audio_mask is None:
            audio_mask = torch.ones(audio.shape[:2], dtype=torch.bool, device=audio.device)
        active = getattr(obs, "correction_active", None)
        active = audio_mask.any(-1) if active is None else active.bool()
        strength = getattr(obs, "correction_strength", None)
        strength = torch.ones(b, device=audio.device) if strength is None else strength.float()
        strength = torch.where(active, strength, torch.zeros_like(strength))

        q = self.queries[None].expand(b, -1, -1).to(audio.dtype)
        tok = q + self.projector(q, self.audio_norm(audio), audio_mask.bool())
        tok = tok + self.proj_mlp(self.proj_norm(tok))
        pooled = tok.mean(1)
        head = self.disp_out(F.silu(self.disp_in(pooled)))
        delta_hat = head[:, :6] * self.delta_scale
        eff_logits = head[:, 6:]

        given_delta = getattr(obs, "correction_delta", None)
        if self.config.use_given_delta and given_delta is not None:
            delta_cond = given_delta.float() / self.delta_scale
        else:
            delta_cond = delta_hat.detach() / self.delta_scale
        given_eff = getattr(obs, "correction_effector", None)
        if self.config.use_given_delta and given_eff is not None:
            eff_emb = self.effector_embed(given_eff.long())
        else:
            eff_emb = torch.softmax(eff_logits, -1) @ self.effector_embed.weight
        tok = tok + self.delta_embed(delta_cond)[:, None] + eff_emb[:, None]
        pooled = tok.mean(1)
        token_mask = active[:, None].expand(b, tok.shape[1])
        return CorrectionContextTorch(tok, token_mask, pooled, strength, delta_hat, eff_logits)

    def condition_suffix(self, ctx, action_tokens, adarms_cond):
        if self.config.mode in ("gated", "gated+suffix_tokens"):
            s = ctx.strength.to(action_tokens.dtype)
            action_tokens = action_tokens + torch.tanh(self.gate_in).to(action_tokens.dtype) * s[
                :, None, None
            ] * self.in_xattn(action_tokens, ctx.tokens.to(action_tokens.dtype), ctx.token_mask)
            adarms_cond = adarms_cond + torch.tanh(self.gate_ada).to(adarms_cond.dtype) * s[:, None].to(
                adarms_cond.dtype
            ) * self.ada_proj(ctx.pooled).to(adarms_cond.dtype)
        return action_tokens, adarms_cond

    def suffix_tokens(self, ctx):
        k = ctx.tokens.shape[1]
        return ctx.tokens, ctx.token_mask, [1] + [0] * (k - 1)

    def pre_head(self, ctx, h):
        if self.config.mode not in ("gated", "gated+suffix_tokens"):
            return h
        s = ctx.strength.to(h.dtype)[:, None, None]
        return h + torch.tanh(self.gate_head).to(h.dtype) * s * self.head_xattn(
            h, ctx.tokens.to(h.dtype), ctx.token_mask
        )

    def gated_residual(self, ctx, residual):
        """Explicit residual (normalized units), applied only to samples with an active correction."""
        return torch.where(ctx.token_mask[:, 0, None, None], residual.float(), torch.zeros_like(residual.float()))

    def aux_loss(self, ctx, obs):
        active = ctx.token_mask[:, 0].float()
        loss = torch.zeros_like(active)
        given_delta = getattr(obs, "correction_delta", None)
        if given_delta is not None:
            loss = loss + ((ctx.delta_hat - given_delta.float()) / self.delta_scale).abs().mean(-1)
        given_eff = getattr(obs, "correction_effector", None)
        if given_eff is not None:
            loss = loss + F.cross_entropy(ctx.effector_logits, given_eff.long(), reduction="none")
        return active * loss
