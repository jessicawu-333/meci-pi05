"""Unit tests for MECI on pi0.5 (openpi @ 215abfb). CPU only, dummy tensors, tiny dummy Gemma.

Run (from this directory, patched openpi tree on PYTHONPATH, pristine reference .npz in MECI_REF):
  PYTHONPATH=<patched>/src:. MECI_REF=/tmp/pristine_ref.npz pytest -q test_meci_pi05.py
"""
import dataclasses
import math
import os

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

import tiny_setup as ts

ts.patch_siglip()
from openpi.models import meci_correction as meci  # noqa: E402
from openpi.models import model as _model  # noqa: E402
from openpi.models import pi0_config  # noqa: E402
from openpi.shared import meci_metric as mm  # noqa: E402
from openpi.shared import normalize as _normalize  # noqa: E402

REF = os.environ.get("MECI_REF", "/tmp/pristine_ref.npz")


# ============================================================================ helpers
def jax_obs(correction=None):
    images, masks, state, tok, tok_mask, *_ = ts.fake_inputs()
    kw = {} if correction is None else {k: jnp.asarray(v) for k, v in correction.items()}
    return _model.Observation(
        images={k: jnp.asarray(v) for k, v in images.items()},
        image_masks={k: jnp.asarray(v) for k, v in masks.items()},
        state=jnp.asarray(state),
        tokenized_prompt=jnp.asarray(tok),
        tokenized_prompt_mask=jnp.asarray(tok_mask),
        **kw,
    )


def make_jax_model(mode="gated", explicit_residual=True, aux_weight=0.1):
    meci_cfg = meci.MECIConfig(audio_dim=ts.DA, num_queries=4, num_heads=4, mode=mode,
                               explicit_residual=explicit_residual, aux_weight=aux_weight)
    cfg = pi0_config.Pi0Config(**ts.base_config_kwargs(), meci=meci_cfg)
    return ts.randomize_adarms_jax(cfg.create(jax.random.key(0)))  # same adaRMS weights as the reference


def set_gates(model, value):
    for name in ("gate_in", "gate_ada", "gate_head"):
        getattr(model.meci_correction, name).value = jnp.asarray(value, jnp.float32)


def randomize_zero_inits(model, seed=3):
    """Make the zero-initialised layers non-zero (as after training) to test masking-based identity."""
    key = jax.random.key(seed)
    mc = model.meci_correction
    for lin in (mc.disp_out, mc.delta_embed):
        key, k = jax.random.split(key)
        lin.kernel.value = jax.random.normal(k, lin.kernel.value.shape) * 0.1
    key, k = jax.random.split(key)
    mc.effector_embed.embedding.value = jax.random.normal(k, mc.effector_embed.embedding.value.shape) * 0.1


@pytest.fixture(autouse=True)
def _free_memory():
    yield
    import gc

    jax.clear_caches()
    gc.collect()


@pytest.fixture(scope="module")
def ref():
    if not os.path.exists(REF):
        pytest.skip("pristine reference not generated")
    return dict(np.load(REF))


# ============================================================================ 1. parser
@pytest.mark.parametrize(
    "phrase, eff, direction, si",
    [
        ("left hand more left half an inch", "left", "left", 0.5 * 0.0254),
        ("right gripper up 1 cm", "right", "up", 0.01),
        ("move forward 1 inch", "default", "forward", 0.0254),
        ("Left arm, back two centimeters", "left", "back", 0.02),
        ("right hand down 5 mm", "right", "down", 0.005),
        ("rotate gripper 10 degrees", "default", "yaw", 10 * math.pi / 180),
        ("left gripper move right a quarter inch", "left", "right", 0.25 * 0.0254),
    ],
)
def test_parser(phrase, eff, direction, si):
    cmd = mm.parse_metric_command(phrase)
    assert cmd is not None
    assert (cmd.effector, cmd.direction) == (eff, direction)
    dp = cmd.delta_p()
    assert dp.shape == (6,)
    assert np.count_nonzero(dp) == 1
    assert math.isclose(abs(dp[cmd.axis]), si, rel_tol=1e-6)


@pytest.mark.parametrize("phrase", ["pick up the other cup", "left hand", "", "go left"])
def test_parser_rejects_non_metric(phrase):
    assert mm.parse_metric_command(phrase) is None


# ============================================================================ 2. residual = delta-p
def _quantile_stats(dim, seed=0):
    rng = np.random.default_rng(seed)
    q01 = -rng.uniform(0.5, 2.0, dim).astype(np.float32)
    q99 = rng.uniform(0.5, 2.0, dim).astype(np.float32)
    return _normalize.NormStats(mean=np.zeros(dim, np.float32), std=np.ones(dim, np.float32), q01=q01, q99=q99)


def test_libero_per_step_residual_sums_to_delta_p_through_openpi_unnormalize():
    """LIBERO: per-step OSC deltas. Sum over the hold window (across chunks) of the residual, after openpi's own
    quantile Unnormalize and the OSC scale, equals delta-p in metres."""
    from openpi import transforms as _transforms

    layout = mm.LIBERO
    stats = _quantile_stats(7)
    unnorm = _transforms.Unnormalize({"actions": stats}, use_quantiles=True)
    cmd = mm.parse_metric_command("move left half an inch")
    w = mm.hold_fade_strengths(16, 8)
    delta_a, routed = mm.raw_action_delta(layout, cmd.delta_p(), cmd.effector)
    assert routed
    total_phys = np.zeros(6)
    base = np.random.default_rng(1).uniform(-1, 1, (ts.H, 32)).astype(np.float32)  # arbitrary policy output
    for chunk_start in range(0, 20, ts.H):  # pi05_libero action_horizon = 10
        r = mm.chunk_residual(layout, delta_a, w, chunk_start, ts.H)
        rn = mm.normalize_residual(r, q01=stats.q01, q99=stats.q99)
        a0 = unnorm({"actions": base.copy()})["actions"]
        a1 = unnorm({"actions": base + rn})["actions"]
        diff = (a1 - a0)[:, :7]
        np.testing.assert_allclose(diff, r, atol=1e-5)  # normalization map is exact
        total_phys += (diff[:, :6] * np.asarray(layout.cartesian_scale)).sum(0)
    np.testing.assert_allclose(total_phys, cmd.delta_p(), atol=1e-6)
    assert np.allclose(total_phys[1], 0.0127)


def _toy_jacobian(n=6, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(6, n)) * 0.3 + np.eye(6, n) * 0.5).astype(np.float64)


def test_aloha_chunk_relative_ramp_and_routing():
    """ALOHA: delta-joint actions relative to the chunk start (DeltaActions(6,-1,6,-1)); the ramp ends at the
    full joint delta, the named arm only, grippers untouched; J dq == delta-p (first order)."""
    layout = mm.ALOHA
    J = {"left": _toy_jacobian(6, 0), "right": _toy_jacobian(6, 1)}
    flip = np.asarray(layout.joint_flip, np.float32)
    for phrase, eff, sl, other in [
        ("left hand more left half an inch", "left", slice(0, 6), slice(7, 13)),
        ("right gripper up 1 cm", "right", slice(7, 13), slice(0, 6)),
    ]:
        cmd = mm.parse_metric_command(phrase)
        delta_a, routed = mm.raw_action_delta(layout, cmd.delta_p(), cmd.effector, mm.make_correction_jacobian_fn(J))
        assert routed and cmd.effector == eff
        dq_aloha = delta_a[sl] * flip[sl]  # undo adapt_to_pi flip -> robot (Interbotix) joint convention
        np.testing.assert_allclose(J[eff] @ dq_aloha, cmd.delta_p(), atol=1e-5)
        assert np.all(delta_a[other] == 0) and delta_a[6] == 0 and delta_a[13] == 0
        w = mm.hold_fade_strengths(16, 8)
        r0 = mm.chunk_residual(layout, delta_a, w, 0, 50)  # pi05_aloha action_horizon = 50 >= window
        np.testing.assert_allclose(r0[-1], delta_a, atol=1e-6)  # cumulative target == full delta
        assert np.all(np.diff(np.abs(r0[:, sl]), axis=0) >= -1e-7)  # monotone ramp
        # window split across two chunks: re-based residuals add up to the full delta
        r_a = mm.chunk_residual(layout, delta_a, w, 0, 8)
        r_b = mm.chunk_residual(layout, delta_a, w, 8, 8)
        np.testing.assert_allclose(r_a[-1] + r_b[-1], delta_a, atol=1e-6)


def test_aloha_unnamed_effector_is_not_guessed():
    cmd = mm.parse_metric_command("move left half an inch")
    J = {"left": _toy_jacobian(), "right": _toy_jacobian(seed=1)}
    delta_a, routed = mm.raw_action_delta(mm.ALOHA, cmd.delta_p(), cmd.effector, mm.make_correction_jacobian_fn(J))
    assert not routed and np.all(delta_a == 0)


def test_single_arm_ignores_effector_name():
    cmd = mm.parse_metric_command("right hand up 1 cm")
    d1, r1 = mm.raw_action_delta(mm.LIBERO, cmd.delta_p(), "right")
    d0, r0 = mm.raw_action_delta(mm.LIBERO, cmd.delta_p(), "default")
    assert r1 and r0 and np.array_equal(d0, d1)


def test_build_correction_inputs_keys():
    stats = _quantile_stats(14)
    J = {"left": _toy_jacobian(), "right": _toy_jacobian(seed=1)}
    out = mm.build_correction_inputs("left hand more left half an inch", mm.ALOHA, action_horizon=50,
                                     norm_stats_actions=stats, jacobian_fn=mm.make_correction_jacobian_fn(J))
    assert set(out) == set(_model.CORRECTION_KEYS) - {"correction_audio", "correction_audio_mask"}
    assert out["correction_residual"].shape == (50, 32) and out["correction_effector"] == 1
    assert np.all(out["correction_residual"][:, 7:] == 0)


# ============================================================================ 3. JAX model: identity / effect
def test_patched_model_without_correction_matches_pristine_openpi(ref):
    m = make_jax_model()
    _, _, _, _, _, actions, noise, _ = ts.fake_inputs()
    obs = jax_obs()
    np.testing.assert_array_equal(np.asarray(m.compute_loss(jax.random.key(1), obs, jnp.asarray(actions))), ref["jax_loss"])
    np.testing.assert_array_equal(
        np.asarray(m.sample_actions(jax.random.key(2), obs, num_steps=3, noise=jnp.asarray(noise))), ref["jax_sample"]
    )


def test_zero_init_gates_identity_with_active_correction(ref):
    """Correction present and active, gates at their zero init, no residual, no aux -> bit-identical."""
    m = make_jax_model(aux_weight=0.0)
    _, _, _, _, _, actions, noise, _ = ts.fake_inputs()
    corr = ts.fake_correction()
    corr.pop("correction_residual")
    obs = jax_obs(corr)
    np.testing.assert_array_equal(np.asarray(m.compute_loss(jax.random.key(1), obs, jnp.asarray(actions))), ref["jax_loss"])
    np.testing.assert_array_equal(
        np.asarray(m.sample_actions(jax.random.key(2), obs, num_steps=3, noise=jnp.asarray(noise))), ref["jax_sample"]
    )


def test_inactive_correction_identity_even_with_trained_gates(ref):
    m = make_jax_model(aux_weight=0.0)
    set_gates(m, 0.7)
    randomize_zero_inits(m)
    _, _, _, _, _, actions, noise, _ = ts.fake_inputs()
    corr = ts.fake_correction(active=(False, False))
    obs = jax_obs(corr)
    np.testing.assert_array_equal(
        np.asarray(m.sample_actions(jax.random.key(2), obs, num_steps=3, noise=jnp.asarray(noise))), ref["jax_sample"]
    )


def test_open_gates_change_only_active_samples(ref):
    m = make_jax_model(aux_weight=0.0)
    set_gates(m, 0.7)
    randomize_zero_inits(m)
    _, _, _, _, _, _, noise, _ = ts.fake_inputs()
    corr = ts.fake_correction(active=(True, False))
    corr.pop("correction_residual")
    out = np.asarray(m.sample_actions(jax.random.key(2), jax_obs(corr), num_steps=3, noise=jnp.asarray(noise)))
    assert np.abs(out[0] - ref["jax_sample"][0]).max() > 1e-4  # active sample changes
    np.testing.assert_array_equal(out[1], ref["jax_sample"][1])  # inactive sample in the same batch unchanged


@pytest.mark.parametrize("which", ["gate_in", "gate_ada", "gate_head"])
def test_each_fusion_path_is_live(which):
    m = make_jax_model(aux_weight=0.0)
    _, _, _, _, _, _, noise, _ = ts.fake_inputs()
    corr = ts.fake_correction()
    corr.pop("correction_residual")
    obs = jax_obs(corr)
    base = np.asarray(m.sample_actions(jax.random.key(2), obs, num_steps=3, noise=jnp.asarray(noise)))
    getattr(m.meci_correction, which).value = jnp.asarray(0.7)
    out = np.asarray(m.sample_actions(jax.random.key(2), obs, num_steps=3, noise=jnp.asarray(noise)))
    assert np.abs(out - base).max() > 1e-5


def test_explicit_residual_added_to_sampled_chunk(ref):
    m = make_jax_model()
    _, _, _, _, _, _, noise, _ = ts.fake_inputs()
    corr = ts.fake_correction()
    out = np.asarray(m.sample_actions(jax.random.key(2), jax_obs(corr), num_steps=3, noise=jnp.asarray(noise)))
    np.testing.assert_allclose(out, ref["jax_sample"] + corr["correction_residual"], atol=1e-6)


def test_suffix_token_mode_identity_when_inactive_and_live_when_active(ref):
    m = make_jax_model(mode="suffix_tokens", aux_weight=0.0)
    randomize_zero_inits(m)
    _, _, _, _, _, actions, noise, _ = ts.fake_inputs()
    off = ts.fake_correction(active=(False, False))
    off.pop("correction_residual")
    np.testing.assert_allclose(
        np.asarray(m.sample_actions(jax.random.key(2), jax_obs(off), num_steps=3, noise=jnp.asarray(noise))),
        ref["jax_sample"], atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(m.compute_loss(jax.random.key(1), jax_obs(off), jnp.asarray(actions))), ref["jax_loss"], atol=1e-6
    )
    on = ts.fake_correction()
    on.pop("correction_residual")
    out = np.asarray(m.sample_actions(jax.random.key(2), jax_obs(on), num_steps=3, noise=jnp.asarray(noise)))
    assert np.abs(out - ref["jax_sample"]).max() > 1e-5


def test_training_loss_aux_and_gradients_reach_gates():
    m = make_jax_model(aux_weight=0.1)
    _, _, _, _, _, actions, _, _ = ts.fake_inputs()
    corr = ts.fake_correction()
    obs = jax_obs(corr)

    def loss_fn(model):
        return jnp.mean(model.compute_loss(jax.random.key(1), obs, jnp.asarray(actions), train=False))

    loss, grads = nnx.value_and_grad(loss_fn)(m)
    assert np.isfinite(float(loss))
    g = grads.meci_correction
    assert abs(float(g.gate_in.value)) > 0 and abs(float(g.gate_ada.value)) > 0 and abs(float(g.gate_head.value)) > 0
    assert float(jnp.abs(g.disp_out.kernel.value).sum()) > 0  # aux delta-p / effector loss trains the head
    # aux loss is ignored for inactive samples
    ctx = m.meci_correction.encode(jax_obs(ts.fake_correction(active=(False, True))))
    aux = m.meci_correction.aux_loss(ctx, jax_obs(ts.fake_correction(active=(False, True))))
    assert float(aux[0]) == 0.0 and float(aux[1]) > 0.0


def test_meci_params_are_merged_by_checkpoint_loader():
    from openpi.training import weight_loaders

    m = make_jax_model()
    params = nnx.state(m, nnx.Param).to_pure_dict()
    flat = {"/".join(map(str, k)): v for k, v in __import__("flax").traverse_util.flatten_dict(params).items()}
    loaded = {k: v for k, v in params.items() if k != "meci_correction"}  # a pretrained pi05 checkpoint
    merged = weight_loaders._merge_params(loaded, params, missing_regex=".*lora.*|.*meci_correction.*")
    mflat = {"/".join(map(str, k)) for k in __import__("flax").traverse_util.flatten_dict(merged)}
    assert {k for k in flat if k.startswith("meci_correction")} <= mflat


def test_rl_helpers():
    w = meci.awr_weights(jnp.array([-1.0, 0.0, 10.0]), beta=1.0, w_max=5.0)
    np.testing.assert_allclose(np.asarray(w), [math.exp(-1), 1.0, 5.0], rtol=1e-6)
    same = meci.flow_dpo_loss(jnp.array(1.0), jnp.array(1.0), jnp.array(1.0), jnp.array(1.0))
    assert math.isclose(float(same), math.log(2), rel_tol=1e-6)
    better = meci.flow_dpo_loss(jnp.array(0.5), jnp.array(1.5), jnp.array(1.0), jnp.array(1.0))
    assert float(better) < math.log(2)


# ============================================================================ 4. data path
def test_observation_from_dict_and_policy_inputs_carry_correction():
    from openpi.policies import aloha_policy, libero_policy

    corr = {k: v[0] for k, v in ts.fake_correction().items()}
    ex = libero_policy.make_libero_example()
    ex.update(corr)
    out = libero_policy.LiberoInputs(model_type=_model.ModelType.PI05)(ex)
    assert all(k in out for k in _model.CORRECTION_KEYS)
    ex = aloha_policy.make_aloha_example()
    ex.update(corr)
    out = aloha_policy.AlohaInputs()(ex)
    assert all(k in out for k in _model.CORRECTION_KEYS)
    images, masks, state, tok, tok_mask, *_ = ts.fake_inputs()
    d = {"image": images, "image_mask": masks, "state": state, "tokenized_prompt": tok, "tokenized_prompt_mask": tok_mask,
         **ts.fake_correction()}
    obs = _model.Observation.from_dict(d)
    assert obs.correction_residual.shape == (ts.B, ts.H, 32)
    pre = _model.preprocess_observation(None, jax_obs(ts.fake_correction()), train=False)
    assert pre.correction_audio is not None and pre.correction_residual is not None


# ============================================================================ 5. PyTorch variant
def torch_obs(correction=None):
    images, masks, state, tok, tok_mask, *_ = ts.fake_inputs()
    kw = {} if correction is None else {k: torch.from_numpy(np.asarray(v)) for k, v in correction.items()}
    return _model.Observation(
        images={k: torch.from_numpy(v).permute(0, 3, 1, 2).contiguous() for k, v in images.items()},
        image_masks={k: torch.from_numpy(v) for k, v in masks.items()},
        state=torch.from_numpy(state),
        tokenized_prompt=torch.from_numpy(tok).long(),
        tokenized_prompt_mask=torch.from_numpy(tok_mask),
        **kw,
    )


@pytest.fixture(scope="module")
def torch_model_factory(ref):
    from openpi.models_pytorch import pi0_pytorch

    state = torch.load(REF + ".torch_state.pt")

    def make(mode="gated", aux_weight=0.0):
        meci_cfg = meci.MECIConfig(audio_dim=ts.DA, num_queries=4, num_heads=4, mode=mode, aux_weight=aux_weight)
        cfg = pi0_config.Pi0Config(**ts.base_config_kwargs(), pytorch_compile_mode=None, meci=meci_cfg)
        torch.manual_seed(0)
        m = ts.fix_torch_projector(pi0_pytorch.PI0Pytorch(cfg)).eval()
        missing, unexpected = m.load_state_dict(state, strict=False)
        assert not unexpected and all(k.startswith("meci_correction.") for k in missing)
        return m

    return make


def _torch_run(m, corr):
    _, _, _, _, _, actions, noise, time = ts.fake_inputs()
    obs = torch_obs(corr)
    with torch.no_grad():
        torch.manual_seed(7)  # forward() applies train-time image augmentation; same seed as the reference
        loss = m(obs, torch.from_numpy(actions), noise=torch.from_numpy(noise), time=torch.from_numpy(time)).numpy()
        sample = m.sample_actions("cpu", obs, noise=torch.from_numpy(noise), num_steps=3).numpy()
    return loss, sample


def test_torch_identity_and_residual(ref, torch_model_factory):
    m = torch_model_factory()
    loss, sample = _torch_run(m, None)
    np.testing.assert_array_equal(loss, ref["torch_loss"])
    np.testing.assert_array_equal(sample, ref["torch_sample"])
    corr = ts.fake_correction()
    res = corr.pop("correction_residual")
    loss, sample = _torch_run(m, corr)  # active correction, zero gates
    np.testing.assert_array_equal(loss, ref["torch_loss"])
    np.testing.assert_array_equal(sample, ref["torch_sample"])
    corr["correction_residual"] = res
    _, sample = _torch_run(m, corr)
    np.testing.assert_allclose(sample, ref["torch_sample"] + res, atol=1e-5)


def test_torch_gates_and_suffix_mode(ref, torch_model_factory):
    m = torch_model_factory()
    with torch.no_grad():
        for g in (m.meci_correction.gate_in, m.meci_correction.gate_ada, m.meci_correction.gate_head):
            g.fill_(0.7)
    corr = ts.fake_correction(active=(True, False))
    corr.pop("correction_residual")
    _, sample = _torch_run(m, corr)
    assert np.abs(sample[0] - ref["torch_sample"][0]).max() > 1e-5
    np.testing.assert_allclose(sample[1], ref["torch_sample"][1], atol=1e-6)

    ms = torch_model_factory(mode="suffix_tokens")
    off = ts.fake_correction(active=(False, False))
    off.pop("correction_residual")
    loss, sample = _torch_run(ms, off)
    np.testing.assert_allclose(sample, ref["torch_sample"], atol=1e-5)
    np.testing.assert_allclose(loss, ref["torch_loss"], atol=1e-5)
    on = ts.fake_correction()
    on.pop("correction_residual")
    _, sample = _torch_run(ms, on)
    assert np.abs(sample - ref["torch_sample"]).max() > 1e-5


def test_torch_training_backward_reaches_gates(torch_model_factory):
    m = torch_model_factory(aux_weight=0.1).train()
    _, _, _, _, _, actions, noise, time = ts.fake_inputs()
    loss = m(torch_obs(ts.fake_correction()), torch.from_numpy(actions), noise=torch.from_numpy(noise),
             time=torch.from_numpy(time)).mean()
    loss.backward()
    mc = m.meci_correction
    assert all(p.grad is not None and p.grad.abs().item() > 0 for p in (mc.gate_in, mc.gate_ada, mc.gate_head))
    assert mc.disp_out.weight.grad.abs().sum().item() > 0
