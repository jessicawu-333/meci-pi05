"""Shared tiny-model setup for the MECI-pi0.5 tests (CPU, dummy Gemma, Ti/16 SigLIP).

Works against both the pristine and the patched openpi tree (whichever is on PYTHONPATH).
The only test-time shim: SigLIP's hard-coded "So400m/14" variant (pi0.py l.84) is swapped for "Ti/16"
to keep CPU memory small; Gemma uses openpi's own "dummy" variant (gemma.py l.60-68).
"""
import numpy as np

H = 10  # action horizon
B = 2
TA, DA = 12, 32  # audio frames, (small) audio feature width for tests (Whisper-small would be 768)


def patch_siglip():
    import openpi.models.siglip as _siglip

    orig = _siglip.Module
    if getattr(orig, "_meci_patched", False):
        return

    def small(num_classes=None, *, variant=None, **kw):
        return orig(num_classes, variant="Ti/16", **kw)

    small._meci_patched = True
    _siglip.Module = small


def base_config_kwargs():
    return dict(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=H,
        max_token_len=16,
        dtype="float32",
    )


def fake_inputs(seed=0):
    """Deterministic non-trivial observation/actions (NumPy)."""
    rng = np.random.default_rng(seed)
    images = {k: rng.uniform(-1, 1, (B, 224, 224, 3)).astype(np.float32) for k in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")}
    masks = {k: np.ones((B,), bool) for k in images}
    state = rng.normal(size=(B, 32)).astype(np.float32)
    tok = rng.integers(0, 1000, (B, 16)).astype(np.int32)
    tok_mask = np.ones((B, 16), bool)
    tok_mask[:, 12:] = False
    actions = rng.normal(size=(B, H, 32)).astype(np.float32)
    noise = rng.normal(size=(B, H, 32)).astype(np.float32)
    time = np.array([0.3, 0.8], np.float32)
    return images, masks, state, tok, tok_mask, actions, noise, time


def fake_correction(seed=1, active=(True, True), strength=(1.0, 0.5)):
    rng = np.random.default_rng(seed)
    amask = np.ones((B, TA), bool)
    amask[1, 9:] = False
    return {
        "correction_audio": rng.normal(size=(B, TA, DA)).astype(np.float32),
        "correction_audio_mask": amask,
        "correction_active": np.array(active, bool),
        "correction_strength": np.array(strength, np.float32),
        "correction_delta": np.array([[0.0, 0.0127, 0, 0, 0, 0], [0, 0, 0.01, 0, 0, 0]], np.float32),
        "correction_effector": np.array([1, 2], np.int32),
        "correction_residual": rng.normal(scale=0.05, size=(B, H, 32)).astype(np.float32),
    }


def fix_torch_projector(model):
    """Test-only shim for the PyTorch port with the 'dummy' Gemma (width 64): gemma_pytorch.py hard-codes the
    SigLIP->LM projection to 2048 (PaliGemmaWithExpertModel.__init__, `vision_config.projection_dim = 2048`),
    which only matches gemma_2b. Replace the projector with a 64-wide one (deterministic init)."""
    import torch
    from torch import nn

    proj = model.paligemma_with_expert.paligemma.model.multi_modal_projector
    g = torch.random.get_rng_state()
    torch.manual_seed(123)
    proj.linear = nn.Linear(proj.linear.in_features, 64)  # dummy Gemma width = 64 (gemma.py l.60-68)
    torch.random.set_rng_state(g)
    return model


def randomize_adarms_jax(model, seed=5):
    """openpi zero-initialises the adaRMS modulation Dense (gemma.py l.128, kernel_init=zeros). In a *fresh*
    model that makes every action-expert residual gate 0 (gemma.py l.453-459), i.e. the expert is an identity
    map and ignores attention entirely. Pretrained pi05 weights are non-zero, so tests emulate that."""
    import flax.nnx as nnx
    import jax

    flat = nnx.state(model, nnx.Param).flat_state()
    key = jax.random.key(seed)
    n = 0
    for path in sorted(flat, key=lambda p: "/".join(map(str, p))):
        name = "/".join(map(str, path))
        if name.startswith("PaliGemma/llm/") and "_1/Dense_0/kernel" in name:
            key, k = jax.random.split(key)
            flat[path].value = jax.random.normal(k, flat[path].value.shape) * 0.05
            n += 1
    assert n == 3, n  # pre_attention_norm_1, pre_ffw_norm_1 (scanned over layers), final_norm_1
    nnx.update(model, nnx.State.from_flat_path(flat))
    return model


def randomize_adarms_torch(model, seed=5):
    """PyTorch counterpart (transformers_replace GemmaRMSNorm.dense is zero-initialised)."""
    import torch

    g = torch.random.get_rng_state()
    torch.manual_seed(seed)
    n = 0
    with torch.no_grad():
        for name, p in sorted(model.named_parameters()):
            if name.startswith("paligemma_with_expert.gemma_expert") and ".dense.weight" in name:
                p.normal_(0.0, 0.05)
                n += 1
    torch.random.set_rng_state(g)
    assert n > 0
    return model
