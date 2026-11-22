"""Run UNPATCHED openpi (pristine tree on PYTHONPATH) and save reference outputs to an .npz.

Usage: PYTHONPATH=<pristine>/src:<tests dir> python reference_pristine.py out.npz
"""
import sys

import jax
import jax.numpy as jnp
import numpy as np
import torch

import tiny_setup as ts

ts.patch_siglip()
from openpi.models import model as _model  # noqa: E402
from openpi.models import pi0_config  # noqa: E402

out = {}
images, masks, state, tok, tok_mask, actions, noise, time = ts.fake_inputs()
cfg = pi0_config.Pi0Config(**ts.base_config_kwargs())
m = ts.randomize_adarms_jax(cfg.create(jax.random.key(0)))
obs = _model.Observation(images={k: jnp.asarray(v) for k, v in images.items()}, image_masks={k: jnp.asarray(v) for k, v in masks.items()},
                         state=jnp.asarray(state), tokenized_prompt=jnp.asarray(tok), tokenized_prompt_mask=jnp.asarray(tok_mask))
out["jax_loss"] = np.asarray(m.compute_loss(jax.random.key(1), obs, jnp.asarray(actions)))
out["jax_sample"] = np.asarray(m.sample_actions(jax.random.key(2), obs, num_steps=3, noise=jnp.asarray(noise)))

from openpi.models_pytorch import pi0_pytorch  # noqa: E402

cfg_t = pi0_config.Pi0Config(**ts.base_config_kwargs(), pytorch_compile_mode=None)
torch.manual_seed(0)
mt = ts.randomize_adarms_torch(ts.fix_torch_projector(pi0_pytorch.PI0Pytorch(cfg_t))).eval()
obs_t = _model.Observation(images={k: torch.from_numpy(v).permute(0, 3, 1, 2).contiguous() for k, v in images.items()}, image_masks={k: torch.from_numpy(v) for k, v in masks.items()},
                           state=torch.from_numpy(state), tokenized_prompt=torch.from_numpy(tok).long(), tokenized_prompt_mask=torch.from_numpy(tok_mask))
with torch.no_grad():
    torch.manual_seed(7)  # forward() applies train-time image augmentation (preprocessing_pytorch, train=True)
    out["torch_loss"] = mt(obs_t, torch.from_numpy(actions), noise=torch.from_numpy(noise), time=torch.from_numpy(time)).numpy()
    out["torch_sample"] = mt.sample_actions("cpu", obs_t, noise=torch.from_numpy(noise), num_steps=3).numpy()
torch.save(mt.state_dict(), sys.argv[1] + ".torch_state.pt")
np.savez(sys.argv[1], **out)
print("saved", {k: v.shape for k, v in out.items()})
