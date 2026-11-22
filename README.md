# MECI audio-correction adapter for π0.5 (openpi @ 215abfb)

This is a proposed design plus code. Nothing has been trained, and no results are claimed. See `DESIGN.md`.

Apply to openpi:

```bash
git clone https://github.com/Physical-Intelligence/openpi && cd openpi
git checkout 215abfb217dbac7d5f1273282331b9b1866c0479
patch -p1 < /path/to/audio_correction_pi05/patches/openpi-pi05-meci.diff
```

Enable it: `pi0_config.Pi0Config(pi05=True, meci=meci_correction.MECIConfig())`. With `meci=None` (the default) the model is unchanged.

Run the tests (CPU). You need a pristine and a patched copy of `src/openpi`, plus `packages/openpi-client/src`:

```bash
OPENPI_PRISTINE=/path/pristine OPENPI_PATCHED=/path/patched OPENPI_CLIENT=/path/openpi/packages/openpi-client/src \
  tests/run_tests.sh
```

The venv used here had Python 3.11, jax 0.5.3 (CPU), flax 0.10.2, torch 2.7.1+cpu, transformers 4.53.2 with openpi's `transformers_replace` copied in, plus einops, jaxtyping 0.2.36, beartype 0.19.0, augmax, orbax-checkpoint 0.11.13, numpydantic, tqdm-loggable, pillow, and pytest.
