#!/usr/bin/env bash
# Run the MECI-pi0.5 tests on CPU. Usage: OPENPI_PRISTINE=... OPENPI_PATCHED=... OPENPI_CLIENT=... ./run_tests.sh
# (JAX and PyTorch groups run in separate processes to keep peak RAM low on small boxes.)
set -euo pipefail
cd "$(dirname "$0")"
REF=${MECI_REF:-/tmp/pristine_ref.npz}
PYTHONPATH="$OPENPI_PRISTINE/src:$OPENPI_CLIENT:." python reference_pristine.py "$REF"
export MECI_REF="$REF" PYTHONPATH="$OPENPI_PATCHED/src:$OPENPI_CLIENT:."
python -m pytest -q -p no:cacheprovider -W ignore test_meci_pi05.py -k "not torch" -rA --tb=short
python -m pytest -q -p no:cacheprovider -W ignore test_meci_pi05.py -k "torch" -rA --tb=short
