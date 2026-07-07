#!/usr/bin/env bash
# TEMP (untracked) — GPU probe for PR #77 on beverin/ds5. Deleted after the run.
# Runs the PR file + immediate logic neighbors, and the GPU-launching lower tests.
# Self-locates its repo root so it can be invoked by absolute path (avoids the
# rcc/ssh quote-stripping that eats `bash -c '... && ...'`).
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
echo "### host: $(hostname)"
echo "### repo: $(pwd)"
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
python3 - <<'PY'
import os, sys
try:
    import torch
    print("torch", torch.__version__)
    print("hip ", torch.version.hip)
    print("cuda", torch.version.cuda)
    print("cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        try:
            print("dev0", torch.cuda.get_device_name(0))
        except Exception as e:
            print("dev0 name err:", e)
except Exception as e:
    print("torch import err:", e)
print("PYTHONPATH", os.environ.get("PYTHONPATH"))
try:
    import xkernels
    print("xkernels", xkernels.__file__)
except Exception as e:
    print("xkernels import err:", e)
    sys.exit(0)
PY
echo "### pytest: PR file (verbose, incl. the 2 GPU-gated tests)"
python3 -m pytest -v tests/test_vkl_schedule_spine.py
echo "### pytest: immediate logic neighbors (quiet)"
python3 -m pytest -q tests/test_vkl_edits.py tests/test_vkl_gate.py tests/test_vkl_cost.py tests/test_vkl_override.py
echo "### pytest: GPU-launching lower/override neighbors"
python3 -m pytest -q tests/test_vkl_lower_gemm.py tests/test_vkl_lower_triton.py 2>&1 | tail -30
echo "### DONE"
