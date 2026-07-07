#!/usr/bin/env bash
# TEMP (untracked) — minimal verify for PR #77 §5 fix on beverin/ds5.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
echo "### host: $(hostname)"
python3 - <<'PY'
import torch
print("torch", torch.__version__, "hip", getattr(torch.version,"hip",None), "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
if torch.cuda.is_available():
    print("dev0", torch.cuda.get_device_name(0))
PY
echo "### pytest: full PR file (29 tests, incl. the 2 GPU-gated §5/§6)"
python3 -m pytest -q tests/test_vkl_schedule_spine.py
echo "### DONE"
