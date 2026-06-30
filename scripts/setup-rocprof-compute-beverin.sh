#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
# Install AMD ROCm Compute Profiler (formerly Omniperf) into PERSISTENT scratch
# on beverin. Run INSIDE the tokenspeed container env on a compute node:
#   scripts/cluster.sh run --host beverin srun --environment=tokenspeed-rocm-aiter-myofi \
#       --partition=mi300 --gpus-per-node=1 --time=00:25:00 \
#       bash -c 'cd /capstor/scratch/cscs/xyao/xkernels && bash scripts/setup-rocprof-compute-beverin.sh'
#
# Why this is not `pip install omniperf`:
#   * AMD never published Omniperf / rocprof-compute to PyPI (both 404 on the
#     simple index); the rocm-7.2.4 GitHub release ships source only.
#   * The app is pure Python and shells out to the SYSTEM rocprofv3 at
#     /usr/bin/rocprofv3 (ROCm 7.2 in the container), so we clone the matching
#     tag and `pip install -r requirements.txt` into a scratch target.
#   * The container image is read-only, so everything goes to scratch.
#
# CRITICAL pinned dep: requirements.txt asks for pandas>=1.4.3 with no upper
# bound, so uv grabs pandas 3.x. rocprof-compute 3.4.0 was written for pandas 2.x
# (object dtype); pandas 3.0's strict str dtype breaks BOTH the v3->v2 counter
# join (Agent_Id merge) and the analyze metric assignment. Pin pandas<3.
#
# Runtime .so's (libdw.so.1 etc. that rocprofv3 dlopens) are staged separately
# from the LOGIN NODE by scripts/stage-rocprof-compute-libs-beverin.sh (the
# container cannot see the host /usr/lib64). This script does NOT touch libs.
#
# Idempotent: re-running skips the clone + dependency install if already present.
set -euo pipefail

: "${TAG:=rocm-7.2.4}"                 # matches the container's ROCm 7.2.x
: "${SRC:=/capstor/scratch/cscs/xyao/rocprof-compute-src}"
: "${PYLIBS:=/capstor/scratch/cscs/xyao/rocprof-compute-pylibs}"

PY=/usr/bin/python3
UV=/usr/local/bin/uv   # container's uv (avoid the host ~/.local/bin/uv leak)

echo "[setup] ROCm Compute Profiler $TAG  (PY=$PY UV=$UV)"
mkdir -p "$(dirname "$SRC")"

# --- 1. fetch source tarball (container git/libcurl is ABI-broken; python
#         urllib reaches codeload.github.com fine) ----------------------------
if [ ! -f "$SRC/VERSION" ] && [ ! -f "$SRC/pyproject.toml" ]; then
    echo "[setup] fetching source tarball for $TAG ..."
    tmp="$(mktemp -d)"
    "$PY" - "$TAG" "$tmp/src.tar.gz" <<'PYEOF'
import sys, urllib.request
tag, out = sys.argv[1], sys.argv[2]
url = f"https://codeload.github.com/ROCm/omniperf/tar.gz/refs/tags/{tag}"
with urllib.request.urlopen(url, timeout=60) as r, open(out, "wb") as f:
    f.write(r.read())
print(f"  fetched {__import__('os').path.getsize(out)} bytes")
PYEOF
    mkdir -p "$SRC"
    tar -xzf "$tmp/src.tar.gz" -C "$SRC" --strip-components=1
    rm -rf "$tmp"
else
    echo "[setup] source already present at $SRC"
fi
echo "[setup] VERSION: $(cat "$SRC/VERSION" 2>/dev/null || echo '?')"

# --- 2. install deps into the persistent scratch target, pinning pandas<3 ---
#      (the .installed sentinel records the pin so a re-run after a source bump
#       still re-resolves deps; delete $PYLIBS to force a clean reinstall.)
if [ ! -f "$PYLIBS/.installed" ] || [ "$(cat "$PYLIBS/.installed" 2>/dev/null)" != "$TAG pandas<3" ]; then
    echo "[setup] installing requirements (pandas<3) into $PYLIBS ..."
    mkdir -p "$PYLIBS"
    "$UV" pip install --python "$PY" --target="$PYLIBS" -r "$SRC/requirements.txt" "pandas<3"
    echo "$TAG pandas<3" > "$PYLIBS/.installed"
else
    echo "[setup] deps already installed at $PYLIBS ($(cat "$PYLIBS/.installed"))"
fi

# --- 3. smoke test: the CLI runs. ------------------------------------------
export PYTHONPATH="$PYLIBS:$SRC/src:${PYTHONPATH:-}"
echo "[setup] CLI version:"
"$PY" "$SRC/src/rocprof-compute" --version | head -4
echo "[setup] pandas: $("$PY" -c 'import pandas;print(pandas.__version__)' 2>&1)"

cat <<EOF

[setup] DONE. Next steps:
  1. Stage runtime libs from the LOGIN NODE (one-time):
       bash scripts/stage-rocprof-compute-libs-beverin.sh
  2. Profile a kernel:
       bash scripts/profile-rocprof-compute-beverin.sh dual_rmsnorm roof

  CLI launcher: python3 $SRC/src/rocprof-compute
  System rocprofv3: /usr/bin/rocprofv3 (ROCm 7.2)
EOF
