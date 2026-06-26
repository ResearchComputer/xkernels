#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
# Stage the runtime .so's that rocprofv3 dlopens (libdw.so.1 + deps) onto shared
# scratch. Run this ON THE LOGIN NODE (it sees the host /usr/lib64 that the
# container cannot, and shares /capstor with the compute nodes):
#
#   rcc run -- bash scripts/stage-rocprof-compute-libs-beverin.sh
#
# The profile driver mirrors these into /opt/rocm/lib per run (rocprof-compute
# resets the profiler subprocess LD_LIBRARY_PATH to the ROCm lib dir only, so
# that's the one path the loader is guaranteed to search).
set -euo pipefail
: "${LIBS:=/capstor/scratch/cscs/xyao/rocprof-compute-libs}"
mkdir -p "$LIBS"
echo "[stage] copying host elfutils/zstd libs -> $LIBS"
for pair in \
    "libdw.so.1:libdw.so.1" \
    "libelf.so.1:libelf-0.185.so" \
    "libzstd.so.0:libzstd.so.0" \
    "libbz2.so.1:libbz2.so.1" \
    "liblzma.so.5:liblzma.so.5"; do
    dst="$LIBS/${pair%%:*}"; src="/usr/lib64/${pair##*:}"
    cp -fL "$src" "$dst" && echo "  + $dst (from $src)" || echo "  ! missing on host: $src"
done
echo "[stage] staged:"; ls -1 "$LIBS"
