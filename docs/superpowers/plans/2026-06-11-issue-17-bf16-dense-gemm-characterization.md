# bf16 dense-GEMM characterization (issue #17, Phase 1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `benchmarks/probe_ffn.py` to characterize which bf16 dense-Linear shapes miss the MFMA/hipBLASLt fast path on gfx942, whether any blas-routing mode restores it, and the Triton `tl.dot` bf16 ceiling — then run it on MI300A and write a findings doc with a Phase-2 recommendation.

**Architecture:** A pure-diagnostic extension to the existing probe: per-shape `F.linear` timing in fp16 vs bf16 (TFLOP/s + ratio + MISS flag), a `--mode` blas switch (`default`/`hipblaslt`/`no-hipblaslt`/`tunableop`) applied per-process, and a single-config Triton bf16 GEMM as the ceiling column. A SLURM job runs all modes; findings land in `docs/issue-17-bf16-dense-gemm.md`. No `src/` changes, no backend registered (that's Phase 2).

**Tech Stack:** Python 3.11, Triton 3.7 (ROCm fork on device), PyTorch (`F.linear`, `torch.backends.cuda`, `torch.cuda.tunable`), SLURM/enroot on CSCS beverin (MI300A, gfx942).

**Local commands:** Python via `.venv/bin/python`; tests via `TRITON_INTERPRET=1 .venv/bin/python -m pytest ...`; lint via `.venv/bin/ruff check .`. (Local box has no GPU — the sweep itself is GPU-only and runs on beverin; unit tests use the interpreter / CPU.)

---

## File structure

- **Modify** `benchmarks/probe_ffn.py` — add `_tflops`, `_apply_blas_mode`, a Triton bf16 GEMM (`_gemm_kernel` + `_triton_gemm`), `_bench_linear`/`_bench_triton`, the `SHAPES` table, `sweep`, and a new `main` with `--mode`/`--regime`. Keep the existing `_ms` timer.
- **Create** `tests/test_probe_dense_bf16.py` — unit tests for the pure helpers, the Triton GEMM (interpreter, fp32), and `_apply_blas_mode` (skipped without triton).
- **Create** `slurm/probe_dense_bf16_beverin.sbatch` — runs the probe across all four modes on the `mi300` partition, one combined log.
- **Create** `docs/issue-17-bf16-dense-gemm.md` — the characterization findings + Phase-2 recommendation (filled from the on-device run in Task 4).

---

### Task 1: Pure helpers, Triton bf16 GEMM, blas-mode switch

**Files:**
- Modify: `benchmarks/probe_ffn.py`
- Test: `tests/test_probe_dense_bf16.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_probe_dense_bf16.py`:

```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Unit tests for the bf16 dense-GEMM probe helpers (issue #17).

Pure / interpreter-level (no GPU). Skipped where Triton is absent because the
probe defines a @triton.jit kernel at import.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

pytest.importorskip("triton")

# probe_ffn lives in benchmarks/, which isn't an installed package.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from benchmarks import probe_ffn as P  # noqa: E402


def test_tflops_math():
    # 2*M*K*N flops; M=K=N=1000 -> 2e9 flops; at 1.0 ms -> 2.0 TFLOP/s
    assert abs(P._tflops(1.0, 1000, 1000, 1000) - 2.0) < 1e-9
    assert P._tflops(0.0, 1, 1, 1) == 0.0  # guard divide-by-zero


@pytest.mark.parametrize("mode", ["default", "hipblaslt", "no-hipblaslt", "tunableop"])
def test_apply_blas_mode_no_raise(mode):
    state = P._apply_blas_mode(mode)
    assert isinstance(state, dict)
    assert state["mode"] == mode


def test_triton_gemm_matches_torch():
    import torch

    # fp32 inputs: the Triton CPU interpreter mis-evaluates bf16 tl.dot, but the
    # tiling/masking/accumulate path is identical, so fp32 validates correctness.
    torch.manual_seed(0)
    M, K, N = 37, 70, 50  # non-tile-aligned -> exercises masking
    a = torch.randn(M, K)
    b = torch.randn(K, N)
    got = P._triton_gemm(a, b)
    ref = a @ b
    torch.testing.assert_close(got, ref, atol=1e-3, rtol=1e-3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `TRITON_INTERPRET=1 .venv/bin/python -m pytest tests/test_probe_dense_bf16.py -v`
Expected: FAIL/ERROR — `AttributeError: module 'benchmarks.probe_ffn' has no attribute '_tflops'` (helpers not defined yet).

- [ ] **Step 3: Add the helpers + kernel to `probe_ffn.py`**

Replace the current header (the module docstring + imports through the `_ms` function) of `benchmarks/probe_ffn.py` with the following, keeping `_ms` intact:

```python
# SPDX-License-Identifier: MIT
"""Diagnose dense bf16 GEMM speed on gfx942: which Linear shapes miss the
MFMA/hipBLASLt fast path, whether a blas-routing mode restores it, and the
Triton tl.dot bf16 ceiling (issue #17, phase 1)."""
from __future__ import annotations

import argparse
import os

import torch
import triton
import triton.language as tl


def _ms(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def _tflops(ms: float, M: int, K: int, N: int) -> float:
    """GEMM throughput in TFLOP/s (2*M*K*N flops). 0.0 if ms is non-positive."""
    if ms <= 0:
        return 0.0
    return 2 * M * K * N / (ms * 1e-3) / 1e12


def _apply_blas_mode(mode: str) -> dict:
    """Apply a bf16 GEMM routing mode and report the resolved state.

    Runtime-settable knobs are applied here; import-time env knobs
    (``TORCH_BLAS_PREFER_HIPBLASLT``, ``PYTORCH_TUNABLEOP_ENABLED``) are set by
    the SLURM job per invocation and only reported. Never raises — unavailable
    knobs are recorded in the returned dict so the log stays honest.
    """
    state: dict = {"mode": mode}
    setlib = getattr(torch.backends.cuda, "preferred_blas_library", None)
    if mode == "hipblaslt" and callable(setlib):
        try:
            setlib("hipblaslt")
        except Exception as exc:
            state["setlib_err"] = str(exc)[:80]
    elif mode == "no-hipblaslt" and callable(setlib):
        try:
            setlib("cublas")  # maps to rocBLAS on ROCm
        except Exception as exc:
            state["setlib_err"] = str(exc)[:80]
    if mode == "tunableop":
        try:
            torch.cuda.tunable.enable(True)
            torch.cuda.tunable.tuning_enable(True)
        except Exception as exc:
            state["tunable_err"] = str(exc)[:80]
    try:
        if callable(setlib):
            state["preferred_blas"] = str(setlib())
    except Exception as exc:
        state["preferred_blas_err"] = str(exc)[:80]
    try:
        state["tunable_enabled"] = bool(torch.cuda.tunable.is_enabled())
    except Exception:
        pass
    state["env_TORCH_BLAS_PREFER_HIPBLASLT"] = os.environ.get("TORCH_BLAS_PREFER_HIPBLASLT")
    state["env_PYTORCH_TUNABLEOP_ENABLED"] = os.environ.get("PYTORCH_TUNABLEOP_ENABLED")
    return state


@triton.jit
def _gemm_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    sam, sak, sbk, sbn, scm, scn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BN)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m[:, None] * sam + offs_k[None, :] * sak
    b_ptrs = b_ptr + offs_k[:, None] * sbk + offs_n[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BK)):
        krem = K - k0 * BK
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < krem), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < krem) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc=acc)
        a_ptrs += BK * sak
        b_ptrs += BK * sbk
    c = acc.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _triton_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C = a @ b for a [M,K], b [K,N]; fp32 accumulate, single fixed tile.

    A ceiling reference for the bf16 MFMA path (issue #17) — not a registered
    backend. Works for any float dtype (tested in fp32 under the interpreter).
    """
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    BM, BN, BK = 64, 128, 64
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _gemm_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=2,
    )
    return c
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `TRITON_INTERPRET=1 .venv/bin/python -m pytest tests/test_probe_dense_bf16.py -v`
Expected: PASS (6 tests: 1 tflops, 4 modes, 1 gemm). The gemm test runs the kernel under the CPU interpreter in fp32.

- [ ] **Step 5: Lint**

Run: `.venv/bin/ruff check benchmarks/probe_ffn.py tests/test_probe_dense_bf16.py`
Expected: no errors. (Note: the old `main` still references removed locals at this point — Task 2 replaces it; if ruff flags unused names in the old `main`, that is fixed in Task 2. If lint fails only inside the old `main`, proceed to Task 2 before committing.)

- [ ] **Step 6: Commit**

```bash
git add benchmarks/probe_ffn.py tests/test_probe_dense_bf16.py
git commit -m "feat(bench): bf16 GEMM probe helpers + Triton ceiling + blas-mode switch (issue #17)"
```

---

### Task 2: Dense sweep + new `main`

**Files:**
- Modify: `benchmarks/probe_ffn.py` (replace the old `main` and the `if __name__` block)

- [ ] **Step 1: Replace the old `main` body with the sweep + new `main`**

In `benchmarks/probe_ffn.py`, delete the existing `def main(): ... ` (the fp16/bf16 matmul + reference-FFN probe) and its `if __name__ == "__main__": main()` footer, and replace with:

```python
# Kimi-K2.6 dense / MLA / shared-expert / head shapes, as (K -> N). F.linear
# weight is [N, K]; flops = 2*M*K*N. lm_head N is a representative large tile.
SHAPES = [
    ("q_a_proj", 7168, 1536),
    ("kv_a_proj", 7168, 576),
    ("shexp_gate_up", 7168, 2048),
    ("shexp_down", 2048, 7168),
    ("lm_head", 7168, 32768),
    ("ffn_gate_up", 4096, 11008),
]
DECODE_M = [1, 2, 4, 8, 16, 32]
PREFILL_M = [512, 2048, 4096]


def _bench_linear(M, K, N, dtype):
    """Time F.linear (x[M,K] @ W[N,K]^T) and return (ms, TFLOP/s)."""
    import torch.nn.functional as F

    x = torch.randn(M, K, device="cuda", dtype=dtype)
    w = torch.randn(N, K, device="cuda", dtype=dtype)
    ms = _ms(lambda: F.linear(x, w))
    return ms, _tflops(ms, M, K, N)


def _bench_triton(M, K, N):
    """Time the single-tile Triton bf16 GEMM and return (ms, TFLOP/s)."""
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    ms = _ms(lambda: _triton_gemm(a, b))
    return ms, _tflops(ms, M, K, N)


def _sanity_check_triton():
    """One small bf16 check vs torch before timing, so a wrong kernel can't pass."""
    a = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(128, 96, device="cuda", dtype=torch.bfloat16)
    got = _triton_gemm(a, b).float()
    ref = (a.float() @ b.float())
    ok = torch.allclose(got, ref, atol=2e-1, rtol=2e-2)
    print(f"# triton bf16 GEMM sanity vs torch: {'OK' if ok else 'MISMATCH'} "
          f"(max|err|={(got - ref).abs().max().item():.3f})")


def sweep(mode, Ms):
    state = _apply_blas_mode(mode)
    print(f"\n# ==== mode={mode} ====")
    print(f"# state: {state}")
    print(f"{'shape':14} {'M':>5} {'K':>6} {'N':>6} "
          f"{'fp16':>8} {'bf16':>8} {'tritbf16':>9} {'bf16/fp16':>10} flag   (TFLOP/s)")
    for tag, K, N in SHAPES:
        for M in Ms:
            try:
                _, f16 = _bench_linear(M, K, N, torch.float16)
                _, b16 = _bench_linear(M, K, N, torch.bfloat16)
                _, tb16 = _bench_triton(M, K, N)
            except Exception as exc:  # OOM / unsupported -> record, keep sweeping
                print(f"{tag:14} {M:5d} {K:6d} {N:6d}  SKIP: {str(exc)[:48]}")
                continue
            ratio = b16 / f16 if f16 > 0 else 0.0
            flag = "MISS" if ratio < 0.5 else ""
            print(f"{tag:14} {M:5d} {K:6d} {N:6d} "
                  f"{f16:8.1f} {b16:8.1f} {tb16:9.1f} {ratio:10.2f} {flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode", default="default",
        choices=["default", "hipblaslt", "no-hipblaslt", "tunableop"],
    )
    ap.add_argument("--regime", default="all", choices=["decode", "prefill", "all"])
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("No GPU; the dense bf16 GEMM sweep requires gfx942 (or any CUDA/ROCm GPU).")
        return

    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
    _sanity_check_triton()
    Ms = []
    if args.regime in ("decode", "all"):
        Ms += DECODE_M
    if args.regime in ("prefill", "all"):
        Ms += PREFILL_M
    sweep(args.mode, Ms)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the no-GPU guard + lint**

Run: `.venv/bin/python benchmarks/probe_ffn.py`
Expected: prints `No GPU; the dense bf16 GEMM sweep requires gfx942 ...` and exits 0.
Run: `.venv/bin/ruff check benchmarks/probe_ffn.py`
Expected: no errors.

- [ ] **Step 3: Re-run unit tests (still green after the edit)**

Run: `TRITON_INTERPRET=1 .venv/bin/python -m pytest tests/test_probe_dense_bf16.py -v`
Expected: PASS (6).

- [ ] **Step 4: Commit**

```bash
git add benchmarks/probe_ffn.py
git commit -m "feat(bench): dense-Linear bf16-vs-fp16 sweep across blas modes (issue #17)"
```

---

### Task 3: SLURM job (all modes)

**Files:**
- Create: `slurm/probe_dense_bf16_beverin.sbatch`

- [ ] **Step 1: Write the SLURM script**

Create `slurm/probe_dense_bf16_beverin.sbatch`:

```bash
#!/bin/bash
# SPDX-License-Identifier: MIT
# Characterize bf16 dense-GEMM MFMA/hipBLASLt routing on beverin (gfx942 /
# MI300A): run benchmarks/probe_ffn.py across all blas modes (issue #17).
#
#   sbatch slurm/probe_dense_bf16_beverin.sbatch
#
#SBATCH --job-name=xk-probe-bf16
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=00:30:00
#SBATCH --output=probe-bf16-%j.out
#SBATCH --error=probe-bf16-%j.out

set -uo pipefail

REPO="${REPO:-/capstor/scratch/cscs/xyao/kernels}"
ENV_NAME="${ENV_NAME:-tokenspeed-rocm-aiter-myofi}"

echo "REPO=$REPO ENV=$ENV_NAME node=$(hostname)"

srun --environment="$ENV_NAME" --cpu-bind=none bash -c '
  set -e
  unset ROCR_VISIBLE_DEVICES || true
  export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="'"$REPO"'/src:${PYTHONPATH:-}"
  PROBE="'"$REPO"'/benchmarks/probe_ffn.py"

  echo "######## mode=default ########"
  python -u "$PROBE" --mode default

  echo "######## mode=hipblaslt ########"
  TORCH_BLAS_PREFER_HIPBLASLT=1 python -u "$PROBE" --mode hipblaslt

  echo "######## mode=no-hipblaslt ########"
  TORCH_BLAS_PREFER_HIPBLASLT=0 python -u "$PROBE" --mode no-hipblaslt

  echo "######## mode=tunableop ########"
  PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=1 python -u "$PROBE" --mode tunableop
'
```

- [ ] **Step 2: Commit**

```bash
git add slurm/probe_dense_bf16_beverin.sbatch
git commit -m "chore(bench): SLURM job to probe bf16 dense GEMM across blas modes (issue #17)"
```

---

### Task 4: Run on beverin, write findings, report

This task runs on the cluster; no TDD. Each step is a real command.

- [ ] **Step 1: Sync the branch to beverin scratch**

Run (from repo root):
```bash
rsync -az --delete \
  --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
  --exclude '.ruff_cache' --exclude '.pytest_cache' \
  ./ beverin:/capstor/scratch/cscs/xyao/kernels/
```
Expected: completes without error. (Renew the CSCS SSH cert at https://sshservice.cscs.ch/ if it expired.)

- [ ] **Step 2: Submit the probe job**

```bash
ssh beverin 'cd /capstor/scratch/cscs/xyao/kernels && sbatch slurm/probe_dense_bf16_beverin.sbatch'
```
Expected: `Submitted batch job <JOBID>`.

- [ ] **Step 3: Wait and read the log**

Poll until the job leaves the queue, then read the full log:
```bash
ssh beverin 'squeue -j <JOBID> -h -o %T; tail -n 200 /capstor/scratch/cscs/xyao/kernels/probe-bf16-<JOBID>.out'
```
Expected: four mode sections, each a TFLOP/s table with a `bf16/fp16` ratio and `MISS` flags, plus the per-mode `state` line and the Triton sanity line. Confirm `default` reproduces the cliff (bf16 ≪ fp16 on at least the FFN-like shapes) and note which modes (if any) raise the bf16 ratio toward 1.0, and the Triton bf16 ceiling column.

- [ ] **Step 4: Write the findings doc**

Create `docs/issue-17-bf16-dense-gemm.md` containing, with the actual numbers from Step 3:
- A one-paragraph result summary (is the cliff systemic across all dense shapes, or shape-specific?).
- The per-mode TFLOP/s tables (paste the decode + a prefill row per shape), highlighting which bf16 shapes are MISS in `default` and whether `hipblaslt` / `tunableop` clear the flag.
- The Triton bf16 ceiling vs the best torch bf16 result.
- A **Phase-2 recommendation** chosen by the data:
  - if a blas mode restores MFMA → document the env/`torch.backends` remedy (cheapest; no kernel), with the exact knob and its measured effect;
  - else → ship the Triton `tl.dot` bf16 GEMM as a registered backend / `F.linear` drop-in, citing the ceiling vs torch gap.
- Follow the structure of `docs/issue-12-hierarchical-all-reduce.md` (problem → method → data → conclusion).

- [ ] **Step 5: Commit the findings**

```bash
git add docs/issue-17-bf16-dense-gemm.md
git commit -m "docs(ffn): bf16 dense-GEMM MFMA characterization on MI300A (issue #17)"
```

- [ ] **Step 6: Push, open PR, report on the issue**

```bash
git push -u origin issue-17-bf16-dense-gemm
gh pr create --repo ResearchComputer/kernels --base main \
  --title "bench(ffn): characterize bf16 dense-GEMM MFMA/hipBLASLt cliff on gfx942 (issue #17 phase 1)" \
  --body "<summary: probe extension + per-mode tables + Triton ceiling + Phase-2 recommendation; references #17>"
```
Then comment on issue #17 with the per-mode table summary, which shapes miss MFMA, whether an env/mode fixes it, the Triton ceiling, and the concrete Phase-2 recommendation. (Squash-merge per repo convention once reviewed.)

---

## Self-review

- **Spec coverage:** dense-shape sweep (Task 2, `SHAPES`/`sweep`) ✓; M decode+prefill (Task 2, `DECODE_M`/`PREFILL_M`) ✓; fp16-vs-bf16 ratio + MISS flag (Task 2, `sweep`) ✓; blas modes (Task 1, `_apply_blas_mode` + Task 3 env per invocation) ✓; Triton bf16 ceiling (Task 1, `_triton_gemm` + Task 2 `_bench_triton`) ✓; SLURM all-modes job (Task 3) ✓; OOM/edge handling (Task 2 `sweep` try/except; Task 1 mode guards) ✓; correctness sanity before timing (Task 2 `_sanity_check_triton`; Task 1 interpreter test) ✓; findings doc + recommendation + issue report (Task 4) ✓. Phase-2 fix correctly deferred (spec out-of-scope) ✓.
- **Placeholders:** none — all code/commands concrete. `<JOBID>` and the PR/findings prose are runtime values filled at execution (the findings doc's structure and required contents are fully specified).
- **Type/name consistency:** `_tflops`, `_apply_blas_mode`, `_gemm_kernel`, `_triton_gemm`, `_bench_linear`, `_bench_triton`, `_sanity_check_triton`, `sweep`, `SHAPES`, `DECODE_M`, `PREFILL_M`, `_ms` are used identically across Tasks 1–3. `_apply_blas_mode` returns a dict with key `mode` (asserted in the test). `_triton_gemm(a, b)` takes a=[M,K], b=[K,N] consistently in test, `_bench_triton`, and `_sanity_check_triton`.
