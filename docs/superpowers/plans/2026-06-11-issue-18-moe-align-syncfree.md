# Sync-free `moe_align_block_size` mode (issue #18) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `truncate=False` mode to `moe_align_block_size` that skips the `num_post.item()` host sync and returns a fixed `max_blocks`-length `expert_ids` (unused tail = sentinel 0), so the Triton align is HIP-graph-capturable on the decode hot path.

**Architecture:** A `truncate: bool = True` flag threaded through the dispatch and both backends. The Triton kernel already computes `expert_ids` for all `max_blocks`; `truncate=False` returns it whole (no `.item()`, no data-dependent slice), and `_align_expert_ids` maps the natural "one-past-range" value of unused trailing blocks to sentinel 0. The reference gains the same flag for parity testing. `truncate=True` stays the default — every existing caller is byte-for-byte unchanged.

**Tech Stack:** Python 3.11, Triton 3.7 (ROCm fork on device), PyTorch (`torch.cuda.graph`), SLURM/enroot on CSCS beverin (MI300A, gfx942).

**Local commands:** Python via `.venv/bin/python`; tests via `TRITON_INTERPRET=1 .venv/bin/python -m pytest ...`; lint via `.venv/bin/ruff check .`.

---

## File structure

- **Modify** `src/xkernels/ops/moe/w4a16.py` — `moe_align_block_size_ref` gains `truncate=True`; when `False`, pad `expert_ids` to `max_blocks` with sentinel 0.
- **Modify** `src/xkernels/ops/moe/triton/align_kernel.py` — `_align_expert_ids` sentinels unused blocks; `moe_align_block_size_triton` gains `truncate=True` (branch the `.item()`+slice).
- **Modify** `src/xkernels/ops/moe/align.py` — `moe_align_block_size(..., truncate=True)` threads the kwarg through `dispatch`.
- **Modify** `tests/test_moe_align_block_size.py` — add `truncate=False` fixed-shape + parity + tail-sentinel tests.
- **Create** `benchmarks/probe_align_capture.py` — capture/replay proof.
- **Create** `slurm/probe_align_capture_beverin.sbatch` — runs the proof on MI300A.

---

### Task 1: Reference `truncate` flag

**Files:**
- Modify: `src/xkernels/ops/moe/w4a16.py`
- Test: `tests/test_moe_align_block_size.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_moe_align_block_size.py`:

```python
@pytest.mark.parametrize(
    "M,top_k,num_experts,block_size",
    [(8, 2, 4, 4), (16, 8, 48, 16), (1, 8, 48, 16), (7, 3, 5, 4)],
)
def test_reference_truncate_false_fixed_shape(M, top_k, num_experts, block_size):
    topk_ids = _make_topk_ids(M, top_k, num_experts)
    s_t, e_t, n_t = moe_align_block_size_ref(topk_ids, block_size, num_experts)  # truncate=True
    s_f, e_f, n_f = moe_align_block_size_ref(topk_ids, block_size, num_experts, truncate=False)
    total = M * top_k
    max_pad = total + (num_experts + 1) * (block_size - 1)
    max_blocks = (max_pad + block_size - 1) // block_size
    used = int(n_f.item()) // block_size
    assert e_f.numel() == max_blocks                       # fixed shape
    torch.testing.assert_close(s_f, s_t, rtol=0, atol=0)   # sorted_ids unchanged
    torch.testing.assert_close(n_f, n_t, rtol=0, atol=0)   # num_post unchanged
    torch.testing.assert_close(e_f[:used], e_t, rtol=0, atol=0)  # used prefix matches
    assert torch.all(e_f[used:] == 0)                      # tail sentinel
```

- [ ] **Step 2: Run to verify it fails**

Run: `TRITON_INTERPRET=1 .venv/bin/python -m pytest "tests/test_moe_align_block_size.py::test_reference_truncate_false_fixed_shape" -q`
Expected: FAIL — `moe_align_block_size_ref()` got an unexpected keyword argument `truncate`.

- [ ] **Step 3: Add `truncate` to the reference**

In `src/xkernels/ops/moe/w4a16.py`, change the `moe_align_block_size_ref` signature and the return. Replace the signature line:

```python
def moe_align_block_size_ref(topk_ids: torch.Tensor, block_size: int, num_experts: int):
```

with:

```python
def moe_align_block_size_ref(
    topk_ids: torch.Tensor, block_size: int, num_experts: int, truncate: bool = True
):
```

Then replace the final `return (...)` block:

```python
    if not expert_ids:
        expert_ids = [0]
    return (
        sorted_ids,
        torch.tensor(expert_ids, dtype=torch.int32, device=topk_ids.device),
        torch.tensor([w], dtype=torch.int32, device=topk_ids.device),
    )
```

with:

```python
    if not expert_ids:
        expert_ids = [0]
    if not truncate:
        # Fixed-shape mode: pad to the full block count with sentinel 0 so the
        # output is graph-shaped. Unused blocks are past num_tokens_post_padded
        # and are never read by the GEMM consumer.
        max_blocks = (max_pad + block_size - 1) // block_size
        expert_ids = expert_ids + [0] * (max_blocks - len(expert_ids))
    return (
        sorted_ids,
        torch.tensor(expert_ids, dtype=torch.int32, device=topk_ids.device),
        torch.tensor([w], dtype=torch.int32, device=topk_ids.device),
    )
```

Also update the docstring `Returns:` line to note `truncate`. Find:

```python
    Returns ``(sorted_token_ids, expert_ids, num_tokens_post_padded)``.
```

Replace with:

```python
    With ``truncate=True`` (default) ``expert_ids`` is trimmed to the used blocks;
    with ``truncate=False`` it is the full ``max_blocks`` length, unused trailing
    blocks set to sentinel 0 (fixed-shape, for graph capture).

    Returns ``(sorted_token_ids, expert_ids, num_tokens_post_padded)``.
```

- [ ] **Step 4: Run to verify it passes**

Run: `TRITON_INTERPRET=1 .venv/bin/python -m pytest "tests/test_moe_align_block_size.py::test_reference_truncate_false_fixed_shape" -q`
Expected: PASS (4).

- [ ] **Step 5: Lint + commit**

```bash
.venv/bin/ruff check src/xkernels/ops/moe/w4a16.py tests/test_moe_align_block_size.py
git add src/xkernels/ops/moe/w4a16.py tests/test_moe_align_block_size.py
git commit -m "feat(moe): truncate flag for moe_align_block_size_ref (issue #18)"
```

---

### Task 2: Triton sentinel + `truncate` + dispatch passthrough

**Files:**
- Modify: `src/xkernels/ops/moe/triton/align_kernel.py`
- Modify: `src/xkernels/ops/moe/align.py`
- Test: `tests/test_moe_align_block_size.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_moe_align_block_size.py`:

```python
@pytest.mark.parametrize(
    "M,top_k,num_experts,block_size",
    [(8, 2, 4, 4), (16, 8, 48, 16), (1, 8, 48, 16), (7, 3, 5, 4), (64, 2, 16, 32)],
)
def test_triton_truncate_false(M, top_k, num_experts, block_size):
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered")
    dev = _device()
    topk_ids = _make_topk_ids(M, top_k, num_experts, device=dev)
    s_t, e_t, n_t = moe_align_block_size(topk_ids, block_size, num_experts, backend=Backend.TRITON)
    s_f, e_f, n_f = moe_align_block_size(
        topk_ids, block_size, num_experts, backend=Backend.TRITON, truncate=False
    )
    total = M * top_k
    max_pad = total + (num_experts + 1) * (block_size - 1)
    max_blocks = (max_pad + block_size - 1) // block_size
    used = int(n_f.item()) // block_size
    assert e_f.numel() == max_blocks                       # fixed shape
    torch.testing.assert_close(s_f, s_t, rtol=0, atol=0)   # sorted_ids unchanged
    torch.testing.assert_close(n_f, n_t, rtol=0, atol=0)   # num_post unchanged
    torch.testing.assert_close(e_f[:used], e_t, rtol=0, atol=0)  # used prefix matches truncate=True
    assert torch.all(e_f[used:] == 0)                      # tail sentinel
    # full triton output equals full reference output in fixed-shape mode
    s_r, e_r, n_r = moe_align_block_size_ref(topk_ids, block_size, num_experts, truncate=False)
    torch.testing.assert_close(e_f, e_r, rtol=0, atol=0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `TRITON_INTERPRET=1 .venv/bin/python -m pytest "tests/test_moe_align_block_size.py::test_triton_truncate_false" -q`
Expected: FAIL — `moe_align_block_size_triton()` got an unexpected keyword argument `truncate` (the dispatch forwards it).

- [ ] **Step 3: Sentinel the unused blocks in `_align_expert_ids`**

In `src/xkernels/ops/moe/triton/align_kernel.py`, in `_align_expert_ids`, replace:

```python
    expert = tl.sum(((cs <= off) & valid).to(tl.int32), axis=0)
    tl.store(expert_ids_ptr + b, expert)
```

with:

```python
    expert = tl.sum(((cs <= off) & valid).to(tl.int32), axis=0)
    # Unused trailing blocks (off >= total padded) count all experts -> num_experts,
    # one past the valid 0-based range. Map to sentinel 0 (matches the tokenspeed
    # contract); used blocks are always < num_experts so they are untouched.
    expert = tl.where(expert >= num_experts, 0, expert)
    tl.store(expert_ids_ptr + b, expert)
```

- [ ] **Step 4: Add `truncate` to `moe_align_block_size_triton`**

In the same file, change the signature:

```python
def moe_align_block_size_triton(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
```

to:

```python
def moe_align_block_size_triton(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    truncate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
```

and replace the final return block:

```python
    # Reference returns expert_ids truncated to the blocks actually used.
    n = int(num_post.item())
    return sorted_ids, expert_ids[: n // block_size], num_post
```

with:

```python
    if truncate:
        # Eager mode: device->host sync to trim expert_ids to the used blocks.
        n = int(num_post.item())
        return sorted_ids, expert_ids[: n // block_size], num_post
    # Sync-free / fixed-shape mode (graph-capturable): no .item(); expert_ids is
    # the full max_blocks length with unused trailing blocks = 0.
    return sorted_ids, expert_ids, num_post
```

Update the docstring `Returns:` block (find the `(sorted_token_ids [max_pad], expert_ids [n // block_size], ...` paragraph) to add:

```python
        With ``truncate=False`` (graph-capturable), ``expert_ids`` is the full
        ``max_blocks = cdiv(max_pad, block_size)`` length with unused trailing
        blocks set to 0 and no device->host sync.
```

- [ ] **Step 5: Thread `truncate` through the dispatch**

In `src/xkernels/ops/moe/align.py`, change the `moe_align_block_size` signature to add `truncate`:

```python
def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    *,
    backend: Backend | str = "auto",
    truncate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
```

and the dispatch call:

```python
    return dispatch("moe_align_block_size", topk_ids, block_size, num_experts, backend=backend)
```

to:

```python
    return dispatch(
        "moe_align_block_size", topk_ids, block_size, num_experts,
        backend=backend, truncate=truncate,
    )
```

Add to the docstring (after the `num_experts:` arg line):

```python
        truncate: if True (default) trim ``expert_ids`` to used blocks (eager); if
            False return the full ``max_blocks`` length with no host sync, for
            HIP/CUDA-graph capture (Triton backend).
```

- [ ] **Step 6: Run to verify it passes (and no regression)**

Run: `TRITON_INTERPRET=1 .venv/bin/python -m pytest tests/test_moe_align_block_size.py -q`
Expected: PASS (all — existing `truncate=True` tests unaffected because the sentinel only changes sliced-off tail blocks, plus the new `truncate=False` cases).

- [ ] **Step 7: Lint + commit**

```bash
.venv/bin/ruff check src/xkernels/ops/moe/triton/align_kernel.py src/xkernels/ops/moe/align.py
git add src/xkernels/ops/moe/triton/align_kernel.py src/xkernels/ops/moe/align.py tests/test_moe_align_block_size.py
git commit -m "feat(moe): sync-free truncate=False mode for moe_align_block_size (issue #18)"
```

---

### Task 3: Capture/replay proof + SLURM job

**Files:**
- Create: `benchmarks/probe_align_capture.py`
- Create: `slurm/probe_align_capture_beverin.sbatch`

- [ ] **Step 1: Write the capture probe**

Create `benchmarks/probe_align_capture.py`:

```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Prove moe_align_block_size_triton(truncate=False) is HIP/CUDA-graph capturable.

Captures the sync-free align into a graph, replays it, and checks the replayed
output matches eager. Contrasts truncate=True (which keeps the .item() sync and
is not capturable). GPU-only; run on gfx942 (slurm/probe_align_capture_beverin.sbatch).
"""
from __future__ import annotations

import torch

from xkernels.ops.moe.triton.align_kernel import moe_align_block_size_triton


def main():
    if not torch.cuda.is_available():
        print("No GPU; graph capture proof requires gfx942 (or any CUDA/ROCm GPU).")
        return
    dev = "cuda"
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
    M, top_k, E, block = 16, 8, 48, 16  # Kimi-K2.6 decode-ish
    g = torch.Generator(device=dev).manual_seed(0)
    topk_ids = torch.randint(0, E, (M, top_k), generator=g, dtype=torch.int32, device=dev)

    s_ref, e_ref, n_ref = moe_align_block_size_triton(topk_ids, block, E, truncate=False)
    torch.cuda.synchronize()

    # Warmup on a side stream (JIT compile is not capturable).
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            moe_align_block_size_triton(topk_ids, block, E, truncate=False)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        s_cap, e_cap, n_cap = moe_align_block_size_triton(topk_ids, block, E, truncate=False)
    graph.replay()
    torch.cuda.synchronize()

    ok = (
        torch.equal(s_cap, s_ref)
        and torch.equal(e_cap, e_ref)
        and torch.equal(n_cap, n_ref)
    )
    print(f"truncate=False capture+replay matches eager: {ok}")
    print(f"  expert_ids fixed length = {e_cap.numel()} (max_blocks), "
          f"num_post = {int(n_ref.item())}, used_blocks = {int(n_ref.item()) // block}")

    # Contrast: truncate=True keeps the .item() sync; capturing it should error.
    try:
        warm = torch.cuda.CUDAGraph()
        with torch.cuda.graph(warm):
            moe_align_block_size_triton(topk_ids, block, E, truncate=True)
        print("  truncate=True captured WITHOUT error (unexpected — .item() sync)")
    except Exception as exc:  # noqa: BLE001
        print(f"  truncate=True not capturable (expected, host sync): {str(exc)[:90]}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the no-GPU guard + lint**

Run: `.venv/bin/python benchmarks/probe_align_capture.py`
Expected: prints `No GPU; graph capture proof requires gfx942 ...` and exits 0.
Run: `.venv/bin/ruff check benchmarks/probe_align_capture.py`
Expected: no errors.

- [ ] **Step 3: Write the SLURM job**

Create `slurm/probe_align_capture_beverin.sbatch`:

```bash
#!/bin/bash
# SPDX-License-Identifier: MIT
# Prove the sync-free moe_align (truncate=False) is HIP-graph capturable on
# beverin (gfx942 / MI300A) — issue #18.
#
#   sbatch slurm/probe_align_capture_beverin.sbatch
#
#SBATCH --job-name=xk-align-capture
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=00:15:00
#SBATCH --output=align-capture-%j.out
#SBATCH --error=align-capture-%j.out

set -uo pipefail

REPO="${REPO:-/capstor/scratch/cscs/xyao/kernels}"
ENV_NAME="${ENV_NAME:-tokenspeed-rocm-aiter-myofi}"

echo "REPO=$REPO ENV=$ENV_NAME node=$(hostname)"

srun --environment="$ENV_NAME" --cpu-bind=none bash -c '
  set -e
  unset ROCR_VISIBLE_DEVICES || true
  export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="'"$REPO"'/src:${PYTHONPATH:-}"
  echo "=== capture proof ==="
  python -u "'"$REPO"'/benchmarks/probe_align_capture.py"
  echo "=== GPU correctness (truncate modes) ==="
  python -m pytest "'"$REPO"'/tests/test_moe_align_block_size.py" -q
'
```

- [ ] **Step 4: Commit**

```bash
git add benchmarks/probe_align_capture.py slurm/probe_align_capture_beverin.sbatch
git commit -m "bench(moe): HIP-graph capture proof for sync-free moe_align (issue #18)"
```

---

### Task 4: Run on beverin, verify, PR

This task runs on the cluster; no TDD. Each step is a real command.

- [ ] **Step 1: Sync the branch to beverin scratch**

```bash
rsync -az --delete \
  --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
  --exclude '.ruff_cache' --exclude '.pytest_cache' \
  ./ beverin:/capstor/scratch/cscs/xyao/kernels/
```
Expected: completes without error. (Renew the CSCS SSH cert at https://sshservice.cscs.ch/ if expired.)

- [ ] **Step 2: Submit the capture/correctness job**

```bash
ssh beverin 'cd /capstor/scratch/cscs/xyao/kernels && sbatch slurm/probe_align_capture_beverin.sbatch'
```
Expected: `Submitted batch job <JOBID>`.

- [ ] **Step 3: Wait and read the log**

```bash
ssh beverin 'squeue -j <JOBID> -h -o %T; tail -n 60 /capstor/scratch/cscs/xyao/kernels/align-capture-<JOBID>.out'
```
Expected: `truncate=False capture+replay matches eager: True`, the fixed `expert_ids` length line, the `truncate=True not capturable` contrast line, and the pytest GPU run passing. If capture+replay is not `True`, stop and investigate (do not claim success).

- [ ] **Step 4: Run the full local interpreter suite once more**

```bash
TRITON_INTERPRET=1 .venv/bin/python -m pytest -q
.venv/bin/ruff check .
```
Expected: all pass, lint clean.

- [ ] **Step 5: Push, open PR, report on the issue**

```bash
git push -u origin issue-18-moe-align-syncfree
gh pr create --repo ResearchComputer/kernels --base main \
  --title "feat(moe): sync-free / fixed-shape moe_align_block_size for HIP-graph decode (issue #18)" \
  --body "<summary: truncate=False mode; capturable; on-device capture/replay proof; references #18>"
```
Then comment on issue #18 with the capture/replay result and the fixed-shape contract. (Squash-merge per repo convention once reviewed.)

---

## Self-review

- **Spec coverage:** `truncate=False` sync-free + fixed shape (Task 2, kernel + dispatch) ✓; unused-block sentinel 0 (Task 2 Step 3) ✓; reference `truncate` for parity (Task 1) ✓; dispatch passthrough (Task 2 Step 5) ✓; `truncate=True` default unchanged (signatures default True; existing tests run in Task 2 Step 6) ✓; interpreter parity/fixed-shape/tail tests (Tasks 1–2) ✓; on-device capture/replay proof (Task 3 probe + Task 4 run) ✓; result reported on #18 (Task 4) ✓.
- **Placeholder scan:** none — all steps carry concrete code/commands. `<JOBID>` and PR body are runtime values.
- **Type/name consistency:** `truncate` is the param name in `moe_align_block_size_ref`, `moe_align_block_size_triton`, and `moe_align_block_size` (dispatch) — identical. Sentinel value is 0 in the reference pad, the Triton `tl.where`, and the test assertions (`e_f[used:] == 0`). `max_blocks = cdiv(max_pad, block_size)` computed identically in reference, kernel allocation, and tests.
