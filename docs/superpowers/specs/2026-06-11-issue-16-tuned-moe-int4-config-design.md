# Issue #16 — Tuned INT4 W4A16 fused-MoE config for Kimi-K2.6 production shapes

**Date:** 2026-06-11
**Status:** Approved
**Issue:** ResearchComputer/kernels#16
**Target hardware:** AMD Instinct MI300A (gfx942, CDNA3)

## Purpose

The INT4 W4A16 grouped fused-MoE GEMM (`ops/moe/triton/moe_int4_kernel.py`,
issues #1/#7) is the single heaviest GPU kernel on the Kimi-K2.6 decode path. In
production (tokenspeed) it runs with Triton's **default (untuned)** config and
logs `Using default MoE kernel config. Performance might be sub-optimal!`. There
is no tuned config checked in for the production shapes on gfx942, so every
captured decode step pays an un-tuned MoE GEMM.

This change produces and checks in **tuned configs** for the two production
shapes, plus the loader and tuning harness needed to (a) use them at launch with
**zero runtime autotune**, and (b) port the same JSON back into tokenspeed's
non-autotuned production launch so the warning goes away.

## Production shapes (Kimi-K2.6, EP=8 → 48 experts/rank)

`top_k=8`, `group_size=32`, symmetric INT4, bf16 activations. Per-rank packed
weights (what the kernel sees):

| GEMM | E | N | K | packed K (int32) |
|------|---|---|---|------------------|
| `w13` (fused gate+up) | 48 | 4096 | 7168 | 896 |
| `w2`  (down)          | 48 | 7168 | 2048 | 256 |

Decode buckets captured into HIP graphs: **M ∈ {1, 2, 4, 8, 16}** (token batch
size; the latency-sensitive regime). We also tune M ∈ {32, 512, 4096} so the
loader has sensible large-M coverage for prefill.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Integration | **(A)** vLLM-style explicit config resolution | Production-faithful (no runtime autotune), portable JSON back into tokenspeed, makes the warning disappear. |
| Persisted format | vLLM-style JSON, one file per `(E,N,K,device,dtype)` | Established pattern; M-bucket → config map; human-diffable. |
| Fallback | Keep the existing `@triton.autotune` entry point | Untuned shapes still autotune → **no regression**; also the engine the offline tuner drives. |
| Tuning method | Explicit `do_bench` sweep over the pruned config space | Full control over timing + lets us record provenance (timing/device/date) in the JSON. |
| Key dimension | Token-batch `M` (= `num_valid_tokens // top_k`) | Matches the issue's captured decode buckets and vLLM's `num_tokens` keying. |

## Architecture

The kernel body becomes a plain `@triton.jit` function. Two launch paths share it:

```
int4_w4a16_moe_gemm(...)                      # production launch wrapper
   │
   ├─ get_moe_int4_config(E,N,K,M,dtype,arch)  # configs.py: read checked-in JSON
   │     │
   │     ├─ HIT  → launch _fused_moe_int4_kernel[grid](..., **cfg)   # direct, no autotune
   │     └─ MISS → launch fused_moe_int4_kernel[grid](...)          # @triton.autotune fallback
```

`fused_moe_int4_kernel` (the autotuned entry point, same name as today) is built
explicitly from the jit body so existing tests (`_pin_single_config` walks
`.fn`/`.configs`) and the tuning harness keep working unchanged.

### Components

1. **`src/xkernels/ops/moe/triton/tuned_configs/*.json`** — checked-in winners.
   Filename: `E=48,N=4096,K=7168,device_name=AMD_Instinct_MI300A,dtype=int4_w4a16.json`
   (and the `N=7168,K=2048` down file). Contents:
   ```json
   {
     "_provenance": {"device": "...", "date": "2026-06-11", "triton": "...", "metric": "median ms over do_bench"},
     "1":  {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1,
             "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "kpack": 2,
             "_ms": 0.0},
     "2":  {...}, "...": {}
   }
   ```
   `_provenance`/`_ms` are metadata (keys starting with `_` are ignored by the loader).

2. **`configs.py`** — add:
   - `_device_name(arch)` → normalized device string (spaces → `_`), defaulting to
     `torch.cuda.get_device_name()`.
   - `load_tuned_config(E, N, K, device, dtype) -> dict | None` — read + cache the JSON for a shape.
   - `get_moe_int4_config(E, N, K, M, dtype, arch) -> dict | None` — pick the entry
     for the **closest bucket ≤ M** (clamp to the smallest bucket when `M` is below
     the table; to the largest when above). `None` when no file exists.

3. **`moe_int4_kernel.py`** — refactor:
   - `@triton.jit def _fused_moe_int4_kernel(...)` — current body, unchanged math.
   - `fused_moe_int4_kernel = triton.autotune(...)(triton.heuristics({"EVEN_K": ...})(_fused_moe_int4_kernel))`.
   - `int4_w4a16_moe_gemm`: compute `M = num_valid_tokens // top_k`, call
     `get_moe_int4_config`; on hit launch `_fused_moe_int4_kernel[grid]` with explicit
     meta-params + `EVEN_K = (K % BLOCK_SIZE_K == 0)` and `num_warps`/`num_stages`;
     on miss keep the current autotuned launch.

4. **`benchmarks/tune_moe_int4_w4a16.py`** — for each shape × M-bucket: build inputs,
   `prune_configs` the candidate space, `do_bench` each via the direct jit path, keep
   the min, write the JSON (with provenance). GPU-only; refuses to run without one.

5. **`slurm/tune_moe_int4_beverin.sbatch`** — runs the tuner on the `mi300` partition,
   writing JSON into the repo's `tuned_configs/` dir (mirrors `bench_all_beverin.sbatch`).

6. **`tests/test_moe_int4_tuned_config.py`** — loader unit tests (exact bucket,
   closest-≤ bucket, clamp below/above, missing file → `None`, `_`-prefixed keys
   ignored, returned config is a member of the candidate space) + an interpreter-level
   integration test that a present tuned config drives the direct path and still
   matches the reference oracle within tolerance.

## Data flow

Launch: `M, top_k → get_moe_int4_config → cfg | None`. On hit, the resolved
meta-params parameterize a single direct kernel launch (one compile, cached). On
miss, behavior is byte-for-byte the current autotuned path.

Tuning (offline, on device): sweep pruned candidates per `(shape, M)`, pick the
fastest, persist. Re-running is idempotent (overwrites the file).

## Error handling / edge cases

- No JSON for a shape → `None` → autotune fallback (current behavior). Never raises.
- Malformed/locked JSON → warn once, treat as miss.
- `M` outside the tabulated buckets → clamp (never extrapolate a missing bucket).
- A tuned config whose `BLOCK_SIZE_K` doesn't divide the group size / pack would be
  invalid; the tuner only ever writes configs drawn from the pruned (valid) space,
  and the loader asserts `BLOCK_SIZE_K % group_size == 0` defensively.

## Testing

- **CPU / `TRITON_INTERPRET=1`:** loader unit tests + tuned-path correctness vs the
  reference oracle on a tiny shape with a hand-written tiny tuned JSON (interpreter
  ignores `num_warps`/stages but exercises the direct-launch parameterization).
- **On device (beverin, gfx942):** run the tuner to produce the real JSON; re-run the
  GPU correctness test to confirm the tuned path matches; spot-check decode ms vs the
  autotuned path to confirm the winners are no worse.

## Out of scope

- Tuning the bf16 dense GEMM (that is issue #17, next).
- Auto-tuning at import time or caching across processes beyond the checked-in JSON.
- A Gluon rewrite of the kernel.

## Deliverable acceptance

- Checked-in JSON for both shapes covering M ∈ {1,2,4,8,16} (plus 32/512/4096).
- Launch path uses the tuned config with no autotune and no default-config warning.
- Existing INT4 MoE tests still pass (interpreter + GPU); new loader tests pass.
- README perf note updated with the tuned decode figures; result reported on #16.
