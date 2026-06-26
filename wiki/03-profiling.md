# Profile findings — roofline, regime, and the "tune me" signals

Roof-mode profiles from 2026-06-26. Two profilers, same 10 ops:

- **beverin (MI300A)**: ROCm Compute Profiler (`rocprof-compute`), roof mode.
  `.omniperf-workloads/<kernel>_roof/` + `<kernel>_roof.analyze.txt`.
- **bristen (A100)**: Nsight Compute (`ncu`), roof mode.
  `.ncu-workloads/<kernel>_roof/<kernel>_roof.report.txt`.

The probes (`benchmarks/probe_omniperf.py` / `probe_ncu.py`) share identical
seeded shapes, so a number on one arch is directly comparable to the other.

## ⚠ Profiler-reliability caveat (read first)

The beverin `rocprof-compute` analyze opens with
`ERROR PC sampling: can not read ps_file_kernel_trace.csv`. That error breaks the
**per-kernel PC-sample attribution** — specifically the *FLOPs-vs-peak %* columns
that need to assign each sample to a dispatch. Concretely, rocprof reports
`mm_fp8_blockscale` at 1.5 TFLOP/s (0.08% of MFMA-F8 peak) while
`bench_fp8_blockscale_gemm.py` measures **363 TFLOP/s** on the identical dispatch.

It does **not** break the aggregate PMC-counter metrics, which collect across
whatever dispatched. So on beverin the trustworthy set is:

- **Arithmetic Intensity (AI, Flops/byte)** — aggregated, robust; the cleanest
  single regime signal (low AI ⟹ memory-bound, high AI ⟹ compute-bound, ≈0 ⟹
  dispatch-bound).
- **`2.1.16 Wavefront Occupancy`** and **`5.2.1 CPC Stall Rate`** — aggregate
  PMC counters (these were missed in the first pass; recovered via the metric
  map in `.agents/skills/use-rocprof-compute/SKILL.md`).
- **Bench-derived TFLOP/s** for the compute-bound GEMMs.

Two beverin caveats remain: (a) these aggregates are **not dispatch-isolated**
(`-d`/`-k` were not used — the skill's "profile the wrong dispatch" pitfall), so
for tiny-work kernels (T=8) the `torch.randn` setup dispatches are mixed in; and
(b) the per-kernel FLOPs-vs-peak % is the one column to ignore.

The bristen **ncu DRAM/Compute % and occupancy are authoritative** (no sampling
error; and `-k regex:$FRAG` isolated the target dispatch).

## Regime table (card regime vs measured)

AI = arithmetic intensity (Flops/byte) on MI300A; DRAM%/Compute% = ncu on A100;
Occ = achieved SM occupancy on A100.

| Kernel | Card regime | AI (F/B) MI300A | DRAM% / Compute% A100 | Occ A100 | Measured verdict |
|---|---|---:|---|---:|---|
AI = arithmetic intensity (Flops/byte) on MI300A; DRAM%/Compute% = ncu on A100;
Occ = achieved SM occupancy on A100; WfOcc = MI300A `2.1.16 Wavefront Occupancy`
(achieved/peak, not dispatch-isolated — see caveat); CPC stall = MI300A `5.2.1`.

| Kernel | Card regime | AI (F/B) MI300A | WfOcc% MI300A | CPC stall% MI300A | DRAM% / Compute% A100 | Occ A100 | Measured verdict |
|---|---|---:|---:|---:|---|---:|---|
| `dual_rmsnorm` | memory | 3.9 | 30% | low | 68 / 54 | 92% | **memory-bound** ✓ |
| `moe_sum_reduce` | memory | 0.47 | 159%† | low | 85 / 19 | 95% | **strongly memory-bound** ✓ |
| `mha_merge_state` | memory | 8.0 | 89% | low | 41 / 66 | 76% | **balanced, compute-leaning** |
| `hc_prenorm_gemm` | memory | 0.40 | **0.95%** | 36% | — | — | **launch/overhead-bound** at T=8 |
| `mhc_pre` | memory | 1.1 | **0.84%** | 42% | — | — | **launch-bound** (tiny T=8, GEMM work) |
| `sparse_mla_attention` | memory | **70.6** | 15% | 41% | — | — | **compute-bound** ⚠ (AI high) |
| `moe_align_block_size` | memory | ≈0 | **0.9%** | 56% | 3 / 4 | **6%** | **dispatch/index-bound** (neither) |
| `fused_ffn` | compute | 141 | 63% | 26% | 87 / 28 | 93% | **memory-bound on A100**; under-uses MFMA |
| `mm_fp8_blockscale` | compute | — | **0.71%** | 50% | N/A (no sm_80 fp8) | N/A | **compute-bound** (363 TFLOP/s bench) |
| `moe_int4_w4a16` | compute | **175.5** | 17% | **83%** | 18 / 35 | **12.5%** | **register-pressure-bound** ⚠ |

† MI300A `Wavefront Occupancy` is a wavefront-count ratio, not the A100's
SM-occupancy that caps at 100%; >100% here means active wavefronts exceed the
counter's normalization (oversubscribed, expected for the large M=8192 reduce).
The interpretable signal is the **sub-1%** cluster (`hc_prenorm_gemm`, `mhc_pre`,
`moe_align`, `mm_fp8`) = deeply under-occupied, consistent with launch-bound
T=8 / tiny-M work.

## The actionable findings

### `moe_int4_w4a16` — occupancy-capped by register pressure (the clearest tune signal)
On A100, **theoretical occupancy is 12.5% and achieved is 12.3%** — ncu: *"this
kernel's theoretical occupancy (12.5%) is limited by the number of required
[registers]"*. DRAM is only 18%, compute 35%, L2/shared memory 66%. The binding
constraint is **VGPR/register pressure**, not bandwidth or FLOPs. On MI300A the
arithmetic intensity is **175.5 Flops/byte** (unambiguously compute-bound), so
this is a compute-bound GEMM whose achievable throughput is gated by occupancy.
→ route to `.agents/skills/diagnose-low-occupancy` (raise occupancy by cutting
register footprint / splitting) and, given it's a GEMM,
`map-to-matrix-cores`. Bench ≈ 22 TFLOP/s at the decode shape — well under the
matrix-engine ceiling, consistent with the occupancy cap.

### `sparse_mla_attention` — card says memory, profile says compute
Card `perf.roofline = memory_bound`, but MI300A **AI = 70.6 Flops/byte** — deep
in the compute regime at this shape (T=8, H=128, D=512, topk=512). The card's
classification likely reflects a different (larger-Kv / prefill) operating point;
at the profiled decode shape the kernel is **compute-bound**. Worth re-grading the
card regime, or noting the shape-dependence.

### `fused_ffn` — the card regime is about the matmuls, which are torch.matmul
Card says `compute_bound` and MI300A AI is high (141 F/B). But the card's own note
is load-bearing here: **the three projection GEMMs are `torch.matmul` in both the
reference and triton paths; the Triton kernel is only the fused SwiGLU
activation.** So the op is matmul-dominated, and on A100 ncu reads the whole
dispatch as **87% DRAM-bound** while on MI300A the matmuls run at **~210 TFLOP/s**
(bench-derived from the 3 projections — well into the MFMA regime, *not* under-
utilizing it). The earlier draft's "15.9% of MFMA-F16 peak" was read from the
broken per-kernel FLOPs column and is wrong (see the caveat above).

This is exactly why the bench speedup over torch fp16 is **~1.03×**: both paths
run the same `torch.matmul` GEMMs at the same ~205–210 TFLOP/s; the fused SwiGLU
Triton kernel only saves one elementwise launch, which is a rounding error
against three GEMMs. **No fix skill applies to the Triton kernel** (it is a
correct, already-fast elementwise fusion); the card's `compute_bound` regime
honestly describes the op the card is embedded in, not the kernel the card ships.

### `moe_align_block_size` — neither bound; its win is overhead, not arithmetic
AI ≈ 0, ncu DRAM 3%, Compute 4%, **occupancy 6%**, and ncu still flags *"all
compute pipelines under-utilized"*. The 33–75× speedup over torch is **launch +
python-loop overhead elimination**, not bandwidth or compute. There is no roofline
headroom to chase; the kernel is already near-optimal for what it is.

### `mha_merge_state` — the most balanced kernel
The only op where ncu Compute (66%) ≫ DRAM (41%) and occupancy is held to 76%
(ncu: theoretical 100%, lost to scheduling/load imbalance). Both pipes move real
work; modest headroom from raising occupancy, not from bandwidth.

## Occupancy summary (A100 / ncu)

| Kernel | Achieved Occ | Notes |
|---|---:|---|
| `moe_sum_reduce` | 95% | excellent |
| `fused_ffn` | 93% | excellent |
| `dual_rmsnorm` | 92% | excellent |
| `mha_merge_state` | 76% | scheduling/load-imbalance gap (ncu flag) |
| `moe_int4_w4a16` | 12% | **register-pressure-capped** (the tune target) |
| `moe_align_block_size` | 6% | tiny per-block work; expected |

The high-occupancy memory-bound kernels (`dual_rmsnorm`, `moe_sum_reduce`) sit at
68–85% DRAM on A100 — i.e. **~1.3–1.6 TB/s achieved of the ~1.94 TB/s peak**.
There is little left to tune on the bandwidth axis for those; the wins in the
bench table are the wins.

## How a profile routes to a fix skill

Filled from this campaign's data (same routing table as
`docs/profiling-on-*.md`, now backed by real numbers):

- DRAM% ≫ Compute% (`moe_sum_reduce`, `dual_rmsnorm`, `fused_ffn` on A100) →
  `diagnose-memory-bound`.
- Low achieved occupancy + a register-pressure cap (`moe_int4_w4a16`) →
  `diagnose-low-occupancy`.
- High AI / compute-card but low MFMA/tensor utilization (`sparse_mla_attention`)
  → `map-to-matrix-cores`. (Not `fused_ffn` — its matmuls are `torch.matmul`;
  see the finding above.)
- ≈0 AI + low everything (`moe_align_block_size`) → nothing to tune (overhead-bound).

## Relation to the profiler skills (honest diff)

This page was produced by running the profilers that
`.agents/skills/use-rocprof-compute` and `use-nsight-compute` document. This
subsection records what was followed vs missed, so the next pass can close the
remaining gaps. Both skills were read **after** the first draft of this page;
the beverin occupancy column above is the direct result of consulting the skill's
metric map (`2.1.16` / `5.2.1`) after the first grep pass used the wrong patterns
and returned nothing.

**Followed the skills on:**
- roof-mode profile on both arches (both skills' default mode).
- Reading the SpeedOfLight ratio (DRAM% vs Compute%) to route — the bristen
  dual_rmsnorm report's own OPT line (`"Memory is more heavily utilized than
  Compute"`) drives the routing, exactly as the skill prescribes.
- Reading occupancy % (bristen from the roof-mode Occupancy section; beverin now
  from `2.1.16`) and the moe_int4 causal OPT line (`Block Limit Registers = 2` →
  theoretical 12.5%, `Est. Local Speedup 87.5%`). That verdict is causal and uses
  the skill's vocabulary.
- The load-bearing runtime gotchas (bristen DCGM `--pause`; beverin `pandas<3`
  pin + `libdw.so.1` staging) — recorded in [`04-gotchas.md`](04-gotchas.md).
- Closing the loop with `record_measurement` (the §6.2 compounding write-back).

**Missed / weaker than the skills prescribe:**
1. **No `sq` mode on either arch** → **no dominant stall reason.** Both skills
   are emphatic: route by the stall reason (causal), not the SpeedOfLight ratio
   alone ("if your two modes disagree, trust the stall reason"). This page's
   routing is **ratio + occupancy only**. The one exception is `moe_int4_w4a16`,
   where the roof-mode Occupancy OPT line names the cause (register pressure), so
   its verdict holds. To close: run `MODE=sq` on bristen (SchedulerStats +
   WarpStateStats) and `-b SQ` on beverin, per the skills' question→mode table.
2. **beverin profiles are not dispatch-isolated** (`-d`/`-k` not used). So the
   `2.1.16`/`5.2.1` aggregates mix the `torch.randn` setup dispatches into the
   tiny-work (T=8) kernels — read the sub-1% occupancy cluster as *indicative*,
   not precise. bristen is isolated (`-k regex:$FRAG`).
3. **No MI300A `achieved_bw_pct` in the cards.** The skill's formula
   (`bw_pct = bytes_moved / (ms/1e3) / peak_HBM`) is analytical and **independent
   of the broken rocprof columns**; the first pass conflated "profiler % broken"
   with "can't report bw%". A naive read+write byte model overshoots 100% for
   `dual_rmsnorm`, which means the *per-op `bytes_moved` derivation* (what the
   fused kernel actually moves to/from DRAM) is the real work — left as a
   follow-up, not fabricated. bristen `achieved_bw_pct` IS recorded (ncu's DRAM
   Throughput % is direct).

Net: the **regime classification and the moe_int4 tune target are sound**; the
**stall-reason depth and the MI300A bw% field are the known gaps** a follow-up
`sq`-mode pass + a per-op byte model would fill.
