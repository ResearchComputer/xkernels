# Collectives — topology-aware hierarchical all-reduce (issue #12)

The distributed-collective module (`xkernels.ops.comm`). Unlike the single-GPU
dispatched kernels, collectives take process groups — so this is a **schedule +
fused-epilogue** contribution, not a per-CTA kernel.

> **One-line summary.** The fused residual-add + RMSNorm epilogue and the
> hierarchical schedule are **correct on real 2-node CXI** (acceptance met), and
> the HIP-graph-capture blocker is fully cracked and documented. **But the
> performance premise — that a hand-rolled hierarchical decomposition beats the
> oracle at decode — does not hold** on this 2-node / 4-NIC-per-node MI300A stack,
> because RCCL's flat all-reduce is *already* topology-aware internally. *(Same
> outcome as the MoE fused combine in `kernels/moe.md`: a correct optimization the
> hardware does not reward.)*

---

## What landed

`xkernels.ops.comm`:

- **`build_topology_groups(ranks_per_node)`** — intra-node (xGMI) + cross-node
  (CXI, same-local-rank) process groups, contiguous block layout.
- **`hierarchical_all_reduce(x, intra, cross)`** — `reduce_scatter` (xGMI) →
  cross-node `all_reduce` of the 1/rpp partial (CXI) → `all_gather` (xGMI).
- **`flat_all_reduce(x, group)`** — the oracle.
- **`residual_rmsnorm`** + Triton `add_rmsnorm` kernel — the fused
  residual-add+RMSNorm epilogue, and `hierarchical_all_reduce_residual_rmsnorm`
  composing the two.

## Validation

| Check | Where | Result |
|-------|-------|--------|
| Schedule correctness, 8-rank logical (2×4) | local gloo/CPU | **PASS** all sizes |
| Fused residual+RMSNorm vs torch oracle | `TRITON_INTERPRET=1` | **PASS** |
| Correctness, single-node 4-rank | beverin MI300A, RCCL | **PASS** (bf16) |
| Correctness, **2-node 8-rank over CXI** | beverin, `myofi` + OFI plugin | **PASS** bs∈{1,2,4,8,16} |
| RCCL uses CXI fabric | beverin | `NET/OFI Selected provider is cxi … (found 4 nics)` |

`hierarchical_all_reduce` is numerically equal to `flat_all_reduce` within bf16
tolerance on real CXI — acceptance met.

## The eager latency finding

Eager (job 380669), 2-node MI300A: **the hierarchical schedule loses (~0.47×) at
decode sizes.** At 14 KiB (`bs×7168` bf16) the collective is *launch-latency-bound*,
and the hierarchical schedule issues **3 collectives** (reduce-scatter + cross
all-reduce + all-gather) vs flat's **1** — 3× the per-launch overhead swamps the
reduced cross-node payload. The win exists only when the per-launch cost is
amortized, i.e. **under HIP-graph decode capture** (the serve's regime).

## Cracking HIP-graph capture

Capturing the **networked RCCL/OFI-CXI** collectives in a HIP graph was blocked
by *two independent* failures on this stack (PyTorch 2.11+rocm7.2, RCCL 2.27.7,
from-source aws-ofi-rccl). Both are now solved:

1. **OFI memory-registration** `invalid device pointer` mid-capture. Fixed by
   making CXI MR registration capture-safe: `FI_CXI_OPTIMIZED_MRS=0` (or the
   libfabric MR cache via `FI_MR_CACHE_MONITOR=memhooks` +
   `NCCL_OFI_MR_CACHE_DISABLE=0`).
2. **PG watchdog** calling `hipEventQuery` on the capturing stream → `operation
   not permitted when stream is capturing`. *Not* env-tunable here
   (`TORCH_NCCL_ASYNC_ERROR_HANDLING=0` does not stop the thread). Fixed by
   driving the collectives through a **watchdog-free raw RCCL communicator**
   (`meta/benchmarks/pynccl_lite.py`, ctypes over `librccl.so`) — the same approach
   vLLM/SGLang use. The unique-id handshake rides the existing torch.distributed
   store; no collective ever goes through the watchdog'd PG.

`meta/benchmarks/bench_capture_pynccl.py` captures both schedules and replays
them; correctness re-validated through the raw comm (flat sum == world, hier ==
flat) before timing.

## The captured latency finding — the premise does not hold

Captured (job 380930), 2-node MI300A, HIP graph:

| bs | MB | flat_ms | hier_ms | speedup |
|---:|---:|---:|---:|---:|
| 1 | 0.014 | 0.0682 | 0.0939 | 0.73× |
| 4 | 0.057 | 0.1123 | 0.1012 | 1.11× |
| 16 | 0.229 | 0.1126 | 0.1177 | 0.96× |
| 256 | 3.670 | 0.1874 | 0.1759 | 1.07× |
| 1024 | 14.680 | 0.3291 | 0.3719 | 0.88× |
| 4096 | 58.720 | 0.8714 | 1.1926 | 0.73× |

**Capture does what we predicted — it amortizes the per-launch penalty.** The
decode (bs=1) flat-vs-hier ratio improved from eager **0.45×** to captured
**0.73×**: with the 3 collectives' launch overhead removed from the graph, most
of the gap closes.

**But hierarchical still does not beat flat — at any size, in either mode.** The
crossover never arrives. The reason is the baseline: on this stack RCCL's *flat*
all-reduce is **already topology-aware** — it discovers the 4 CXI NICs, builds NIC
groups, and runs an internally hierarchical schedule (`Selected provider is cxi …
found 4 nics`, NIC groups 0–3, `SENDRECV`). Our hand-written
`reduce_scatter`(xGMI) → `all_reduce`(CXI, ¼ payload) → `all_gather`(xGMI) is
doing manually what RCCL already does inside one launch:

- **Decode (≤14 KiB, bs≤2):** latency-bound. Even captured, 3 serial dependent
  graph nodes cost more than flat's one → **0.73–0.75×**.
- **Mid (bs 4–256, 57 KiB–3.7 MB):** roughly even, occasionally +7–11% — within
  run-to-run noise, no robust win.
- **Bandwidth (bs≥1024, ≥14.7 MB):** the extra xGMI passes are pure overhead on
  top of an already-NIC-saturating flat collective → **0.73× at 58.7 MB.**

## Conclusion

A manual decomposition would be expected to pay off only where the vendor
collective is *not* topology-aware, or at larger node counts / different NIC ratios
where the cross-node payload reduction outweighs the extra intra-node passes.
Capture amortization is real (0.45×→0.73×) but insufficient to cross 1.0× here.

## Reproduce

```bash
# eager + captured all-reduce on 2-node MI300A
scripts/cluster.sh submit --host beverin scripts/archive/issues/bench_allreduce_beverin.sbatch
scripts/cluster.sh submit --host beverin scripts/archive/issues/bench_capture_beverin.sbatch
```

> **Env note** (for whoever maintains the EDFs): `~/.edf/tokenspeed-rocm-aiter-myofi.toml`
> no longer sets `LD_LIBRARY_PATH` to the from-source plugin at
> `/capstor/store/cscs/swissai/infra02/xyao/tokenspeed-beverin/aws-ofi-rccl/lib`.
> Without it RCCL can't find `librccl-net-ofi.so` and fails with *"Failed to
> initialize any NET plugin."* The bench sbatch sets it explicitly; consider
> restoring it in the EDF `[env]`.
