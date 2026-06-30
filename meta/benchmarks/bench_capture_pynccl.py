# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""HIP-graph-captured flat-vs-hierarchical all-reduce via a watchdog-free comm.

The decode-regime measurement for issue #12. Drives the collectives through
``pynccl_lite`` (raw RCCL, no PG watchdog) so the sequence is HIP-graph
capturable, and pairs it with the OFI capture fix (``FI_CXI_OPTIMIZED_MRS=0``)
found by ``scripts/archive/issues/probe_graph_beverin.sbatch``. Captures both schedules, replays,
and reports per-replay latency — the regime where per-launch overhead is
amortized and only the collective's own cost (incl. the hierarchical schedule's
4x-smaller cross-node payload) remains.

Launch via SLURM srun (see scripts/archive/issues/bench_capture_beverin.sbatch). Reads rank/world
from SLURM. Applies the OFI + watchdog env in-process before init.
"""
from __future__ import annotations

import argparse
import os
import statistics

# OFI capture fix + watchdog-off, applied before any RCCL/torch init. The raw
# comm has no watchdog, but the default PG (used only for eager barriers) does;
# keep it from interfering. FI_CXI_OPTIMIZED_MRS=0 makes OFI MR registration
# capture-safe on the CXI provider.
os.environ.setdefault("FI_CXI_OPTIMIZED_MRS", "0")
os.environ.setdefault("NCCL_OFI_MR_CACHE_DISABLE", "0")
os.environ.setdefault("FI_MR_CACHE_MONITOR", "memhooks")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "0")
os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from pynccl_lite import NcclComm  # noqa: E402


def _env(*names, default=0):
    for n in names:
        if n in os.environ:
            return int(os.environ[n])
    return default


def _capture(run_inplace):
    """Capture run_inplace into a HIP graph (connections pre-established eagerly).
    Returns the graph, or raises."""
    for _ in range(10):  # eager warmup: establish RCCL connections + OFI MRs
        run_inplace()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run_inplace()
    torch.cuda.synchronize()
    return g


def _time_replay(g, iters, warmup):
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start.record()
        g.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks-per-node", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=30)
    args = ap.parse_args()

    rank = _env("RANK", "SLURM_PROCID")
    world = _env("WORLD_SIZE", "SLURM_NTASKS", default=1)
    local_rank = _env("LOCAL_RANK", "SLURM_LOCALID")
    torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}")
    # Bind device_id so the default PG's first collective (an eager barrier) does
    # NOT "guess device ID based on global rank" -> on node 1 that guesses
    # cuda:4..7, which don't exist (4 GPUs/node) -> "invalid device pointer".
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=dev)
    dtype = torch.bfloat16

    rpp = args.ranks_per_node
    num_nodes = world // rpp
    node, local = rank // rpp, rank % rpp

    # Raw comms: flat (all), intra (same node, xGMI), cross (same local, CXI).
    flat_ranks = list(range(world))
    intra_ranks = [node * rpp + i for i in range(rpp)]
    cross_ranks = [local + n * rpp for n in range(num_nodes)]
    if rank == 0:
        print(
            f"world={world} rpp={rpp} num_nodes={num_nodes} dtype={dtype}\n"
            f"  flat={flat_ranks} (rank0 intra={intra_ranks} cross={cross_ranks})",
            flush=True,
        )
    comm_flat = NcclComm(flat_ranks, rank, "uid_flat")
    comm_intra = NcclComm(intra_ranks, rank, f"uid_intra_{node}")
    comm_cross = NcclComm(cross_ranks, rank, f"uid_cross_{local}")
    dist.barrier()
    if rank == 0:
        print("[comms built]", flush=True)

    # ---- correctness: ones -> flat sum == world; hier out == world ----
    n0 = rpp * args.hidden
    xb = torch.ones(n0, device=dev, dtype=dtype)
    comm_flat.all_reduce(xb)
    torch.cuda.synchronize()
    flat_ok = abs(xb.float().mean().item() - world) < 1e-3

    xb = torch.ones(n0, device=dev, dtype=dtype)
    rsb = torch.empty(n0 // rpp, device=dev, dtype=dtype)
    outb = torch.empty(n0, device=dev, dtype=dtype)
    comm_intra.reduce_scatter(rsb, xb)
    comm_cross.all_reduce(rsb)
    comm_intra.all_gather(outb, rsb)
    torch.cuda.synchronize()
    hier_ok = abs(outb.float().mean().item() - world) < 1e-3
    flag = torch.tensor([1 if (flat_ok and hier_ok) else 0], device=dev)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if rank == 0:
        print(
            f"correctness: flat={'OK' if flat_ok else 'BAD'} "
            f"hier={'OK' if hier_ok else 'BAD'} "
            f"all_ranks={'PASS' if flag.item() else 'FAIL'}",
            flush=True,
        )

    # ---- captured latency: flat vs hierarchical ----
    itemsize = torch.empty((), dtype=dtype).element_size()
    if rank == 0:
        print(
            f"\n[graph-captured]\n{'bs':>5} {'MB':>8} {'flat_ms':>10} {'hier_ms':>10} "
            f"{'speedup':>8}",
            flush=True,
        )
    for bs in args.sizes:
        n = bs * args.hidden
        mb = n * itemsize / 1e6
        xb = (torch.randn(n, device=dev) * 0.1).to(dtype)
        rsb = torch.empty(n // rpp, device=dev, dtype=dtype)
        outb = torch.empty(n, device=dev, dtype=dtype)

        def _flat(xb=xb):
            comm_flat.all_reduce(xb)

        def _hier(xb=xb, rsb=rsb, outb=outb):
            comm_intra.reduce_scatter(rsb, xb)
            comm_cross.all_reduce(rsb)
            comm_intra.all_gather(outb, rsb)

        try:
            gf = _capture(_flat)
            gh = _capture(_hier)
        except Exception as exc:  # noqa: BLE001
            if rank == 0:
                print(f"{bs:>5} capture failed: {exc}", flush=True)
            continue
        flat_ms = _time_replay(gf, args.iters, args.warmup)
        hier_ms = _time_replay(gh, args.iters, args.warmup)
        if rank == 0:
            print(
                f"{bs:>5} {mb:>8.3f} {flat_ms:>10.4f} {hier_ms:>10.4f} "
                f"{flat_ms / hier_ms:>7.2f}x",
                flush=True,
            )

    dist.barrier()
    for c in (comm_flat, comm_intra, comm_cross):
        c.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
