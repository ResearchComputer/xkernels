# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Minimal HIP-graph capture probe for RCCL collectives (issue #12 deep-dive).

Self-contained (only torch + torch.distributed). Isolates the two known blockers
to capturing networked collectives on the beverin ROCm/RCCL/OFI-CXI stack:

  * watchdog blocker  -> hipEventQuery on the capturing stream
  * OFI MR blocker    -> libfabric registers a device pointer during capture

Topology:
  * 1 node, PROBE_MODE=full  -> group=None over 4 ranks = xGMI only, NO OFI.
        Use this to isolate the *watchdog* blocker.
  * 2 nodes, PROBE_MODE=full -> group=None spans nodes -> OFI/CXI.
        Use this to isolate the *OFI MR* blocker.
  * 2 nodes, PROBE_MODE=intra -> same-node subgroups (xGMI sanity in a 2-node job).

Env knobs:
  PROBE_CFG    = key into CFGS below (applies env overrides BEFORE PG init)
  PROBE_MODE   = full | intra                 (default full: group=None)
  NUMEL        = elements in the bf16 buffer  (default 7168)
  WARMUP       = side-stream warmup iters     (default 20)
  RANKS_PER_NODE = ranks/node for intra mode  (default 4)

The env overrides MUST be applied before init_process_group: ProcessGroupNCCL
reads TORCH_NCCL_* in its constructor, and RCCL/libfabric read NCCL_*/FI_* at
comm/provider creation (first collective). Setting os.environ here, inside the
container at runtime, also beats the EDF [env] (e.g. NCCL_NET_GDR_LEVEL).
"""
from __future__ import annotations

import os
import traceback

# --- env-override configs, keyed by PROBE_CFG; applied before any RCCL init ---
_WD_OFF = {
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "0",
    "TORCH_NCCL_ENABLE_MONITORING": "0",
    "TORCH_NCCL_DESYNC_DEBUG": "0",
    "TORCH_NCCL_BLOCKING_WAIT": "0",
    "TORCH_NCCL_TRACE_BUFFER_SIZE": "0",
}
_MR_CACHE = {
    "NCCL_OFI_MR_CACHE_DISABLE": "0",
    "FI_MR_CACHE_MONITOR": "memhooks",
    "FI_MR_CACHE_MAX_COUNT": "8192",
}
CFGS: dict[str, dict[str, str]] = {
    "baseline": {},                                   # watchdog ON: expect abort
    "debug": {**_WD_OFF, "NCCL_DEBUG": "INFO",        # verbose OFI/REG trace
              "NCCL_DEBUG_SUBSYS": "NET,REG,GRAPH,INIT"},
    "wd_off": _WD_OFF,                                # disable watchdog only
    "wd_off_mrcache": {**_WD_OFF, **_MR_CACHE},       # + libfabric MR cache
    "wd_off_no_devrdma": {**_WD_OFF, "NCCL_OFI_RDMA_USE_DEVICE_RDMA": "0"},
    "wd_off_cxi_optmr": {**_WD_OFF, "FI_CXI_OPTIMIZED_MRS": "0"},
    "wd_off_all": {**_WD_OFF, **_MR_CACHE,
                   "NCCL_OFI_RDMA_USE_DEVICE_RDMA": "0",
                   "FI_CXI_OPTIMIZED_MRS": "0"},
}

_CFG = os.environ.get("PROBE_CFG", "baseline")
for _k, _v in CFGS.get(_CFG, {}).items():
    os.environ[_k] = _v  # force-override (beats EDF [env])

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402


def _env(*names, default=0):
    for n in names:
        if n in os.environ:
            return int(os.environ[n])
    return default


def main():
    rank = _env("RANK", "SLURM_PROCID")
    world = _env("WORLD_SIZE", "SLURM_NTASKS", default=1)
    local_rank = _env("LOCAL_RANK", "SLURM_LOCALID")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    dev = torch.device(f"cuda:{local_rank}")

    mode = os.environ.get("PROBE_MODE", "full")
    rpp = _env("RANKS_PER_NODE", default=4) or 4
    numel = _env("NUMEL", default=7168) or 7168
    nwarm = _env("WARMUP", default=20) or 20

    if rank == 0:
        print(f"torch={torch.__version__} hip={torch.version.hip}", flush=True)
        try:
            print(f"rccl_version={torch.cuda.nccl.version()}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"rccl_version_err={e}", flush=True)
        print(
            f"[probe cfg={_CFG}] world={world} mode={mode} numel={numel} "
            f"warmup={nwarm} GDR_LEVEL={os.environ.get('NCCL_NET_GDR_LEVEL')} "
            f"ASYNC_EH={os.environ.get('TORCH_NCCL_ASYNC_ERROR_HANDLING')}",
            flush=True,
        )

    group = None
    if mode == "intra":
        node = rank // rpp
        for nd in range(world // rpp):
            g = dist.new_group(ranks=list(range(nd * rpp, (nd + 1) * rpp)))
            if nd == node:
                group = g

    # group size for the value check: full -> world, intra -> ranks/node
    gsize = world if group is None else dist.get_world_size(group)
    x = torch.ones(numel, device=dev, dtype=torch.bfloat16)

    def run():
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=group)

    # Warm up on a side stream: RCCL establishes connections + OFI registers the
    # buffer MR *before* capture (so capture hits the registration cache).
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(nwarm):
            run()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        print("[probe] warmup complete, attempting capture ...", flush=True)

    graph = torch.cuda.CUDAGraph()
    try:
        x.fill_(1.0)  # reset before capture so the replay value is deterministic
        with torch.cuda.graph(graph):
            run()
        torch.cuda.synchronize()
        if rank == 0:
            print(f"[probe cfg={_CFG}] CAPTURE OK", flush=True)
        x.fill_(1.0)  # ones -> all_reduce SUM over the group -> gsize
        graph.replay()
        torch.cuda.synchronize()
        val = x.float().mean().item()
        if rank == 0:
            ok = abs(val - float(gsize)) < 1e-3
            print(
                f"[probe cfg={_CFG}] REPLAY {'OK' if ok else 'WRONG'} "
                f"mean={val} (expect {float(gsize)})",
                flush=True,
            )
    except Exception:  # noqa: BLE001
        if rank == 0:
            print(f"[probe cfg={_CFG}] CAPTURE FAILED:\n" + traceback.format_exc(), flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
