# scripts/archive/issues/ — per-issue one-shots

Reproducibility trail for closed issues. Two kinds live here:

## Flat `*.sbatch` (Tier-A SLURM jobs, committed)

The supported per-issue SLURM jobs (bench/probe/tune/test on beverin), one per
closed `meta/docs/kernels/*.md`. Cited from those kernel docs by path; submit
via `scripts/cluster.sh submit --host beverin <job>.sbatch`.

## Per-issue probe subdirs (one-shot Python diagnostics)

The `_*.py` / `diag_*.py` probes that diagnosed a specific issue, moved out of
the supported `scripts/` surface once the issue closed and its conclusion was
written to a card / `meta/docs/wiki/` / `meta/docs/kernels/` doc. Not
maintained — do not assume they run cleanly without the exact inputs/cluster
state of their original investigation. Nothing in `src/` or `tests/` imports
them.

| subdir | issue(s) | conclusion lives in |
|---|---|---|
| `rope-68/` | #68 apply_rope reference-backed lowering + OOB diagnosis | `meta/docs/kernels/attention.md`, wiki 02/03 |
| `int4-moe-50-65/` | #50 fused INT4/MXFP4 MoE, #65 fused_combine default | `meta/docs/kernels/moe.md`, wiki §16 |
| `paged-attn-71-52/` | #71 varlen paged GQA, #52 reusable output-buffer workspaces | `meta/docs/kernels/attention.md`, wiki §15 |
| `sampling-topk-70/` | #70 fused topk_softmax MoE-gating | `meta/docs/kernels/moe.md` |
| `ffn-82/` | #82 fused_ffn + moe_sum_reduce numerics (pre-fix diagnosis) | commit `fed0579`, wiki 02/03/04, `norm.md` |
| `misc-probes/` | #75 H1/H2 count, #77, generic diag/parity/perf-record | wiki 02/03, `meta/benchmarks/` |

## campaigns/2026-07-05-sweep/

Raw output (`sweep-*.{json,md}`, `tune-*.log`) of the 2026-07-05 sweep campaign
produced by `meta/benchmarks/sweep_all.py` + `tune_pointwise.py`. The curated
tables are in `meta/docs/wiki/07-campaign-2026-07-05.md`; these are the raw
structured data kept for re-verification. Regenerable via the
`sweep_{beverin_cdna3,bristen_sm80}.sbatch` jobs.
