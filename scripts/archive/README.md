# scripts/archive/ — one-shots & exploration scratch

Scripts moved out of the supported `scripts/` harness to keep that surface small.
They are kept (not deleted) because they are the **reproducibility trail** for
findings already baked into the registry and docs. They are not maintained — do
not assume they run cleanly without the exact inputs/cluster state of their
original campaign. Nothing in `src/` or `tests/` imports them.

## ds5-probes/

The `ds5_*.py` probes (+ `m128_bf16_rootcause.py`) that mapped the CUTE-DSL
(`cutlass.cute`) surface on the ds5 Grace+Blackwell GB10 (sm_121) test bed while
the first native-CUDA cards were authored.

- **Findings live in:** `meta/docs/usage/ds5-testbed.md`,
  `meta/wiki/05-cutedsl-authoring.md`, and the `source`/`notes`/`regime` fields of
  the `*.cuda.card.json` cards (which cite individual probes by path).
- **Still runnable** (with the ds5 venv + `CUDA_HOME=/usr/local/cuda-13.0`): the
  analysis/tools — `ds5_verify_card.py`, `ds5_roofline_survey.py`,
  `ds5_dsl_math_probe2.py`, `ds5_dsl_rowsum_probe.py`, `ds5_bf16_load_probe.py`.
- **Superseded iterations were deleted** during consolidation: `ds5_dsl_math_probe.py`
  (→ cited `…probe2.py`) and `ds5_jit_cache_probe.py` (→ cited `ds5_jitcache_probe.py`).
  The remaining `*_probe2.py` / `*_final*` / `*_solved*` files are the **cited keepers**.

## campaigns/

Dated, single-purpose drivers:

- `record_campaign_measurements.py` — one-shot write-back of the 2026-06-26
  benchmark/profile campaign into card `perf.measured` (via
  `xkernels.registry.writeback.record_measurement`). Its 17 rows are already in the
  cards; re-running would just re-record the same points.
- `bench-pr-on-beverin.sh` — push + run a single PR's benchmark file on an MI300A
  node. Superseded for general use by `scripts/cluster.sh run --host beverin`
  + `scripts/cluster.sh submit --host beverin`.
