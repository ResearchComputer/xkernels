# Benchmarks

Each `bench_<type>.py` sweeps representative shapes across every registered
backend and prints a markdown timing table.

```bash
python meta/benchmarks/bench_ffn.py --dtype float16
```

Backends only appear if their deps/build are present on the machine
(`reference` always; `triton`/`cuda` on supported hardware). Timing uses
`xkernels.utils.benchmarking.benchmark` (Triton `do_bench` when available).

`bench_all.py` is the consolidated runner: it times every single-GPU kernel's
optimized backend against its naive-PyTorch baseline and prints the markdown
speedup table used in the top-level README's **Performance** section.

```bash
python meta/benchmarks/bench_all.py            # single GPU
sbatch scripts/slurm/bench_all_beverin.sbatch     # beverin (MI300A)
```
