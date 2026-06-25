# Documentation Index

This directory contains documentation for the xkernels project, organized by issue and feature.

## Quick Links

- [Document Drift Report](./DRIFT_CHECK_REPORT.txt) - Latest `check_document_drift.py` output (auto-generated)

## By Issue

### Core Features
- **[adding-a-kernel.md](./adding-a-kernel.md)** - Guide for adding new kernels to the codebase
- **[benchmarking-on-beverin.md](./benchmarking-on-beverin.md)** - How to benchmark/test on the beverin MI300A cluster via `rcc`

### Distributed & Collectives
- **[issue-12-hierarchical-all-reduce.md](./issue-12-hierarchical-all-reduce.md)** - Topology-aware hierarchical all-reduce with HIP-graph capture results

### DeepSeek-V4 Support (AMD gfx942)
- **[issue-17-bf16-dense-gemm.md](./issue-17-bf16-dense-gemm.md)** - bf16 dense GEMM MFMA characterization and `TORCH_BLAS_PREFER_HIPBLASLT=0` recommendation
- **[issue-32-sparse-mla-attention.md](./issue-32-sparse-mla-attention.md)** - Sparse-MLA attention compute for DeepSeek-V4
- **[issue-36-mhc-prenorm-gemm.md](./issue-36-mhc-prenorm-gemm.md)** - MHC hidden-compression prenorm GEMM
- **[issue-38-fp8-blockscale-gemm.md](./issue-38-fp8-blockscale-gemm.md)** - fp8 block-scale dense GEMM (portable gfx942 path)
- **[issue-39-v4-perf-pass.md](./issue-39-v4-perf-pass.md)** - Tunable launch knobs for the V4 sparse-MLA + MHC prenorm GEMM
- **[issue-41-fp8-mfma-blockscale-gemm.md](./issue-41-fp8-mfma-blockscale-gemm.md)** - Native fp8 MFMA fast path for the block-scale GEMM
- **[issue-43-mxfp4-moe-gemm.md](./issue-43-mxfp4-moe-gemm.md)** - Fast MXFP4 grouped fused-MoE GEMM for DeepSeek-V4
- **[issue-44-mhc-pre-post.md](./issue-44-mhc-pre-post.md)** - Full MHC `mhc_pre` / `mhc_post` fusions for DeepSeek-V4

### MoE (Mixture of Experts)
- **[issue-26-mxfp4-moe-ep.md](./issue-26-mxfp4-moe-ep.md)** - Expert parallelism for quantized fused-MoE
- **[issue-28-mxfp4-paged-gather.md](./issue-28-mxfp4-paged-gather.md)** - MXFP4 paged KV gather for DeepSeek-V4 DSA indexer

## Directory Structure

```
docs/
├── DRIFT_CHECK_REPORT.txt            # Auto-generated code vs documentation consistency analysis
├── adding-a-kernel.md                # Developer guide for adding new kernels
├── issue-12-hierarchical-all-reduce.md
├── issue-17-bf16-dense-gemm.md
├── issue-20-fused-combine.md
├── issue-26-mxfp4-moe-ep.md
├── issue-27-dsa-indexer.md
├── issue-28-mxfp4-paged-gather.md
├── issue-32-sparse-mla-attention.md
├── issue-36-mhc-prenorm-gemm.md
├── issue-43-mxfp4-moe-gemm.md
├── issue-44-mhc-pre-post.md
└── benchmarking-on-beverin.md
```

## Documentation Standards

### Issue Docs (`issue-XX-*.md`)
Each issue document follows this structure:
1. **Context** - What problem is being solved
2. **What ships** - Description of the implementation
3. **API** - Function signatures and parameters
4. **Correctness (acceptance)** - Test criteria and validation results
5. **Notes / scope** - Limitations and related work

## Related Resources

- **Tests**: See `tests/` directory for unit tests and integration tests
- **Benchmarks**: See `benchmarks/` directory for performance benchmarks
- **SLURM Jobs**: See `slurm/` directory for on-cluster test scripts
- **Source Code**: See `src/xkernels/` directory for implementation

## Document Drift Status

For the latest analysis of code vs documentation consistency, run
`python docs/check_document_drift.py` (or see the auto-generated
[DRIFT_CHECK_REPORT.txt](./DRIFT_CHECK_REPORT.txt)).

**Current Status**: ⚠️ Minor drift is tracked automatically. Known items:
- Several internal helpers / backend functions are intentionally undocumented.
- `mhc_pre` / `mhc_post` (issue #44) now have a dedicated doc; previously they
  were implemented but only mentioned in the MHC module docstring.
- The README bf16 performance note has been scoped to the `torch.matmul` (NN)
  layout following the issue #17 characterization.
