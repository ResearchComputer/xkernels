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
- **[issue-43-mxfp4-moe-gemm.md](./issue-43-mxfp4-moe-gemm.md)** - Fast MXFP4 grouped fused-MoE GEMM for DeepSeek-V4
- **[issue-44-mhc-pre-post.md](./issue-44-mhc-pre-post.md)** - Full MHC `mhc_pre` / `mhc_post` fusions for DeepSeek-V4

### MoE (Mixture of Experts)
- **[issue-26-mxfp4-moe-ep.md](./issue-26-mxfp4-moe-ep.md)** - Expert parallelism for quantized fused-MoE
- **[issue-28-mxfp4-paged-gather.md](./issue-28-mxfp4-paged-gather.md)** - MXFP4 paged KV gather for DeepSeek-V4 DSA indexer

### Technical Specs & Plans
- **[superpowers/plans/2026-06-11-issue-18-moe-align-syncfree.md](./superpowers/plans/2026-06-11-issue-18-moe-align-syncfree.md)** - Implementation plan for sync-free `moe_align_block_size`
- **[superpowers/specs/2026-06-11-issue-18-moe-align-syncfree-design.md](./superpowers/specs/2026-06-11-issue-18-moe-align-syncfree-design.md)** - Detailed design spec for issue #18

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
├── benchmarking-on-beverin.md
└── superpowers/
    ├── plans/                        # High-level implementation roadmaps
    │   ├── 2026-06-10-xkernels-scaffold.md
    │   ├── 2026-06-11-issue-16-tuned-moe-int4-config.md
    │   ├── 2026-06-11-issue-17-bf16-dense-gemm-characterization.md
    │   ├── 2026-06-11-issue-18-moe-align-syncfree.md
    │   ├── 2026-06-11-issue-20-fused-combine-epilogue.md
    │   ├── 2026-06-11-issue-32-sparse-mla-attention.md
    │   └── 2026-06-12-issue-36-mhc-prenorm-gemm.md
    └── specs/                        # Detailed design specifications
        ├── 2026-06-10-xkernels-scaffold-design.md
        ├── 2026-06-11-issue-16-tuned-moe-int4-config-design.md
        ├── 2026-06-11-issue-17-bf16-dense-gemm-characterization-design.md
        ├── 2026-06-11-issue-18-moe-align-syncfree-design.md
        ├── 2026-06-11-issue-20-fused-combine-epilogue-design.md
        ├── 2026-06-11-issue-32-sparse-mla-attention-design.md
        └── 2026-06-12-issue-36-mhc-prenorm-gemm-design.md
```

## Documentation Standards

### Issue Docs (`issue-XX-*.md`)
Each issue document follows this structure:
1. **Context** - What problem is being solved
2. **What ships** - Description of the implementation
3. **API** - Function signatures and parameters
4. **Correctness (acceptance)** - Test criteria and validation results
5. **Notes / scope** - Limitations and related work

### Superpowers Specs (`superpowers/specs/`)
Detailed design specifications include:
1. **Purpose** - What this change achieves
2. **Design** - Technical approach and algorithm details
3. **Components** - Files and functions to modify/create
4. **Data flow** - How data moves through the system
5. **Testing** - Unit tests and on-device validation
6. **Deliverable acceptance** - Concrete success criteria

### Superpowers Plans (`superpowers/plans/`)
High-level implementation plans:
1. **Goal** - What needs to be done
2. **Architecture** - High-level design
3. **Tech Stack** - Tools and libraries used
4. **File structure** - Directory layout
5. **Task breakdown** - Step-by-step implementation checklist

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
