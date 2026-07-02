---
name: mixed-precision-convert
description: >
  Take an existing kernel (typically fp32) to a lower precision — bf16/fp16
  inputs with fp32 accumulation, or fp8/int4 with dequant — and RE-DERIVE the
  numerics contract honestly: the reference STAYS the high-precision source of
  truth, only the backend cards move precision, and tolerances + cross-backend
  parity are re-validated against the mixed-precision math (not hand-waved). The
  numerics/tolerance work is CPU-doable and is the meat of this skill; the
  kernel-side dtype plumbing is GPU-gated. Use when an op has an fp32 card and a
  task asks for a bf16/fp16/fp8/int4 card on the same arch, or when an issue
  specifies a mixed-precision acceptance bar.
license: Apache-2.0
x-kernel-lib:
  id: mixed-precision-convert@1.0.0
  backend_scope: agnostic
  when_to_use:
    triggers:
      - "an fp32 (or higher-precision) card exists and a task needs a bf16/fp16/fp8/int4 card for the same op + arch"
      - "an issue doc specifies a mixed-precision acceptance tolerance (e.g. bf16 2e-2, fp8 dot 1e-2 rel)"
      - "find_impl is asked for a dtype the op has no card for"
    preconditions:
      - "the op has an Op Spec and a backend-neutral reference (author-an-op-spec if not)"
      - "the reference is the high-precision source of truth (NOT itself converted — see Pitfalls)"
  inputs_required:
    - "impl_card_id (the source-precision card)"
    - "target dtype + accumulation dtype"
    - "the honest tolerance source for the target precision"
  tools:
    - get_op_spec
    - get_impl_card
    - verify
    - verify_parity
    - record_measurement
  validation:
    must_pass:
      - "the new lower-precision card verifies against the UNCHANGED high-precision reference at the re-derived tolerance (verify().correctness.passed == true)"
      - "numerics.by_dtype carries the target dtype with a tolerance traced to a real source (issue doc / reference library / mixed-precision math), cited in numerics.notes"
      - "verify_parity agrees across the now-mixed backends (precision conversion is numerics-relevant)"
      - "the reference's numerics.reference path and reduce_dtype are UNCHANGED (only backend cards moved precision)"
  references:
    - "src/xkernels/ops/gemm/reference.py (fp8 dequant -> fp32 matmul; dot_bf16 knob; e4m3fn vs e4m3fnuz)"
    - "src/xkernels/ops/moe/w4a16.py + reference.py (uint4b8 dequant via group scale, fp32 grouped-GEMM accumulate)"
    - "meta/docs/library.md §5.4 (fp16/bf16 accumulation order differs across vendors -> cross_backend_rtol is looser), §10 (CUDA-shaped reference anti-goal)"
    - "meta/docs/kernels/gemm.md (mixed-precision acceptance bars — the fp8 block-scale portable/mfma paths)"
  metrics:
    uses: 0
    success_rate: null
    median_iterations: null
    regression_count: 0
  provenance:
    authored_by: human
    created: "2026-06-25T00:00:00Z"
    supersedes: []
---

> **Run it on a GPU — ds5 via rcc + docker.** `verify` / `verify_parity` are
> device calls. Sync and run them inside the NGC container on the GB10
> (`arch="nvidia_sm121"`):
> ```bash
> rcc --profile ds5 push
> rcc --profile ds5 run --docker -s 'python -c "from xkernels import verify; print(verify(\"<new_card>@1.0.0\", arch=\"nvidia_sm121\")[\"correctness\"][\"passed\"])"'
> ```
> `-s` = shell snippet (heredocs/pipes ok); `--docker` uses the profile container
> (`PYTHONPATH=/workspace/src` set, no venv). DSL ops not yet imported by
> `ops/<x>/__init__.py` need `register_dsl(spec_of(<body>),"triton")` first.
> AMD/gfx942 → `scripts/cluster.sh run --host beverin`. Full recipe:
> `meta/docs/usage/ds5-testbed.md`.

## Procedure

1. **Read the op's numerics story.** `get_op_spec(op_id)`. Note `numerics.reference`
   (the high-precision oracle), `reduce_dtype`, and existing `by_dtype` entries.
   The invariant for this whole skill: **the reference does not move precision.**
   It is the fp32 (or fp64) source of truth every lower-precision card is graded
   against. Converting the reference itself is the cardinal sin (§10: a reference
   that drifts toward one precision silently tilts "correct").

2. **Pick the precision policy and justify it.** For each candidate state the
   accumulation dtype and where the precision drops:
   - **bf16/fp16 inputs, fp32 accumulate:** the standard. Inputs cast at the
     kernel boundary; the inner product / reduction stays fp32. Tolerance ~1e-2
     rel for bf16, tighter for fp16 — driven by the mantissa width, not guesswork.
   - **fp8 inputs (block-scale / per-token-group):** dequant to fp32 (or bf16 for
     a `dot_bf16` fast path), accumulate fp32. The dequant MUST be bit-identical
     across backends (route operands through the reference's quant helpers — see
     author-an-op-spec step 5), so divergence is fp32 accumulation order only.
   - **int4/uint4b8 weights (W4A16):** weight dequant via the group scale; the
     activation stays 16-bit. Same bit-identical-dequant rule.
   Cite the tolerance source in `numerics.notes` (issue doc, reference library's
   bar, or the explicit mantissa/accumulation math).

3. **Set `numerics.by_dtype` for the target precision.** Do NOT widen the
   existing dtype's tolerance to cover the new one — give the new dtype its own
   entry. `cross_backend_rtol` stays looser than any single-backend `rtol` (§5.4:
   accumulation order differs across vendors; tightening it to chase bit-equality
   false-fails parity). If the new precision is fp8, also record the *encoding*
   expectation in notes (e4m3fn vs e4m3fnuz — see step 5).

4. **Author the lower-precision Impl Card.** `specialization_knobs` should expose
   any precision-switch knob the entry callable actually accepts (e.g.
   `dot_bf16`, fp8 `path`), but NOT internal tiles (author-an-op-spec pitfall).
   `provenance.derived_from: <higher-precision card>`. The card resolves to a
   runtime callable via `(op_spec.kernel, backend)`, so the kernel source must
   plumb the new dtype through to the matrix engine — that source edit is the
   GPU-gated part.

5. **Encoding is part of precision on AMD.** For fp8 on CDNA3 (gfx942), the
   matrix-core op decodes `float8_e4m3fnuz` natively; the OCP `e4m3fn` encoding
   upcasts to a slower f16 MFMA (see `src/xkernels/ops/gemm/triton/entry.py` and
   `configs.py`). If the target arch is AMD CDNA3, record the fnuz expectation in
   the card `notes` and make the input generator produce fnuz operands for that
   arch — else the card "runs" but silently misses its perf regime (a
   `tune-for-cdna` / `map-to-matrix-cores` problem, not a correctness one).

6. **Verify + parity.** `verify(<new_card>, arch)` against the *unchanged*
   reference at the new tolerance. `verify_parity(op)` — now that backends are
   mixed-precision, this is the gate that catches a backend that dropped
   precision somewhere you didn't intend (e.g. accumulated in fp16). On GPU,
   `record_measurement` the new card per (arch, shape, dtype).

7. **Honest no-GPU branch.** The numerics contract (steps 1–3) and the reference
   CPU gate are fully doable without a GPU. The kernel-side dtype plumbing
   (step 4) and any perf claim are GPU-gated. Ship the card `compiled:false`
   honestly once the contract is right; the dtype-plumbed kernel + measurement
   land on a GPU.

## Pitfalls

- **Converting the reference to the target precision.** The reference is the
  high-precision oracle; if it moves to bf16, "correct" silently means "matches
  bf16," and no card can ever catch a precision bug. Keep it fp32+.
- **Widening an existing dtype's tolerance to cover the new one.** Each precision
  gets its own `by_dtype` entry with its own justification. A blanket rtol that
  passes both fp32 and fp8 carries no information.
- **Collapsing single-backend rtol and cross_backend_rtol.** Cross-backend is
  looser on purpose (§5.4). If you set them equal you either false-fail parity
  (too tight) or stop catching real numeric drift (too loose).
- **fp8 on AMD with the wrong encoding.** `e4m3fn` operands on gfx942 upcast to a
  slow f16 MFMA; the card "works" but the speedup never appears and you'll
  misdiagnose it as a tiling problem. Use `e4m3fnuz` on CDNA3 (fnuz family).
- **Quantized operands re-randomized per backend.** fp8/int4 inputs must come
  from the reference's exact-inverse quant helpers so all backends dequant
  identical bits (else parity measures your RNG, not the kernel).
- **Treating a precision conversion as numerics-irrelevant.** It is the opposite
  — it is THE most numerics-relevant change. Always re-run verify_parity.
