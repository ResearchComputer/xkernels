---
name: port-across-arch
description: >
  Adapt a kernel that is correct on one arch of a vendor to ANOTHER arch of the
  SAME vendor (NVIDIA sm_80 -> sm_90: TMA, thread-block clusters, wgmma; AMD
  cdna2 -> cdna3: larger MFMA, the fnuz fp8 path), RE-VALIDATING the unchanged
  contract. The Op Spec NEVER moves — portability lives in the contract, not the
  source (§10); you add a new Impl Card + re-record measurements on the new arch,
  and confirm cross-arch parity. Today the repo's backends are Triton (no native
  CUDA/HIP source yet), so the concrete path is: retune the Triton autotune
  configs for the new arch's matrix-engine shapes (matrix_instr_nonkdim/kpack on
  AMD; num_warps/num_stages/tile sizes on NVIDIA) and re-record — the native
  sm80->sm90 (TMA/clusters/wgmma) and cdna2->cdna3 intrinsics path is documented
  for when native cards exist. GPU-gated on the target arch. Use when find_impl
  returns a card for one arch of a vendor but missing_arch for a sibling arch.
license: Apache-2.0
x-kernel-lib:
  id: port-across-arch@1.0.0
  backend_scope: [cuda]
  when_to_use:
    triggers:
      - "find_impl returns a card for one arch of a vendor (e.g. nvidia_sm80) but missing_arch for a sibling (nvidia_sm90), same vendor + backend"
      - "a card exists on cdna2 and a task targets cdna3 (or vice versa) — same family"
    preconditions:
      - "source card verify(...).correctness.passed == true on its arch"
      - "source and target arch are the SAME vendor + backend (cross-vendor is port-cuda-to-hip; cross-backend is a different skill)"
      - "the Op Spec exists and does NOT change (portability is in the contract, §10)"
  inputs_required:
    - "source impl_card_id + its arch"
    - "target arch (same vendor: nvidia_sm80<->sm90, amd_cdna2<->cdna3)"
  tools:
    - get_impl_card
    - get_op_spec
    - verify
    - verify_parity
    - record_measurement
  validation:
    must_pass:
      - "new-arch card verify(...).correctness.passed == true on the target arch (vs the UNCHANGED reference)"
      - "verify_parity agrees ACROSS ARCHS of the same vendor (the two arch cards must agree at cross_backend_rtol — same op, same reference, same tolerance)"
      - "new-arch card is recorded with its own (arch, shape, dtype) measurement; the old arch's measurement is untouched (do not overwrite cross-arch)"
  references:
    - "registry/schema/impl_card.schema.json (arch.family enum: nvidia_sm80, nvidia_sm90, amd_cdna2, amd_cdna3)"
    - "src/xkernels/ops/gemm/triton/configs.py (per-arch autotune config selection: decode vs prefill tiles, MFMA shapes)"
    - "meta/docs/library.md §10 (portability in the contract, not the source; never hardcode arch constants), §9 (cross-arch parity is part of the milestone acceptance)"
    - ".agents/skills/autotune-knob-sweep/SKILL.md (re-tune + record the new-arch winner)"
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

## Procedure

1. **Confirm same-vendor + same-backend.** This skill is within a vendor
   (NVIDIA sm80<->sm90, AMD cdna2<->cdna3). Cross-vendor (CUDA->HIP) is
   `port-cuda-to-hip`; cross-backend (triton->native-hip) is author-an-op-spec +
   a new card. Read the source card's `arch.family` and the target; if the vendor
   differs, re-route.

2. **Read the contract once; it does not change.** `get_op_spec`. The
   constraints, numerics, reference, and tolerances are arch-invariant — that is
   the whole point of "portability lives in the contract" (§10). Do NOT touch the
   Op Spec to "make it work on the new arch"; if it doesn't meet the spec, fix the
   card, not the spec.

3. **Author the new-arch card** (`<op>.<backend>.<targetarch>.card.json`):
   `arch.family` = target, `arch.wave_size` = 32 (NVIDIA) / 64 (AMD) read from
   the family, `provenance.derived_from: <source_card>`, `provenance.skill_used:
   [port-across-arch]`. Copy the source `specialization_knobs` verbatim — the
   tunable surface is the same across archs of a vendor; only the WINNER differs.

4. **Adapt the kernel to the new arch's matrix engine / features.**
   - **Triton path (today's repo reality):** the kernel source is arch-portable;
     what changes is the autotune config space. Retune for the new arch:
     - AMD cdna2 -> cdna3: larger MFMA shapes (32x32 fp8), the fnuz fp8 path
       (`matrix_instr_nonkdim`, `kpack`), wider LDS. See `configs.py`.
     - NVIDIA sm80 -> sm90: prefer larger tiles + more `num_stages` (sm90 has
       more smem), and on native builds TMA/wgmma (not yet in repo).
   - **Native path (when native cards exist):** sm80 -> sm90 adds TMA (tensor
     memory accelerator for the async global->smem copy), thread-block clusters,
     and wgmma (warp-group matrix multiply); cdna2 -> cdna3 adds the larger MFMA
     and the native fp8 fnuz decode. Each is a real source edit, gated on a GPU.

5. **Verify on the target arch + cross-arch parity.** `verify(<new_card>,
   target_arch)` vs the unchanged reference. Then `verify_parity(op,
   archs=[source_arch, target_arch])`: the two arch cards of the same vendor must
   agree at `cross_backend_rtol` (same reference, same tolerance — this is part of
   the §9 milestone acceptance). A divergence here is a genuine arch-specific
   numerics difference (e.g. cdna3's wider MFMA reorders the fp32 sum); keep it
   within the looser tolerance or widen with justification (open question §11:
   setting cross_backend_rtol).

6. **Autotune + record on the new arch.** `autotune-knob-sweep` the declared knob
   space for the target arch; `record_measurement(new_card, target_arch, shape,
   dtype, knobs=<winner>, ms=..., source=<run_id>)`. The source arch's measurement
   stays put — measurements are per-arch and never overwritten cross-arch.

## Pitfalls

- **Editing the Op Spec "for the new arch."** The contract is arch-invariant
   (§10). A spec change ripples to every arch's card; if only one arch struggles,
   fix that card. This is the most common and most damaging mistake.
- **Overwriting the source arch's measurement.** `perf.measured` entries are keyed
   by (arch, shape, dtype); the new arch gets its own entry. Clobbering the old
   one destroys the cache that lets the next task skip autotuning on that arch.
- **Assuming cross-arch is automatically numerically identical.** A wider MFMA
   (cdna3) or wgmma (sm90) changes contraction association; verify_parity across
   archs is mandatory, not optional. Divergence within cross_backend_rtol is
   genuine; outside it is a bug.
- **Hardcoding the source arch's tile/wave constants.** `wave_size` flips between
   families; tile sizes that were optimal on sm80/cdna2 are usually wrong on
   sm90/cdna3. Retune; don't copy the winner.
- **Conflating with cross-vendor porting.** sm80 -> sm90 is this skill; CUDA ->
   HIP is port-cuda-to-hip. They have different success criteria (this one keeps
   the backend and checks cross-ARCH parity; the other changes backend and checks
   cross-BACKEND parity). Mixing them ships "works on the new arch but slow."
- **Treating Triton-arch-portability as done because "Triton is portable."**
   Portable means "compiles," not "fast." The autotune configs still need
   per-arch retuning and re-recording, or the new-arch card runs but misses its
   regime (the "it runs ≠ it's good" lesson, §10, applied within a vendor).
