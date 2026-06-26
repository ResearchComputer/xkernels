---
name: diagnose-wrong-results
description: >
  Diagnose a kernel that FAILS on a real GPU after passing on CPU / Triton-interpreter
  — either a hard crash (illegal-memory-access / SIGSEGV / CUDA/HIP error) OR a
  numerical mismatch in verify() / verify_parity() — BEFORE any perf work. The
  peer of diagnose-low-occupancy / diagnose-memory-bound (which assume verify
  PASSES): those fire on a correct-but-slow card; this fires on a wrong-or-crashing
  one. Encodes the GPU debugging ladder the interpreter cannot surface: (1)
  reproduce in isolation, NOT in the pytest harness (which adds autotune-pinning +
  parametrize-ordering confounders); (2) the isolation ladder (in-isolation ->
  in-sequence -> on-main -> on-stash) to separate your change from pre-existing /
  state-pollution failures; (3) the crash-localization trichotomy — CUDA_LAUNCH_BLOCKING
  / HIP_LAUNCH_BLOCKING, compute-sanitizer / rocgdb, and bypass-autotune — whose
  three-way PASS/FAIL signature fingerprints the root cause; (4) pin/bypass/mask
  to confirm. The load-bearing gotcha this captures: the autotune wrapper corrupts
  certain dispatches under concurrency (PASSES under blocking, CLEAN under the
  sanitizer, PASSES the resolved-config direct path) — the most time-consuming
  failure mode in an autotune-heavy repo, and one no single profiler tool exposes.
  GPU-gated on the failing backend. Use whenever interpreter-validated code crashes
  or fails verify on first contact with a real GPU.
license: Apache-2.0
x-kernel-lib:
  id: diagnose-wrong-results@1.0.0
  backend_scope: [cuda, hip]
  when_to_use:
    triggers:
      - "a kernel that passed on CPU / TRITON_INTERPRET=1 raises illegal-memory-access / SIGSEGV / a CUDA or HIP runtime error on a real GPU"
      - "verify() or verify_parity() reports a numerical mismatch on GPU that did not reproduce under the interpreter"
      - "a kernel is correct in isolation but crashes / drifts when run after other dispatches in the same process (state pollution)"
    preconditions:
      - "a failing dispatch on a real GPU (this is a correctness/crash skill, NOT a perf skill — if verify PASSES but ms is bad, leave now and use use-nsight-compute / use-rocprof-compute + a diagnose-*perf* skill)"
      - "a standalone seeded reproduction script (NOT the pytest harness) for the failing shape"
      - "shell access to the failing backend's debugger: compute-sanitizer (NVIDIA) / rocgdb + ASan (AMD); and the ability to set CUDA_LAUNCH_BLOCKING / HIP_LAUNCH_BLOCKING"
  inputs_required:
    - "the failing shape, dtype, backend (and the dispatch path: autotuned vs pinned vs direct)"
    - "a CPU reference or oracle value to compare the GPU output against"
    - "a git-clean baseline (main) you can stash onto to answer 'did I break this'"
  tools:
    - verify
    - verify_parity
    - get_impl_card
  validation:
    must_pass:
      - "the failing dispatch now passes verify() on GPU AND in the process state that used to crash (run it after the small buckets, not just in isolation)"
      - "verify_parity still agrees (a crash fix that shifts numerics changed the math — back it out)"
      - "the full test suite (not just the single failing case) is green on GPU"
      - "the root cause is NAMED, not just masked: you can state which of {true OOB in the kernel / autotune-wrapper corruption / harness artifact / logic-value bug} it was, and why the fix is correct"
  references:
    - ".agents/skills/use-nsight-compute/SKILL.md (graduate here once correctness passes — it profiles a correct card)"
    - ".agents/skills/use-rocprof-compute/SKILL.md (the AMD twin)"
    - ".agents/skills/diagnose-low-occupancy/SKILL.md, .agents/skills/diagnose-memory-bound/SKILL.md (the perf-diagnosis peers — both REQUIRE verify().correctness.passed == true, which this skill exists to restore)"
    - "src/xkernels/ops/moe/triton/moe_int4_kernel.py, src/xkernels/ops/moe/triton/moe_mxfp4_kernel.py (the masked-gather fix that resolved the issue-#50 autotune-wrapper case — concrete worked example)"
    - "wiki/04-gotchas.md (entry: autotune-wrapper corruption under concurrency — the case study that produced this skill)"
  metrics:
    uses: 0
    success_rate: null
    median_iterations: null
    regression_count: 0
  provenance:
    authored_by: agent
    created: "2026-06-26T00:00:00Z"
    supersedes: []
---

> **Routing note — this is the "wrong/crashing" peer of the diagnose-*perf* skills.**
> `diagnose-low-occupancy` and `diagnose-memory-bound` both hard-require
> `verify().correctness.passed == true` — they fix a *correct-but-slow* card. This
> skill is what runs *before* that gate, to turn a wrong/crashing kernel back into a
> correct one. The cardinal sin is to reach for a profiler (`ncu`/`rocprof`) on a
> crashing dispatch: the profiler either dies with the crash, hides the bug under
> its own serialization, or reports an irrelevant stall reason. **Restore
> correctness in a standalone repro first; only then graduate to perf work.**

> **The interpreter lulls you into false confidence.** `TRITON_INTERPRET=1`
> executes the kernel in Python: every `tl.load` is in-bounds (it materializes the
> whole tensor), there is no autotune (no config trials), no async overlap, no
> concurrency between dispatches. So interpreter-green proves the *math* is right;
> it proves nothing about masks, autotune-wrapper interaction, OOB gathers, or
> cross-dispatch state. A kernel that is interpreter-green and GPU-red is the
> canonical trigger for this skill.

## Procedure

1. **Reproduce in a standalone seeded script, NOT in pytest.** This is step zero
   and the most-skipped. The pytest harness adds three confounders that either
   *hide* a real bug or *invent* a fake one:
   - **autotune config pinning** (`_pin_single_config` and friends) — forces one
     `BLOCK_SIZE_*`; valid only where the launcher's `align_block_m(M)` matches
     the pin (see Pitfalls). A pinned harness can turn a real misroute invisible,
     or fabricate a numerical failure on a larger shape.
   - **parametrize ordering** — earlier buckets populate the autotune cache and
     module-level buffers; a failure only in the Nth bucket is often *state
     pollution*, not a bug in that bucket's kernel.
   - **fixtures / scope / shared tensors** — silent aliasing across tests.

   Write a 30-line script: seed, build the one failing shape, call the kernel
   once, compare to a CPU oracle, print `max abs err`. If it passes standalone
   but fails in pytest, you have a harness/state bug — skip to the isolation
   ladder to prove which.

2. **Run the isolation ladder** to separate *your change* from pre-existing /
   order-dependent failures. Answer, in order:
   - **In isolation** (single shape, single call, fresh process) — does it fail?
     If NO but it fails in-suite → state pollution (autotune cache / module
     buffer). Reproduce by running the small buckets *then* the failing one.
   - **In sequence** — run the exact bucket order pytest uses; does the failure
     appear only after earlier buckets? That signature = pollution, not logic.
   - **On main** (`git stash`, re-run) — does main *also* fail it? If YES, the
     bug is pre-existing; stop debugging your diff and bisect main. If NO, your
     change caused it — but *which* part? Stash and re-apply file-by-file.
   - **On the other backend/arch** — fails everywhere → kernel *logic*; fails on
     one arch only → that arch's *dispatch* (autotune configs, wave size, dtype
     path).

   The ladder's payoff: this skill's author spent ~30 min "fixing" the M=128 EP
   numerical failure before the ladder proved it was a harness config-pin
   artifact (pin=16 vs `align_block_m(128)=64`) — *the kernel was correct*. The
   ladder exists to stop you fixing correct code.

3. **Run the crash-localization trichotomy** — for a crash or wrong-results, run
   the failing dispatch under three conditions and read the **signature** (the
   pattern across all three), not any single outcome:

   | condition | env / tool | what it removes | if it now PASSES, the culprit is … |
   |---|---|---|---|
   | serialize launches | `CUDA_LAUNCH_BLOCKING=1` / `HIP_LAUNCH_BLOCKING=1` | async overlap + cross-launch concurrency | a **race / ordering** between dispatches (classic: autotune trials vs the real launch) |
   | memory sanitizer | `compute-sanitizer` (NVIDIA) / `rocgdb` + ASan (AMD) | hides nothing; reports the first true OOB/UAF | if it PASSES with **0 errors**, the crash is NOT a memory-safety bug the sanitizer can reach → a **wrapper / scheduling** issue |
   | bypass autotune | resolve one config by hand, call the kernel directly (no `@triton.autotune`) | the autotune wrapper's trial-then-cache path | if it PASSES, the **autotune wrapper** corrupts this dispatch |

   Read the three-way signature, e.g.:
   - **PASSES/blocking + CLEAN/sanitizer + PASSES/direct** → **autotune-wrapper
     corruption** (the signature this skill was written from). No single tool
     shows this; only the *combination* does. → step 4, mask + pin/bypass.
   - **FAILS/blocking + sanitizer ERROR** → a **true OOB** in the kernel hot loop
     (the sanitizer names the line). → add the mask, confirm the sanitizer clears.
   - **FAILS/blocking + no sanitizer error** → a **logic / value** bug (no memory
     violation; the math is wrong). → diff the GPU output against the CPU oracle
     element-wise, localize the divergence to a tile/expert/block.
   - **PASSES/blocking + sanitizer ERROR** → the OOB is real but benign under
     serialization (a race exposes a narrower window). → mask the OOB *and*
     investigate the race.

4. **Confirm by pin / bypass / mask, and name the root cause.** The skill does
   not end at "it passes now" — it ends at "I can state why." Concretely:
   - **autotune-wrapper corruption:** mask every gather by its true extent (the
     defensive fix — autotune trials read OOB even when the cached winner is
     fine), AND if the wrapper still corrupts for this dispatch, pin/bypass it
     (resolve one config, or `_pin_single_config`) and **file the wrapper bug**
     separately. Document the fallback in the launcher so the next agent does
     not re-trip it (the issue-#50 INT4 EP path keeps the reference align for
     exactly this reason — see the launcher comment).
   - **true OOB:** add `mask=offs < N, other=<pad>` on the offending load; the
     sanitizer run must now be clean.
   - **harness artifact:** fix the *test* (e.g. drop the shape whose
     `align_block_m` mismatches the pin), not the kernel.
   - **logic/value bug:** fix the kernel; re-derive against the CPU oracle.

5. **Re-run the FULL suite + verify_parity on GPU, in process order.** Confirm
   the fix holds in the state that used to crash (run the small buckets *then*
   the failing one), not just in isolation. A crash fix that shifts numerics
   changed the math — back it out and redo. Only then graduate to
   `use-nsight-compute` / `use-rocprof-compute` for perf.

## Pitfalls

- **Debugging in the harness instead of a standalone repro.** The single biggest
  time sink. pytest's autotune-pinning + parametrize-ordering + shared fixtures
  invent and hide bugs in equal measure. Standalone seeded script FIRST, always.
- **Autotune-wrapper corruption under concurrency** (the load-bearing gotcha).
  Signature: PASSES under `CUDA_LAUNCH_BLOCKING=1`, CLEAN under `compute-sanitizer`
  (0 errors), PASSES the resolved-config direct path — but crashes under the
  autotuned launch. Mechanism: Triton's autotune *trials* configs whose tile
  shapes exceed the dispatch, and the kernel's gather/load reads OOB *during the
  trial*; the cached winner may be fine, but the trial corrupted state/output.
  Two-part fix: (a) **mask every gather by the true extent** (defensive, always —
  see next pitfall), and (b) if the wrapper still corrupts, **pin/bypass** the
  config for that dispatch and file the wrapper bug. This is the most
  time-consuming failure mode an autotune-heavy repo will hit, and no single
  profiler tool surfaces it — only the trichotomy does.
- **Unmasked gathers are latent bugs.** A `tl.load(ptr + offs)` where `offs` can
  exceed the array length (last block overshoot, autotune trial with a larger
  `BLOCK_SIZE_*`) MUST carry `mask=offs < N, other=<pad_id>`. An unmasked gather
  that "works" is passing by luck of the config/shape; it passes until a trial or
  a new shape reads past the end. Default to masking; it is result-preserving
  when `other=` equals the value `token_mask`/`<mask>` already drops.
- **Treating a clean sanitizer run as exoneration.** `compute-sanitizer` runs
  without the multi-pass counter machinery, and serializes enough that
  instrumentation-sensitive corruption (autotune trials) can vanish under it.
  "0 errors under the sanitizer" + "crashes without it" is the autotune-wrapper
  signature, NOT proof of innocence.
- **Treating `CUDA_LAUNCH_BLOCKING=1` passing as a fix.** It is a *diagnosis*
  (the bug is a race/ordering), never a fix — production never runs blocking.
  Blocking-pass must be followed by the real fix (mask + pin/bypass), then
  re-confirmed with blocking OFF.
- **Config-pin must agree with the align block.** A test that pins one autotune
  config (`BLOCK_SIZE_M=16`) is valid ONLY for shapes where the launcher's
  `align_block_m(M)` equals the pin. Extending a pinned parametrize list with a
  larger shape silently misroutes (the kernel builds a 16-wide dispatch for a
  64-wide block). Check `align_block_m(M)` before adding a bucket to a pinned
  test — CPU-checkable, catches it instantly.
- **Profiling a crashing dispatch.** `ncu`/`rocprof` replay counters and will
  either die with the crash or report a stall reason that has nothing to do with
  the bug. This skill runs BEFORE the profiler; the profiler runs once
  correctness is green.
- **Stopping at "it passes now."** A masked-out OOB, a config pin, or a
  harness-shape removal all *mask*; only the named root cause is a real fix.
  State which of {true OOB / autotune-wrapper / harness artifact / logic bug} it
  was, and why the fix is correct, before declaring done.
