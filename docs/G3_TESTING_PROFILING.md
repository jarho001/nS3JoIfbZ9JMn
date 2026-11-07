# G3 intermediate matched-battle diagnostic

2026-11-07 implementation follow-up: the optional lean wrapper and portable CLIs
are implemented, with validation/measurement pending. No tests or benchmarks were
run for that change by explicit user instruction. Use the current commands and
output contracts in [the results report](G3_TESTING_PROFILING_RESULTS.md#portable-follow-up-commands-not-executed)
for a later authorized run; the completed full-output diagnostic below is history.

This user-approved diagnostic supersedes the broad campaign below for this run.
Preserve all completed work and uncommitted simulator repairs. Work directly with
Sol 6.1/high, without delegation. The requested `/workspace/nS3JoIfbZ9JMn` does not
exist on this executor; use its existing checkout and evidence in `/workspace`.

Compare pinned CPU Pokémon Showdown with compiled JAX CPU and GPU on exactly
`F01-p1`, `F02-p1`, `F03-p1`: 34 + 33 + 61 = 128 request boundaries per pack.
Freeze the existing independent native expectations, teams, initial conditions,
seed words and every recorded choice into the diagnostic evidence directory.
Do not generate another campaign. Verify these three trajectories once on the
measured compiled CPU/GPU code: state, ordered RNG, requests/legal actions,
events/cumulative histories, terminal outcomes and actual device placement.
Save the first divergence and stop if the gate fails.

Use only one pack (3 battles) and 32 identical packs (96 battles). A fixed CPU
allocation of logical CPUs 0–7 applies to every backend, with OMP/BLAS budgets of
8 threads; Node uses one simulation thread. Record observed process thread
counts separately from active simulation parallelism. Use the existing isolated
Python environment, hardware and current uncommitted source; record hashes.

Reuse the existing initialization, projection, exact comparison and benchmark
synchronization helpers with a small fixed-pack adapter. JAX uses one compiled
scan over exactly 128 useful transitions per lane, selecting the fresh starting
state between the three battles. There are no padded or absorbing transitions.
Retain every full Result/state/event/draw trace, cumulative history and legal
mask. Showdown runs uninstrumented in one long-lived Node process with its normal
validation, PRNG, requests and string log retained. It does not retain JAX-style
numeric full-state snapshots and per-draw trace arrays; disclose this output
contract difference instead of stripping JAX work.

Prepare fresh independent input buffers/Battle objects outside each timed call.
Warm once, then measure three repetitions. The one-pack warm result also serves
as the trajectory gate. Record initialization, host preparation, compilation,
input transfer and warmup separately. Time only simulation through complete
output readiness; synchronize all GPU inputs before and all outputs after.
Exclude assertions, oracle observation, host projection, file I/O, subprocess
startup and cleanup. Report median and full range of elapsed time, completed
battles/s and useful boundaries/s. Sequential CPU/GPU execution avoids contention.

A single 300-second process-group timeout bounds execution including imports,
fixture freezing, validation, compilation, warmup and measurement. Save compact
checkpoints after each completed backend/point. Budget exhaustion means incomplete
evidence, with no automatic extension or retry. No fuzzing, suites, ablations,
optimization or mechanics work. No profiler unless timings expose a clear issue
and the same remaining budget permits one short capture.

Deliver one compact six-row comparison table, timing boundaries/hardware/source,
a short evidence-based interpretation and one useful follow-up measurement.
This restricted-slice diagnostic does not accept G3 or establish peak throughput.
Artifacts: `test-logs/g3-testing-profiling/diagnostic/`; progress:
`docs/ORCHESTRATION_STATE.json`; report: `docs/G3_TESTING_PROFILING_RESULTS.md`.

## Authorized follow-up: one warmed 96-battle GPU trace

The user authorized the proposed single trace after the diagnostic publication.
Reuse the frozen three-battle pack and exact recorded `run` function, 32 copies,
with all existing outputs retained. Keep affinity 0–7 and the existing software
and device. Cap this execution, including compilation/warmup/digest checks, at
300 seconds. Warm once, compare every lane with the previously native-validated
trajectory digest, capture one complete loop, then stop. Do not run an output
elision comparison or implement a faster mode in this follow-up. Inspect actual
hardware stream events separately from enclosing GPU annotations; do not treat
nested XLA spans as GPU busy time. Record output bytes and compiler memory
analysis as context, not a causal attribution of time. Artifacts are under
`test-logs/g3-testing-profiling/diagnostic-profile-96/`.

<details>
<summary>Superseded broad handoff (historical scope; do not execute)</summary>

# Handoff: extensive G3 testing and CPU/GPU bottleneck profiling

**Execution stopped by user, 2026-11-06.** The extensive CPU/GPU sweep below
is no longer authorized to continue. Establish a matched, uninstrumented legacy
Showdown-versus-GPU benchmark before any further benchmark execution. Existing
CPU/GPU timings do not establish speed relative to Showdown. Preserve the
intermediate evidence in [the results report](G3_TESTING_PROFILING_RESULTS.md)
and [compact state](ORCHESTRATION_STATE.json); G3 remains open.

Prepared 2026-11-06. This is an execution handoff, not an active campaign or new
performance evidence. Assign it explicitly to start the work below. Work directly
with Sol 6.1/high, without delegation or an independent review gate.

## Outcome and starting point

Establish how reliable the current restricted simulator is under a substantially
broader test campaign, verify it on an actual GPU, and identify its measured CPU
and GPU bottlenecks. Deliver reproducible benchmarks and a ranked engineering
recommendation. Do not promise a speedup or expand the supported game.

Use `/workspace/nS3JoIfbZ9JMn`. Read `AGENTS.md`, applicable cloud runtime/network
and GPU guidance, `PROJECT_PLAN.md` G3/D3/V0–V4/V6, current compact state,
`docs/SHOWDOWN_BEHAVIOR_CASES.md`, `docs/shared-engine.md`, and
`rules/shared-engine-v1.json`. Preserve local work and synchronize safely with
`origin/main`; never reset, force-push, or rewrite remote history.

The published starting point is `9b98ee8`; resolve the actual current commit/tree
rather than assuming it is unchanged. The accepted CPU evidence covers 102 native
scenarios / 448 boundaries, including eight complete three-member battles and
four complete legacy duels. All 155 catalog boundaries passed eager/JIT and
32-lane execution; full battles passed batches 1/8/32. Evidence is in
`test-logs/g3-first-playable/` and `docs/ORCHESTRATION_STATE.json`.

The semantic oracle is Showdown
`1242c1e2d9763ba5b85d33f13aecb89bae19ae3d`. The supported game is the versioned
1–3-member custom singles slice, not the random-team distribution or full format.
G3 remains open for actual GPU correctness/placement and profiling. G1 remains
open, G2 learning remains deferred, and the historical accepted task count is 52.
`benchmarks.shared_engine` is a proposed entry point, not an existing harness.

## Scope and resources

This assignment, when given for execution, includes long CPU/native regression,
new differential tests, GPU correctness checks, and performance/profiler runs
within this document. Build the missing benchmark harness and discriminating
tests as needed. Correctness repairs within the existing allowlist are in scope;
turn each discovered bug into a minimized independent native regression.
Production optimization is not the default deliverable: report bottlenecks and
recommend changes first. Keep any diagnostic ablation explicitly experimental.

Use available CPU resources and one already available compatible GPU. Record the
selected hardware, CPU thread budget, RAM/VRAM and software stack before running.
Use isolated compatible environments; preserve the existing CPU environment and
exact battle arithmetic/RNG. Do not provision paid hardware, change managed
infrastructure, or add multi-device work under this assignment.

Keep allocations below 80% of reported free host/device memory, allowing for
compiler and temporary buffers. Estimate state/trace/output sizes before large
runs. Stop increasing a sweep at its resource boundary and report skipped points;
do not blindly retry OOMs. There is no tiny per-mechanic time budget. Set practical
command timeouts, and checkpoint at the integrated boundaries below.

## Integrated checkpoint: freeze and reproduce the baseline

Record source commit/tree or relevant dirty-file hashes, reference identity,
manifest version, commands, environment versions, device placement, CPU affinity/
thread configuration, compilation-cache settings and artifact paths. Ensure the
reference checkout is clean and guarded. Use a fresh evidence directory and fresh
native output filename; the runner intentionally refuses to overwrite evidence.

Existing entry points, from the repository root:

```sh
node reference/oracle.cjs provenance
node reference/oracle.cjs format
node --test test/reference.test.cjs test/shared-engine.test.cjs
node reference/shared-engine-cases.cjs --batch all --output NEW_NATIVE_FILE
G3_NATIVE_CASES=NEW_NATIVE_FILE JAX_PLATFORMS=cpu "$JAX_PYTHON" -m unittest \
  test.test_shared_engine_cases test.test_shared_engine_battles \
  test.test_shared_engine_lifecycle -v
JAX_PLATFORMS=cpu "$JAX_PYTHON" -m unittest \
  test.test_accuracy_choice test.test_accuracy_choice_duel test.test_burn_choice \
  test.test_two_move_session_duel test.test_request_session_batch \
  test.test_two_move_session_reset -v
```

Set `JAX_PYTHON` to a verified interpreter; the current `.venv-jax/bin/python` is
CPU-only evidence, not proof of CUDA support. Run the broader primitive/legacy
suite once as a baseline if compatible: `JAX_PLATFORMS=cpu "$JAX_PYTHON" -m
unittest discover -s test -p 'test_*.py' -v`. Do not rerun every historical suite after each small change.
Separate setup/tooling failure from a semantic test failure.

## Integrated checkpoint: expand correctness evidence

Extend the existing shared native runner/harness instead of adding an oracle per
mechanic. Preserve exact relevant state, stable identity/request order, pending
choices/queue, HP/PP/status/counters/items/stats, RNG contexts/ranges/values/pre-
and post-states, requests/legal masks, both cumulative recipient histories and
outcomes. Normalize only documented timing/diagnostic paths. Expected outputs
must come from pinned native execution, never the JAX implementation.

Predeclare and persist seeds, team construction and policies before the campaign:

- Exercise one-, two- and three-member supplied teams, both seats, mixed move
  slots, levels 1/50/100 and sampled interior levels, allowed items and Speed EVs.
  Include tied speeds, changed speed after paralysis/switching, critical/miss/
  immunity, last PP, Struggle/Protect/recoil, repeated Protect, status reflection/
  cure, toxic saturation, residual KO order, voluntary and forced switches, and
  faster/slower/KO U-turn interruptions.
- Use random legal play and scripted tactical/edge policies. Target 1,000 seeded
  battle starts or 100,000 resolved request boundaries, whichever occurs first,
  with at most 512 request boundaries per start. Report actual starts, completed
  battles, decisions, boundaries, truncations, capacity diagnostics and effect/
  pairwise-interaction counts. No mismatch-free count proves equivalence.
- Keep ordinary starts separate from trusted HP/PP/status/counter fixtures.
  Exchange seats with explicit seed/policy identities; do not assume naive
  symmetry of tied RNG draws. Exercise choice replacement/undo, both submission
  orders, stale tokens, absorbing terminals, privacy pairs and live/stale resets.
- Add invariants for HP/PP bounds, legal transitions, RNG draw continuity, unused
  trace padding, zero-sum terminal rewards, atomic rejection/overflow and lane
  independence. Test event, draw, queue, history and token limits. A cap is
  truncation or a diagnostic, never an invented draw or win.
- Replay new regressions and representative interaction/long traces through
  eager/JIT, heterogeneous batches and saved continuation/reset. Keep full raw
  traces for failures and selected fixed samples, plus compact campaign coverage;
  stream comparisons rather than storing every enormous snapshot in Git.

At a divergence, save the original seed/sets/choices/source identity, then minimize
it to the first differing boundary/field/draw. Ordinary repairs and oracle
clarification remain in scope. Pause performance conclusions until the measured
workload passes again. Do not shrink the allowlist, skip a required family, or
relax assertions to obtain a pass.

## Integrated checkpoint: prove actual GPU correctness

Use the compatible GPU interpreter with `JAX_PLATFORMS=cuda` for the shared case,
battle and lifecycle modules above, adjusting the platform only for explicitly
selected hardware. Assert a GPU backend and actual device placement of inputs,
eager/JIT results and batched state/trace leaves; prohibit silent CPU fallback.

Compare CPU/native and GPU integer outputs exactly for the accepted corpus,
new minimized regressions, mixed normal/pending/pivot/forced/terminal/rejected
lanes and representative new campaign traces. Include continuation/reset and
capacity behavior. Host-side oracle generation, serialization and protocol
rendering are allowed; they are not device-resident engine execution.

If required GPU execution is unavailable, preserve exact source identity,
commands, versions, required hardware and CPU results in `gpu_block` with
`awaiting_gpu`. Continue independent CPU analysis, then stop GPU-dependent work.
Do not repeat blocked attempts without new readiness evidence.

## Integrated checkpoint: measure and explain bottlenecks

Implement one reusable harness, preferably `benchmarks/shared_engine.py`, with
versioned configuration, machine-readable results and separate profiler traces.
Benchmark identical supported workloads, action semantics, output contracts and
source revisions on compiled JAX CPU and GPU. Keep three measurements distinct:

1. Device kernel: prepared device inputs, `advance`, and explicitly consumed state
   and event/draw outputs; no oracle, Python assertions or file I/O in the timer.
2. Sustained execution: bounded compiled rollout chunks, real phase mixtures,
   terminal handling and supplied-state reset costs. Prevent terminal/padded lanes
   from making mostly no-op execution appear fast.
3. Host integration: submissions, legal-action handling, transfers and player
   projection/history/serialization. Report this separately from kernel timing.

For an additional native CPU comparison, use uninstrumented pinned Showdown on
the same supplied teams/choices and comparable outputs under a documented CPU
budget. The oracle's per-draw stacks/snapshots are validation overhead and must
not be presented as native simulator speed.

Start with batches 1/8/32, then 128/512/2048/8192 where memory permits. Sweep chunk
lengths 1/8/32/128 for the primary ordinary and mixed workloads; use selected
operating points for diagnostic cohorts. Cover ordinary unequal-speed turns,
tied turns, statuses/items/residuals, switches/replacements/pivots, exhaustion,
and realistic mixed lanes; measure terminal/rejected lanes separately.

Separate tracing/lowering and cold compilation from warmed execution; disclose
persistent-cache hits. Warm each shape/configuration and synchronize completion
with `block_until_ready` before and after timed work. Use at least ten measured
repetitions, each long enough to amortize timer overhead (target >=200 ms), and
report median and spread. Exclude setup/input placement from kernel timing and
measure those costs independently. Record output-retention policy: allowing XLA
to discard unused traces or retaining all full states across a scan changes the
workload. A checksum-only streaming experiment is labeled separately from an API
that materializes complete results.

Report request boundaries, player decisions, completed battles, useful versus
padded/absorbing work, compile time, warm latency/throughput, peak memory, reset
cost and host/device transfers/synchronizations. Do not call every kernel call a
turn or infer training speed from simulator-only throughput.

Use bounded JAX profiler traces, HLO/compiler inspection, built-in host profiling
and available GPU tools to test these hypotheses rather than assert them:

| Suspected cost | Evidence to seek |
|---|---|
| History and padded event/draw buffers | Bytes per lane, allocation/write traffic, peak temporaries, occupancy/capacity effects; inspect `empty_trace`, `_event`, `_draw`, history append. |
| Dynamic queues, sorts and nested loops | Kernel/HLO time, launch count, compilation growth, ordinary versus switching/residual cohorts; inspect `_qsort`, `_residual`, `_run_queue`. |
| Mixed-phase batching | Homogeneous versus heterogeneous cost and useful-lane fraction; inspect how `vmap` lowers conditionals, rather than assuming a particular GPU divergence mechanism. |
| Exact PRNG arithmetic | Draw-count-normalized cost and fused kernel behavior; preserve the exact recurrence and draw order. |
| Repeated validation | Cost of `_supported`, stat derivation and roster sorting per advance; do not silently remove input validation from the primary benchmark. |
| Host integration | Python dispatch/projection, per-leaf conversions and synchronizations in `shared_engine_io.py`; distinguish actor-interface cost from the device kernel. |
| Static shapes and dispatch | Compilation/recompilation counts, batch/chunk crossover, memory scaling and Python-loop versus compiled-chunk overhead. |

For each top bottleneck, provide a trace/counter or controlled measurement, its
share of the relevant workload, the proposed intervention, correctness risks and
the next discriminating experiment. Label unverified hypotheses. Diagnostic
ablations may estimate tracing/validation/buffer costs, but dropping required
history, RNG, views or diagnostics does not establish a valid production speedup.

## Completion, evidence and stopping

Save meaningful checkpoints in the existing compact state, and keep one concise
results report with links to configs, machine-readable measurements, coverage,
minimized regressions and selected traces. Capture source identity before its
evidence record; invalidate affected measurements after a relevant source change.
Commit/push under the existing authorization, preserving unrelated work. Do not
create numbered tasks, duplicate reports or a new review ceremony.

Finish with: executed versus pending checks; complete/truncated/failing campaign
counts and coverage; proven GPU placement/parity; CPU/GPU batch/chunk crossover;
compile/warm/memory/host-cost breakdowns; and a ranked, evidence-backed bottleneck
list with recommended next work. Assess the existing G3 closure criteria without
inventing a speed threshold. Missing device proof remains pending; no G4/full-
format, generator-distribution, training or headline strength claim follows.

Stop or checkpoint for an unresolved minimized correctness failure, an excluded
mechanic/scope change, unavailable required GPU evidence or a concrete resource
limit. Never convert a timeout/capacity failure into successful battle completion.

Copyable execution assignment:

```text
Use /workspace/nS3JoIfbZ9JMn and execute docs/G3_TESTING_PROFILING.md directly
with Sol 6.1/high, without delegation. I approve its extensive native/CPU tests,
GPU correctness checks and CPU/GPU profiling scope on already available hardware.
Record it as the active approved goal, preserve local work, and synchronize safely
with origin/main. Continue through the integrated checkpoints, saving compact
resumable progress. Deliver measured bottlenecks and recommendations; preserve
exact simulator semantics and follow the handoff's acceptance/stopping conditions.
```

</details>
