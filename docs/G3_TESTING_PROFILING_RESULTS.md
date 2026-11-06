# G3 fixed-battle intermediate diagnostic

Completed in **79.16 seconds** of the 300-second execution allowance, including
imports, fixture setup, CPU/GPU validation, compilation, warmup and measurements.
Exit status 0. During that comparison, no profiler, additional campaign,
optimization or further timing was run. The broad campaign below remains stopped; G3 is not accepted by this
intermediate result. The requested nested checkout is absent; execution used
`/workspace`, preserving the existing local work.

| Complete battles | Backend | Simulation ms: median (min–max) | Battles/s | Useful boundaries/s |
|---:|---|---:|---:|---:|
| 3 | Showdown CPU | 36.94 (33.82–46.62) | 81.2 | 3,465 |
| 3 | JAX CPU | 50.29 (50.03–50.80) | 59.7 | 2,545 |
| 3 | JAX GPU | 540.73 (539.53–826.83) | 5.5 | 237 |
| 96 | Showdown CPU | 840.74 (840.06–949.37) | 114.2 | 4,872 |
| 96 | JAX CPU | 599.81 (596.13–612.48) | 160.1 | 6,829 |
| 96 | JAX GPU | 871.75 (871.70–875.80) | 110.1 | 4,699 |

Showdown has the lowest median for three battles: JAX CPU takes **1.36×** its time,
and GPU **14.64×**. For 96 battles, JAX CPU is fastest (**1.40×** Showdown's
throughput). GPU takes **1.037×** Showdown's median time; their observed ranges
overlap, so these three repetitions do not establish a stable ordering between
those two backends at that point. Only one warmup was permitted; the falling
small-pack native/GPU samples show remaining run-to-run variation.

There are exactly 128 useful request boundaries per pack and 4,096 for 32 packs.
GPU elapsed time grows only **1.61×** for 32× the battles, versus **11.93×** on JAX
CPU and **22.76×** on native CPU. This supports poor amortization at the small GPU
operating point. It does not identify whether kernel launch/scheduling, memory
traffic, control flow or some combination dominates. The device loop has 128
sequential dependency steps per lane, and retains complete state/event/draw
results at each step. One state is 83,427 bytes, mostly fixed history storage;
those representation/retention costs differ from native string logs and mutable
state. These are observed implementation/contract facts, not a measured causal
breakdown. The most useful single follow-up is **one short GPU profiler trace of
the same warmed 96-battle loop**, separating kernel execution, device idle gaps
and memory work. That follow-up is now completed below.

The fixed workload is `F01-p1` (34 boundaries), `F02-p1` (33), `F03-p1` (61).
[The frozen pack](../test-logs/g3-testing-profiling/diagnostic/native-pack.json)
contains exact teams, seed words, initial conditions, every recorded choice and
independent native expectations; SHA-256
`400ad378b14d5be83d78b5b3523f8a3c973f4a83365127806de3088b37b7a60f`.
The compiled one-pack warm execution on each device passed every boundary's
state, ordered RNG, private requests/legal masks, events and cumulative histories,
then genuine terminal outcomes. Actual device placement was checked on every
input/output leaf. Full CPU/GPU output digests match. Legal masks were evaluated
in the compiled loop and bulk-fetched with the results for host assertions. The
measured native runs also checked terminal outcomes and final RNG outside timing.

Hardware: AMD Ryzen 7 7800X3D, 19 GiB host RAM, NVIDIA RTX 3060 Ti (8 GiB).
All processes have affinity to logical CPUs 0–7; OMP/OpenBLAS budgets are 8.
Showdown runs serially in one long-lived Node v24.22.1 process (7 observed OS
threads, one simulation thread). The shared JAX CPU/GPU process had 145 OS
threads, including runtime/compiler/device helpers; this is not 145-way simulation
parallelism. Affinity limits simultaneous host execution to 8 logical CPUs. JAX
CPU uses XLA's existing worker configuration; no CPU budget tuning was performed.
JAX/JAXLIB 0.12.0.1, NumPy 2.5.4, Python 3.13.18. Backends ran sequentially.

Source: working tree based on `51da7a941bcd301367e2c904f049c5a377248867`, with the
previous parity repairs included in the measured working tree. Measured engine SHA-256
`8fe364fd9e0b84d8e61daa717c27a0a4e4ce8f65b81b793ffd2fbb5198b3c599`.
Native Showdown pin: `1242c1e2d9763ba5b85d33f13aecb89bae19ae3d`, identity and
clean checkout verified before native execution. Full source hashes and raw
samples are in [results.json](../test-logs/g3-testing-profiling/diagnostic/results.json).

Each cell uses fresh independent starting buffers/Battle objects, one warmup,
and three timed complete executions. JAX is a compiled device scan over precisely
128 transitions per lane, replacing its carry with the prepared starting state
between battles. No padded or absorbing transitions are executed or counted.
Timing begins after inputs are ready and ends when all required outputs are
ready; GPU completion is synchronized. Host output transfer/projection,
assertions, setup, subprocess startup, cleanup and file I/O are excluded.
JAX retains every full numeric Result/state/trace and legal mask on device;
Showdown submits the recorded command strings through `Battle.choose`; JAX
consumes numeric choice plans encoded outside timing. Showdown retains its normal
mutable state, string event log and generated requests without observer hooks, per-draw stacks or oracle snapshots. Thus this
compares matched complete battles with explicitly different output
representations/retention; it is not a proof of equivalent-output core speed.

Excluded costs: initial preparation of the three JAX states took CPU **0.661 s**,
GPU **1.226 s**. Per-repetition fresh buffer preparation/placement is recorded
separately; median input readiness cost was about CPU **0.8 ms / 2.4 ms** and GPU
**3.0 ms / 4.1 ms** for 3/96 battles (copies and transfer included, outside timing).
XLA lowering+compilation took CPU **11.81 / 11.10 s**, GPU **14.70 / 12.82 s**.
Warm simulation took CPU **58.6 / 652.8 ms**, GPU **864.8 / 1,218.6 ms**;
Showdown **111.9 / 874.1 ms**. Native first cold initialization took **0.832 s**;
measured repetitions' initialization took **1.17–3.27 ms** for 3 battles and
**21.33–22.99 ms** for 96. Compilation for the one-pack measurements is recorded
under the correctness gates and reused by their timing cells; a zero incremental
compile field in those cells does not mean compilation was free.

Reproduction command (creates a new evidence directory; do not automatically
rerun or overwrite the existing one):

```sh
timeout --signal=TERM --kill-after=1s 300s env XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_COMPILATION_CACHE=false OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 taskset -c 0-7 /opt/jax-env/bin/python -m benchmarks.g3_diagnostic
```

The [revised plan](G3_TESTING_PROFILING.md),
[compact state](ORCHESTRATION_STATE.json) and
[execution log](../test-logs/g3-testing-profiling/diagnostic.log) retain the scope,
stopping condition and checkpoints. All earlier evidence is preserved below.

## Follow-up: one warmed 96-battle GPU capture

The user authorized the proposed trace after publication. The exact recorded
`run` function was loaded without executing the diagnostic entry point. Engine,
loop and fixture source hashes match the measured diagnostic. GPU placement and
every output leaf in all 32 packs matched the previously CPU/GPU/native-validated
trajectory digest. One warmed complete call was captured, with **96 battles /
4,096 useful boundaries**, all original outputs retained. No faster mode,
elision experiment, new campaign or additional performance comparison was run.

The capture setup first failed because this JAX version does not expose
`ProfileOptions.device_tracer_level`. Supported attributes were inspected and
that unsupported assignment removed. The failed setup consumed 20.27 seconds;
the corrected execution consumed 29.01 seconds, including compilation, warmup,
validation and export. Total machine execution was **49.29 seconds**, within the
shared 300-second limit; only one actual capture ran. The corrected process was
limited to 270 seconds rather than restarting a full budget. The setup failure
log and checkpoint remain alongside the successful artifacts.

The [capture results](../test-logs/g3-testing-profiling/diagnostic-profile-96/results.json)
and [offline hardware-stream analysis](../test-logs/g3-testing-profiling/diagnostic-profile-96/analysis.json)
show:

- Retained outputs occupy **381,706,240 bytes (364.0 MiB)**. Per-transition state
  snapshots account for 325.9 MiB; **320.0 MiB** of that is repeated fixed history
  buffers. Draw/event traces add **38.0 MiB**. These are materialized output bytes,
  not measured peak memory or bytes transferred to the host during simulation.
- **570,544 hardware kernel executions** occur within the one call, including
  **214,180 `memcpy32_post` kernels**. Those copy-named kernels account for 213.7 ms
  of summed device durations, about 31.5% of summed kernel durations. Their names
  alone do not identify which copies come from output retention versus internal
  state/control-flow handling.
- There are **1,483 single-byte device-to-host transfers** inside the compiled
  call. These are consistent with runtime checks of loop conditions; they are
  distinct from the bulk host output fetch, which occurs after the capture.
  The host also executes 11,835 command buffers. A compiled JAX loop therefore
  does not imply one fused GPU kernel or the absence of runtime synchronization.
- The stream records include **826 device-to-device copies of 2.5 MiB each**,
  totaling 2.02 GiB. That size equals one 32-lane history buffer. This is consistent
  with repeated history-buffer copies but is not a complete attribution of them.
  Their summed transfer duration in this capture is 9.4 ms.
- The profiled call takes **2.882 seconds**, versus the earlier unprofiled median
  of 0.872 seconds. Hardware stream interval union is 676.4 ms, about 23.5% of the
  captured call. This is a property of the perturbed capture, **not an estimate of
  unprofiled GPU utilization**. Only hardware stream events were counted; nested
  XLA annotations were excluded. Summed durations can overlap and are not wall
  time. Use the original diagnostic for backend speed comparisons.

The evidence points to very fine kernel granularity, runtime synchronization and
repeated data movement as issues to investigate. It does not establish that
removing saved snapshots alone will resolve the GPU gap or predict a speedup.

### Proposed leaner rollout API (recorded; unimplemented)

The current benchmark asks `lax.scan` to return every full `Result`, including
its state, plus legal masks. As a result it saves all intermediate states, each
with its own complete fixed-size history buffer, as well as every draw/event
trace. This is a benchmark output contract, not a requirement imposed by JAX's
immutable programming model. A compiled loop can carry the current state forward
without returning all earlier states.

Forward prediction still needs the complete current causal battle state: HP,
PP, statuses/items, active/reserve roster, pending actions and suspended queue,
turn counters and exact RNG state. It does not inherently need old state
snapshots or a record of every past RNG draw. A policy may need its observation
history or recurrent memory, so its required observations/events must still be
supplied; full trusted internal snapshots are a separate concern.

A reasonable first design is an optional rollout output contract that returns
final states and status codes, plus selected observations or transitions the
caller requests, while using the same exact transition implementation. Keep the
full trace path for parity/debugging. Initially retain the existing state history
and capacity/validation behavior: removing history or trace accounting from the
engine itself would change current observable or rejection behavior unless
handled explicitly. Compare final states, RNG, terminal outcomes and requested
outputs before accepting such a path. This could reduce storage and permit
compiler elimination of unused work **while preserving forward prediction**;
it has not been implemented or timed. It should not be described as a mode that
sacrifices forward simulation capability.

The proposed first increment is a wrapper around the existing transition logic:

- Accept existing initial states and supplied choice plans, with explicit stopping
  limits. Carry only the current state through the loop instead of returning a
  full state snapshot at each transition.
- Return final states, per-lane completion/status information and useful
  transition counts. Preserve failure status and its location, so a later result
  cannot hide an earlier invalid, unsupported or capacity-rejected submission.
- Allow callers to request selected observations, events, rewards or sampled
  states. Keep complete state/draw/event trajectories as an explicit diagnostic
  output option. Policy-visible outputs must retain their existing visibility
  rules; trusted simulator state is not a policy observation.
- Preserve exact PRNG evolution, validation, atomic rejection, pending choices,
  interrupted move queues, terminal outcomes and the current history/capacity
  contract. The first increment changes returned data, not battle semantics.

This is an output-retention proposal, not a second approximate simulator. Do not
remove engine history, trace overflow accounting or other observable behavior as
an incidental optimization. A later implementation should first demonstrate
exact final-state/RNG/status and selected-output agreement on the same frozen
three trajectories, plus relevant existing rejection/capacity fixtures. Only
then would one matched 96-battle comparison quantify any benefit. Savings in
memory are plausible from the retained-byte counts; runtime gains remain
unmeasured because the capture also exposes substantial fine-grained execution
and synchronization. Implementation and measurement remain future work.


<details>
<summary>Earlier broad campaign evidence (preserved; superseded scope)</summary>

# G3 testing and profiling results

The user stopped the extensive CPU/GPU sweep on 2026-11-06. All profiling
and device-monitor processes are stopped. No further benchmark may run until a
matched, uninstrumented legacy Showdown-versus-GPU benchmark is established.
The measurements below time the new JAX simulator on CPU and GPU; Showdown
served only as the instrumented correctness oracle. **No speedup over Showdown
has been measured.** G3 remains open. Compact continuation state is in
[ORCHESTRATION_STATE.json](ORCHESTRATION_STATE.json).

Intermediate results, logs, source/config hashes, traces and correctness fixes
are preserved locally and have not yet been committed or pushed. The final-source
CPU profile completed 28 kernel and 72 scan points. The interrupted final-source
GPU profile saved 13 kernel and 33 scan points, plus 18 conservative guard
records; its last saved checkpoint was the tied cohort, batch 1, chunk 8,
compile/measure scan phase. That checkpoint does not imply the in-flight point
completed. Raw checkpoints are in
[CPU results](../test-logs/g3-testing-profiling/profile-cpu/results.json) and
[partial GPU results](../test-logs/g3-testing-profiling/profile-gpu/results.json).
The GPU log and one-second device monitor are retained alongside them.

The clean starting commit was `51da7a941bcd301367e2c904f049c5a377248867`,
synchronized with `origin/main`. The checkout is `/workspace`; the handoff's
former nested path is absent. Source captures are consolidated in
[source-identities.json](../test-logs/g3-testing-profiling/source-identities.json),
with per-campaign hashes alongside each summary. The native oracle remains
Showdown `1242c1e2d9763ba5b85d33f13aecb89bae19ae3d`, guarded and clean.

All 1,000 final-source starts completed with exact native agreement over 30,342
boundaries and 57,257 player decisions: 900 ordinary supplied starts and 100
explicit trusted fixtures, zero truncations, capacity diagnostics or mismatches.
The [predeclared inputs](../benchmarks/g3-campaign.json) contain both seats,
1–3-member teams, levels/Speed EVs/items/move slots and explicit battle/policy
seeds. See [campaign summary](../test-logs/g3-testing-profiling/campaign-cpu-summary.json).
The independent [native coverage census](../test-logs/g3-testing-profiling/native-campaign-coverage.json)
agrees with every start/boundary count: 41 observed effect categories and 726
within-boundary effect pairs. Those counts describe observed co-occurrence,
not exhaustive interaction equivalence.

| Correctness repair | Independent native regressions |
|---|---|
| Physical type/Levitate immunity precedes Protect; status checks retain their timing. | `protect-immunity-Tackle/Earthquake/Toxic` |
| Incoming effective Speed refreshes before `runSwitch` queue insertion, including forced replacements. | `paralyzed-forced-return` |
| Residual status/item callbacks enter sorting even for fainted holders; their effects are skipped afterward. Native `checkFainted` changes status only after the residual action. | `fainted-leftovers-handler-sort`, complete `fainted-brn/tox-handler-sort` traces |
| Trusted immune `Pokemon.setStatus` installation is silently rejected; the fixture helper must not disclose an attempted move. | `trusted-toxic-immune-Gengar/Snorlax` |

Before-repair failures and full minimized/selected native traces are retained
losslessly under [test-logs/g3-testing-profiling](../test-logs/g3-testing-profiling).
The final [native corpus](../test-logs/g3-testing-profiling/final-native-summary.json)
has 111 scenarios / 464 boundaries, including 171 catalog boundaries and 293
complete three-member-battle boundaries. Current checks cover exact state,
queues, PP/status/counters, ordered RNG arguments/values/pre/post states,
requests/legal masks, both event histories, outcomes, padding, atomic rejection
and capacities. All catalog boundaries have CPU/GPU eager/JIT and 32-lane
agreement; complete battles also pass batches 1/8/32.

The starting full CPU suite passed 128 tests in 487.875 seconds. After the final
engine repair, 13 affected CPU checks passed in 137.934 seconds and 13 CUDA checks
in 498.389 seconds. Focused complete status-family continuation/reset checks
add the fourteenth covered check, including saved states at every boundary,
eager/JIT and 32 lanes (CPU 24.987s, CUDA 48.231s). The benchmark preparation
contract passed separately. Logs preserve the exact commands and source scope;
no unchanged legacy suite was rerun after each repair.

GPU execution asserts the backend and device of initialized inputs and all
compared eager/JIT/batched state/trace leaves; `JAX_PLATFORMS=cuda` forbids CPU
fallback. The [selected replay](../benchmarks/g3-replay.json) has 12 complete
traces / 686 boundaries, including the original discoveries, queue insertion and
the paired longest native traces (174/177 boundaries). The final CPU replay
passes saved first/middle/last continuations and resets. The corresponding GPU
replay also passes all 12 traces / 686 boundaries in 184.640 seconds.

Hardware/software are in
[environment.json](../test-logs/g3-testing-profiling/environment.json): Ryzen 7
7800X3D, 19.0 GiB host RAM, one RTX 3060 Ti with 8 GiB VRAM; Python 3.13.18,
JAX/JAXLIB 0.12.0.1 and NumPy 2.5.4. Commands use the existing isolated
`/opt/jax-env/bin/python`, affinity 0–7, eight-thread BLAS/OMP budgets and
`XLA_PYTHON_CLIENT_PREALLOCATE=false`. Profiling disables persistent compilation
caching. The state payload is 83,427 bytes/lane, including 81,920 bytes of
history; a trace adds 9,737 bytes before compiler/layout overhead.

The [reusable harness](../benchmarks/shared_engine.py) and
[versioned sweep](../benchmarks/g3-profile.json) separate prepared-input advance,
full-output sustained scans with real terminal/reset handling, and host
submission/projection/serialization. Host stacking prepares identical buffers
before device placement and outside all timers. Measurements retain all state
and event/draw outputs, synchronize completion, use ten repetitions targeting
at least 200 ms and preserve raw spread/counters. Terminal/rejection cohorts
have zero useful work; their submitted-choice counters are attempts. Memory
preflight uses 80% of reported free memory with a conservative safety factor
and compiler reserve; skipped points are not measured OOMs or maximum capacity.

The first CPU sweep and partial CUDA sweep are preserved as
`profile-*-before-status-fix` and are superseded for the final engine revision.
Their native fixed workloads passed, but a further native family review exposed
the missing fainted status callbacks. No performance conclusions are accepted
from that revision. Further sweep execution and CPU/GPU analysis are paused by user instruction.
The next performance prerequisite is a matched Showdown-versus-GPU contract: same
workloads and useful work counts, explicit equivalent output/retention and
transfer costs, CPU resource budget, warmup/compilation accounting, and
synchronized GPU completion. No replacement benchmark has been launched.

</details>
