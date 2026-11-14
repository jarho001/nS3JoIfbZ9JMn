# Development workflow

[AGENTS.md](../AGENTS.md) governs execution. Sol 6.1/medium owns the work directly;
[configuration](../.codex/config.toml) retains only the explicitly requested Luna
implementer and Janitor. There is no automatic delegation, escalation role,
Astra review, or periodic independent-review gate. The primary reviews its own
diff and evidence. Independent review happens only when the user requests it.

## Prepare, then pursue a goal

Use [MILESTONE_PREPARATION.md](MILESTONE_PREPARATION.md) for new or materially
changed goals. Initial milestone preparation is complete; resume an approved
goal from `active_task`, not from the historical first-G3 assignment. Model/effort
and Goal-mode rules are in [AGENTS.md](../AGENTS.md); repository edits do not
change the execution surface.

For an approved goal, investigate, implement, validate, and finish directly.
Internal checkpoints should produce useful integrated behavior, not another
numbered task for each tiny mechanic. Continue within the approved goal until
its acceptance conditions pass, the run must checkpoint, or a concrete blocker
requires external evidence or a user decision. Preserve the simulator-first
direction in [PROJECT_PLAN.md](../PROJECT_PLAN.md); G1 stays open and G2 learning
stays deferred. Do not infer milestone completion from task counts.

## Establish behavior in families

Before coding a family, complete the lightweight reachability precheck in
[SHOWDOWN_BEHAVIOR_CASES.md](SHOWDOWN_BEHAVIOR_CASES.md#reachability-audit-and-family-precheck-2026-11-13).
Record the required branch/witness, existing shared implementation/evidence and
new dependency edges in `active_task.reachability_precheck`; unresolved membership
calls for investigation first. This does not add a review or approval gate.

Before coding a family, collect a compact set of expected cases from pinned
Showdown. Cover ordinary behavior, boundaries, and interactions with existing
mechanics. Distinguish a native reachable battle from a synthetic injected
state. Capture inputs/choices, state transitions, random draws and resulting
RNG state, requests/actions, each player's visible events, and termination as
applicable. Expected outputs come from the reference, not the implementation
under test. Add cases when new interactions are discovered; do not prewrite a
prose specification for the whole game or build another oracle per tiny slice.

## Validate proportionally

[AGENTS.md — Validation and stopping](../AGENTS.md#validation-and-stopping)
owns the validation policy: grouped focused CPU checks during implementation,
checkpoint consolidation and the current deferrals. Reuse existing tests and
native cases; do not turn example or historical command lists into a per-patch
checklist. Final milestone acceptance remains in PROJECT_PLAN.md.

[GPU_HANDOFF.md](GPU_HANDOFF.md) is the single maintained home for exact deferred
G3/G4 device, broad-check and campaign commands. Historical plans/results retain
their reproducible commands as evidence, not fresh authorization to execute them.

For executed evidence keep the command/result and relevant source commit/tree
identity; include pin, environment/device versions and output location when
needed to reproduce it. A source revision predates its evidence record; do not
use a self-referential final commit hash. Separate deferred commands from
executed results. Required missing evidence leaves the relevant goal pending.

## Compilation-aware G4 checkpoints

Adopted 2026-11-14 following the [bounded investigation](../test-logs/g4-cpu-conformance/compilation-investigation.json).
The next work package and later G4 packages should contain two to four related
reachable mechanic families sharing a scheduler path or dependency. Choose the
members and checkpoint before implementation; use a smaller group for a concrete
shared-path risk or dependency. Do not add unrelated mechanics to fill a quota.

1. Precheck each member's reachability and establish compact pinned native cases
   before its implementation. Keep exact state/RNG/requests/events/history checks;
   synthetic installations do not prove generator membership. A discovered
   stimulus defect may be corrected after implementation: regenerate independent
   native expectations, record the reason, retain intended branch coverage and
   exact assertions, and validate the corrected fixture before claiming parity.
   Run cheap native fixture/inventory contracts per meaningful revision, without a full engine
   compile per member. Prepare the next member's native cases during a compile
   when useful; avoid concurrent JAX compiles and overlapping simulator edits.
2. Implement members in dependency order. Closely coupled members may share an
   edit batch; separate edits/status writes per member are not acceptance gates.
   Record member implementation versus pending CPU validation at meaningful
   checkpoints or interruption, without per-member documentation busywork.
   At most one small group may accumulate without integration evidence. Add only
   targeted cross-member/shared-path cases.
   Use cheap helper-level checks where they diagnose a local risk; these do not
   replace whole-step parity. Do not disable JIT as a presumed shortcut.
3. Run one focused CPU integration invocation for all group cases, using one
   `jax.jit(advance)` executable per required shape/dtype signature. Prefer one
   group fixture/method or a shared test-support executable; merely listing old
   unittest modules together does not ensure reuse. Clear/release caches only
   after the group, not between members. Keep signature variants/broad batching
   out of this implementation check unless a concrete change requires them.
4. A divergence stops further families: minimize and repair it, then rerun affected
   group cases once against the changed source. Do not weaken assertions or claim
   unrun members passed. If interrupted, save source/fixture identity, the pending
   group command and first failure/cursor. After a pass, save one compact group
   checkpoint, update ledger/GPU handoff, commit/push and continue the next group
   within authorized execution. A published group is an internal checkpoint,
   not a reason to end an authorized sustained run. Select/precheck the next
   group and continue unless acceptance is reached, the user stops/limits scope,
   execution resources require checkpointing, or a concrete blocker arises.
   Record the actual stopping reason; do not manufacture extra checks or status
   updates to keep busy. Do not pause for approval after every family.

Correctness runs should use a trusted persistent compilation cache:

```sh
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR="$PWD/work/jax-cache"
```

Use these settings before importing JAX; create the ignored directory if needed.
Remove hard-coded `JAX_ENABLE_COMPILATION_CACHE=false` from new correctness
commands. The deferred runner now supports `--cache-dir` and explicit `--cold`,
records effective cache settings, and defaults to persistent correctness caching.
Preserve per-module process isolation for historical suites because
combined execution previously exceeded 8 GiB; persistent cache survives process
exit and `jax.clear_caches()`. Capture cache hit/miss diagnostics on normal runs
rather than launching repeated suites solely to demonstrate reuse. A changed
lowered program, shape/dtype, compiler/options or device can miss the cache;
never describe caching as avoiding compilation of newly changed mechanics.

Keep optimized settings unchanged. The reduced-optimization 180s trial did not
finish; neither its speed nor parity is established. Explicit cold-compilation
experiments must disable caching and say so, separately from correctness and
simulation runtime measurements. No new profiling campaign is authorized here.
Record compile-attempt count, group members/cases, combined lower/compile time,
and initialization/replay/assertion time from the normal run. Combined timers
are sufficient for development decisions when labeled accurately; unavailable
subphase/cache-load/wall-time metrics stay unknown. Do not rerun checks or add
profiling just to split timers. Expected changed-source cache misses are not a
workflow failure. Report compile cost per validated family as a descriptive
metric, not a controlled speedup claim. Use actual cloud
CPU quota/memory limits when choosing concurrency; more threads do not grant
more cores. Larger hosts and graph deduplication remain possible future work,
not prerequisites for continuing G4 or proven improvements.

All final eager/JIT, mixed batches, full/lean, lifecycle/migration, directed and
fixed-campaign gates remain required. Grouping changes feedback cadence only;
CPU completion and full G4 acceptance criteria do not change.

## Keep documentation small

- [PROJECT_PLAN.md](../PROJECT_PLAN.md): milestone contracts, priorities, and
  lasting decisions; behavior cases live in tests/fixtures or one linked catalog.
- [ORCHESTRATION_STATE.json](ORCHESTRATION_STATE.json): the current goal,
  completed/next checkpoint, evidence pointers, and concrete blockers. Reuse
  `active_task` for the active goal; use a stable milestone/goal identifier.
- [TASKS.md](TASKS.md) and existing archives: historical acceptance index and
  pointers. Existing task count and review records stay historical; new goal
  progress does not require new TASK### files or task-count review cycles.

Update facts in their authoritative home, with links instead of repeated prose.
Keep a short continuation note only when work must resume. Do not append status
narratives, duplicated blob inventories, or separate handover/review reports for
every change. Historical acceptance and unique evidence are preserved, not
rewritten to match the new policy.

## Optional roles and entry points

Luna implements only when the user explicitly requests Luna. Janitor audits or
cleans up only when the user explicitly requests Janitor; its contract is in
[janitor.toml](../.codex/agents/janitor.toml). It reduces redundant code, tests,
and documentation while protecting unique regression evidence and provenance.
It is not a recurring gate. Neither role recursively delegates.

The [manual/scheduled entry point](SCHEDULED_PROMPT.md) follows these rules and
resumes only an explicitly approved goal. Cadence, concurrency, and Goal mode
are controlled outside the repository. This rewrite does not change schedules.
Commit/push authorization, preservation of unrelated work, and blocker handling
remain in AGENTS.md.
