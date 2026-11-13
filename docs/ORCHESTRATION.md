# Development workflow

[AGENTS.md](../AGENTS.md) governs execution. Sol 6.1/medium owns the work directly;
[configuration](../.codex/config.toml) retains only the explicitly requested Luna
implementer and Janitor. There is no automatic delegation, escalation role,
Astra review, or periodic independent-review gate. The primary reviews its own
diff and evidence. Independent review happens only when the user requests it.

## Prepare, then pursue a goal

First use [MILESTONE_PREPARATION.md](MILESTONE_PREPARATION.md) to sharpen G0–G8,
identify behavior families, and propose the first coherent G3 goal. This is
planning work; it does not start implementation or grant milestone acceptance.
The user then selects Goal mode with Sol 6.1/medium on the execution surface and
assigns the prepared goal. Configuration changes do not switch a running agent
or start Goal mode. If the requested model/effort is unavailable, stop and report
it. Experiment with this workflow before adding wall-clock budgets.

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

Choose meaningful checks for the changed behavior and shared dependencies. Run
them once per meaningful revision; rerun only when changes or failures justify
it. Reuse existing tests and a shared parity harness. Do not require all prior
suites, a new timing probe, or every execution-mode matrix after each patch.
Use representative JIT/batch checks at integration boundaries where relevant.

The 2026-11-12 user-approved default is one focused CPU differential pass for
the changed mechanic family, plus a narrow regression when a concrete shared
path needs it. Broad execution-mode, batch, lifecycle, full/lean and historical
suites are consolidated at agreed milestone checkpoints. Preserve missing
coverage as pending; do not weaken final acceptance or automatically resume the
interrupted Substitute gate. The user approved this scope change without
requesting implementation or test execution to resume in the same turn.

The 2026-11-13 continuation resumes sequential mechanic families. Defer GPU
correctness, broad CPU matrices, complete-battle campaigns and profiling until
reachable mechanics and inventory closure are complete. Keep their acceptance
requirements and exact commands in [GPU_HANDOFF.md](GPU_HANDOFF.md).

Agree the goal's validation scope in its brief, including known-long runs and
any actual device requirement. Long integration, exhaustive, GPU, and performance
runs outside that scope need a separate requested session. No fixed 30/90-second
microtask budget applies. Set practical command timeouts; investigate failures
instead of extending limits blindly. A performance claim still needs measured
performance evidence. Start with CPU; apply AGENTS.md's required GPU blocking
rules without treating optional GPU absence as failure.

For executed evidence keep the command/result and relevant source commit/tree
identity; include pin, environment/device versions and output location when
needed to reproduce it. A source revision predates its evidence record; do not
use a self-referential final commit hash. Separate deferred commands from
executed results. Required missing evidence leaves the relevant goal pending.

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
