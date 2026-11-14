# Working policy

Use GPT-6.1 Sol with medium reasoning effort as the primary agent, as configured
in `.codex/config.toml`. The primary investigates, implements, tests, inspects
the diff, and finishes the work directly. If this model/effort is unavailable,
report the blocker rather than silently substituting. Do not automatically raise
reasoning effort; the user selected Medium to control usage.

## Ownership and scope

- No automatic delegation, model escalation, or independent review gates.
  Invoke Luna `implementer` or `janitor` only when the user explicitly requests
  that role. Availability in configuration is not authorization to invoke it.
  Astra reviews and task-count/milestone review checkpoints are retired.
- Work toward a coherent user-approved goal or milestone package. Break work
  into internal checkpoints as useful; do not turn each mechanic, correction,
  or checkpoint into a separately numbered task and handoff cycle.
- Before Goal mode, sharpen the milestones and behavior cases using
  [the preparation handoff](docs/MILESTONE_PREPARATION.md). Goal mode is selected
  on the execution surface; repository configuration does not activate it.
  The current workflow experiment uses Sol 6.1/medium without a new wall-clock
  budget. Do not launch Goal mode or invent an approved goal on a scheduled wake.
- Preserve unrelated work and avoid overlapping edits. Stop for a concrete
  blocker or a decision outside the approved goal; ordinary implementation
  choices belong to the primary agent.

## Context and documentation

Start at [CURRENT.md](docs/CURRENT.md). After Trail preflight, run
`python scripts/read_context.py` for the live goal/package/blockers. Use its
`--field`, `--headings` and `--section` modes or bounded rg/sed reads for needed
detail. Do not load the whole plan, state, case catalog, GPU handoff or archives
for routine orientation. Read applicable scope/acceptance sections before acting;
selective loading does not narrow the goal or retire any pending gate.

Trail-code is required in every environment used for this project (local,
cloud, worktree, and CI when agents perform project work). Before investigation
or edits, verify `node scripts/trail.cjs --version` and run
`node scripts/trail.cjs ask "<current task>"`. Queries refresh the graph against
current source. Use the Trail MCP server when exposed; the pinned CLI is the
mandatory fallback when the host cannot expose or reload MCP tools. A copied
`.trail/cache` alone is not a working installation. On a fresh checkout or when
the pin changes, run `node scripts/setup-trail.cjs` (authorized project tooling
setup), then repeat the preflight. If setup or a real query fails, stop project
work, repair Trail or report the concrete blocker; do not silently continue
using only source searches. After orientation, use targeted source reads and
`rg` for exact text or non-indexed files as needed. No deep/LLM or semantic
model build is required. Setup and host configuration: `docs/TRAIL_SETUP.md`.
Do not read all archives merely for routine orientation.

Keep useful evidence, not a narrative. The plan owns scope and acceptance;
`docs/ORCHESTRATION_STATE.json` owns current progress/blockers; CURRENT.md routes
bounded reads of it without duplicating status. Tests and their referenced
outputs own reproducible evidence. Completed records belong in linked history,
not active state; unfinished obligations and resumable WIP stay live. Update
these records only when something material changes. Record lasting design decisions once.
Do not create per-patch TASK documents, duplicate handoffs, repeated manifests,
or a transcript of investigation. Existing task/review archives remain history,
not instructions or mandatory reading. Preserve recorded acceptance evidence;
this workflow change does not accept any milestone or reset the task count.

## Validation and stopping

Establish pinned Showdown behavior by mechanic family before implementation,
then add interaction cases as the goal evolves. Follow
[the workflow](docs/ORCHESTRATION.md) for evidence. Preserve exact relevant RNG,
state, request/legal-action, event, visibility, and terminal behavior. Keep
unsupported behavior explicit; never weaken assertions to obtain a pass.

Run meaningful affected checks once per meaningful revision. Choose checks and
reasonable timeouts for the agreed goal, rather than the retired 30-second /
90-second tiny-task limits. Do not rerun every historical suite after each
edit. Long integration, exhaustive, GPU, or performance runs must be within the
approved goal's validation scope or a separately requested test session. Do
not blindly retry slow or failing checks. Missing required evidence remains
pending, with the exact command, relevant source identity, and blocker.

The 2026-11-14 compilation-aware workflow supersedes one full JIT differential
per family: plan small groups of two to four related reachable families, retain
native prechecks per family, then validate every member against one shared
compiled step at the group checkpoint. A concrete dependency/risk can justify a
smaller group; do not pad groups with unrelated work. Keep implemented members
CPU-unvalidated until the group passes; stop for a divergence and diagnose it.
Use persistent caching for unchanged-source correctness reruns, preserving
process isolation where needed for memory. Do not clear caches between focused
members or compile separate equivalent wrappers within a group. Exact workflow,
cache commands and limits: [ORCHESTRATION.md](docs/ORCHESTRATION.md#compilation-aware-g4-checkpoints).
Do not adopt unproven reduced-optimization settings or start compile ablations.
Use lightweight group checkpoints and accurately labeled combined timers; do not
add runs for accounting alone. A passed/published group does not end an authorized
sustained run: continue the next group or record the concrete stopping reason.

GPU correctness, broad CPU matrices, complete-battle campaigns and profiling
remain deferred until reachable mechanics and inventory closure are complete.
Preserve final acceptance and pending commands in `docs/GPU_HANDOFF.md`. Do not
restart the paused Substitute gate or rerun historical matrices after each group.

Start on CPU. Require GPU only for a stated acceptance criterion or a concrete
device-specific property. Missing optional GPU evidence is not a blocker. If
required GPU execution is unavailable, persist `gpu_block` as `awaiting_gpu`
with source commit/tree identity, commands, versions, hardware need, expected
result, and existing CPU evidence. Stop dependent work until new evidence is
tied to unchanged relevant source. A blocked wake without new evidence stops
without repeating tests or replanning. Never claim unrun checks or scheduler
changes. Preserve unaccepted WIP explicitly when needed for external validation.

During the user's travel GPU deferral, keep all G3/G4 device obligations, notes and
exact ready/pending commands in [GPU_HANDOFF.md](docs/GPU_HANDOFF.md).
Update that same handoff and its runnable checkpoint test list as G4 CPU work
adds mechanics. Keep remaining G3 GPU acceptance checks in the same record.
Missing future release/device evidence does not block the active
CPU-only goal; record it in `deferred_gpu_work`, and do not execute GPU work,
profiling or benchmarks while the deferral remains active.

Before finishing, inspect the diff against the goal and actual evidence. Update
one compact progress/blocker record and report changes, checks, and unresolved
limits. No separate reviewer or acceptance ceremony is required.

## Commits and pushes

The user grants standing authorization to commit and push project changes in
this repository to the remote default branch without confirmation, including
workflow changes. Follow explicit local-only/no-commit/no-push requests. Preserve
unrelated changes. Fetch and synchronize safely; never reset local work,
force-push, or rewrite remote history. If synchronization conflicts, preserve
both sides and resolve only within the authorized scope or report the blocker.
