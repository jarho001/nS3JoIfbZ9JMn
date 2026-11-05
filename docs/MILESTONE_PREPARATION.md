# Handoff: sharpen milestones before Goal mode

Give this file to one Sol 6.1/high agent. This is a planning assignment, not a
simulator implementation run. Work directly without subagents. Read AGENTS.md,
PROJECT_PLAN.md, current orchestration state, the current task summary, pinned
reference metadata, and relevant code/tests. Consult only relevant historical
evidence, including the G0 specification and T050 review where useful. Preserve
unrelated work. If the requested model/effort is unavailable, report it.

## Deliverable

Sharpen every G0–G8 milestone in PROJECT_PLAN.md. Keep its purpose, useful
outcome, dependencies, explicit exclusions, acceptance conditions, and evidence
needed to call it complete concrete and compact. Separate already demonstrated
facts from proposals and missing proof. Preserve G0's recorded acceptance and
all open milestones; do not invent approvals or change the simulator-first
priority. G3 comes next while G1 remains open and G2 learning is deferred.

For each milestone name:

- What a user or developer can actually do when it is complete.
- The coherent implementation package and required predecessors, including what
  can proceed in parallel conceptually without requiring delegated agents.
- Observable acceptance cases and exact validation entry points, reusing what
  exists. Mark commands not yet implemented as proposed, never as runnable.
- Necessary reference, CPU, GPU, learning, performance, or transfer evidence;
  distinguish required proof from optional follow-up and from unavailable proof.
- Explicit non-goals, uncertain decisions, and when to stop or seek user input.

Make G3 ready for a substantial Goal-mode run: choose an integrated playable
simulator outcome with meaningful mechanic families and interaction coverage.
Avoid another tiny restricted variant per move/status. Inspect whether a shared
engine state and parity harness can absorb the existing slices; identify what
should be reused, migrated, or later retired. Do not perform that cleanup now.
Do not promise full-format support or accept a large goal without evidence.

## Prepare behavior cases in batches

Create one compact linked behavior catalog if the existing tests/fixtures do
not already serve that purpose. Prioritize detailed native cases for the first
G3 package and adjacent interactions. For later milestones list the relevant
case families and evidence gaps; do not exhaustively specify the whole game.
Use pinned Showdown source and, where needed, small reference probes to settle
expected behavior before implementation. Source inference without an observed
oracle result is provisional and must be labeled. Do not run long suites or
GPU/performance experiments during this planning assignment.

Consider these families where relevant to the proposed package:

- Moves: hit/miss, damage boundaries, critical hits, PP, immunity and KO.
- Status: application/failure, burn/poison damage, paralysis, duration/recovery,
  residual ordering, double KO and termination.
- Turn order: priority, speed ties, random draw order, skipped actions after KO.
- Switching: voluntary choices, forced replacement, pivots, entry effects.
- Interactions: abilities/items with damage, status, switching, and visibility.
- Protocol: simultaneous choices, legal-action masks, private/public requests,
  per-player event/history views, phase transitions and terminal outcomes.
- Execution: deterministic continuation/reset, scalar/batched equivalence,
  supported bounds and explicit overflow/unsupported behavior.

For each detailed case record a compact table row or fixture reference with:
setup and preconditions; whether native-reachable or synthetic; choices;
expected state transition (HP/PP/status/stats as relevant); RNG draws, bounds,
order and before/after state; requests/legal actions and each player's visible
events; terminal result or rejection/boundary behavior; pin and oracle evidence
location; and which assertion/test will consume it. Use shared schemas and
reference helpers instead of a bespoke oracle for each mechanic. Expected
results must be independent of the JAX implementation being tested.

## Finish

Edit planning documents only: sharpen PROJECT_PLAN.md, link the case catalog
if needed, and update one compact preparation status in orchestration state.
Use `workflow_policy.milestone_preparation_status` for pending/complete status.
Do not create numbered tasks, implement mechanics, rewrite archival evidence,
start Goal mode, or mark `active_task` as an approved goal. Surface decisions
that actually need the user; resolve routine planning choices yourself.

Return a short first-goal brief: outcome, scope/exclusions, behavior families,
acceptance/validation scope (including any known-long or required GPU work),
meaningful integration checkpoints, and stopping conditions. Identify the plan
and catalog paths plus remaining decisions. Leave user approval and Goal-mode
selection as the next step. No wall-clock budget or new review gate is required.

Copyable assignment:

```text
Use docs/MILESTONE_PREPARATION.md to sharpen every G0–G8 milestone and prepare
batched pinned-Showdown behavior cases, with the first G3 goal ready for Goal
mode. Use Sol 6.1/high directly, without delegation. Edit planning documents
only. Return the proposed goal brief and remaining decisions; do not start
implementation or Goal mode.
```
