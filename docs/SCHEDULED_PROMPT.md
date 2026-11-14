# Task entry point

Use this prompt for manual or scheduled continuation. The execution surface
selects model/effort, cadence, concurrency, and Goal mode; editing this file does
not activate Goal mode or change a saved schedule.

```text
Use the existing cloud environment and repository /workspace/nS3JoIfbZ9JMn.
Read AGENTS.md and applicable runtime guidance; perform its Trail preflight.
Start at docs/CURRENT.md and run python scripts/read_context.py for the live goal,
package, blockers and pending acceptance. Read only the applicable workflow,
plan/acceptance and behavior sections via --section/--field or bounded rg/sed;
do not cat whole state, plan, case catalog or handoff files. Inspect Git status
and safely synchronize with the remote default branch while preserving local
and unrelated work, then refresh the live view if source changed. Load historical
evidence only to answer a concrete question.

Use the primary model/effort specified by AGENTS.md and .codex/config.toml.
If it is unavailable, report the blocker rather than substituting. Work directly; do not
invoke Luna, Janitor, or another agent unless the user explicitly requested it.
There are no automatic Astra or task-count/milestone independent-review gates.

First inspect existing blockers and newly supplied evidence. If required
external/GPU evidence is still absent, retain the block and stop without
repeating tests or replanning.

Resume only a user-approved goal recorded in active_task with scope, acceptance
conditions, and validation scope. If milestone preparation is incomplete or no
goal is approved, stop with that prerequisite and refer to
docs/MILESTONE_PREPARATION.md. Do not select a tiny Task053, launch Goal mode,
or treat this generic entry point as approval of a new milestone-sized goal.

For an approved goal, follow the simulator-first direction, prepared native
behavior cases, and the documentation/validation policy. Implement directly
through useful integrated checkpoints within that goal; do not stop after each
mechanic merely to create another handoff. Stop when the goal is complete, a
run limit requires a continuation checkpoint, or a concrete blocker prevents
progress. Keep unsupported mechanics explicit and required missing evidence
pending. Do not silently narrow scope, weaken tests, or infer milestone approval.

Inspect the final diff and actual checks. Update the existing compact goal
record with progress, evidence pointers, next step or blocker. Commit and push
only this goal's changes under AGENTS.md authorization; preserve unrelated work
and label any unaccepted WIP clearly. Return a brief result. Never claim a
schedule was paused or Goal mode activated unless independently verified.
```
