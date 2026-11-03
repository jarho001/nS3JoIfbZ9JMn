# Task entry point

Use this entry point identically for manual or scheduled work. Select model and
effort in the UI/role configuration; the scheduler owns cadence, concurrency,
and limits. Do not edit the schedule.

```text
Continue one bounded unit of development in the existing environment and
repository: /workspace/nS3JoIfbZ9JMn.

Read `AGENTS.md`, applicable runtime guidance, and `docs/ORCHESTRATION.md`.
Read current `docs/ORCHESTRATION_STATE.json`, `docs/TASKS.md`, and
`PROJECT_PLAN.md` as needed. Synchronize safely with the remote while
preserving local and unrelated work.

Resolve pending review corrections or a due global review first. Otherwise,
follow the current simulator-first direction in `PROJECT_PLAN.md` and
`project_direction_decision` in the orchestration state. Resume unfinished
work or choose exactly one bounded deliverable that advances the JAX battle
simulator or its Showdown parity validation. Do not select additional PPO,
tournament, scoring, trajectory, or baseline-reporting work unless it directly
enables simulator semantics or validation. G1 remains open; this priority does
not accept or waive any milestone. Keep the minimal request, legal-action,
simultaneous-choice, player-view/history, and outcome contract needed for
correctness. State scope and acceptance criteria before implementation. Follow
the repository's delegation, focused validation, review, and acceptance
policies. Measure compiling/batching on representative simulator slices early.

If progress needs external evidence, including necessary GPU results, and none
is available, preserve the blocker and stop.

Update the task handoff and durable state. Commit and push this entrypoint's
task changes to the remote default branch, then stop with a brief result and
any unresolved blocker.
```
