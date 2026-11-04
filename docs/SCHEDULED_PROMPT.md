# Task entry point

Use this entry point identically for manual or scheduled work. Select model and
effort in the UI/role configuration; the scheduler owns cadence, concurrency,
and limits. Do not edit the schedule.

```text
Continue one bounded unit of development in the existing environment and
repository: /workspace/nS3JoIfbZ9JMn.

Read `AGENTS.md`, applicable runtime guidance, and `docs/ORCHESTRATION.md`.
Read the compact current `docs/ORCHESTRATION_STATE.json`, `docs/TASKS.md`, and
relevant `PROJECT_PLAN.md` sections. Open archived history only when needed.
Synchronize safely with the remote while preserving local and unrelated work.

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
policies. Use Luna/xhigh with a fresh scoped worker packet, not a full-history
fork; avoid repeated status polling and interim reviews.

Run only relevant fast checks: 30 seconds per command, 90 seconds per task.
Skip known-long suites/probes; do not extend timeouts or retry slow checks.
Reserve long integration, exhaustive, GPU and performance checks for an explicit
long-test session. Record deferred commands/source identity and recommend the
useful long session in the final summary for the user to run. Required missing
acceptance evidence remains pending; do not claim it passed.

If progress needs external evidence, including necessary GPU results, and none
is available, preserve the blocker and stop.

Update the task handoff and durable state. Commit and push this entrypoint's
task changes to the remote default branch, then stop with a brief result and
any unresolved blocker.
```
