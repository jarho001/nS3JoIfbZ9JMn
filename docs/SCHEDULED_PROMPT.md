# Scheduled development prompt

Copy the full prompt below into the existing scheduled workflow. It assumes
the scheduler is attached to the stated workspace and does not change or
recreate the schedule.

```text
Continue the scheduled development workflow in the same managed environment
and repository: /workspace/nS3JoIfbZ9JMn. Preserve local work and full task
history. Do not clone a replacement repository, reset, clean, or overwrite
existing work.

Confirm the environment and read `AGENTS.md`, applicable managed-cloud runtime
guidance, and `docs/ORCHESTRATION.md`. Verify runtime role/model from metadata;
self-identification is not proof. Expected roles: parent
`gpt-6.1-sol`/high, implementer `gpt-6-luna`/max, advanced implementer
`gpt-6.1-sol`/medium, global reviewer `gpt-6-astra`/medium. Repo files do
not prove scheduler selection. If a required role is unavailable or
unverifiable, report it and stop unattended development; never substitute.

Prevent overlap with a whole-run lock tied to the actual Git common directory.
It covers that repository's worktrees, not separate clones/cloud instances;
require scheduler-level single-invocation control or verified shared
coordination, and stop if overlap is uncertain. Do not clear locks without
owner evidence. Preserve local changes.

Before selecting gates/work, discover the actual remote/default branch; do not
assume `origin/main`. If ambiguous, stop. Inspect status/worktrees, fetch, and
sync safely: fast-forward a clean target, or use an isolated worktree only if
ownership is clear.
Never reset, clean, force-switch, or overwrite local work. Re-fetch before
push; merge a moved default into the task commit without rewriting history and
rerun affected checks. Conflicts or rejected pushes block.

Preserve the scheduler's recurrence, end time, and used count. The declared
cap is 30 runs in five days; initial start/usage are unknown. Do not reset,
extend, or invent an ordinal. Verify actual usage before advancement; otherwise
stop. Do not change the schedule.

Now read latest state, `docs/TASKS.md`, and `PROJECT_PLAN.md`; reconcile
accepted numbered tasks without overwriting newer state or inventing approval.
Initial snapshot: 20 accepted, bootstrap review at 20 pending; newer state
supersedes it. First inspect GPU state and new user evidence. If an
`awaiting_gpu` block has no evidence,
retain it and stop without tests/replanning; review cannot accept incomplete
GPU work. Validate CPU first; optional GPU absence is deferred, not a blocker.
Follow the evidence protocol in `ORCHESTRATION.md`.

If `pending_review_corrections` is nonempty, do only those corrections under
the existing owner, run affected checks, request Astra re-review, and stop; do
not create/count a task. If a prior disposition is `BLOCKED` and no new
evidence arrived, stop without repeating review. If Astra review is due, use
`global_reviewer` to sample code, tests, and oracle
evidence; assess direction/milestones and record disposition. One review
covers coincident gates. `NEEDS_CHANGES` allows only recorded corrections,
affected checks, and Astra re-review on a later run; persist corrections and
stop. `BLOCKED` records needed evidence; wait for new input. No new task/count
until accepted. Otherwise perform only the due review and stop; do not start
Task021 or other work. Parent cannot bypass blocking findings. If gates permit
work, resume the active/incomplete or planned deliverable. If none exists, use
the project plan and task history to define and record one justified bounded deliverable before
delegating one owner. Follow `ORCHESTRATION.md`; never narrow scope or weaken
tests.

Use Trail before source search: `trail map`, then `trail ask "<question>"
--source`; fallback to `/workspace/tools/trail/node_modules/.bin/trail`. Run
relevant validation once per meaningful revision. Assess each Trail finding
against evidence, fix real conflicts, explain lexical matches, and never bypass
all findings. Stage only inspected task files if needed. Finish with
`git diff --check`.

Before handoff, inspect the diff, update the task/state, and get the parent's
focused acceptance review before committing. Use a short subject and useful
body; keep review checkpoints fixed until Astra accepts. The user authorizes
this scheduled workflow alone to push finished commits to the discovered
default branch. Never force-push, rewrite history, or publish elsewhere. Push
unaccepted WIP only when required GPU evidence is pending, preserving its block.

Report the single deliverable's disposition, changed files, exact validation,
current task/review/GPU state, actual remote/default branch and push result,
scheduler-cap evidence or blocker, and uncertainty. Stop at the task boundary.
Never claim unverified role selection, scheduler pause, GPU evidence, run
ordinal, or cap count.
```
