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

If `pending_review_corrections` is nonempty, read the durable findings and
perform only those corrections under the single named owner. Reuse the current
owner unless the primary explicitly authorizes a handover; name the new owner
and confirm the prior owner has stopped. Do not overlap corrections with
parent-owned implementation. The Astra reviewer may turn required findings
into bounded acceptance criteria and route them to one Luna/max implementer;
the reviewer never owns implementation. Run affected checks and request an
independent Astra re-review in the same invocation when feasible. There is no
later-run-only correction rule. This loop creates/counts no project task and
stops at acceptance or a blocker. Allow at most two correction rounds (initial
implementation plus one follow-up), then report `BLOCKED` or request
architectural authority. Stop sooner for external evidence, GPU/access, a
decision beyond delegated authority, or execution limits. Persist owner,
finding, reproducer/evidence, and checks if unfinished. Optional suggestions
are not auto-implemented; do not spawn recursive reviewers. If a prior
disposition is `BLOCKED` and no new evidence arrived, stop without repeating
review. One review covers coincident gates. No new task/count until accepted.
After the review/correction loop for any due Astra checkpoint, stop at
acceptance or a blocker; do not begin another deliverable in that invocation.
For the due bootstrap review, do not start Task021. Parent cannot bypass
blocking findings.

When no correction loop is pending and Astra review is due, use
`global_reviewer` to sample code, tests, and oracle/reference evidence, assess
project direction and milestones, and record the disposition. One review covers
coincident gates. If no review is due and no corrections are pending, resume
the active/incomplete or planned deliverable. If none exists, use the project
plan and task history to define and record one justified bounded deliverable
before delegating one owner. Follow `ORCHESTRATION.md`; never narrow scope or
weaken tests.

Use Trail before source search: `trail map`, then `trail ask "<question>"
--source`; fallback to `/workspace/tools/trail/node_modules/.bin/trail`. Run
relevant validation once per meaningful revision. Assess each Trail finding
against evidence, fix real conflicts, explain lexical matches, and never bypass
all findings. Review only the focused diff/evidence once per meaningful
revision; do not rerun unrelated mechanics checks. Stage only inspected task
files if needed. Finish with `git diff --check`.

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
