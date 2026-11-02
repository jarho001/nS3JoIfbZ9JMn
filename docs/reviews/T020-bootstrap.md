# T020 bootstrap review

## Original disposition

The bootstrap review returned `NEEDS_CHANGES` against source commit
`d770de5ce188c0fbe5d80090c850e5253b607a7d`. This is the reviewed source
identity, not a later correction or final commit. This initial disposition is
historical: on 2026-11-02, the parent independently reviewed the required
documentation corrections and policy diff and recorded `ACCEPTED`. T020 remains
an accepted project task and the accepted-task count remains 20; the review
baseline is 20 and the next regular checkpoint is 30. No milestone acceptance
or review commit is recorded. The scheduled-run cap remains 30 in five days
with actual start and usage unknown.

## Required findings

1. `docs/MECHANICS_COVERAGE.md`, `docs/REUSE_AUDIT.md`, and the ending of
   `docs/BENCHMARK_CHARTER.md` conflated existing G0 specification artifacts
   with unresolved G0 decisions and later implementation gaps. The corrections
   must distinguish the pinned target/metadata, mechanics ledger, accepted
   reuse decision, and draft charter/source manifest from the still-needed G0
   specification-closure review and from G1/G3/G4 implementation. G0 must not
   require a full engine or imply milestone acceptance. State which unresolved
   target/scope decisions need resolution and that operational benchmark
   budgets do not all automatically gate G0.
2. `docs/BENCHMARK_CHARTER.md` incorrectly said the runner summary has no turn
   field. The terminal value is already at `summary.outcome.turns`; the latest
   public `|turn|` marker is tracked internally as `turnCount` but is not
   exported in the summary. No source change was requested.

## Routing and evidence

Root authorized this bounded correction handoff to the sole owner
`/root/global_reviewer/t020_corrections`, excluding overlapping parent-owned
implementation. The explicit Luna/max child route and requested Astra/medium
review route were accepted, but no independent backend identity or automatic
scheduler-role-loading proof was available. Repository role configuration and
a requested route are not proof of runtime role selection.

The original reviewer sampled code, oracle, and player-privacy evidence. The
reviewer's prior command
`node --test test/protocol.test.cjs test/battle-runner.test.cjs test/neutral-full-damage-oracle.test.cjs`
passed all three file entries with zero failures in about 93 seconds, and its
`git diff --check` passed. These are prior-review results, not checks rerun by
the correction owner. No Node or Python mechanics rerun is part of this
correction task.

## Correction disposition

The independent parent re-review accepted the corrections. The pending findings
and correction owner are cleared in `docs/ORCHESTRATION_STATE.json`; the initial
`NEEDS_CHANGES` and pending state remain only as historical context. The
correction work did not create or count a project task. No source or test files
were in scope.

## Validation and review evidence

- Before acceptance, a Python standard-library check parsed
  `docs/ORCHESTRATION_STATE.json` and `.codex/agents/global_reviewer.toml`,
  verified the pending original disposition, owner, two findings, count 20,
  next checkpoint 30, no milestone approval/commit, and unknown scheduler
  start/usage; 120 Markdown links across the changed documentation resolved.
- After recording acceptance, a final Python standard-library check parsed
  `docs/ORCHESTRATION_STATE.json` and verified current `ACCEPTED`, historical
  original `NEEDS_CHANGES`, cleared pending corrections/owner, baseline 20,
  next checkpoint 30, count 20, no milestone approval/commit, null active task,
  and unknown scheduler start/usage. Six links in this report and the task
  handoff were scanned; every local target resolves.
- `git diff --check HEAD && git diff --cached --check` passed after the final
  acceptance records.
- `/workspace/tools/trail/node_modules/.bin/trail check --base HEAD` reported no
  conflicts and four non-blocking lexical suggestions. They matched unrelated
  damage/event learnings at `MECHANICS_COVERAGE.md:48`, `REUSE_AUDIT.md:35`,
  `SCHEDULED_PROMPT.md:58`, and this report; none concerns the G0 status or
  reviewer-routing content.
- The stale-claim scan for the removed turn-field and later-run-only wording
  returned no matches. The parent independently reviewed the focused document
  and policy diff and recorded `ACCEPTED`. No Node or Python mechanics tests
  were rerun.
