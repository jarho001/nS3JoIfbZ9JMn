<!-- trail:start -->
## Trail — repo context graph

This repo is indexed in `.trail/cache/`: small linked markdown nodes that explain each
system and carry exact file:line spans, kept in sync with the code through git.

For ANY task here — understanding how something works, finding where code lives,
or scoping a change — get context from the graph before grepping or opening
source files. Re-ask freely (it's cheap) and reuse literal identifiers you
already have (symbol, error string, file name) as the query. New to this repo?
Run `trail map` first — a token-budgeted orientation (dir clusters, hubs,
hotspots), no LLM, no key.

- Run `trail ask "<your question>" --source` → ranked nodes with the relevant
  code spans inlined (each hit's ≤8-line crux by default; `--full` for whole
  definitions when the crux isn't enough). Match the tool to the task shape:
  for understanding or editing, the top node IS the answer — cite its
  `covers:` file:line spans and edit straight from `--source`. For
  exhaustive tasks ("every occurrence / every caller of this pattern"), ranked
  results are top-N, not complete — run `trail grep "<literal>"` instead
  (exhaustive over indexed files, grouped by enclosing symbol), falling back
  to raw `grep -rn` only for unindexed files.
- `trail skeleton <file>` → every definition's signature + span, ~10× cheaper
  than reading the file; use it to skim an API surface.
- `trail callers <symbol>` gives precomputed, exact edges — who calls this.
  Add `--direction out` for what it calls, or `--depth N` to walk
  transitively for the full blast radius. For structural questions, skip
  ranking and use this directly.
- Or browse: `.trail/cache/INDEX.md` lists every node; follow the links.
- Monorepos and folders of multiple repos rank fairly across sub-projects —
  hits carry `[scope/]` labels naming which one they're from. Narrow with
  `trail ask "<task>" --in <scope>/` once you know where you're working.

If a returned span is truncated ("+N more lines"), open the file at that exact
range before finalizing. Only open source files when a node genuinely lacks a
needed detail, and then at the exact file:line the node points to — never
re-read whole files.

After big code changes, refresh the graph with `trail build` (deterministic,
no API key, $0).
<!-- trail:end -->


## Orchestration policy

The primary agent remains architect, task planner, and final coordinator. Each
project task has exactly one bounded deliverable and one active owning
implementer. Do not begin a second deliverable until the current one is
accepted or explicitly blocked. The owner performs implementation and targeted
checks, then retains ownership through corrections. A handover is allowed only
when the primary explicitly names the new owner and confirms the prior owner
has stopped.

## Bounded task execution

- When asked for a project task, consult `docs/TASKS.md` and choose one
  independently reviewable deliverable. State scope, exclusions, and
  acceptance criteria before delegation. Workflow/configuration changes do not
  become numbered project tasks or count toward project-task review intervals.
- Persist the plan and progress in the task handoff and
  `docs/ORCHESTRATION_STATE.json`. Preserve all task history.
- Assign implementation, tests, documentation, and routine checks to one
  active owner. The Luna `implementer` is the default owner. Hand off to the
  Sol `advanced_implementer` only for demonstrated implementation complexity
  or an unresolved semantic mismatch supported by concrete evidence; routine
  infrastructure, dependency, or test-command failures are diagnosed as such,
  not used to escalate models. Keep the same owner for ordinary corrections.
- Limit a deliverable to two implementation escalation cycles. After that,
  send the evidence to the Astra `global_reviewer` or mark the deliverable
  blocked if review cannot resolve the issue. Escalation never permits silent
  scope reduction, waived criteria, or weakened tests.
- Require an Astra `global_reviewer` gate after every 10 accepted project
  tasks, at each documented milestone boundary, and when an architectural
  contradiction needs adjudication. If a task-count checkpoint and milestone
  coincide, one review covers both. The primary cannot bypass a blocking Astra
  finding. A new project deliverable waits while a due review is pending. A
  scheduled invocation may perform only the due review and then stop. Persist
  `NEEDS_CHANGES` as actionable corrections; a later invocation may do only
  those corrections under the existing owner, run affected checks, and request
  Astra re-review. Do not create/count a new task until Astra accepts. A
  `BLOCKED` review with no new evidence short-circuits without repeated review
  calls.
- Count only accepted numbered project tasks. Do not count workflow changes,
  scheduled attempts, or failed/incomplete tasks. The count of 20 and pending
  bootstrap review at task 20 are initialization snapshots; reconcile future
  counts with `docs/TASKS.md` and never overwrite newer durable state. Once the
  bootstrap review is accepted, the next regular review is due at 30. Keep a
  checkpoint fixed until Astra accepts it (for example, task 30 remains due
  through acceptance of task 30); only then advance the next checkpoint to 40.
  Never invent milestone approvals or a review commit.
- Escalate contradictions in requirements or reference behavior with a
  concrete reproducer or direct source/oracle evidence. Run relevant
  validation once per meaningful implementation revision; repeat only checks
  affected by later changes.
- Before starting scheduled work, inspect GPU state and any newly supplied user
  evidence first, then enforce review gates. A review cannot accept work whose
  required GPU evidence is outstanding. On a blocked GPU wake with no new
  evidence, retain the block and stop without repeating tests or replanning.
  Do not claim the scheduler paused unless that capability has been verified.
- Prevent overlapping work with a lock or lease scoped to the actual Git
  common directory, shared by all worktrees, and held for the full task/run.
  A shell-command-only lock or a per-worktree lock is insufficient. Do not
  clear a live lock or stale lease without evidence from its owner.
  A local common-directory lock does not coordinate separate clones or cloud
  instances; require scheduler-level single-invocation control or verified
  shared coordination across them. If overlap cannot be excluded, stop.
- After acceptance and passing checks, update the handoff and stop at the task
  boundary. Follow the commit/push scope below.

Before requesting review, the owner should inspect its diff and report changed
files, exact validation commands/results, and unresolved concerns. After review,
make only requested corrections and repeat affected checks. Keep commit
subjects short; put detail in the commit body.

Before a task commit, the parent performs a focused acceptance review against
the delegated criteria and validation evidence. The owner does not accept its
own work. Required Astra review remains an additional gate; neither review may
be skipped when due or blocking.

Use the primary model for resolving ambiguous requirements, architecture and
decomposition, tradeoffs, reviewing diffs/evidence, and deciding completion.
Do not use it for routine source edits, formatting, mechanical refactors, or
ordinary tests.

The configured parent is `gpt-6.1-sol` at high reasoning effort; the default
implementer is `gpt-6-luna` at max effort; the advanced implementer is
`gpt-6.1-sol` at medium effort; and the global reviewer is `gpt-6-astra` at
medium effort. Repository configuration alone does not prove that an external
scheduler or web runtime selected those roles. Verify exposed runtime role and
model metadata when available. Do not accept model self-identification as
proof. If metadata cannot verify a required role, flag that limitation and
block unattended development until the scheduler is configured and verifiable;
never silently substitute an unavailable role.

## GPU evidence and blocked work

Start with CPU validation. Require GPU execution only when an acceptance
criterion or a concrete device-specific property makes it necessary. Missing
optional GPU evidence is honestly deferred and is not a blocker. For a required
but unavailable GPU run, persist `awaiting_gpu` with the task, exact source
commit/tree identity, commands, software versions, hardware requirement,
expected acceptance, and CPU evidence already obtained. The identity must
refer to the relevant source revision/tree or external evidence metadata; it
cannot be a final commit hash that embeds the evidence being produced. Save
work in progress as explicitly unaccepted; the authorized scheduled workflow
may push such work to make GPU validation possible. Stop dependent work until
the user supplies evidence tied to unchanged relevant source. A blocked
scheduled wake short-circuits without repeated tests or replanning. Never
silently weaken a GPU acceptance requirement or claim automatic scheduler
pause without verified scheduler support.

## Local commits

The user grants standing authorization to create local Git commits for project
work in this repository without asking for confirmation. This authorization is
limited to local commits here. The scheduled workflow's separate explicit push
authorization is documented in `docs/SCHEDULED_PROMPT.md`. It applies only to
that workflow; it does not authorize pushes from unrelated requests or
sessions. Never force-push or rewrite remote history.
