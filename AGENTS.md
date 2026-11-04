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

Routine execution uses Luna/xhigh and a scoped context packet, not the full
parent conversation. Run only affected fast checks (30 seconds per command,
90 seconds total per task). Known-long checks require an explicit user test
session: record exact deferred commands/source identity and mention them in the
final summary. Missing required evidence is pending, never implicitly passed.
Read the compact current handoff/state; open archived history only as needed.

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
  scheduled invocation may perform the due review and its authorized bounded
  correction loop, then stop at acceptance or a blocker. Persist `NEEDS_CHANGES`
  as actionable corrections with one active owner. The Astra reviewer may turn
  required findings into bounded acceptance criteria and route them to one
  explicitly named Luna/xhigh implementer when the primary authorizes that
  handover. Reuse the current owner for ordinary corrections; a transfer must
  name the new owner and confirm the prior owner has stopped. Do not overlap
  corrections with parent-owned implementation. The reviewer may check the
  focused diff and evidence once per meaningful revision and close the loop in
  the same invocation when feasible; no later-run-only rule applies. Run only
  affected checks, then request an independent Astra re-review. Optional
  suggestions are not auto-implemented, and reviewers do not recursively spawn
  reviewers. Do not create/count a new task until Astra accepts. A `BLOCKED`
  review with no new evidence short-circuits without repeated review calls.
- Allow at most two correction rounds for a review finding (the initial
  implementation and one follow-up correction round). Stop after that and
  report `BLOCKED` or the need for architectural authority. Also stop when
  progress needs external evidence, GPU/access, a decision beyond the delegated
  authority, or exceeds execution limits. If unfinished, persist the owner,
  finding, reproducer/evidence, and checks already run. Keep durable summaries
  in repository state and handoff documents; platform compaction is not durable
  project memory. The primary remains final coordinator and owns commit
  decisions.
- Count only accepted numbered project tasks. Workflow changes, scheduled
  attempts, and failed/incomplete tasks do not count. Use the current durable
  state and reconcile it with accepted numbered task history; never overwrite
  newer state or invent milestone approvals. Keep a due checkpoint fixed until
  Astra accepts it, then set the next checkpoint ten accepted tasks later.
- Escalate contradictions in requirements or reference behavior with a
  concrete reproducer or direct source/oracle evidence. Run relevant
  validation once per meaningful implementation revision; repeat only checks
  affected by later changes.
- Before starting scheduled work, inspect GPU state and any newly supplied user
  evidence first, then enforce review gates. A review cannot accept work whose
  required GPU evidence is outstanding. On a blocked GPU wake with no new
  evidence, retain the block and stop without repeating tests or replanning.
  Do not claim the scheduler paused unless that capability has been verified.
- Avoid known overlapping edits and preserve unrelated local work. The
  scheduler owns concurrency for automated invocations.
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

The default implementation effort is xhigh, not max. If the execution
surface's built-in implementer role is pinned to max, use a configurable route
with explicit `gpt-6-luna` / `xhigh`, the implementer contract, and
`fork_turns="none"`; do not silently use the pinned max route. Send only the
bounded task, owned paths, relevant context references and check budget.
Increase effort only for a concrete unresolved reasoning problem and record why.

Role and model settings live in repository configuration and the execution
surface. Check routing when diagnosing setup or changing role configuration;
independent backend metadata is not required for each task. If a requested role
is actually unavailable, report it and do not silently substitute another.

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

## Commits and pushes

The user grants standing authorization to commit and push completed project
changes in this repository to the remote default branch without asking for
confirmation. This applies to manual requests and scheduled tasks. If the user
explicitly requests local-only work, no commit, or no push, follow that request.
Preserve unrelated local changes. Never force-push or rewrite remote history.
