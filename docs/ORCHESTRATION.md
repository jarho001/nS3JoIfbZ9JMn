# Project orchestration

This repository advances in one accepted, independently reviewable project
deliverable at a time. The durable state is
[`ORCHESTRATION_STATE.json`](ORCHESTRATION_STATE.json); numbered project-task
history remains in [`TASKS.md`](TASKS.md). The workflow and its configuration
are not project tasks and do not affect the accepted-task counter.

## Ownership and escalation

The parent agent is the architect and final coordinator. It defines one
deliverable, exclusions, and acceptance criteria, then assigns one active
implementation owner. The Luna `implementer` owns ordinary edits and focused
validation. It retains ownership through corrections. A different owner may
start only after an explicit handover that names the new owner and stops the
prior owner.

Keep coordination proportional: one concise delegation, checkpoints only for
unresolved architecture or evidence, and one complete review package plus
required corrections. Batch bookkeeping and affected checks; avoid repeated
status polling and interim reviews. Preserve required validation and review
gates.

Hand off from Luna to the Sol `advanced_implementer` only when concrete
evidence demonstrates implementation complexity or an unresolved semantic
mismatch. A reproducer, focused failure, exact source evidence, or pinned-oracle
comparison should make the escalation reviewable. Routine infrastructure,
dependency, and test-command failures are diagnosed as blockers; they do not
justify model escalation. Allow at most two implementation escalation cycles
for one deliverable. Then request Astra review or record a blocker. No handoff
can silently reduce scope, waive criteria, or weaken tests.

The Astra `global_reviewer` is an independent gate with no implementation
ownership. It samples the actual code, tests, and oracle/reference evidence,
assesses project direction and documented milestone progress, and returns
`ACCEPTED`, `NEEDS_CHANGES`, or `BLOCKED` with evidence and blockers. It may
compact durable project documentation when useful, not platform internals. The
parent cannot bypass a blocking finding.

For required corrections, Astra may translate findings into bounded acceptance
criteria and route them to one explicitly named Luna/max implementer when the
primary authorizes that handover. Reuse the existing owner for ordinary
corrections; a handover names the new owner and confirms the prior owner has
stopped. Do not overlap correction work with parent-owned implementation. The
reviewer retains no implementation ownership, optional suggestions are not
auto-implemented, and reviewers do not recursively spawn reviewers. The parent
remains the final coordinator and commit owner.

## Review checkpoints

Run the Astra gate after every 10 accepted numbered project tasks, at each
documented milestone boundary, and when an architectural contradiction needs
adjudication. One review covers coincident task-count and milestone
checkpoints. A new deliverable waits while any gate is due.

Persist `NEEDS_CHANGES` as actionable corrections with one active owner. The
same invocation may complete the bounded corrections, run affected checks, and
request independent Astra re-review when feasible; there is no later-run-only
rule. The reviewer checks the focused diff/evidence once per meaningful
revision. This correction loop creates or counts no project task. Do not start
another task until Astra accepts. Allow at most two correction rounds (initial
implementation plus one follow-up); after that, stop and report `BLOCKED` or a
need for architectural authority. Stop sooner if external evidence, GPU/access,
a decision beyond delegated authority, or execution limits are required. If the
loop is unfinished, persist the owner, required finding, reproducer/evidence,
and checks run. Optional suggestions are not auto-implemented, and no recursive
reviewers are spawned. A `BLOCKED` review with no new evidence short-circuits
without repeating review calls. After acceptance, the parent performs a
focused acceptance review before commit.

A scheduled invocation whose only work is a due review may perform that review
and its authorized bounded correction loop, then stop at acceptance or a
blocker. It must not start a new project deliverable in that invocation.

On every wake, check `gpu_block` and any newly supplied user evidence before
review. Do not certify a deliverable whose required GPU evidence is outstanding;
return `BLOCKED` if that evidence is part of the review scope. If no new evidence
arrived for an `awaiting_gpu` block, stop without rerunning tests or replanning.

Only accepted numbered project tasks count. Workflow edits, scheduled
invocations, and failed or incomplete tasks do not count. Use the current
durable state and reconcile it with accepted numbered entries in `TASKS.md`;
preserve newer state and do not infer approvals or review commits from silence.
Milestones count only when recorded in the project handoff; do not invent
milestone boundaries.

## Validation and GPU evidence

Validate on CPU first. GPU execution is required only when an acceptance
criterion or a specific device behavior makes it necessary. Optional GPU
absence is recorded as deferred, without making it a blocker. If required GPU
validation is unavailable, save explicitly unaccepted work and set
`gpu_block.status` to `awaiting_gpu`. Record the task, exact source revision or
tree identity, commands, software versions, hardware need, expected acceptance,
and CPU evidence. Identify the source revision/tree or external evidence
metadata; never use a final commit hash that embeds the evidence as its own
identity. Stop dependent work until the user supplies GPU evidence for the
unchanged relevant source.

A blocked scheduled wake without new evidence short-circuits: retain the block
and stop without rerunning tests or replanning. Do not claim that the scheduler
paused unless that behavior was verified. A scheduled workflow may save,
commit, and push explicitly unaccepted WIP when that is needed to obtain the
required GPU evidence; the block remains active until evidence is reviewed.

## Non-overlap and repository state

Avoid known overlapping edits and preserve unrelated local work. The scheduler
controls concurrency for automated invocations.

## Durable state update protocol

Before work, read this document, `ORCHESTRATION_STATE.json`, `TASKS.md`, and the
Git status/history. Reconcile the accepted count from numbered task statuses
against durable state; preserve newer state and recorded history. If they
disagree, investigate recorded evidence and do not fabricate acceptance or
review approvals.

When a project task starts, record its exact ID, scope, owner, and state in the
top-level `active_task`. On acceptance, update `TASKS.md`, increment the
accepted count, and clear `active_task`. Keep the next regular review checkpoint
fixed until Astra accepts that due review; then set the next checkpoint ten
accepted project tasks after the accepted review baseline. Record
review dispositions and commits only when evidence exists; keep actionable
Astra corrections in `review_state.pending_review_corrections` until accepted.
For a failed, abandoned, or incomplete task, preserve its history without increasing
the accepted count. For a GPU block, populate `gpu_block`; clear it only after
the evidence has been tied to unchanged source and accepted. Keep unknowns as
`null` or explicitly `unknown`.

## Role configuration

Model/effort roles are selected through repository configuration and the
execution surface. Check their setup when diagnosing routing or changing role
configuration; independent backend metadata is not a per-task requirement. If
a requested role is actually unavailable, report it and do not silently
substitute another.
