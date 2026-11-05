# Current task handoff

Accepted numbered tasks: **45**. Active task: **none**. GPU block: **none**.
Global review baseline: **40**; next checkpoint: **50**. No pending review corrections.
G0 is accepted; G1/G3 and all implementation milestones remain open.

## Current position

Prioritize integrated JAX simulation and pinned Showdown parity. Restricted
Mew/Tackle v1 and speed-tie v2 turns/duels are accepted. Task037 adds staged
private submissions, bounded local episode tokens, and separate current-request
and previous-event projections around unchanged v2 kernels. Task038 adds a
read-only initial session view; details: [TASK038.md](TASK038.md). Task039
validates complete staged-session battles to terminal against pinned Showdown
in both submission orders: [TASK039.md](TASK039.md). Task040 validates those
sessions across a heterogeneous `jit(vmap(submit_choice))` batch:
[TASK040.md](TASK040.md). Task041 adds a trusted terminal-session reset that
requires cleared commitments and accepts a supplied supported v2 request
state: [TASK041.md](TASK041.md).

Task042 adds the separately versioned Pound-only Gen 5 Mew v3 turn slice with
native ordinary-turn, draw, and recipient-view parity: [TASK042.md](TASK042.md).
It expands one playable move case without accepting a project milestone.

Task043 adds the separately versioned Gen 5 two-move Tackle/Pound Mew v4 turn
API with simultaneous choice, per-slot PP, native request/event/RNG parity,
and isolated recipient views: [TASK043.md](TASK043.md). Task044 validates that
API through real native terminal outcomes with dynamic Tackle/Pound choices
and last-Tackle-PP continuation: [TASK044.md](TASK044.md). Struggle and the
broader format boundary remain unsupported. Task044 was accepted by primary
`/root` on 2026-11-05; this does not claim an Astra review or milestone
acceptance.

Task045 adds private staged request sessions for v4 two-move choices, keeping
the current request separate from prior resolved events: [TASK045.md](TASK045.md).
It was accepted by primary `/root` on 2026-11-05. This does not claim an Astra
review or milestone acceptance.

Relevant APIs: `simulator/neutral_tackle.py`, `simulator/neutral_tackle_ties.py`,
`simulator/neutral_pound.py`,
`simulator/neutral_two_move.py`,
`simulator/two_move_session.py`, `simulator/neutral_duel.py`,
`simulator/request_session.py`. API contract and
supported boundaries: [neutral-tackle-variant.md](neutral-tackle-variant.md).
Continue one justified simulator deliverable; no next task is prescoped here.

## Routine execution

Use Luna/xhigh with a scoped context packet, not the parent's full conversation.
Run only relevant fast checks: at most 30 seconds per command and 90 seconds
of test execution per task. Use `timeout --signal=TERM --kill-after=5s 30s ...`;
skip known-long checks instead of starting them. Do not extend the limit or
rerun timed-out checks. Record exact deferred commands, source identity and
reason, and recommend an explicit long-test session in the final summary.
Long suites, exhaustive grids, GPU validation and performance probes require
an explicit test-session request. Required missing acceptance evidence remains
pending; do not describe a deferred check as passed.

Useful dedicated integration session (not part of routine development):

```sh
JAX_PLATFORMS=cpu python -m unittest test.test_request_session test.test_neutral_tackle test.test_neutral_duel test.test_neutral_tackle_ties -v
```

## Accepted task index

Full Task001–037 contracts, evidence and acceptance history are preserved
byte-for-byte in [TASKS_ARCHIVE_001_037.md](TASKS_ARCHIVE_001_037.md).
Read only a relevant archived task when needed. This table is the compact
accepted-count ledger; archive records are its detail, not additional tasks.

| Task | Status | Deliverable |
|---|---|---|
| 001 | Accepted | pinned Showdown reference |
| 002 | Accepted | player-view projection and visibility fixtures (G0/G1 portion) |
| 003 | Accepted | request-to-request baseline battles |
| 004 | Accepted | reference protocol trace and replay harness |
| 005 | Accepted | first JAX arithmetic slice |
| 006 | Accepted | stored-stat derivation slice |
| 007 | Accepted | compose Attack stored-stat derivation and stage scaling |
| 008 | Accepted | first damage arithmetic primitive |
| 009 | Accepted | base-damage integer core |
| 010 | Accepted | neutral base-damage composition |
| 011 | Accepted | fixed-point modifier primitive |
| 012 | Accepted | Gen 5 damage-finalization tail |
| 013 | Accepted | Gen 5 type-effectiveness arithmetic |
| 014 | Accepted | burn-reduction kernel |
| 015 | Accepted | Gen 5 STAB arithmetic |
| 016 | Accepted | bounded post-core damage composition |
| 017 | Accepted | neutral full damage composition |
| 018 | Accepted | G0 mechanics coverage ledger |
| 019 | Accepted | simulator and integration reuse audit |
| 020 | Accepted | benchmark charter and source/version manifest |
| 021 | Accepted | versioned per-battle runner result records |
| 022 | Accepted | deterministic paired-seed manifests |
| 023 | Accepted | reuse seed-bound prepared teams |
| 024 | Accepted | execute one paired seed case |
| 025 | Accepted | sequential paired-manifest runner |
| 026 | Accepted | offline recorded-pair outcome accounting |
| 027 | Accepted | offline recorded-work counter accounting |
| 028 | Accepted | request-aware eleven-slot singles action adapter |
| 029 | Accepted | completed-battle player decision trajectory collector |
| 030 | Accepted | versioned move-first reference baseline policy |
| 031 | Accepted | named baseline paired matchup execution |
| 032 | Accepted | single named baseline matchup CLI |
| 033 | Accepted | offline named baseline outcome report |
| 034 | Accepted | restricted seeded JAX neutral Tackle turn |
| 035 | Accepted | bounded compiled restricted Tackle duels |
| 036 | Accepted | seeded speed ties in restricted JAX Tackle duels |
| 037 | Accepted | staged simultaneous submissions for restricted JAX turns |
| 038 | Accepted | initial actor view for a request session — [evidence](TASK038.md) |
| 039 | Accepted | complete staged-session battle parity — [evidence](TASK039.md) |
| 040 | Accepted | heterogeneous batched staged request sessions — [evidence](TASK040.md) |
| 041 | Accepted | trusted terminal request-session reset — [evidence](TASK041.md) |
| 042 | Accepted | versioned neutral Pound Mew turn slice — [evidence](TASK042.md) |
| 043 | Accepted | two-move neutral Tackle/Pound Mew turn API — [evidence](TASK043.md) |
| 044 | Accepted | exact complete-battle parity for two-move Mew — [evidence](TASK044.md) |
| 045 | Accepted | private staged submissions for two-move Tackle/Pound turns — [evidence](TASK045.md) |

## Active work

None. Task045 was accepted by primary `/root` on 2026-11-05. This does not claim
an Astra review or project milestone acceptance. The accepted count is 45; the
review baseline remains 40 and the next checkpoint remains 50. Optional
regressions are recorded in `deferred_validation` in the state; they remain
unrun and are not acceptance dependencies.
