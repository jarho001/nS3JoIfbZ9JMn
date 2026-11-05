# Current task handoff

Accepted numbered tasks: **39**. Active task: **none**. GPU block: **none**.
Global review baseline: **30**; next checkpoint: **40**. No pending review corrections.
G0 is accepted; G1/G3 and all implementation milestones remain open.

## Current position

Prioritize integrated JAX simulation and pinned Showdown parity. Restricted
Mew/Tackle v1 and speed-tie v2 turns/duels are accepted. Task037 adds staged
private submissions, bounded local episode tokens, and separate current-request
and previous-event projections around unchanged v2 kernels. Task038 adds a
read-only initial session view; details: [TASK038.md](TASK038.md).
Task039 validates complete staged-session battles to terminal against pinned
Showdown in both submission orders: [TASK039.md](TASK039.md).

Relevant APIs: `simulator/neutral_tackle.py`, `simulator/neutral_tackle_ties.py`,
`simulator/neutral_duel.py`, `simulator/request_session.py`. API contract and
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

## Active work

None. Optional long regressions are recorded in `deferred_validation` in the
state; they were not run and do not block the focused Task038/039 acceptance.
