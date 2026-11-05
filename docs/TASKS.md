# Current task handoff

Accepted numbered tasks: **52**. Active task: **none**. GPU block: **none**.
Automatic independent reviews: **disabled**. Baseline 50 is historical; no next checkpoint.
No pending review corrections.
G0 is accepted; G1/G3 and all implementation milestones remain open.

## Current position

Prioritize integrated JAX simulation and pinned Showdown parity. Tasks 048–049
add the restricted Mega Punch-only v5 turn and validate complete native battles
through terminal outcomes: [TASK048.md](TASK048.md), [TASK049.md](TASK049.md).
Task049 was accepted by primary `/root` on 2026-11-05. Task050 adds the
restricted Tackle/Mega Punch simultaneous-choice turn with real accuracy misses
and was accepted by primary `/root` and independent Astra T050Global on
2026-11-05; see [TASK050.md](TASK050.md) and the [T050Global review report](reviews/T050-global.md).
Task051 extends native/JAX parity for that unchanged API through successive
turns to genuine terminal outcomes and was accepted by primary `/root` on
2026-11-05; see [TASK051.md](TASK051.md).
Task052 adds one pre-existing public burn flag and burn residual to the
restricted Tackle/Mega Punch Mew turn and was accepted by primary `/root` on
2026-11-05 ([evidence](TASK052.md)).
Earlier bounded simulator
variants and staged request sessions remain documented in [TASKS_ARCHIVE_001_037.md](TASKS_ARCHIVE_001_037.md)
and the Task038–047 handoffs. Full-format support, the team generator, training,
and later milestones remain open; G0 is the only accepted milestone.

## Next step and workflow

Milestone preparation is pending; no Goal-mode implementation is active or
approved. Use [MILESTONE_PREPARATION.md](MILESTONE_PREPARATION.md) to sharpen
G0–G8 and prepare native behavior cases, then propose an integrated G3 goal.
Sol 6.1/high owns execution directly under [ORCHESTRATION.md](ORCHESTRATION.md).
Luna and Janitor require an explicit user request; automatic Astra gates and
per-patch task/handoff cycles are retired. Historical acceptance remains intact.

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
| 046 | Accepted | complete-battle parity through staged two-move request sessions — [evidence](TASK046.md) |
| 047 | Accepted | trusted terminal reset for v4 request sessions — [evidence](TASK047.md) |
| 048 | Accepted | restricted neutral Mega Punch Mew turn with native misses — [evidence](TASK048.md) |
| 049 | Accepted | exact complete-battle parity for Mega Punch Mew — [evidence](TASK049.md) |
| 050 | Accepted | restricted simultaneous Tackle/Mega Punch accuracy-choice turn — [evidence](TASK050.md), [T050Global review](reviews/T050-global.md) |
| 051 | Accepted | complete-battle parity through accuracy-choice v6 — [evidence](TASK051.md) |
| 052 | Accepted | restricted single-burn Tackle/Mega Punch Mew turn — [evidence](TASK052.md) |
