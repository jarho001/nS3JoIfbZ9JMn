# Restricted neutral Pound Mew turn variant

**Status:** Task042 was accepted by primary `/root` on 2026-11-05. This does
not accept an Astra review or project milestone.

`gen5-neutral-pound-mew-1v1-v3` is a separate fixed-state turn API for two
Mew with Pound as their only move. Each side submits the single request action
`move 1`; the simulator does not choose among moves. Both Mew are Psychic, so
Pound is neutral Normal damage and receives no STAB.

The API exports `make_pound_state`, `transition_pound`, and
`project_pound_view` from `simulator`. The state retains the accepted fixed
`BattleState` layout and Gen 5 four-limb PRNG. Its explicit variant ID is 3 and
move ID is 1. The boundary checks Mew, Pound, supported initialized stat/HP
ranges, positive PP up to 56, and the normal move-request phase. For active
nonterminal states, the v1/v2 Tackle APIs reject v3 Pound states and the Pound
API rejects Tackle states. Terminal inputs remain absorbing in each API:
dispatch returns `STATUS_TERMINAL` before checking variant or move identity.

The pinned Gen 5 Dex identifies Pound as Normal, Physical, base power 40,
accuracy 100, and base PP 35. Three PP Ups give the maximum PP of 56. Its
default critical ratio is one in sixteen. The turn uses the accepted exact
accuracy, critical, and damage-roll LCG draws, Gen 5 physical damage order,
action order, speed-tie phases, events, and recipient visibility projection.
The v3 transition returns the same fixed sixteen-slot `TieTransition` trace
shape used by the speed-tie API.

`PlayerView` deliberately keeps one legal-action bit and has no move-name
field. A view returned by `project_pound_view` interprets that bit as Pound
because the state and projection API enforce v3's Pound-only boundary. The
projection contains no opponent trusted state or raw draw trace.

`reference/neutral-pound-oracle.cjs` uses actual pinned Gen 5 Battles created
from directly supplied one-Mew, Synchronize, Hardy, perfect-IV, itemless,
Pound-only sets. It captures normal requests, original PRNG values and limb
traces, recipient-selected events, and post-turn state. The focused differential
test is `test/test_neutral_pound.py`.

This variant adds one ordinary move case only. It does not add mixed moves,
team or switch handling, effects, Struggle, request sessions or reset,
duel-running, policies/training/reporting, or full-format support. It makes no
milestone, performance, or GPU claim. [Task042](TASK042.md) records the accepted
contract and evidence.
