# Restricted two-move neutral Mew turn variant

`gen5-neutral-two-move-mew-1v1-v4` is a separate fixed-state Gen 5 turn API
for two Mew with Tackle in slot 1 and Pound in slot 2. Each side submits its
move choice simultaneously: choice `1` selects Tackle and choice `2` selects
Pound. Choice `0` is pending, negative values are invalid, unknown positive
values are unsupported, and selecting a known move with zero PP is invalid.

The public `TwoMoveState` is independent of the accepted one-move
`BattleState` layout. It stores a fixed move order `[33, 1]` and an int32 PP
matrix shaped `[player, move slot]`. Both moves are neutral Normal physical
attacks against Psychic Mew, with no STAB: Tackle uses power 50 and Pound
power 40. The transition reuses the existing neutral action and v2 speed-tie
phase engine through a private variant-4 boundary with internal move ID 0.
Each actually executed actor spends only its chosen PP. An actor skipped after
a first-action knockout spends none. Spending a slot's final PP leaves the
other slot available. If either side has no PP in either slot, the next
nonterminal boundary is unsupported; this API does not execute Struggle.

The API exports `make_two_move_state`, `transition_two_move`, and
`project_two_move_view` from `simulator`. Variant ID is 4. Move IDs are fixed
to Tackle 33 then Pound 1; each move's maximum supported PP is 56. The
transition retains the shared fixed sixteen-slot tie draw trace and ordinary
Gen 5 accuracy, critical, and damage draws.

`TwoMoveTransition.event_move_ids` identifies the selected move for each
executed event row and uses `-1` for unused rows. A `TwoMovePlayerView`
contains only that recipient's two PP values, known move IDs, legal mask,
common own HP/stats, and recipient-visible event projection. It does not
contain the opponent's PP, an unexecuted choice, trusted full state, or RNG
trace.

`reference/neutral-two-move-oracle.cjs` initializes actual pinned Gen 5
random battles from direct one-Mew Synchronize/Hardy/perfect-IV/itemless sets
with both moves in order. It captures all four choice pairs, both unequal
initiative directions, a tied mixed pair, an actual critical, a first-action
knockout, and the last Tackle PP boundary. That boundary sets slot 1 PP to 1
through the live native Pokemon, regenerates normal requests with
`battle.makeRequest('move')`, and asserts the original PRNG limbs do not
change. Native requests retain both rows and mark an exhausted move disabled.
The focused differential test is `test/test_neutral_two_move.py`.

This variant adds only Tackle/Pound choice for Mew. It does not add other
moves, switches/teams, status, items, weather, Struggle execution, session or
reset integration, policies/training/rollouts/tournaments/reporting, a duel
runner, GPU/performance claims, or full-format support. [Task043](TASK043.md)
records the contract and evidence. Task043 was accepted by primary `/root` on
2026-11-05. This acceptance does not claim an Astra review or milestone
approval.
