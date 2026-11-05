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

Task045 adds the opt-in local session adapter in `simulator/two_move_session.py`:
`make_two_move_request_session`, `submit_two_move_choice`,
`project_two_move_request_view`, and `project_two_move_submission_view`. The
first valid choice only changes that actor's private pending slot. The second
accepted choice resolves the unchanged v4 transition once, clears both slots,
and increments a bounded local episode. The actor view contains their current
request and own submitted bit; the submission view carries prior resolved
events in a separate projection. The episode is a local token, not a native
Showdown request ID. Stale-token and duplicate-commit rejection belong to this
adapter; native `Side.choose` can replace an earlier commitment. Task045 was
accepted by primary `/root` on 2026-11-05; this adapter does not expand battle
mechanics or claim a milestone.

Task047 adds `reset_two_move_request_session` and its fixed-shape
`TwoMoveSessionReset` result. This trusted coordinator operation accepts a
supported caller-supplied nonterminal v4 state only after the local session is
terminal with both pending choices clear. It copies that state exactly,
clears commitments, and advances the local episode once. Stale tokens take
precedence over unsupported metadata or replacement state; a valid live
session reports not-terminal, and overflow is checked before reset. Rejections
preserve the entire input session and run no turn or RNG draw. This remains a
local trusted reset, not native `rqid` or global replay security. Primary
`/root` accepted Task047 on 2026-11-05; it does not change the v2 reset or v4
numerical engine, and does not claim an Astra review or milestone approval.

`reference/neutral-two-move-oracle.cjs` initializes actual pinned Gen 5
random battles from direct one-Mew Synchronize/Hardy/perfect-IV/itemless sets
with both moves in order. It captures all four choice pairs, both unequal
initiative directions, a tied mixed pair, an actual critical, a first-action
knockout, and the last Tackle PP boundary. That boundary sets slot 1 PP to 1
through the live native Pokemon, regenerates normal requests with
`battle.makeRequest('move')`, and asserts the original PRNG limbs do not
change. Native requests retain both rows and mark an exhausted move disabled.
The focused one-turn differential test is `test/test_neutral_two_move.py`.
Task044 adds an opt-in `--duels` native fixture mode and a separate
`test/test_neutral_two_move_duel.py` check for exact traces through genuine
battle outcomes, including alternating Tackle/Pound choices and continuation
after the last Tackle PP is spent. Its result does not expand the move or
effect boundary.

This variant adds only Tackle/Pound choice for Mew. It does not add other
moves, switches/teams, status, items, weather, Struggle execution, reset or
networked session integration, policies/training/rollouts/tournaments/reporting,
a production simulator duel runner, GPU/performance claims, or full-format
support. [Task043](TASK043.md)
records the v4 turn contract and evidence; [Task044](TASK044.md) records the
complete-battle parity validation. Task043 and Task044 were accepted by primary
`/root` on 2026-11-05. Task045 was also accepted by primary `/root` on that
date; none of these records claims Astra review or milestone approval.

[Task046](TASK046.md) adds an opt-in `--session-duels` fixture and focused CPU parity test
that thread the accepted staged session API through paired native battles to
genuine terminal outcomes in both submission orders. It checks first-commit
privacy, local episode progression, separate current-request and prior-event
views, full state/RNG/draw parity, and Pound continuation after p1 spends its
last Tackle PP. It was accepted by primary `/root` on 2026-11-05; this
does not claim an Astra review or milestone acceptance. It changes no battle
mechanics, simulator API, or numerical engine.
