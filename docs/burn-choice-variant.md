# Restricted single-burn Tackle/Mega Punch variant v7

Accepted by primary `/root` as Task052 on 2026-11-05; see [the task evidence](TASK052.md).

`gen5-single-burn-choice-mew-1v1-v7` adds a fixed turn slice for two directly
initialized Gen 5 Mew with fixed slots `[Tackle, Mega Punch]`. Its nested
`BurnChoiceState` wraps the v6 `TwoMoveState` layout with variant ID 7 and
holds two publicly known native burn flags. At most one flag may be true.
`make_burn_choice_state`, `transition_burn_choice`, and
`project_burn_choice_view` are exported from `simulator`; v5 and v6 retain
their existing action constants and APIs.

Choice `1` selects Tackle and choice `2` selects Mega Punch. Choice `0` is
pending, negative values are invalid, unknown positive values are unsupported,
and a selected exhausted slot is invalid. Each side must retain one positive
PP slot; nonterminal all-PP-zero states are unsupported. PP maxima are 56 and
32, and speeds must be distinct. Rejected and terminal submissions preserve
the entire nested state, including burn flags, with empty attack, draw, and
residual traces.

The implementation delegates attack order, move accuracy, critical checks,
damage rolls, PP use, and RNG updates to the existing v6 turn path. Its only
shared-helper extension passes an optional burn bit to the existing damage
composition call for physical moves. That call applies the accepted Gen 5
fixed-point 0.5 burn modifier after the roll/type steps. The no-burn default
keeps the existing v5/v6 behavior unchanged. There is no Facade, Guts,
Synchronize reflection, status infliction/cure, or other status effect in this
variant.

After the attack phase, a terminal battle skips residual completely. Otherwise
the one live pre-existing burned Pokémon loses
`max(1, floor(baseMaxHP / 8))`, clipped to current HP, with no PP or RNG use.
The wrapper recomputes final HP, phase, terminal flag, winner, and turn from
the original pre-turn state. Burn bits track native `Pokemon.status` even if
the Pokémon faints: the pinned native boundary can report HP condition `0 fnt`
while persistent status remains `brn`. The oracle records both values rather
than inferring status from the protocol condition.

`BurnChoiceTransition.events`, `draws`, and `event_move_ids` retain the
existing two-row attack contract. Its separate one-row `residual` event
reports the target, final target HP, maximum HP, and actual clipped HP loss.
Visible event order is attempted moves and their miss/hit/critical/damage/faint
rows, then residual damage/faint if present, then the native turn or winner
outcome. Current outcome and next request are derived from final
post-residual state; a terminal projection does not invent a repeated winner
event.

`BurnChoicePlayerView` wraps the accepted v6 accuracy-choice view, exposes the
two public burn flags, and adds a recipient-safe residual view. The damaged
owner sees exact residual HP and clipped loss. The opponent sees ceiling
percentage HP and loss `-1`. Both recipients see the public status flags,
attempted move, miss, and event order. Views omit RNG, complete state, opponent
PP, concealed exact HP, and unattempted choices. Terminal legal masks are
false for both players.

## Native evidence boundary

The guarded oracle uses pinned Showdown commit
`1242c1e2d9763ba5b85d33f13aecb89bae19ae3d`, unmodified
`gen5randombattle`, directly supplied itemless Synchronize/Hardy/perfect-IV
Mew sets, and no random-team generation. For a burn case it calls the native
`mon.setStatus('brn', mon)` before `makeRequest('move')`, then asserts installed
status, the other Mew unburned, public setup status lines, request HP conditions,
and unchanged original PRNG state. This is a trusted pre-existing status
boundary, not status-infliction or Synchronize behavior. The HP-1 fixture uses
native `sethp(1)` as trusted prior damage, followed by a live request; it does
not forge a terminal state.

| Native case | Observed turn behavior |
|---|---|
| `no-burn-control` | Both Tackle actions; no residual event. |
| `p1-burned-mega-punch-critical-seed-21` | P1's critical Mega Punch leaves P2 at 143/175 after burn reduction; P2's Tackle damages P1 before P1 takes its 21-HP residual. |
| `p2-burned-tackle-p2-faster` | P2 attacks first with burned Tackle; P1's Mega Punch hits, then P2 loses 21 residual HP. |
| `p1-burned-mega-punch-miss-tick-seed-0` | Mega Punch misses on accuracy roll 93, P2 still acts, and P1 takes its surviving 21-HP residual. |
| `burned-direct-attack-ko-skips-residual-seed-21` | The burned faster P1 directly KOs P2; attack-phase terminal state skips residual. |
| `burned-residual-ko-after-accuracy-miss-seed-0` | P2's Mega Punch misses at accuracy roll 93, P1 acts at 1 HP, and native burn residual KOs P1 for one clipped HP. Native status remains `brn` alongside condition `0 fnt`. |

The native trace records before/after HP, both move PP slots, stats, RNG,
turn/phase/winner, public status flags, live current/next requests, both
recipient event streams, exact draw callbacks, and the residual `[from] brn`
cause. Residual consumes no extra PRNG draw. The focused CPU test compares
complete numerical transition PyTrees in one heterogeneous `jit(vmap(...))`
batch and checks both recipients' current/next legal requests, status/HP
conditions, event order, terminal masks, and residual privacy. This is one
restricted public-status turn and does not claim broader Gen 5 status or
full-format support.
