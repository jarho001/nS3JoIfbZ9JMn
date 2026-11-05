# Restricted Tackle/Mega Punch accuracy-choice variant v6

Accepted by primary `/root` on 2026-11-05 after independent T050Global review; see the [review report](reviews/T050-global.md).

`gen5-accuracy-choice-mew-1v1-v6` adds one separately versioned ordinary turn
for two directly initialized Mew that simultaneously select a move from fixed
slots `[Tackle, Mega Punch]`. It reuses the `TwoMoveState` fixed layout with
variant ID 6; this is layout reuse, not v4 semantic compatibility. The public
API is `make_accuracy_choice_state`, `transition_accuracy_choice`, and
`project_accuracy_choice_view` from `simulator`.

Choice `1` selects Tackle (power 50, accuracy 100, maximum PP 56). Choice `2`
selects Mega Punch (power 80, accuracy 85, maximum PP 32). Choice `0` is
pending, negative values are invalid, unknown positive values are unsupported,
and choosing a known exhausted slot is invalid. Each side needs at least one
positive PP slot. Partial depletion is supported, but a nonterminal state
where either side has exhausted both slots is unsupported; the native request
may offer Struggle there, and this API does not execute it. Only distinct
positive Speed values are supported.

The accepted v5 accuracy/miss action and turn machinery provide the numerical
work. The v6 adapter supplies the selected action's power and accuracy, then
maps the internal action, RNG, HP, turn, winner, and terminal fields back to
the public two-slot state. It deducts PP only from each executed actor's chosen
slot, including on a miss. A first-action knockout skips the other actor's PP,
event, and draws. A final nonterminal move in one slot leaves the other slot
available and the public phase remains a move request while each side has a
usable move.

Every attempted action consumes the native `random(100)` accuracy draw.
Tackle's 100 accuracy always hits after consuming that draw. Mega Punch rolls
85–99 miss; a miss still spends its selected PP and records actor, target, and
attempted move ID, while consuming no critical or damage draw. A hit consumes
the native `random(16)` critical check followed by `random(16)` damage. Draws
are packed into the existing six-slot v5 trace with bounds, values, contexts,
and four-limb before/after RNG state. Rejected, pending, unsupported, and
terminal results preserve the public state and return empty event/draw traces.

`AccuracyChoiceTransition.event_move_ids` identifies each executed row and
uses `-1` for unused rows. `AccuracyChoicePlayerView` wraps the recipient-safe
two-move view and exposes an explicit miss mask. The miss actor, target, event
order, and attempted move ID remain visible; damage, target HP, critical, and
faint fields are hidden for a miss. A view contains the recipient's own PP,
fixed move IDs, and legal mask, with no RNG, opponent PP, full state, or
unexecuted opponent choice.

The native oracle creates unmodified pinned Gen 5 random-battle requests from
direct one-Mew Synchronize/Hardy/perfect-IV/itemless sets and normal two-move
move slots. It bypasses random team generation, checks live 56/32 PP, and
captures current/next requests, complete state, exact native draw chains, and
both recipient event projections. Its bounded fixtures cover all four choice
pairs, both initiative directions, a Mega Punch accuracy miss alongside a
Tackle hit on the same roll, a critical, first-action knockout, and both
last-slot PP boundaries. One native ordinary follow-up selects Tackle after
the same actor's final Mega Punch PP is spent.

This variant adds only that direct-Mew choice turn. It does not add status,
items, weather, boosts, other moves, Struggle execution, sessions, reset,
runner, training, GPU, full-format, or milestone support. Existing v4 semantics
and sessions do not change; existing v5 default behavior remains the same.

## Successive native duels (Task051)

Accepted by primary `/root` on 2026-11-05; see [TASK051.md](TASK051.md) for
the full validation and terminal-view correction evidence.

The oracle's opt-in `--duels` mode records six fixed full battles through native
terminal outcomes. It uses seeds 0, 1, and 21; distinct speeds in both
initiative directions; the same direct, itemless, unmodified Gen 5 Random
Battle Mew sets; and live `makeChoices` requests with the original PRNG
observer. A fixture-local repeating choice schedule mixes both moves and all
four submitted pairs. If its scheduled slot is disabled, it chooses the first
slot that the current native request marks legal. Each battle is capped at 32
turns and fails on truncation or unsupported all-PP-zero before native end.

| Native battle | Turns | Attempts | Tackle / Mega Punch | Hits / misses | Criticals | First-action KO skips | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| `complete-p1-faster-seed-0` | 6 | 12 | 7 / 5 | 11 / 1 | 1 | 0 | p2 win; terminal after second action |
| `complete-p2-faster-seed-1` | 8 | 15 | 7 / 8 | 13 / 2 | 0 | 1 | p2 win |
| `complete-p2-faster-critical-seed-21` | 5 | 9 | 4 / 5 | 9 / 0 | 2 | 1 | p2 win |
| `complete-last-mega-punch-seed-0` | 7 | 14 | 9 / 5 | 13 / 1 | 0 | 0 | p2 win; Mega Punch depletion continued with Tackle |
| `complete-last-tackle-seed-1` | 8 | 15 | 4 / 11 | 12 / 3 | 0 | 1 | p1 win; Tackle depletion continued with Mega Punch |
| `complete-first-action-ko-seed-21` | 1 | 1 | 0 / 1 | 1 / 0 | 1 | 1 | p1 win |

Across the six complete outcomes, 35 turns produced 66 attempts: 31 Tackle and
35 Mega Punch. Tackle hit 31 times and had 2 criticals; Mega Punch hit 28
times, missed 7 times, and had 2 criticals (4 criticals total). Mega Punch
misses were 5 by p1 and 2 by p2; neither actor missed with Tackle. Submitted
pair counts were Tackle/Tackle 7, Tackle/Mega Punch 11, Mega Punch/Tackle 10, and
Mega Punch/Mega Punch 7. Four first-action knockouts skipped the other action;
two terminal turns occurred after both actions. There were no double-miss
turns and no same-actor misses on consecutive turns. Both final-PP setups
spent the selected slot on a live native request, exposed that row as disabled
in the current/next own request, and executed the remaining move on a later
turn.

Each turn records both submitted choices, the complete before/after HP, PP,
stats, RNG, turn, phase and winner snapshots, ordered draw bounds/contexts/
values and four-limb pre/post RNG states, selected move IDs, and both split
recipient event projections. The focused JAX test threads the actual returned
state through all 35 turns and compares each complete transition, recipient
views, live requests/legal masks, and terminal state. Every battle also checks
one rejected post-terminal submission with unchanged state/RNG and an empty
trace. The native and focused evidence is recorded in [TASK051.md](TASK051.md);
it validates this bounded v6 slice and does not claim broader format support.
