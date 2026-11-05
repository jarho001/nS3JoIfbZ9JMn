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
