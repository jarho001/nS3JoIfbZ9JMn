# Restricted neutral Mega Punch Mew variant v5

`simulator.neutral_mega_punch` adds a separately versioned pure-JAX normal
turn for two directly initialized Mew. The only legal action is Mega Punch.
This is a restricted development slice with no full-format, random-team
generator, move-pool, speed-tie, session, reset, or Struggle support.

The public API is `make_mega_punch_state`, `transition_mega_punch`, and
`project_mega_punch_view`. The fixed `BattleState` layout carries variant ID 5
and move ID 5. Each side must have positive PP at or below 32, positive HP and
stats within the existing Mew bounds, and a distinct positive Speed. Choices
are `[p1, p2]`: `1` submits request `move 1`, `0` means pending, negatives are
invalid, and unknown positive values are unsupported. Earlier variant IDs,
speed ties, malformed state layouts, and unsupported states fail closed.

The pinned Gen 5 Dex defines Mega Punch as Normal Physical, power 80, accuracy
85, base PP 20, and critical ratio 1/16; three PP Ups raise the maximum to 32.
The native oracle confirms those facts and builds unmodified
`gen5randombattle` battles from direct one-Mew sets with Synchronize, Hardy,
perfect IVs, no item, and no team-generation call. No status, item, weather,
boost, or relevant passive effect is in the supported matchup. Normal damage
against Psychic Mew is neutral and receives no STAB.

The faster Mew acts first. Each attempted move spends one PP and consumes the
native `random(100)` accuracy call. Rolls 85–99 miss: a miss logs its move and
target, consumes no critical or damage draw, and leaves HP unchanged. If the
target survives, the other Mew still acts. A hit consumes `random(16)` for the
critical check and then `random(16)` for damage, using the accepted integer
damage helpers. A first-action KO skips the other actor's PP and draws. A
nonterminal action that spends the final PP returns phase unsupported so this
slice does not claim the later Struggle request.

`MegaPunchTransition` carries at most two attempted-action event rows and six
fixed draw slots. `draws.count` marks the packed native-order prefix; a miss
therefore leaves no empty critical or damage slots. The trace records every
bound, value, context, and four-limb before/after RNG state. A rejection leaves
the full input state unchanged and returns empty event and draw traces.

`MegaPunchPlayerView` wraps the existing recipient-safe `PlayerView` and an
explicit two-row `missed` mask. It includes own exact HP, PP and stats, legal
action state, public action order and miss targets, recipient-visible damage
and HP conditions, critical/faint information, and turn/outcome. For a miss it
retains the move attempt, actor order, and target while masking damage, target
HP, critical, and faint fields. It contains no trusted state, RNG, opponent PP,
or concealed exact HP or damage. The local native oracle includes the original
split-selected `-miss` actor and target when comparing both recipient views.

`reference/neutral-mega-punch-oracle.cjs` checks the pinned Showdown commit,
normal live requests, exact native RNG draws, state and event order, both
recipient projections, and both initiative directions. Its last-PP fixture
changes a live Mega Punch move slot to one PP and calls `battle.makeRequest`;
the native continuation request is explicitly recorded as Struggle while the
JAX v5 transition boundary remains unsupported. The focused CPU test compares
native success lanes and a heterogeneous compiled `jit(vmap(...))` batch,
including miss, hit, critical, KO, pending, invalid, unsupported, and terminal
cases. This coverage makes no full-format or training claim.

Task049 adds an opt-in `--duels` mode to the same oracle; invoking it without
arguments retains the original one-turn fixture semantics. The bounded duel
fixture contains six actual native battles, 28 turns, 53 attempted moves, 48
hits, five misses, five criticals, 43 noncritical hits, three first-action KO
skips, and three terminal second-action outcomes. It covers both faster
players, misses by both actors, and one miss from the second acting slot. Three
battles contain misses, one contains multiple misses across turns, and no turn
contains two misses. All six begin with ordinary 32 PP and end through native
battle resolution within the 32-turn cap.

The separate focused duel test threads each compiled transition's actual
returned state into the next native turn and compares the complete transition
PyTree, draw prefix and unused slots, both recipient event views, current and
next legal requests, and terminal state. Each finished state also rejects a
post-terminal submission without changing state or RNG. The coverage is still
limited to this direct-Mew Mega Punch variant and does not imply other move,
format, session, Struggle, or milestone support.

Task048 was accepted by primary `/root` on 2026-11-05 after focused review of
the implementation, native oracle, tests, and documentation. The acceptance is
limited to this declared v5 slice; G0 remains the only accepted project
milestone.

Task049's complete-battle parity deliverable was accepted by primary `/root` on
2026-11-05. The accepted project-task count is 49; this does not expand the
variant beyond its declared Mega Punch-only boundary or accept a later
milestone.
