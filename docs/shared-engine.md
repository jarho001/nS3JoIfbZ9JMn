# Shared singles engine

The approved first G3 package introduced [schema 1](../rules/shared-engine-v1.json),
[one JAX state/scheduler](../simulator/shared_engine.py), and a
[host player adapter](../simulator/shared_engine_io.py). Legacy APIs remain intact.
The completed [schema 2](../rules/shared-engine-v2.json) package supports custom
1–6-member singles with Stealth Rock, Spikes and Gen 5 Rapid Spin. [Schema 3](../rules/shared-engine-v3.json) added generic immutable tables,
per-mon IV/EV/nature/ability/types/Hidden Power/shiny/happiness/name/boost storage,
special/type-dependent damage and general set decoding to the same scheduler.
`make_state` retains the original custom-set contract; `make_gen5_state` accepts
general sets and returns `UNSUPPORTED` before draws/events for missing handlers.
The runtime capability masks are incomplete; the full G4 goal remains active.
Full-format G4
coverage remains separate acceptance work. Historical evidence keeps its original
source/schema identity; it does not imply validation of later engine changes.

The current runtime uses [schema 22](../rules/shared-engine-v22.json). Schemas
4–6 integrated major statuses and effect provenance, boosts/weather/screens,
and observed ability/item hooks. Schema7 adds declarative hit counts, per-hit
updates, capped drain/recoil arithmetic, self-destruction and reactive damage.
Rough Skin, Iron Barbs, Rocky Helmet and Liquid Ooze inherit native Gen6
callbacks with immediate faint resolution in Gen5. Resolve those faints before
the next hit update/handler; the resulting double-KO order differs from later
recoil. Life Orb remains once per move. Move-driven thaw occurs only after
BeforeMove succeeds; Gen5 Scald's explicit `thawsTarget:false` leaves a frozen
target frozen. These transitions use the same pure JAX scheduler and masks.

Schema8 appends a fifth volatile record for `mustrecharge`, preserving native
source, global effect order and residual duration. All six declarative recharge
moves share the hit path. Successful hits install recharge before contact damage;
misses, immunity and Protect do not. Recharge requests contain one minimal
`Recharge` row and `trapped: true`; public action slot 0 queues an explicit
internal recharge action rather than underlying move slot 0. Its BeforeMove
handler runs before sleep/freeze/confusion/paralysis, spends no PP and consumes
no status RNG. Pending/undo, replacements and retained queues preserve this
identity. Fainting clears the phase. Accepted schema6/7 continuations pad the
new slot with zeros; fabricated legacy recharge requires replay/provenance.
Rampage, charge moves, other locks and the broader causal counters remain open.

Schema7 appends trace event/draw IDs without changing schema6 state storage or
canonical data IDs, so schema6 continuation is lossless. Earlier schema
migrations retain their explicit provenance requirements; missing advanced
status/field history is not reconstructed by guessing. The current 2048-row
history is known to be insufficient for full G4: a reference-generated legal
switch-only battle reaches 6541 visible rows and the native turn-limit tie.
Capacity/terminal repair, remaining callbacks and fixed conformance campaigns
remain active work. Current CPU evidence and exact source identities belong in
[the progress record](ORCHESTRATION_STATE.json); later device checks belong in
[the GPU handoff](GPU_HANDOFF.md).

Use `save_state`/`load_state` in the host adapter for keyed NPZ serialization
with schema and table hash. Older untagged keyed states require an explicit
`schema=1` or `schema=2` migration. Original stable IDs and observable state are
preserved. The old schema did not record IV/EV data: migration reconstructs its
declared perfect-IV/Hardy domain and infers Speed EVs from stored stats, choosing
zero when low-level rounding makes both permitted spreads identical. Unknown
fields/table hashes fail; positional tuples are not a migration format.

```python
import jax
from simulator.shared_engine import make_state, advance, i32
from simulator.shared_engine_io import player_view

teams = [[{'species': 'Mew', 'moves': ['Tackle', 'U-turn']},
          {'species': 'Snorlax', 'moves': ['Earthquake']}],
         [{'species': 'Gengar', 'moves': ['Toxic', 'U-turn']}]]
state = make_state(teams, seed=[1, 2, 3, 4]).state
# Policy receives only player_view(state, its_side), never the trusted State.
view = player_view(state, 0)
step = jax.jit(advance)
result = step(state, i32([0, 1]))
state = result.state
```

Actions use the existing eleven-slot mapping. `-1` submits nothing; `-2` undoes
an unexecuted choice. A single submitted action is private and replaceable.
Supply optional `int32[generation, episode]` to reject stale submissions.
Forced/wait phases use the same API: submit only for acting sides. The state
preserves the opposing action through a pivot, rather than soliciting it again.
The adapter emits owner-exact HP and native opponent percentages, including the
99% cap for damaged HP. Initial public reveals and all later events are retained;
pending choices, RNG and hidden reserves/items/PP are excluded from player views.
Terminal requests and masks are empty. `winner` is -1 while live, 0 for a tie,
and 1/2 for the winning side; `rewards` supplies the terminal +1/-1/0 contract.

## Optional lean rollout wrapper

`simulator.rollout.rollout` applies a supplied choice plan through the same
`advance` kernel. It is implemented as of 2026-11-07, but its tests and timings
were explicitly not run in that implementation task. The later approved CPU
validation passed all 10 focused rollout/fixture checks on `bda1df1`
(`test-logs/g4-roster-hazards/lean-rollout-cpu.log`). GPU correctness and performance
remain deferred; existing timing tables describe the full-output loop.

```python
from functools import partial
from simulator.rollout import rollout, player_view as rollout_view

# choices is int32[T,2]; one row may contain a private partial commitment.
run = jax.jit(rollout)
summary = run(state, choices)
# Continue an unfinished plan with summary.state; check failed before continuing.
view = rollout_view(summary, 0)  # host projection, outside compiled timing

# Static options select extra output; samples are zero-based plan indices.
debug = jax.jit(partial(rollout, retain=('events', 'draws', 'codes'),
                        sample_steps=(0, 3)))
```

The default result contains final state/code, attempted-row and useful-transition
counts, first-failure information, and terminal/truncated flags. It retains no
per-step arrays by default. The complete causal state, including current history,
is carried forward. A rejected advance leaves that input unchanged; the rollout
keeps the preceding successful prefix, records its first failing plan index, and
freezes that lane. Subsequent rows cannot overwrite the error. Terminal lanes
also stop advancing. A live state at the plan or `max_steps` limit is truncated;
it is not labeled a draw. Independent lanes use `jax.vmap(rollout)`.

`max_steps`, `retain` and `sample_steps` are static options. Optional `tokens`
are int32[T,2] local generation/episode pairs; `enabled` is bool[T] and skips
disabled rows without consuming RNG. Outputs have at most `max_steps` rows.
Available retained fields are `states`, `events`, `draws`, `trace`, `rewards`,
`legal_actions` (after the submission), and `codes`. Their `valid` mask denotes
attempted rows; stopped rows have code `SKIPPED`, empty event/draw traces and
zero reward/mask. State samples have separate `sample_valid` and `sample_steps`
arrays, so callers never interpret unused zero storage as a valid battle.

Raw states/events/draws are trusted diagnostic data, not policy observations.
Use the wrapper's host `player_view` adapter for recipient-safe final and sampled
views. It delegates to the existing projection and exposes neither pending
choices nor RNG nor hidden opposing roster data. No neural encoding or policy
inference has been added. Retaining all `states` and `trace` arrays remains an
explicit debugging option; the scalar transition API is unchanged.

States are fixed-shape NamedTuples of arrays and can be stacked with
`jax.tree.map` and advanced with `jax.jit(jax.vmap(advance))`. Persist all state
leaves, their field names, manifest/schema identity and source revision; restoring
only HP or RNG would lose history, command order, counters and suspended actions.
The lifecycle tests round-trip every field through an `allow_pickle=False` NPZ
at ordinary, private-commit, forced and pivot boundaries. `reset` accepts a
validated initialized replacement only after termination and increments the local
generation. It replaces the old history with the new battle's initial reveals.

The buffers are explicit capacities, with checked atomic rollback. They are not
claims about unlimited battles. On `CAPACITY`, preserve the input state and
resolve capacity outside the kernel; do not retry the same step as though it
completed. Invalid/unsupported/stale submissions similarly return unchanged state
and zero event/draw counts. A driver-imposed request cap is truncation, not a tie.
Host input errors reject malformed shapes/dtypes or undeclared set fields before
execution. Trusted HP/PP/status/counter installations are synthetic test boundaries.

Reproduce independent expectations with:

```sh
node reference/shared-engine-cases.cjs --batch all --output /tmp/g3-cases-new.json
G3_NATIVE_CASES=/tmp/g3-cases-new.json JAX_PLATFORMS=cpu \
  .venv-jax/bin/python -m unittest test.test_shared_engine_cases \
  test.test_shared_engine_battles test.test_shared_engine_lifecycle -v
node --test test/shared-engine.test.cjs
```

Without `G3_NATIVE_CASES`, tests run the guarded native runner themselves. Use a
new output path: the runner refuses to overwrite evidence. `--batch boundaries`,
`--batch battles`, or an exact case ID selects a subset. Native snapshots retain
complete requests, stable IDs/order, private commitments, suspended queues,
HP/PP/status/items/stored stats/speed/counters, raw draw contexts/stacks and both
projected histories. Comparisons ignore only diagnostic stack paths and protocol
`|t:|` timing, and compare all ordered numeric draw arguments/values/pre/post
states and classified contexts. Terminal native historical choice records are
not future work; the supported engine treats terminal states as absorbing.

The behavior catalog maps the required families to cases and tests. Tests never
import JAX into the native runner or derive expected damage from this engine.
Full traces cover eight three-member battles across both seats and multiple
seeds; exhaustion starts explicitly label depleted PP as synthetic. Passing
those scripts does not establish untested full-format equivalence or performance.

## Six-member roster and hazards

Schema 2 retains move IDs 1–11 and appends Stealth Rock (12), Spikes (13)
and Rapid Spin (14). Struggle stays a forced action, never a stored moveslot.
State adds `hazards: int32[2,3]`: Rock present, Spikes layers, first installed
hazard (0 none, 1 Rock, 2 Spikes). Placement order determines entry-damage
order. Rapid Spin clears own Spikes then Rock only after a successful hit;
it uses Gen 5 power 20 and has no Speed boost. Gengar's Levitate avoids Spikes
without an ability event, while all supported species take neutral Rock damage.

Persist schema 2 and all named State fields with snapshots. A schema 1 snapshot
requires an explicit zero-hazard field migration; positional schema 1 tuples
are incompatible. Existing G3 fixture inputs initialize fresh schema 2 states.
Those schema2-specific bounds did not implement Toxic Spikes, ordinary poison,
additional species or other unlisted callbacks. Later schemas extend that domain;
the current masks still reject all remaining unsupported handlers. The fixed event/draw/history buffers retain atomic rejection.

```sh
node reference/shared-engine-cases.cjs --batch g4 --output /tmp/g4-new.json
G4_NATIVE_CASES=/tmp/g4-new.json JAX_PLATFORMS=cpu \
  .venv-jax/bin/python -m unittest test.test_roster_hazards -v
```

`--batch all`, `boundaries` and `battles` preserve the historical G3 catalog;
`--batch g4` selects the new package. Both native catalogs contain full snapshots,
ordered draws and recipient events. The G4 checks add complete six-member games,
hazard placement/removal/entry KOs, mixed batches, saved continuation/reset and
lean/full rollout agreement. These correctness checks are not performance runs.

The trusted native snapshot copies each request at capture time. It omits
`request.update`, the resend notification added by native `Side.emitRequest` on
an unavailable command; the raw notification/error packet remains in each case's
`privatePackets`. The same decision projection omits an empty `disabledSource`
string, which carries no source disclosure; every nonempty source is preserved.
No request move, PP, disabled/trapped disclosure, side data or
action mask is omitted. This prevents later native notifications from mutating
an earlier captured request. It does not accept G1's broader transport protocol.

Schema9 stores ordered attacks received, typed move outcomes, move snapshots,
per-turn/lifetime action counters and side faint references as hidden JAX state.
It integrates rampages, eight charge moves, semi-invulnerability, Power Herb,
Taunt/Encore/Disable/Torment and declarative flinching under explicit capability
masks. Volatile records retain source kind, move, target location and duration.
Advanced pre-schema9 saves cannot reconstruct this causal history: loading them
requires an independently replayed state that matches every recorded old field.
Fresh initial states retain automatic migration. These additions do not close
the reachable-mechanics audit, long-history capacity gap or full-format G4 gates.

The [schema9 CPU checkpoint](../test-logs/g4-cpu-conformance/schema9-checkpoint.json)
records 18 affected checks and exact scalar replay of 526 scenarios and 2895 boundaries.
The full 61-check checkpoint suite and full-format G4 gates remain pending.

## Gen5 item lifecycle and provenance

Schema16 appends `item_provenance[side,mon,2]`: immutable initial item and pending
native item staleness (undefined/internal/external). Trick/Switcheroo take both
items and set the target's new item before the source's, preserving callback
creation order and native messages. Choice acquisition runs the same item Start
hook as entry, removing an old lock even when exchanging two Choice items.
Knock Off uses Gen5 power20 and removes after contact only if the source survives;
removal does not set lastItem or make an item recyclable. Recycle clears consumed
lastItem and runs item Start/Update on restoration. Sticky Hold keeps its distinct
Trick immunity and Knock Off activation behavior. All remain under current item
capability masks; species-locked and unimplemented item hooks stay unsupported.

Old states can infer original item from held/consumed history under their earlier
masks; independently matching replay retains missing-provenance safeguards.
Journal6 binds the appended field, with explicit older layout/replay handling.
Focused28/44 native parity and current persistence passed; broad legacy migration
and execution matrices remain deferred to the consolidated gate.

## Gen5 Pursuit and pre-switch work

Schema15 appends Pursuit volatile slot22 and causal columns21–23 for
`beingCalledBack`, `pursuitActivated` and `skipPursuit`. A before-turn-move queue
entry installs its duration1 callback. Interception removes the source's queued
attack and schedules the same move dispatcher with a Pursuit source annotation;
voluntary switches also schedule finish work. This avoids nested full-engine
calls and preserves native Updates, cancellation and the KO hint. U-turn can be
intercepted before its replacement prompt; phazing bypasses interception.
Pursuit doubles base power before Technician and skips accuracy on a switching
target, retaining ordinary behavior otherwise. Activation/skip flags survive a
canceled switch until replacement, including exact source/target KO handling.

Old state layouts pad missing causal columns and volatile slots; old journals
require independently matching replay of every recorded field. Current pivot
flags and both recipient histories round-trip under journal5. Native capture
includes the three flags for Pursuit cases; historical cases retain their
original projection contract. Focused23/41 parity is not broad G4 acceptance.

## Gen5 phazing

Schema14 adds transient force-switch flags to Trace and resolves drag entry
synchronously after the move, sharing entry callbacks with normal queued switches.
Drag cancels the outgoing actor's queued move, samples the native command roster,
and applies hazards/entry abilities before the action's Update. The flag is set
before contact callbacks, so even lethal Rough Skin or Life Orb can be followed
by a drag. Native terminal histories can contain drag after the win message.
Roar/Whirlwind bypass Substitute and fail without reserves; Dragon Tail/Circle
Throw still damage a doll or a target without reserves. Suction Cups and
Soundproof preserve their observed stage and messages; type immunity precedes
Soundproof. This is focused36/46 evidence, not full callback or G4 closure.

## Gen5 partial trapping

Schema13 appends volatile slot21 for binding moves, preserving source identity,
move, creation order, duration in column5 and captured damage divisor in column7.
Ordinary counters sample5/6 before the first residual decrement; Grip Claw uses8
without a draw. Residual order13 deals maxHP/16, or maxHP/8 for Binding Band,
until expiration. Source departure/faint/new-entry cleanup emits the native
silent end. Reapplication does not refresh the timer. Substitute creation clears
binding silently; an existing doll prevents binding installation. Rapid Spin
clears binding on both actual hits and AfterSubDamage, and U-turn can escape.
New-turn requests disclose the hard trap and reject switching without battle RNG.

The focused27-case/105-boundary CPU check includes all seven moves, tied timers,
item variants, mirrored private retries and save/restore. State schemas9–12 pad
missing slots with zero; journal schema4 requires matching replay for older
journals and validates each recorded shape/dtype/value. This is a reduced
implementation checkpoint; broad execution modes and full G4 acceptance remain
pending. Exact evidence is in
[schema13-checkpoint.json](../test-logs/g4-cpu-conformance/schema13-checkpoint.json).

## Gen5 Substitute interception

Schema12 appends volatile slot20 with bounded remaining doll HP in column7;
other slots and the shared full/core transition kernel stay stable. Creation
requires HP strictly above one quarter of maxHP and spends its floored quarter
through native direct damage, preserving `hurtThisTurn` and `lastDamage`.
Already-present and weak failures retain their distinct messages and move result.

Primary interception respects the pinned `bypasssub` flag. Gen5 sound moves and
Struggle still hit the doll. Every absorbed hit computes ordinary damage/draws,
caps it to doll HP, updates the attacker's `lastDamage`, and emits activation or
end without damaging the target or appending its `attackedBy` record. A later
hit of the same multi-hit move can hit the real target after the doll breaks.
Recoil and ceil drain run per absorbed hit; contact and secondary application
are skipped, while secondary RNG rolls still occur. Native AfterSubDamage
installs recharge/locked-move effects and lets Rapid Spin clear hazards.
Blocked status moves retain the native outer move result and hit-loop updates;
Taunt/Encore/Disable/Torment use their observed bypass behavior.

Legacy state schemas9–11 extend their twenty volatile slots with zero; matching
independent replay retains the older provenance rules. Journal schema3 binds the
expanded core shape and complete recipient pages; older journal schemas1/2
require matching independent replay, including every recorded shape/dtype/value.
Transform/Illusion, partial trapping, called moves and complete callback closure
remain separate pending work. Directed cases do not prove generator membership.

## G4 architecture decisions — 2026-11-12

This is the design pass for the remaining CPU goal on engine `567f02c` /
record `e42ad98`. It does not enable mechanics or accept G4. The user selected
High for this pass; the repository's Medium default remains unchanged.
Independent source and native evidence are frozen in
[g4-design-frontier.json.gz](../test-logs/g4-cpu-conformance/g4-design-frontier.json.gz)
and [g4-design-native.json.gz](../test-logs/g4-cpu-conformance/g4-design-native.json.gz).
Reproduce them with `reference/gen5-closure-audit.cjs OUTPUT.json` and
`reference/g4-design-cases.cjs OUTPUT.json`, run through Node. Both refuse to
overwrite an existing output. `test/g4-design-native.test.cjs` checks reproduction.

### Reachability and exclusions

Use a typed dependency worklist. Keep four independent facts per entry:
identity candidate/reachability, callback execution or query precondition,
implementation coverage, and independent validation. A partial implementation
mapping never proves reachability, and candidate absence never proves exclusion.
The existing ledger remains open; its callback totals are a conservative universe,
not a percentage of the game completed.

The audited initial bounds are 223 move IDs, 127 ability IDs and 45 possible item
IDs from generator return expressions. Generator eligibility remains a separate
proof obligation. Add implicit Struggle. Nature Power calls Earthquake; Sleep
Talk selects eligible current move slots, including virtual slots after Transform.
Transform and Imposter copy existing opposing move IDs rather than draw arbitrary
Dex moves. Pursuit reuses its queued move, and Magic Bounce reuses the incoming
move with `hasBounced` preventing repeated reflection. Do not expand to every
Dex move merely because these mechanisms exist.

Metronome, Assist, Sketch, Copycat, Mimic, Mirror Move, Me First and Instruct are
absent from the initial move bound. This is a checked initial fact, **not yet a
transitive exclusion**. The expanded audit enumerates mutations/calls in root
moves, abilities, items, discovered conditions and exported rules. Resolve every
computed argument and helper/engine edge before promoting absence to exclusion.
The previous narrower scan missed obligations such as Color Change, Cursed Body,
Bug Bite/Pluck and Knock Off. Literal method scanning alone cannot close the graph.

Track item lineage separately from current ownership: Trick/Switcheroo transfer
existing items; consumption/removal, Bug Bite/Pluck and Harvest need original
item identity, last item and berry-use provenance. Do not infer that an item
mentioned in an immunity check can be created. Any Leppa/restoration or endlessly
recycled-item exclusion requires this complete lineage proof, not sampled teams.
Forms need explicit species guards and transition outputs: Forecast/Castform,
Flower Gift/Cherrim, Relic Song/Meloetta, Multitype/type and initial forme helpers.
Transform/Illusion expose current/apparent species without granting arbitrary
forme abilities. Keep dynamic form edges open until their guards are audited.

**Engine queries are separate roots.** `Battle.nextTurn` invokes
`FoeMaybeTrapPokemon` on alternate abilities of a foe's apparent species, subject
to its rule/release checks. The audit finds Arena Trap, Magnet Pull and Shadow Tag
queries. These callbacks can affect disclosure even when the actual ability is
Frisk or another non-trapping ability. Querying them does not make those ability
IDs transferable by Trace. Use current/apparent species for this lookup, preserve
unknown-type conditions and Gen5 rules; Ghost immunity to trapping must not be
imported from later generations. The native Frisk/Gothitelle and Ghost/Shadow Tag
witnesses establish these distinctions on constructed sets, not generator membership.

Medium implementation should discharge one worklist edge with its source hash,
argument bound/precondition and native witness at a time. An unclassified identity
creator or recursive call edge is the reason to return to High. Do not require
High for routine callback implementation once those edges are classified.

### History, capacity and terminal behavior

**Separate mechanical state from the lossless event journal before increasing
full-battle history arrays.** Keep one transition kernel and one numeric event
format. A new core advance returns the mechanical/request state and bounded
trace; the existing full-history API becomes a compatibility adapter over that
same kernel. No second mechanics implementation or altered policy observations.
Version this state/API migration explicitly; historical schema9 evidence stays
attached to its original source.

Keep immutable journal pages outside the scan carry. Use the already approved
16-boundary conformance chunk, a page budget of `16 * E` event rows per lane,
and flush initialization separately; `E` is the audited per-advance event bound.
The current `E=128` is provisional. Pages may remain on device until the chunk
boundary; no host callback is required inside JIT. A chunk boundary is not a
battle terminal. GPU policies consume recipient-safe requests/events inside the
chunk; journal persistence and CPU differential comparison happen between chunks.
The compatibility adapter must reconstruct the complete player history, including
after save/restore. Full and lean rollout modes use the same core and differ only
in explicitly retained outputs. Default lean carry must not contain whole-game
history or allocate all intermediate core states.

The authoritative checkpoint is `(core, private request/disclosure state, journal
pages/cursor, generation/episode)`. Commit all outputs transactionally; failure
must not append half a trace. Never evict a page silently or substitute a digest
for required history. Digests validate resumed pages but do not replace them.
Persist enough event-time identity/display information for Transform and Illusion;
rendering old events against the final transformed species can corrupt history.
A legacy schema9 save is valid input to the migration only with its complete
recorded prefix; absent pages or causal state require replay/provenance.

This avoids guessing a whole-battle event constant from the 6541-row generated
witness (or the 6041-row neutral witness). Do not enlarge 2048 to an arbitrary
power of two and call the capacity audit done. The 4096 campaign boundary cap
bounds a test run, not native game semantics. Audit queue, draw, trace, call-frame,
volatile and attack-record capacities separately. Retain atomic guarded failures
until each accepted bound has a source proof; a required reachable overflow still
fails G4. The existing 1024 attack records are not an accepted final bound.
Do not replace ordered `attackedBy` with only the latest hit without proving all
consumers, including Assurance/Avalanche/Revenge, Counter/Mirror Coat and abilities.

Use an explicit bounded active-move frame stack, not Python recursion or another
queued turn, for called/reflected moves. Frame fields include source effect,
actor/target, original chosen target location, active-move mutation flags,
PP/action-accounting mode, and return phase. Sleep Talk -> Taunt -> Magic Bounce reflection
requires three frames including the outer move. Nature Power -> Earthquake is
a separate two-frame path: Gen5 Nature Power has `nosleeptalk`, so Sleep Talk
cannot select it. The called/reflected native witness also checks PP accounting.
Three is a witnessed structural requirement,
not a final bound until all call edges and interrupted Pursuit paths are closed.
Charge locks/rampages reuse the ordinary scheduler. Never charge called-move PP
or increment root action counters by applying `runMove` rules indiscriminately
to `useMove`; pinned `BattleActions` distinguishes those paths.

Implement native turn-limit behavior at the next-turn boundary: increment/reset
causal fields, evaluate clauses, then emit a turn/request only if still live.
Warnings occur at 500/600/700/800, every ten turns from 900, and every turn from
990 through 1000. At turn 1001 emit the native message and tie before another
turn event/request. Preserve exact ordering and recipient histories from the
existing witnesses. Earlier Endless Battle Clause outcomes require staleness and
item/PP-restoration provenance; implementing the hard turn limit alone does not
close terminal rules.

### Mutable identity and retry disclosure

Store immutable original set/base identity separately from mutable battle
identity: base species/stats/ability/move slots versus effective species, non-HP
stats, types, ability and virtual move slots. Keep stable team indices for source
references across switches, forms, Transform and Illusion. Base identity is not
necessarily player-visible identity. Current single arrays cannot safely serve
all three purposes.

The native Transform witness keeps Mew's HP/max HP, level and Gen5 Hidden Power
identity, copies Snorlax's non-HP stats/ability/moves, installs virtual PP capped
at five, and retains original `baseAbility`. Implement native failure guards,
boost and `timesAttacked` copying, virtual move metadata, reset on switch and
ability End/Start ordering. Add capture fields for original/virtual state,
transformed and illusion identity before differential acceptance. Item/form-driven
base changes must follow native permanence flags; a generic switch reset must
not undo permanent changes. Save/restore must retain both identities.

**A native unavailable choice can change private disclosure without advancing
the battle.** In the frozen hidden-Shadow-Tag witness, `switch 2` is rejected,
RNG/draws stay unchanged and the mechanical `trapped` value remains `hidden`,
but the player's request changes from `maybeTrapped` to `trapped` and a private
error/resend is emitted. This is observable state, not removable transport noise.

Add an explicit retry/disclosure result (for example `RETRY`) distinct from
syntactic `INVALID`, stale tokens, unsupported mechanics and capacity failures.
Keep existing no-change guarantees for those errors. For a native retry, preserve
mechanics and RNG while updating only the affected private request/disclosure
and native commitment state; retain the opponent's accepted commitment exactly.
Specify token advancement and reset behavior before exposing the API: advance
the episode token when retry disclosure or rejected-command commitment clearing
changes private state, so old request tokens cannot apply to a new disclosure
revision. Ordinary pending submissions retain the G3 request episode contract. This is adapter bookkeeping, not a
native battle RNG/action increment. Use ordered per-side submissions for retry
fixtures, never an ambiguous mixed invalid/valid atomic batch. Existing commands
remain compatible; add the result handling to rollout/policy adapters explicitly.

Trapping state needs source/escape conditions, effective hidden trap truth,
`maybeTrapped`/`maybeLocked`/`maybeDisabled` and recipient disclosure state. Do not
compute all legal switch availability from publicly disclosed `request.trapped`.
Keep the distinction between selectable candidates, privately rejected commands
and actions that actually resolve. Return to native cases for switch/undo/retry,
Pursuit, pivot and forced replacement interactions; do not reveal an opponent's
ability simply to make an action mask exact.

### Implementation order and acceptance

1. Implement the shared core/journal adapter and native turn-limit path; replay
   both long witnesses to genuine terminal with exact RNG, requests and histories.
   Check chunk continuation, save/restore and atomic page failures on CPU.
2. Add private retry/disclosure state and results; cover Frisk query, actual hidden
   traps, Gen5 Ghost, rejected switch followed by a valid choice, undo and an
   already-pending opponent. Preserve existing INVALID/stale/overflow guarantees.
3. Integrate Substitute/per-hit interception, trapping/phazing and interrupted
   Pursuit/pivots. Extend native branch/interaction witnesses before implementation.
4. Introduce base/effective/virtual identity, ability lifecycle and item provenance;
   implement Transform/Imposter/Illusion, forms, transfer/consumption/restoration.
5. Execute called/reflected moves with the shared frame stack. Close identity,
   callback/query and engine/rule worklists; prove each final capacity or keep it open.
6. Freeze the candidate and run the existing full directed, eager/JIT/1/8/32,
   lifecycle, 64-development and 128-held-out gates unchanged. Do not open held-out
   starts to develop these designs. Later GPU correctness remains in GPU_HANDOFF.

These decisions are ready for Medium implementation. Unknown native branches
remain evidence obligations, not architecture debates to settle by approximation.
No throughput claim or profiling run is needed for this pass.

## Schema10 core/journal implementation checkpoint

`CoreState` contains the same mechanical fields as the compatibility `State`,
without `history` or `hcount`. Both call the same `advance` transition code;
`advance_core` enforces the core input type. Legacy states retain their2048-row
atomic capacity guard. `simulator.journal.Checkpoint` pairs a core with immutable
raw and recipient-rendered event pages. The host adapter renders each boundary
at event time, preserves the complete prefix, and returns the original checkpoint
on invalid/stale/unsupported/capacity/terminal errors. `journal.save/load` binds
all core fields and journal metadata with a checksum and table/schema identity.
Use this adapter to persist or display core histories; the legacy IO functions
explicitly reject bare core inputs that lack a journal.

Schema10 preserves the schema9 state field layout and accepts schema9 saves
without causal reconstruction. Core/journal persistence has its own version1.
The hard turn-limit path emits native warnings and the turn1001 tie before a new
turn/request. Earlier Endless Battle Clause conditions remain unsupported work.
The external journal removes the full-game history allocation from the scan carry;
per-advance traces, draws, queue and ordered-attack bounds remain independently
guarded and unaudited for full-format acceptance.

The [first CPU integrated gate](../test-logs/g4-cpu-conformance/schema10-checkpoint.json) passed three checks, including the1000-boundary native replay. GPU correctness remains deferred.
Device chunk collection, journal-level reset and private retry/disclosure are next;
this checkpoint does not claim the entire architecture or G4 complete.

## Schema11 trapping/private retry checkpoint

The candidate adds Arena Trap, Magnet Pull, Shadow Tag and Gen5 Frisk. Trapping
truth stays distinct from species-based possible-trap queries and the request's
disclosed `trapped` bit. The current capability masks exclude Transform/Illusion
and type-changing callbacks; known/apparent-type query handling must be extended
before enabling them. Ghosts do not gain later-generation trapping immunity.
Frisk retains native single-target sampling, including its draw when no item is
announced.

`request_state:int32[2,2]` stores trap disclosure and the cannot-undo flag for an
accepted speculative switch. `RETRY=7` returns recipient-only numeric feedback in
`Trace.private_errors`; `shared_engine_io.private_packets` renders that side's
native error/resend. Public event history receives no private error packets.
A first hidden-trap rejection changes disclosure, clears only that side's pending
choice and advances the local episode token. A repeated known rejection or a
blocked undo/replace may have no state change. Neither draws battle RNG nor
counts a resolved transition. A successful speculative switch becomes committed
until resolution; undo and replacement are rejected exactly as native. Forced
replacement and pivots can escape ability trapping.

Retries isolate one ordered side submission. Ambiguous combined invalid/valid
vectors remain atomic INVALID; the future generated campaign recorder must
split per-side submissions and retain intermediate receipts. Do not remove
hidden-trap switch candidates from policy inputs or use true opposing abilities
to avoid retries. Syntactic invalid/stale/unsupported/capacity errors retain their
no-change guarantees. Validly formed hard-lock switch rejections now return RETRY
and private feedback; their mechanics, public trace and RNG remain unchanged.

Rollout continues through RETRY, counting attempts but no resolved transition.
Journal checkpoints persist the private state in core schema2 and use atomic
file replacement. Old core-schema1 checkpoints require an independently matching
replay with identical recorded fields and complete pages. State-schema9/10 saves
involving trap/query species or new abilities likewise need replay/provenance;
trap-free legacy saves use zero disclosure. The old schema10 core checkpoint and
its evidence retain their original source identity.

The new native corpus has31 scenarios/215 boundaries and two complete six-member
battles. Native `choose()` can flush ordinary private requests after500 log rows;
those request JSON objects are compared to canonical boundary requests. Exact
retry error/resend packets are compared separately. Raw packets remain captured;
BattleStream delivery timing is G1 work, not an ignored mechanical field.
Scalar parity and eight affected CPU checks passed; see the [schema11 checkpoint](../test-logs/g4-cpu-conformance/schema11-checkpoint.json).
The full68-check suite, full-format G4 and GPU acceptance remain pending.

Schema17 separates base/effective identity for Transform and Imposter. Original
requests/stats and original moves/PP survive virtual replacement; copied ability
Start and cached effect identity follow native lifecycle timing. Known/apparent
types preserve speculative private trap queries. Switch/faint restore the base
identity. Focused17/36 CPU parity and live persistence pass; Illusion/event-time
identity, forms, calls, broad regressions and full G4 acceptance remain pending.

Schema18 adds Illusion and captures display names plus switch details in six
additional event columns at emission time. A damaging-hit reveal cannot rewrite
earlier moves, damage or recipient history. Disguise references survive switching
out, are reselected on entry, and clear on damaging-hit reveal/faint. The pinned
format includes Illusion Level Mod. Private requests retain true identities.
Journal8 binds the new field and16-column pages; older journal promotion requires
independent matching replay. Focused12/25 CPU parity and live persistence pass;
broad migration/execution gates remain pending.

Schema19 adds weather-driven Castform/Cherrim forms and Flower Gift stat hooks.
Effective species/types/stored and request stats change while base identity and
original switch details persist. WeatherChange callbacks follow the existing
native speed shuffle; Transform suppresses these form changes, and switching
restores the base species. Focused16/43 CPU parity and live persistence pass.
Other forms, broad execution/migration gates and full-format acceptance remain open.

Schema20 adds Relic Song and Zen Mode through the same effective-form operation.
Successful Relic Song toggles Meloetta before held-item AfterMove callbacks. Zen
Mode runs in residual order29, persists an ability-source volatile, reverses on
recovery and emits the native silent form restoration on ability End at switching.
Volatile slot23 and journal9 preserve its source/order and live continuation.
Focused17/30 CPU parity passes; broader regressions and full G4 remain pending.

Schema21 implements Role Play, Skill Swap and Entrainment over separate base and
effective abilities. Start/End ordering, Illusion reveal, private original ability
and native external volatile staleness persist through switching/fainting and
saved continuation. Skill Swap starts the recipient of the source ability first;
acquiring Imposter does not invoke its SwitchIn callback. Move protection respects
the resolved move flag. Journal10 binds ability provenance. Focused29/58 CPU parity
passes; suppression/ability creation and broad/full-format gates remain pending.

Schema22 adds six status-immunity abilities and acquisition-time Update cures.
The shared Update scheduler visits holders in native speed/shuffle order. Rest
respects Insomnia/Vital Spirit; primary and secondary prevention retain exact
visibility and RNG. Focused18/39 CPU parity and live persistence pass. Yawn/Own
Tempo, broad execution/migration and full G4 conformance remain pending.
