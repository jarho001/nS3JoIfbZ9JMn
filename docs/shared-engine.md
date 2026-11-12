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

The current runtime uses [schema 9](../rules/shared-engine-v9.json). Schemas
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
