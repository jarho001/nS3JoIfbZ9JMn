# Shared restricted singles engine

The approved first G3 package uses [one manifest](../rules/shared-engine-v1.json),
[one JAX state/scheduler](../simulator/shared_engine.py), and a
[host player adapter](../simulator/shared_engine_io.py). Legacy APIs remain intact.
This is a custom 1–3-member singles game; G3 GPU closure and full-format G4
coverage remain separate acceptance work.

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
were explicitly not run in that implementation task. Parity and performance
acceptance remain pending; existing timing tables describe the full-output loop.

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
