# Restricted seeded JAX Tackle duel

**Status:** Task034 and Task035 were accepted after primary review. Task036's separate speed-tie v2 is accepted after primary review. This restricted slice does not constitute a complete battle engine or close G1, G3 or any implementation milestone.

## Variant and initialization boundary

The named variant is `gen5-neutral-tackle-mew-1v1-v1`. It has one active Mew per side, one enabled move (`Tackle`), neutral non-STAB physical damage, positive HP and PP, ordinary level/stat bounds, and distinct effective speeds. Only accuracy, critical, damage, HP clamp, faint, terminal outcome, and next-turn request behavior are modeled.

The guarded oracle at [`reference/neutral-turn-oracle.cjs`](../reference/neutral-turn-oracle.cjs) checks the pinned Showdown checkout and initializes the unmodified `[Gen 5] Random Battle` rule table (`gen5randombattle`) with directly supplied one-Mew teams. It bypasses random team generation; it does not override any format rule. Each set has Synchronize, no item, Hardy nature, perfect IVs, and Tackle only. Ordinary cases use level 50 with either zero or 252 Speed EVs. The first-action-KO case uses level 1 versus level 100 with no Speed EVs. At each turn boundary, the oracle captures `battle.p1.activeRequest` and `battle.p2.activeRequest`, plus each side's own `getRequestData()` projection, then submits `move 1` for both players through actual `Battle.makeChoices`.

The battle constructor receives the pinned sets and seed directly. The supported oracle boundary begins after Showdown initialization has produced the normal move requests and its live RNG state. JAX receives the initialized HP/max HP, PP, level, Attack, Defense, Speed, turn, phase, and four PRNG limbs. JAX does not implement team generation, species stat derivation, request parsing, or full format validation.

## Python API and bounds

[`simulator/neutral_tackle.py`](../simulator/neutral_tackle.py) exposes a `BattleState` `NamedTuple` PyTree, `make_state(...)`, `transition(state, choices)`, and `project_view(before, result, audience)`. State arrays are fixed p1/p2 vectors of `int32`; RNG is four big-endian `uint16` limbs; turn, phase, winner, species, move, and variant IDs are scalar integer leaves. `Transition` contains the new trusted state, a diagnostic status, at most two action-event rows, and six fixed draw slots.

`make_state` checks structure and exact integer types. The compiled transition returns `STATUS_UNSUPPORTED` without advancing for value-level boundary violations. Its declared value bounds are: variant ID 1, species ID 151 (Mew), move ID 33 (Tackle), level `[1,100]`, HP `[1,maxHP]`, max HP `[1,404]`, Attack/Defense/Speed `[1,299]`, PP `[1,56]`, turn `[1,2^31-2]`, and unequal speeds. The HP/stat maxima match level-100 Hardy Mew with legal 31 IVs and 252 EVs; the turn ceiling prevents a continuing transition from overflowing int32. The state boundary does not prove arbitrary HP/stat values came from a legal set; the differential scenarios use pinned initialized Pokemon values.

Both sides choose from the same captured request boundary using a p1/p2 `int32` vector: `1` is the sole legal Tackle action, `0` means a choice is still pending, negative values are malformed, and unknown positive action IDs are unsupported. Pending, invalid, unsupported, and terminal results leave state and RNG unchanged and return fixed-shape empty event/draw data. Terminal input is absorbing. If a surviving side spends its last Tackle PP, the returned phase is unsupported because Showdown would next request Struggle, which is outside this variant.

The following uses default JAX32 arithmetic and does not enable x64:

```python
import jax
import jax.numpy as jnp
from simulator.neutral_tackle import make_state, project_view, transition

state = make_state(
    hp=(175, 175), max_hp=(175, 175), pp=(56, 56),
    level=(50, 50), attack=(120, 120), defense=(120, 120),
    speed=(120, 152), rng=(1, 2, 3, 4),
)
result = jax.jit(transition)(state, jnp.asarray([1, 1], dtype=jnp.int32))
p1_view = jax.jit(project_view, static_argnums=2)(state, result, 0)
```

`jax.vmap(transition)` batches independent p1/p2 lanes. `project_view` accepts only a static audience ID (`0` for p1, `1` for p2); compile it with `static_argnums=2`. Unsupported or dynamic audience values fail closed.

## Bounded duel API (Task035)

[`simulator/neutral_duel.py`](../simulator/neutral_duel.py) exposes `run_duel(initial_state, *, max_turns, turn_budget=None)`. It composes the unchanged `transition` with a fixed-shape `jax.lax.scan` and conditional turn execution. Both sole legal Tackle choices are submitted on each attempted turn. The explicit static trace capacity `max_turns` must be a Python integer from 1 through 56, the accepted maximum PP. Compile with `jax.jit(run_duel, static_argnames=("max_turns",))`; for `vmap`, close over one common static capacity and pass a per-lane scalar `int32` `turn_budget`.

`turn_budget` defaults to the capacity and accepts scalar integers from 0 through that capacity. Zero budget returns the unchanged nonterminal state with `DISPOSITION_BUDGET_EXHAUSTED`, zero turns, and an empty valid prefix. A structurally malformed capacity, state PyTree, or budget type/shape raises before simulation. A representable signed `int32` budget outside `[0, max_turns]` returns `DISPOSITION_INVALID_BUDGET` with `DIAGNOSTIC_INVALID_BUDGET`, unchanged state, zero turns, and an empty prefix in eager, JIT, and vmapped execution. A valid initially terminal state completes with zero turns and `STATUS_TERMINAL`, including at zero budget.

`DuelResult` returns the trusted final `BattleState`, disposition, successful-turn count, final `diagnostic_status`, and fixed-capacity `DuelTrace`. `trace.valid` is the only turn-validity signal and is a contiguous successful prefix. Each valid row contains its before state, accepted `Transition` (after state, events, and draws), and both recipient projections from the existing `project_view`. False rows are zero placeholders; their enum-like fields are not requests or outcomes and must never be consumed. Select only one valid recipient view for a given actor. The pair of player views is trusted combined trace data and is not a joint actor observation.

If the capacity/budget ends while the state is nonterminal, the result is budget-exhausted with no synthetic draw, event, winner, loss, or tie; `diagnostic_status` is the last successful `STATUS_OK`, or `DIAGNOSTIC_NONE` if no transition ran. Completion on the final slot takes precedence over budget exhaustion. Resumption passes the returned state to a new call and concatenates only rows selected by each call's valid prefix. At the existing last-PP boundary, a one-slot run can finish one valid turn and report budget exhaustion at the continuing state; a larger budget attempts the next kernel step, stops as `STATUS_UNSUPPORTED`, and adds no row for that failed attempt. Struggle remains unsupported.

## Damage and random state

Pinned Gen 5 move data resolves Tackle to base power 50 and accuracy 100. The pinned move constructor defaults its critical ratio to 1; Gen 5's ratio table makes that a `1/16` critical chance. Each executed Tackle consumes exactly three live calls in this order:

1. `random(100)` for accuracy, even though Tackle cannot miss in this variant.
2. `random(16)` for critical.
3. `random(16)` for the Gen 5 damage randomizer.

The transition uses the existing arithmetic kernels for the neutral base damage and final non-critical stages. On a critical it doubles neutral pre-roll damage before the explicit random roll. Mew is Psychic, so Normal Tackle has no STAB and neutral type effectiveness. No boosts, status, weather, held item, or relevant passive ability is active.

[`simulator/gen5_rng.py`](../simulator/gen5_rng.py) implements the pinned 64-bit LCG with four big-endian 16-bit limbs:

```text
x' = (0x5D588B656C078965 * x + 0x00269EC3) mod 2^64
```

After advancing, Showdown returns the upper 32 bits. For bounds `[1,65535]`, JAX computes `floor(output32 * bound / 2^32)` by splitting the multiply into 16-bit halves. The state multiply uses base-256 limbs, so each JAX uint32 intermediate stays within range with x64 disabled. The transition uses only the concrete bounds 100 and 16.

The oracle observes `PRNG.prototype.random` after normal request initialization, captures actual bound/context and pre/post four-limb state, delegates to the original implementation, and returns its original result unchanged. It does not replace PRNG values, patch `getDamage`, or replay a random tape. The current seeded fixture includes six seeds across bounds 1, 2, 16, 100, and 65535, including high/low and carry patterns. The turn fixtures include both speed orders, an actual critical, three surviving turns, and a first-action KO.

## Player views

`Transition.state`, `Transition.events`, and `Transition.draws` are trusted simulator data. Actors receive the narrower `PlayerView` returned by `project_view`:

- The request projection has only that side's exact HP/max HP, PP, level, relevant stats, one legal-action mask, and Tackle row.
- Each action projection retains move order, critical occurrence, damage-event HP condition, faint, next turn, and winner.
- The damaged owner sees its exact HP condition. The opponent sees the pinned HP Percentage Mod condition (`ceil(100 * hp / maxHP) / 100`, with a non-full 100 clamped to 99).
- Exact theoretical damage is never exposed to the attacking opponent. The damaged owner receives only observed HP loss, clamped to HP actually removed; overkill damage stays in trusted `Transition.events`.

The oracle normalizes only relevant semantic records from the actual recipient-selected split branch. It does not claim byte-for-byte protocol parity and does not put raw packets or trusted state in a player view. A caller must pass only `PlayerView` to an actor; the transition result remains privileged engineering data.

## Current validation

The current focused checks are:

```sh
node --check reference/neutral-turn-oracle.cjs
node --check test/neutral-turn-oracle.test.cjs
node --test test/neutral-turn-oracle.test.cjs
JAX_PLATFORMS=cpu python -m unittest test.test_neutral_tackle -v
```

The guarded Node oracle tests pass 3/3 and the focused CPU JAX differential tests pass 6/6. They compare live PRNG vectors and every turn's internal HP/PP/order/turn/winner, draw bounds/context/pre/post state, and both semantic player views. A separate handwritten arithmetic check computes nested-truncation Tackle damage, critical doubling, and randomization. The tests also cover invalid/pending/unsupported/terminal no-advance, hidden overkill, eager/JIT parity, and heterogeneous JIT/vmap lanes.

One synchronized CPU compile/warm probe ran `jax.jit(jax.vmap(transition))` for eight heterogeneous battle lanes on Python 3.13.18, JAX/JAXLIB 0.12.0.1, backend `cpu`, device `cpu:0`. The first synchronized call took 0.536254 seconds, including compilation; one warmed synchronized call took 0.000157 seconds. Every lane returned `STATUS_OK`. These are measurements of that exact probe configuration only, not a performance or throughput claim. The measured source was based on `19616d6c0a5f0401ceca28bf2928f0dcc06f22b2`; current source blob IDs are recorded in the Task034 handoff. The exact command, including each seed and initialized state, was:

```sh
JAX_PLATFORMS=cpu python - <<'PY'
import platform
import sys
import time
import jax
import jaxlib
import jax.numpy as jnp
from simulator.neutral_tackle import make_state, transition

assert jax.default_backend() == 'cpu'
seeds = [
    [1, 2, 3, 4], [9, 2, 3, 4], [31, 41, 59, 26], [65535, 0, 65535, 0],
    [0, 65535, 0, 65535], [65535, 65535, 65535, 65535], [4, 3, 2, 1], [101, 202, 303, 404],
]
states = [
    make_state(
        hp=(175 - 3 * i, 175 - 2 * i), max_hp=(175, 175), pp=(56 - i, 56 - i),
        level=(50, 50), attack=(120, 120), defense=(120, 120),
        speed=(120 + (20 if i % 2 == 0 else 0), 120 + (20 if i % 2 else 0)),
        rng=seed,
    )
    for i, seed in enumerate(seeds)
]
batch = jax.tree_util.tree_map(lambda *values: jnp.stack(values), *states)
choices = jnp.ones((8, 2), dtype=jnp.int32)
compiled = jax.jit(jax.vmap(transition))

def synchronize(value):
    return jax.tree_util.tree_map(lambda leaf: leaf.block_until_ready(), value)

start = time.perf_counter()
first = synchronize(compiled(batch, choices))
compile_warmup = time.perf_counter() - start
start = time.perf_counter()
second = synchronize(compiled(batch, choices))
warmed = time.perf_counter() - start
assert first.status.tolist() == [0] * 8
assert (first.state.hp == second.state.hp).all()
assert (first.state.rng == second.state.rng).all()
print('python', sys.version.split()[0], 'jax', jax.__version__, 'jaxlib', jaxlib.__version__)
print('backend', jax.default_backend(), 'device', jax.devices()[0])
print('configuration', 'jit(vmap(transition))', 'batch=8', 'two one-active Mew lanes per battle', 'int32 arrays / uint16 rng limbs')
print('first_synchronized_call_seconds', f'{compile_warmup:.6f}')
print('warmed_synchronized_call_seconds', f'{warmed:.6f}')
print('result_statuses', first.status.tolist())
PY
```

The first synchronized call was 0.536254 seconds and the warmed synchronized call was 0.000157 seconds; all eight statuses were `0`. The code files in this run are identified by their Git blob IDs in the handoff, alongside the source base. GPU is optional and is not an acceptance criterion. Full `gen5randombattle`, random-team generation, switching, speed ties, Struggle, other moves, effects, full request/protocol encoding, PPO, and G1/G3 milestone acceptance remain out of scope for v1.

## Task035 full-duel evidence (accepted)

The guarded `--duels` mode generated four complete actual native Battles from normal move requests: `p2-first-full-duel` seed `[1,2,3,4]` (8 turns), `p1-first-full-duel` seed `[31,41,59,26]` (8), `critical-full-duel` seed `[9,2,3,4]` (8), and `one-action-ko-full-duel` seed `[2718,2818,2845,9045]` (1, p2 wins on its first action). Each has a pinned terminal winner; the surviving cases cover both speed orders and true criticals. The JAX differential compares every row's before/after state, event order, critical/damage, draw bounds/context/value/pre/post limbs, and both recipient-specific `project_view` projections. It also checks independent Tackle damage, exact outcome state, resume concatenation, PP=1 unsupported behavior, terminal/pending/unsupported/invalid-budget states, and mixed-lane JIT/vmap independence.

The original no-argument oracle output is byte-identical before and after adding the explicit mode: 124744 bytes, SHA-256 `d1a219a9483d8d1c0859338089a8d980372e79987c796f155e5c210395828157`. Focused results were: new duel Python module 4/4 before the final three-lane diagnostic-case extension, that changed mixed-lane method 1/1 afterward, new full-duel plus unchanged default Node oracle tests 5/5, and unchanged Task034 JAX tests 6/6. The newly added diagnostic lanes cover initial terminal/zero budget, pending phase, and a runtime budget outside capacity; each stays unchanged with an empty prefix.

One synchronized CPU compile/warm probe measured the four native initial states above with `jax.jit(jax.vmap(run_duel))`, static capacity 8, and per-lane budgets `[8,8,8,8]`. It ran on Python 3.13.18, JAX/JAXLIB 0.12.0.1, backend `cpu`, device `cpu:0`; counts were `[8,8,8,1]`. The first synchronized call took 1.121906 seconds and one warmed synchronized call took 0.000532 seconds. These are measurements for this exact configuration, not a performance or throughput claim. Source Git blobs: `simulator/neutral_duel.py` `81d290fbd60179a559667df9d2ad84463f90dbc8`, unchanged `simulator/neutral_tackle.py` `67b707d8b089fc7e2173d17497cbf1a62b10bf2f`, and `simulator/gen5_rng.py` `ed8c8cddd8f001d2c7da63ff2bd2ef3c4cf060d7`.

The probe can be reproduced with:

```sh
JAX_PLATFORMS=cpu python - <<'PY'
import sys
import time
import jax
import jaxlib
import jax.numpy as jnp
from simulator import make_state, run_duel

assert jax.default_backend() == 'cpu'
rows = [
    ((175, 175), (175, 175), (56, 56), (50, 50), (120, 120), (120, 120), (120, 152), (1, 2, 3, 4)),
    ((175, 175), (175, 175), (56, 56), (50, 50), (120, 120), (120, 120), (152, 120), (31, 41, 59, 26)),
    ((175, 175), (175, 175), (56, 56), (50, 50), (120, 120), (120, 120), (120, 152), (9, 2, 3, 4)),
    ((13, 341), (13, 341), (56, 56), (1, 100), (7, 236), (7, 236), (7, 236), (2718, 2818, 2845, 9045)),
]
states = [make_state(hp=r[0], max_hp=r[1], pp=r[2], level=r[3], attack=r[4], defense=r[5], speed=r[6], rng=r[7]) for r in rows]
batch = jax.tree_util.tree_map(lambda *values: jnp.stack(values), *states)
budgets = jnp.asarray([8, 8, 8, 8], dtype=jnp.int32)
compiled = jax.jit(jax.vmap(lambda state, budget: run_duel(state, max_turns=8, turn_budget=budget)))
def synchronize(value):
    return jax.tree_util.tree_map(lambda leaf: leaf.block_until_ready(), value)
start = time.perf_counter()
first = synchronize(compiled(batch, budgets))
first_seconds = time.perf_counter() - start
start = time.perf_counter()
second = synchronize(compiled(batch, budgets))
warm_seconds = time.perf_counter() - start
assert first.disposition.tolist() == [0, 0, 0, 0]
assert first.turn_count.tolist() == [8, 8, 8, 1]
assert first.trace.valid.sum(axis=1).tolist() == [8, 8, 8, 1]
assert (first.state.hp == second.state.hp).all()
assert (first.state.rng == second.state.rng).all()
print('python', sys.version.split()[0], 'jax', jax.__version__, 'jaxlib', jaxlib.__version__)
print('backend', jax.default_backend(), 'device', jax.devices()[0])
print('configuration', 'jit(vmap(run_duel))', 'lanes=4', 'static capacity=8', 'turn budgets=[8,8,8,8]', 'int32 state / uint16 rng limbs')
print('initial seeds', [[1, 2, 3, 4], [31, 41, 59, 26], [9, 2, 3, 4], [2718, 2818, 2845, 9045]])
print('result turn counts', first.turn_count.tolist())
print('first_synchronized_call_seconds', f'{first_seconds:.6f}')
print('warmed_synchronized_call_seconds', f'{warm_seconds:.6f}')
PY
```

Primary accepted Task035 with count35; this does not require GPU, claim performance, or close G1/G3 or a milestone.

The delivered executor blob `38efee781f8dc711931669eb0e202c90fca4e870` differs from the measured/tested blob `81d290fbd60179a559667df9d2ad84463f90dbc8` only by removal of one blank EOF line. Adding that newline back reproduces the measured hash; executable code is unchanged, and tests/probe were not repeated for this formatting correction.

## Task036 speed-tie variant (accepted)

[`simulator/neutral_tackle_ties.py`](../simulator/neutral_tackle_ties.py) exports the separately named `gen5-neutral-tackle-speed-ties-mew-1v1-v2` variant (ID `2`) through `make_tie_state`, `transition_with_speed_ties`, `project_tie_view`, and `run_tie_duel`. It reuses the accepted `BattleState`, Tackle executor, recipient projection semantics, and bounded duel core while returning a separate `TieDrawTrace`; the v1 API and six-slot trace remain unchanged. Equal speeds take the pinned seeded queue-shuffle path. Unequal speeds use deterministic ordering and consume no scheduler shuffle draw.

The per-turn draw trace has a fixed capacity of 16: six Tackle draws plus up to ten scheduler draws. Native scheduler calls are observed as raw `PRNG.random(0, 2)` ranges and delegated unchanged; the JAX trace stores the effective bound `2`, with context IDs 4–8 for queue sort, BeforeTurn, move-hit Update, action Update, and residual Update. The native observer separately identifies the turn-start Update and action Update using the delegated `runAction` choice and `eachEvent` event metadata. Scheduler draws remain interleaved with the two attacks. A surviving tied turn uses all 16 slots. A first-action KO ends at 7 calls and skips the second action; a second-action KO ends at 13. The native observer records actual phase names, raw ranges, values, and pre/post RNG limbs without replacing random returns.

The `--ties` oracle fixture contains three distinct-seed, seven-turn complete tied battles: seeds `[1,2,3,4]`, `[31,41,59,26]`, and `[9,2,3,4]`. They cover both initial action orders, a surviving turn, a critical hit, and a first-action KO. The Python differential chains actual JAX results and checks every turn state, event/action order, draw bound/context/value/pre/post limbs, recipient-specific request/event projections, tie initiative mapping, and terminal result. Separate CPU tests cover eager/JIT equality; mixed tie/unequal/terminal/pending/unsupported/invalid-budget `jit(vmap)` lanes and individual-lane equivalence; valid-prefix padding, lane permutation, state/RNG freeze, budget resumption, exact-capacity completion, PP=1 stop behavior, and diagnostic no-advance. The unequal-speed v2 control uses the original native `--duels` scenario and matches its full state/draw/recipient trace.

Focused validation on CPU: new tie Python suite 3/3; existing duel suite 4/4; existing v1 Tackle suite 6/6; tie, duel, and unchanged-default Node oracle suites 6/6. Python/Node syntax checks and `git diff --check` passed. Before/after output for both the default oracle and `--duels` is byte-identical to the saved pre-Task036 captures: default SHA-256 `d1a219a9483d8d1c0859338089a8d980372e79987c796f155e5c210395828157`; `--duels` SHA-256 `baeb3b3bd3f1eccf35f59f2d4fa4e9f23f933ec8cefb4fd7f6f60b2ead2eaf51`.

One synchronized CPU compile/warm probe used `jax.jit(jax.vmap(run_tie_duel))`, four lanes, static capacity 8, and per-lane budget 8. Three tied native initial states completed in 7 turns each; the legacy unequal-speed native control completed in 8, producing counts `[7,7,7,8]`. On Python 3.13.18, JAX/JAXLIB 0.12.0.1, CPU backend/device `cpu:0`, compile plus synchronized execution took 3.185057 seconds and the warm call took 0.000741 seconds. This is a measurement of that configuration, not a speed or throughput claim. It used the frozen source blobs recorded in Task036's handoff; no source or test file changed after the probe.

The probe can be reproduced from the native fixtures and frozen API:

```sh
node reference/neutral-turn-oracle.cjs --ties > /tmp/task036-ties.json
node reference/neutral-turn-oracle.cjs --duels > /tmp/task036-duels.json
JAX_PLATFORMS=cpu python - <<'PY'
import json
import time
import jax
import jaxlib
import jax.numpy as jnp
from simulator import make_tie_state, run_tie_duel

def initial(path, scenario_id):
    rows = json.load(open(path))['scenarios']
    raw = next(row for row in rows if row['id'] == scenario_id)['initial']['state']
    return make_tie_state(
        hp=raw['hp'], max_hp=raw['maxHp'], pp=raw['pp'], level=raw['level'],
        attack=raw['attack'], defense=raw['defense'], speed=raw['speed'],
        rng=raw['rng'], turn=raw['turn'],
    )

states = [
    initial('/tmp/task036-ties.json', 'equal-speed-p1-initial-order-and-first-action-ko'),
    initial('/tmp/task036-ties.json', 'equal-speed-p2-initial-order'),
    initial('/tmp/task036-ties.json', 'equal-speed-critical'),
    initial('/tmp/task036-duels.json', 'p2-first-full-duel'),
]
batch = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states)
budgets = jnp.full((4,), 8, dtype=jnp.int32)
compiled = jax.jit(jax.vmap(lambda state, budget: run_tie_duel(state, max_turns=8, turn_budget=budget)))
sync = lambda value: jax.tree_util.tree_map(lambda leaf: leaf.block_until_ready(), value)
start = time.perf_counter(); first = sync(compiled(batch, budgets)); first_s = time.perf_counter() - start
start = time.perf_counter(); second = sync(compiled(batch, budgets)); warm_s = time.perf_counter() - start
assert first.turn_count.tolist() == [7, 7, 7, 8]
assert first.trace.valid.sum(axis=1).tolist() == [7, 7, 7, 8]
print('python', __import__('platform').python_version(), 'jax', jax.__version__, 'jaxlib', jaxlib.__version__)
print('backend', jax.default_backend(), 'device', jax.devices()[0])
print('configuration', 'jit(vmap(run_tie_duel))', 'lanes=4', 'capacity=8', 'budgets=[8,8,8,8]')
print('turn_counts', first.turn_count.tolist())
print('first_synchronized_call_seconds', f'{first_s:.6f}')
print('warmed_synchronized_call_seconds', f'{warm_s:.6f}')
PY
```

No GPU, throughput, full-format, G1/G3, or milestone acceptance claim is made. Primary accepted Task036 at count36; review baseline30/next40 remain unchanged.

## Task037 staged submissions (accepted)

[`simulator/request_session.py`](../simulator/request_session.py) wraps the unchanged v2 `BattleState` and `transition_with_speed_ties` with a fixed-shape `RequestSession`. The trusted session stores two private pending choices in p1/p2 order and an int32 `episode`. `make_request_session` accepts a local episode from 0 through `MAX_REQUEST_EPISODE` (`2^31-1`). That token identifies this adapter's next submission boundary; it is not a native Showdown `rqid`.

Call `submit_choice(session, player, choice, expected_episode)` with integral scalar values. Player IDs are 0 (p1) and 1 (p2); action 1 is the sole legal `move 1` Tackle. Structural shape/type errors raise before simulation. Runtime statuses are `SESSION_WAITING`, `SESSION_RESOLVED`, `SESSION_STALE_EPISODE`, `SESSION_DUPLICATE`, `SESSION_INVALID_PLAYER`, `SESSION_INVALID_ACTION`, `SESSION_TERMINAL`, `SESSION_UNSUPPORTED`, `SESSION_OVERFLOW`, and `SESSION_STOPPED` (a v2 transition rejected an otherwise accepted resolution). For well-formed calls, status precedence is stale episode, invalid player, invalid action, terminal state, unsupported state/session, episode overflow, then duplicate commit. A successful first submission returns `SESSION_WAITING`, records only that player's pending choice, and leaves the battle, events, and RNG unchanged. The other player's valid submission resolves the existing v2 transition once, returns `SESSION_RESOLVED`, clears both pending choices, and increments the local episode. The max int32 episode is an exhausted sentinel: episode `MAX_REQUEST_EPISODE-1` may resolve and advance to it; further submissions return `SESSION_OVERFLOW` without changing the session.

Stale tokens, repeated submissions, invalid player/action values, terminal states, unsupported states, and overflow return explicit `SESSION_*` statuses and preserve the session. Duplicate rejection is a local single-commit policy. Native `Side.choose` can replace a prior choice; replacement/retraction is outside this API and its status behavior is not claimed as native parity. `SessionSubmission` is trusted coordinator data because it includes the full before-state and transition.

Before either player submits, project an initial request with `project_request_view(session, audience)`, where `audience` is a static 0/1 integer (use `jax.jit(project_request_view, static_argnums=1)`). The typed `SessionRequestView` contains that player's `PlayerView`, local episode and own `own_choice_submitted` bit. It reads the unchanged battle state and does not run a transition. It exposes no opposing commitment, combined view, trusted RNG or full battle state.

After a submission, `project_submission_view(result, audience)` keeps its `SessionView` return type and uses the same current-request projection for `player_view`; after resolution it reflects the next request's HP, PP, phase and legal-action mask. `resolved_events` separately contains the prior transition projected through the existing `project_tie_view` rules. The result includes the local episode and only the recipient's own `own_choice_submitted` bit. Terminal and exhausted-PP requests have no legal Tackle action.
