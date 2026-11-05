# Task043 two-move neutral Mew turn API

## Scope and status

Implement one separately versioned `gen5-neutral-two-move-mew-1v1-v4` turn
API for two Gen 5 Mew with Tackle in move slot 1 and Pound in slot 2. Both
players submit one simultaneous move choice each turn. Reuse the exact shared
neutral action, speed-tie, RNG, damage, event, and recipient projection
machinery; preserve the accepted v1-v3 typed state layouts and boundaries.

Owner: `/root/task043`. Source base:
`177b35a8b1a4a9c65368c9914149ec3d7e03c639`. Status: accepted by primary
`/root` on 2026-11-05. Accepted task count is 43; review baseline is 40 and
next checkpoint is 50. This does not claim Astra review or milestone
acceptance. GPU evidence is not required.

Owned paths: `simulator/neutral_two_move.py`, `simulator/__init__.py`,
`reference/neutral-two-move-oracle.cjs`, `test/test_neutral_two_move.py`,
`docs/neutral-two-move-variant.md`, this handoff, `docs/TASKS.md`, and
`docs/ORCHESTRATION_STATE.json`. An optional cross-link in
`docs/neutral-pound-variant.md` is allowed. Existing Tackle/Pound engines,
oracles, tests, and dependency configuration are outside the write scope.

## Acceptance criteria

- Provide an independent fixed typed state with variant ID 4, two-slot int32
  PP per player, and fixed ordered move IDs `[33, 1]`; preserve the accepted
  single-move `BattleState` contract. Enforce exact shape/dtype validation,
  Mew and initialized stat/HP bounds, PP in `[0, 56]`, normal move-request
  phase, and fixed move identities. Reject unsupported active states without
  changing state or RNG; preserve terminal absorption.
- Accept simultaneous choices 1/2, with 0 pending, negative invalid, and
  values above 2 unsupported. A selected known move with zero PP is an invalid
  choice. Rejected and pending results have no events or draws and do not
  change state or RNG.
- Execute Tackle at power 50 and Pound at power 40 through the existing
  neutral action and shared tie-phase engine. Decrement only the PP slot of an
  actor whose move actually executed. An actor skipped after a first-action
  knockout does not spend PP. A side that spends its last PP in one slot may
  still select its other move. If either nonterminal side has no PP in either
  slot, mark the next request boundary unsupported; do not invent Struggle or
  an outcome. Preserve the fixed tie draw trace and terminal absorption.
- Return event move identities only for executed rows. Provide a separate
  flat typed player view with that player's two PP values, legal mask, and
  known move IDs; preserve common HP and recipient visibility rules. Mask
  unexecuted event move IDs with the invalid sentinel. Never expose the
  opponent's unexecuted choice, opponent PP, trusted full state, or RNG/draw
  trace through an actor view.
- Add a guarded native oracle using the pinned Gen 5 random battle engine and
  actual one-Mew Synchronize/Hardy/perfect-IV/itemless sets with both real
  moves in the prescribed order. Capture normal request rows, original PRNG
  values and limb traces, makeChoices for the actual four move pairs, and
  recipient-split events with real move identities. Include all four action
  pairs, both unequal initiative directions, a tied mixed pair, an actual
  critical, first-action KO, and a last-Tackle-PP boundary that retains Pound
  availability. Any PP setup must use the actual engine API, regenerate normal
  requests, prove request regeneration consumes no RNG, and disclose setup.
- Focused CPU JIT tests compare native HP/maxHP/all PP/stats/turn/phase/winner,
  exact RNG draw count/bounds/contexts/values/limbs, order, identities,
  critical/faint events, current and next requests, and both recipient views.
  Prove move choices select the correct power and PP slot, the other move
  remains available at the last-PP boundary, and an unexecuted KO actor keeps
  its PP. Cover heterogeneous `jit(vmap(...))` lanes, malformed structure,
  unavailable slots, unsupported moves/variants, masked partial views, and
  all-PP exhaustion without a fabricated result.
- Inspect the complete diff and report exact commands, pass/failure evidence,
  elapsed times, source hashes, changed paths, and limitations to the primary.
  Keep the accepted task count at 42 until independent primary review. Do not
  self-accept, commit, or push.

## Exclusions

No additional moves, damage mechanics, switches/teams, status, items, weather,
Struggle execution, request-session/reset integration, policies/training/
rollouts/tournaments/reporting, duel runner, GPU/performance/long-suite work,
or milestone acceptance. No broad shared framework or duplication of the
existing tie phase engine. The existing `neutral_tackle.py`,
`neutral_tackle_ties.py`, Pound implementation and old oracle files should not
need edits; report concrete evidence to `/root` before any out-of-scope change.

## Plan and progress

1. Inspect the cached repository graph and exact shared action/tie/projection
   seams, pinned native helper exports, and prior Pound oracle/test shape. The
   `trail` CLI and simulator cards are missing in this workspace, so use the
   documented targeted source fallback; do not rebuild/install the toolchain.
2. Implement the independent v4 state/transition/view API and package exports
   using a transient internal boundary with variant 4 and neutral move ID 0.
3. Add the guarded native two-move oracle and focused JIT/differential tests.
4. Update the variant contract and task evidence, run only required affected
   fast checks under 30 seconds per command / 90 seconds total test execution,
   inspect the full diff, and report one evidence package to `/root`.

## Implementation and evidence

Implementation is complete and awaits primary review. The public v4 API uses
`TwoMoveState` with `pp[2,2]` and fixed move IDs `[33, 1]`; a private adapter
uses variant 4, internal move ID 0, and PP placeholders to enter the existing
tie-phase engine. The selected move power is dynamic per actor. Public PP is
decremented only for event rows that actually executed. If either side reaches
zero PP in both slots, the next nonterminal request phase is unsupported.
`TwoMovePlayerView` contains only own PP/moves and executed move IDs, with
unused event rows masked as `-1`.

The native oracle uses the pinned Gen 5 Dex (not the default Dex) and records
actual Tackle power 50 and Pound power 40, each Normal/Physical, accuracy 100,
base PP 35, critical ratio 1 and priority 0. Its seven one-turn cases cover
all four action pairs, p1-first and p2-first unequal speeds, tied mixed
choices, an actual critical, a first-action knockout, and spending the last
Tackle PP while Pound remains legal. The PP setup uses the live Pokemon move
slot and native `battle.makeRequest('move')`; the fixture confirms request
regeneration preserves all four PRNG limbs.

Environment: Python 3.12.14, JAX/JAXLIB 0.12.0.1, CPU backend; pinned Showdown
oracle. No GPU run was required. The exact focused commands and unittest
reported durations were:

```sh
timeout --signal=TERM --kill-after=5s 30s env JAX_PLATFORMS=cpu /tmp/task040-jax-env/bin/python -m unittest test.test_neutral_two_move.NeutralTwoMoveTests.test_native_turns_match_full_state_trace_move_identity_and_both_views -v
```

Passed: 1 test in 17.835s. This compares all seven native before/after states,
all PP slots, requests, event move identities, recipient event views, action
order, exact tie/action draw bounds, contexts, values and PRNG limbs, plus
compiled transitions and projections. An independent Gen 5 damage formula
checks both selected base powers and the critical results.

```sh
timeout --signal=TERM --kill-after=5s 30s env JAX_PLATFORMS=cpu /tmp/task040-jax-env/bin/python -m unittest test.test_neutral_two_move.NeutralTwoMoveTests.test_boundaries_and_masking -v
```

Passed: 1 test in 14.860s. This covers malformed structures, old/new typed
state boundaries, pending and invalid choices, unavailable PP, unsupported
variants/moves/species/PP bounds, terminal absorption, all-PP exhaustion,
masked partial views, and preserved unused PP after a KO.

```sh
timeout --signal=TERM --kill-after=5s 30s env JAX_PLATFORMS=cpu /tmp/task040-jax-env/bin/python -m unittest test.test_neutral_two_move.NeutralTwoMoveTests.test_jitted_vmap_isolates_heterogeneous_lanes -v
```

Passed: 1 test in 11.357s. Seven vmapped lanes cover ordinary and tied
success, pending, unavailable-slot rejection, the last-Tackle-PP boundary,
all-PP exhaustion, and terminal absorption. Every batched lane equals its
scalar transition; rejected lanes keep their complete state and empty traces.

The initial combined boundary/batch command was:

```sh
timeout --signal=TERM --kill-after=5s 30s env JAX_PLATFORMS=cpu /tmp/task040-jax-env/bin/python -m unittest test.test_neutral_two_move.NeutralTwoMoveTests.test_boundaries_masking_and_heterogeneous_vmap_lanes -v
```

It hit the 30s timeout and did not pass. The test methods were then split into
the two narrower commands above; the optional lane-permutation compile was
removed because lane-by-lane scalar equality already verifies isolation. No
implementation code changed between the timeout and those two passing
commands.

Failed attempts are retained in the budget accounting. The first standalone
oracle command was:

```sh
timeout --signal=TERM --kill-after=5s 30s node reference/neutral-two-move-oracle.cjs >/tmp/task043-native-oracle.json
```

It failed in 0.133s because it incorrectly queried the default Dex, which
reported modern Tackle power 40; the oracle was changed to query
`Dex.mod('gen5')` and to assert priority zero. The corrected standalone
`timeout --signal=TERM --kill-after=5s 30s node reference/neutral-two-move-oracle.cjs >/tmp/task043-native-oracle.json`
passed in 0.172s. The first native/JIT test attempt failed in 1.352s because
the fixed one-dimensional move-ID vector was indexed as if it were per-player;
the implementation now indexes it by the selected slot. The next attempt
failed in 2.483s because native ordinary draws omit a `phase` property; the
test now uses the draw context for ordinary actions and phase for speed-sort
draws. The final native/JIT command passed afterward.

The two focused native/JIT failures (1.352s + 2.483s), capped combined
boundary/batch timeout (30s), final three focused tests (17.835s + 14.860s +
11.357s), and review-correction test below total 79.392 unittest-reported
seconds, below the 90s budget. Adding the failed and passed standalone native
oracle commands gives 79.697s of recorded validator runtime. End-to-end shell
elapsed was not captured for the earlier commands; their unittest durations
are reported above and the combined command was capped at 30s.

The following bounded static checks passed before the primary-review source
correction; the correction itself was compiled and exercised by the focused
test below:

```sh
timeout --signal=TERM --kill-after=5s 30s python3 -m py_compile simulator/neutral_two_move.py simulator/__init__.py test/test_neutral_two_move.py
timeout --signal=TERM --kill-after=5s 30s node --check reference/neutral-two-move-oracle.cjs
timeout --signal=TERM --kill-after=5s 30s git diff --check
timeout --signal=TERM --kill-after=5s 30s python3 -m json.tool docs/ORCHESTRATION_STATE.json >/dev/null
```

Each final static command passed; measured tool runtimes were 0.000044s for
`py_compile`, 0.000027s for `node --check`, 0.000031s for `git diff --check`,
and 0.000021s for the JSON parse. The same four checks also passed in an
earlier pretest batch (0.000028s, 0.000025s, 0.000023s, and 0.000018s,
respectively). The initial task-state validation command
`git status --short && git rev-parse HEAD && git rev-parse origin/main && tail -20 docs/TASKS.md && python -m json.tool docs/ORCHESTRATION_STATE.json >/dev/null`
passed in 0.018s. After the final state bookkeeping, `git diff --check` and
the JSON parse passed again in 0.000055s and 0.000024s. The guarded native
fixture ran once in each of six unittest command invocations (three native
test attempts, the capped combined attempt, and the final boundary and vmap
methods); these runs are included in the unittest durations. It also ran
standalone twice: one default-Dex failure and one corrected Gen 5 pass.

Primary-review correction, 2026-11-05: imported `TurnEvents` so public type
introspection resolves the `TwoMoveTransition.events` annotation. Added one
continuation test that spends p1's last Tackle PP while p2 uses Pound, confirms
the first turn is nonterminal with phase unsupported and p1 PP `[0, 0]`, then
checks the next compiled call returns `STATUS_UNSUPPORTED` with unchanged full
state/RNG and empty events/draws. The same method checks
`get_type_hints(TwoMoveTransition)['events'] is TurnEvents`. No earlier test
method was rerun.

```sh
time -p timeout --signal=TERM --kill-after=1s 8s env JAX_PLATFORMS=cpu /tmp/task040-jax-env/bin/python -m unittest test.test_neutral_two_move.NeutralTwoMoveTests.test_all_pp_exhaustion_is_unsupported_on_continuation -v
```

Passed: 1 test in 1.505s; shell `real` time was 1.92s. The correction used one
`jax.jit(transition_two_move)` object for both the accepted turn and its
unsupported continuation. The cumulative recorded validator time is 79.697s,
leaving 10.303s under the 90s cap. No earlier source test was rerun for this
correction.

Final source-base and blob identities (the handoff's own blob is reported
separately to avoid a self-referential hash; TASKS/state hashes are omitted
because acceptance bookkeeping changes those metadata files):

| File | Blob |
|---|---|
| `docs/neutral-two-move-variant.md` | `f2728826d41e85b0c0d4500f4e24f410c307f3c0` |
| `reference/neutral-two-move-oracle.cjs` | `3f8e7759bbbefea5055699b23b8634a7d27e01d1` |
| `simulator/__init__.py` | `6188814abc05806503185d8ebfe18cc3de928bfd` |
| `simulator/neutral_two_move.py` | `013dda42c232b47b2e5bd815b7c57a43d817199f` |
| `test/test_neutral_two_move.py` | `6ba18e596c1a18761f9d42a7787b9e351a13b6ab` |

Source base: `177b35a8b1a4a9c65368c9914149ec3d7e03c639`. No commit or push.

Changed files: `simulator/neutral_two_move.py`, `simulator/__init__.py`,
`reference/neutral-two-move-oracle.cjs`, `test/test_neutral_two_move.py`,
`docs/neutral-two-move-variant.md`, `docs/TASK043.md`, `docs/TASKS.md`, and
`docs/ORCHESTRATION_STATE.json`. The optional Pound cross-link was not needed.
The implementation changes only these assigned paths. Accepted task count is
43; review baseline 40 and next checkpoint 50 remain unchanged. Primary
acceptance by `/root` is recorded above; no Astra review or milestone
acceptance is claimed. No commit or push was made.
