# Task050: restricted Tackle/Mega Punch accuracy-choice turn

Status: **accepted by `/root` on 2026-11-05** after independent Astra T050Global acceptance. Owner: `/root/task050_resume` (Luna/xhigh).
Base: `e5f4b21e52da78d7bcef37a1b4551808adec0b50`. Accepted task count is 50;
the regular review baseline is 50 and the next checkpoint is 60. G0 remains
the only accepted milestone. See the [T050Global review report](reviews/T050-global.md).

Ownership handover: on 2026-11-05, `/root/task050_resume` took over from
`/root/task050` after the prior owner stopped following a usage-limit
failure and left the live agent tree. The prior owner is stopped; the
deliverable, owned paths, criteria, exclusions, and source base are
unchanged.

## Deliverable

Add the separately versioned `gen5-accuracy-choice-mew-1v1-v6` API in
`simulator/accuracy_choice.py` for an ordinary simultaneous choice turn by two
directly initialized Mew. Reuse the fixed `TwoMoveState` layout from v4 with
variant ID 6, fixed move IDs `[33, 5]` (Tackle, Mega Punch), per-slot PP, and
the accepted v5 accuracy/miss and turn machinery. This is layout reuse only;
v6 does not claim v4 semantic compatibility.

The fixed actions are `1 = Tackle` (power 50, accuracy 100, maximum PP 56) and
`2 = Mega Punch` (power 80, accuracy 85, maximum PP 32). Each side has a real
per-slot PP vector. Only distinct positive speeds are supported; speed ties
are unsupported. Both sides are actual direct initialized Mew with
Synchronize, Hardy, perfect IVs, no item, in unmodified pinned
`gen5randombattle`; bypass random team generation.

The v6 adapter uses `neutral_two_move._adapt_state` to create a transient
`BattleState`, then invokes generalized v5 `_execute_action` and `_run_turn`
with the selected move's power and accuracy. Internal dummy PP and final
request phase are private. Public PP is deducted only for each executed actor's
selected slot, including misses, using executed event actors and move IDs.
First-action knockout skips the other actor's PP, event, and draws. Public
nonterminal phase reflects whether each side still has at least one usable
move. Do not introduce a second numerical battle implementation.

`AccuracyChoiceTransition` contains the public `TwoMoveState`, status,
`MegaPunchEvents`, packed `MegaPunchDrawTrace` of at most six draws, and
`int32[2] event_move_ids` (`-1` for unused rows). Attempted move IDs remain
present for misses. Invalid, pending, unsupported, and terminal submissions
preserve public state and RNG and return empty traces. Structural validation
precedes runtime status dispatch; runtime precedence follows the v4 dispatcher,
including terminal absorption. Reject wrong variant/move IDs/PP caps/ties;
allow an exhausted slot while another is usable; reject all-PP-zero
continuation without Struggle.

`AccuracyChoicePlayerView` wraps the existing recipient-safe `TwoMovePlayerView`
and exposes a two-row miss mask. Derive the common projection from the
transient state and reusable v5 miss masking. Preserve the miss actor, target,
order, and attempted move ID while masking hit-only damage, target HP, critical,
and faint fields. Expose only recipient own PP/IDs/legal mask and public event
information; never expose full state, RNG, opponent PP, or an unexecuted
opponent choice. Validate type hints, malformed shapes/dtypes, and static
audience fail-closed. Old APIs must fail closed on v6 states.

## Scope exclusions

No status, items, weather, boosts, other moves, Struggle execution, sessions,
reset, runner, training, GPU claim, full-format support, or milestone support.
No edits to v4 engine/session code or shared neutral-turn code. In
`simulator/neutral_mega_punch.py`, only add optional `base_power` and `accuracy`
parameters to `_execute_action`, optional `action_fn` to `_run_turn`, and if
needed extract the existing seven-line miss mask into one private reusable
helper. Existing v5 defaults and public behavior must remain identical.

## Owned paths

- `simulator/accuracy_choice.py`
- `simulator/neutral_mega_punch.py` (minimal internal generalization only)
- `simulator/__init__.py`
- `reference/accuracy-choice-oracle.cjs`
- `reference/neutral-mega-punch-oracle.cjs` (only optionally export the
  unchanged generic event projector)
- `test/test_accuracy_choice.py`
- `docs/accuracy-choice-variant.md`
- `docs/TASK050.md`, `docs/TASKS.md`, `docs/ORCHESTRATION_STATE.json`

## Acceptance evidence

The native oracle uses the guarded final exports `assertReady`, `baseSnapshot`,
and `withObservedBattleDraws` from the pinned Mega Punch oracle at
`1242c1e2d9763ba5b85d33f13aecb89bae19ae3d`; assert the Gen 5 Dex facts, move
IDs, and actual live PP (56/32). Use real two-slot requests and choices,
preserve the original PRNG, and capture before/after state and PP, current and
next requests, and split-selected move/miss/crit/damage/faint/turn/outcome
events. Include the four choice pairs, both faster players, a first Mega Punch
miss (`[0,2,3,4]`, accuracy roll 93) while Tackle hits on the same roll,
mixed-pair orderings, a critical (seed 21), first-action knockout, last Mega
Punch PP and last Tackle PP. At least one actual native follow-up turn after
slot depletion must choose the remaining move and check the state/PP/RNG chain.
Document native Struggle versus this typed unsupported boundary for both PP
slots exhausted; do not claim equivalent request phase there.

The focused CPU test compares complete native state/PP/RNG/turn/phase/winner
parity and exact bounded draw prefixes, context, values, pre/post limbs and
stable unused slots. It checks both recipients' current/next requests and legal
masks, all event/miss move IDs, rejection isolation, zero-PP slot disabling,
miss PP spending, and KO skip behavior. Compare actual numeric results in one
heterogeneous `jit(vmap(transition_accuracy_choice))` batch containing native
cases and pending/invalid/exhausted-slot/unsupported-state/tie/cross-variant/
terminal lanes. Include malformed shape/dtype and static audience checks.
Independent hand arithmetic may validate overkill damage; native HP, events,
and draws remain authoritative. Use one compiled scalar callable only if needed
for an actual depleted-state follow-up. Do not run an eager/JIT/vmap
cross-product.

Run focused `test.test_accuracy_choice` and, because its internal engine is
generalized, existing FAST `test.test_neutral_mega_punch` once after the final
source revision. Every validation command is capped at 30 seconds, with a 90
second aggregate including Astra's later review commands. For this resumed
handover, reserve 30 seconds conservatively for the interrupted owner's
unknown work, keep new-owner validation under 40 seconds, and leave at least
20 seconds for primary review. Skip the known-slow `test.test_neutral_two_move`
module. Record exact commands, wall times, environment, source blob IDs and any
deferred evidence. A missing required criterion remains `awaiting_validation`,
never waived.

## Validation record (candidate remains unaccepted)

Source identity: repository base `e5f4b21e52da78d7bcef37a1b4551808adec0b50`
(tree `5cd0b0d4117002a6f94b218149d1a1e5f32eb873`); current workspace HEAD
`cf11f44b6afd1ae671266c354a052c521d8ef669` adds only the supplied long-test
logs over that base. The native oracle pins Showdown commit
`1242c1e2d9763ba5b85d33f13aecb89bae19ae3d`. Candidate source blobs used by the
passing checks:

- `simulator/accuracy_choice.py`: `2480a85c574c604a06a705d39ebbf5ab38dbca19`
- `simulator/neutral_mega_punch.py`: `44bac3dae2a4a6ef8fb7b7cc0519412ea5037098`
- `simulator/__init__.py`: `fe5d497a96b36ef7fa3c078a916d461560c7f3f3`
- `reference/accuracy-choice-oracle.cjs`: `901d14cc35c234e72cb46238452c7e44e9a70f8b`
- `reference/neutral-mega-punch-oracle.cjs`: `698d668f67ba44e55186b0f5adc87760d6c2109e`
- `test/test_accuracy_choice.py`: `f97b747dceae9572af7df65f8612048b027c957e`
- `docs/accuracy-choice-variant.md`: `725b1117dc00352eb3c257ab4f1b8460774a58de`

Environment: Python 3.12.14, JAX/JAXLIB 0.12.0.1 on CPU, Node v24.22.1. The
following command passed all four focused v6 tests (5.357 seconds in unittest;
5.80 seconds outer wall time) and invoked the pinned native oracle:

```sh
time -p timeout --signal=TERM --kill-after=5s 30s /tmp/task040-jax-env/bin/python -m unittest test.test_accuracy_choice -v
```

The required fast v5 regression passed both tests (2.096 seconds in unittest;
2.49 seconds outer wall time):

```sh
time -p timeout --signal=TERM --kill-after=5s 30s /tmp/task040-jax-env/bin/python -m unittest test.test_neutral_mega_punch -v
```

All seven focused-v6 invocations and the v5 regression are recorded here because
the earlier runs exposed test-authoring issues before the final passing run:
focused-v6 attempt 1 exited during import after 0.04 seconds with an
`IndentationError`; attempt 2 ran 4 tests in 3.238 seconds and had two errors
(a traced Python-list choice and an incorrectly assumed nested continuation
fixture); attempt 3 ran 4 tests in 5.061 seconds and had two mismatches from
using native post-state as the transition input; attempt 4 ran 4 tests in
5.156 seconds and had one mismatch because overkill damage was clamped to
remaining HP instead of checked with hand arithmetic. Attempt 5 passed 4/4 in
5.393 seconds (5.89 seconds outer); attempt 6 passed 4/4 in
5.378 seconds (5.88 seconds outer) after adding the explicit critical-event and
public-constant assertions. Attempt 7, the final attempt above, passed 4/4 in
5.357 seconds (5.80 seconds outer) after adding direct native recipient-event
projection assertions. Each focused attempt used the exact command above. The
v5 command above ran once and passed 2/2. Final known/reserved execution
budget is 70.95 seconds of the 90-second limit: 35 seconds of owner test
commands, 30 seconds reserved conservatively for the interrupted owner, and
5.95 seconds for Astra review. No further tests were run after acceptance.

The files `/tmp/task050-accuracy-choice-native.json` and
`/tmp/task050-commit-message.txt` existed at handover. The fixture artifact has
no recorded source identity or execution metadata and is not counted as
validation; no prior-owner test result is known. The user-supplied
`long-tests.log` and `test-logs/*.log` report the same 21 legacy tests passing
in the combined run (511.757 seconds) and across the separate module runs.
Commit `cf11f44b6afd1ae671266c354a052c521d8ef669` has base
`e5f4b21e52da78d7bcef37a1b4551808adec0b50` and changes only those logs. They
cover the accepted base, not this candidate; they include no JAX/device
version or placement provenance, so they provide no GPU evidence and are not
repeated here.

The pre-existing v4 suite `test.test_neutral_two_move` took 111.047 seconds in
those base logs and is explicitly excluded by this task's 30-second command
cap. It is not an acceptance dependency and no rerun is recommended:

```sh
JAX_PLATFORMS=cpu /tmp/task040-jax-env/bin/python -m unittest test.test_neutral_two_move -v
```

That command identifies the excluded legacy v4 suite. Task050 does not change
the v4 engine or session, and the base log is historical evidence only; this
suite is not a Task050 dependency and no rerun is recommended. No GPU,
performance, full-format, Trail-graph, or milestone claim is made.

## Final acceptance

Astra independent CPU checks passed 4/4 in 5.516 seconds (5.95 seconds outer),
verified all six candidate source blobs, and found no required corrections or
blockers. Primary `/root` accepted Task050 on 2026-11-05. No source or test changes
followed review; see the [T050Global report](reviews/T050-global.md).
