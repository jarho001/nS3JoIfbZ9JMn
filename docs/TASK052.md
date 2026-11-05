# Task052 — restricted single-burn Tackle/Mega Punch Mew turn

**Status:** accepted by primary `/root` on 2026-11-05
**Owner:** `/root/task052` (Luna/xhigh implementer)
**Primary:** `/root` (architect, reviewer, and commit owner)
**Source base:** `24997e4a7117897e18c6c78b04533ed6d0ddd612`
**Accepted count:** 52 (review baseline 50; next checkpoint 60)
**Milestone / GPU:** G0 remains the only accepted milestone; GPU is not required.

## Deliverable

Add versioned variant 7, `gen5-single-burn-choice-mew-1v1-v7`, for one
pre-existing public native burn flag and ordinary simultaneous Tackle/Mega
Punch turns. Reuse the existing v6 action/RNG/damage path and the accepted
fixed-point burn primitive. Resolve the once-per-turn burn residual after the
attack phase, with native-compatible terminal and event behavior, and provide
recipient-safe views for attack and residual events. Add the pinned native
oracle, focused heterogeneous JAX parity test, and variant documentation.

Owned paths are limited to:

- `simulator/burn_choice.py` (new)
- `simulator/neutral_mega_punch.py` (optional private burned flag on the shared
  action helper; default behavior remains unchanged)
- `simulator/accuracy_choice.py` (optional burned vector on its private turn
  runner; public v6 default behavior remains unchanged)
- `simulator/__init__.py` (uniquely prefixed v7 exports)
- `reference/burn-choice-oracle.cjs` (new)
- `test/test_burn_choice.py` (new)
- `docs/burn-choice-variant.md` (new)
- `docs/TASK052.md`
- `docs/TASKS.md`
- `docs/ORCHESTRATION_STATE.json`

Do not edit mechanics, v4/session/shared neutral-turn sources, existing
tests/oracles, pinned reference/configuration, user logs, or scheduler/config
files. The scope excludes burn infliction/cure/reflection, multiple burned
actors, other effects/statuses, Struggle, ties, sessions/reset/runner/training,
GPU/performance/full-format claims, and milestone claims.

## Acceptance criteria

- Use a nested fixed-layout `BurnChoiceState(battle: TwoMoveState tagged 7
  with move IDs 33/5, burned: bool[2])`. `burned` mirrors live native `brn`
  flags and has at most one true bit. Build via the v6 two-slot layout and
  rewrite the private outer variant tag to 7. Existing v6 APIs reject the new
  wrapper or wrong variant.
- Support a no-burn control and at most one pre-existing burned actor. Reject
  malformed layout, invalid variant/caps, ties, and dual burn without state or
  RNG changes. Preserve v6 structure-first choice precedence: negative is
  invalid, unknown positive unsupported, zero pending, exhausted selected slot
  invalid, and terminal submissions are absorbing. Preserve the all-PP-zero
  unsupported rule and PP caps 56/32.
- Reuse the existing engines. Add only a private optional `burned=False` to
  Mega Punch's `_execute_action`, passed to `damage_from_preroll` as
  `burned=...`, `physical=True`, `has_guts=False`. Add optional `burned=None`
  to accuracy-choice's private `_run_accuracy_choice_turn`, used only by its
  action closure. Public v5/v6 APIs retain exact default behavior.
- In Gen 5, burn halves eligible physical damage through the accepted
  fixed-point primitive after the normal roll/type path; no Facade or Guts is
  present. If attack resolution is already terminal, skip residual entirely.
  Otherwise, if one actor is live and burned, subtract
  `max(1, floor(baseMaxHP/8))`, clipped to current HP, without PP/RNG changes.
  Recompute terminal/winner/phase/turn from the original pre-turn number.
  Preserve burn flags from actual native `Pokemon.status` after fainting; the
  protocol HP condition may say `0 fnt` while persistent status remains `brn`
  in this one-Mew boundary.
- Keep attack events, move IDs and draws unchanged, with stable empty slots and
  one fixed residual event row. Record actual clipped HP loss consistently.
  Preserve full input state and burn bits on all rejected or terminal
  submissions.
- Projection wraps the v6 accuracy-choice player view, public known-burn flags,
  and a recipient-safe residual view. Reuse the v6 projector only by adapting
  inner battle copies to tag 6 and constructing a private v6 result. Common
  turn/winner output reflects final post-residual state. Event order is attack
  attempt/miss/hit/crit/damage/faint, residual damage/faint, then turn/winner.
  Show exact HP/loss only to its owner; show opponent HP as ceiling percent and
  loss as -1. Never expose RNG, full state, opponent PP/exact HP, or an
  unattempted choice. Terminal legal masks are false for both actors; do not
  invent a repeated native winner event.
- Add a guarded native oracle using the final neutral-turn guard/snapshot/draw
  helpers, pinned Showdown commit `1242c1e2d9763ba5b85d33f13aecb89bae19ae3d`,
  and unmodified itemless Gen 5 Random Battle Mew with generator bypassed.
  Establish pre-existing status through native `setStatus('brn', mon)` and
  verify installed status, the other Mew remaining unburned, unchanged RNG,
  setup status protocol, current/next requests, native draw callbacks, and full
  HP/PP/RNG/turn/phase/winner/events. Keep trusted status/low-HP setup separate
  from simulated attack/residual turns. No replacement RNG tape or generic
  projector loss of residual cause/order.
- Cover roughly six fixed bounded cases: no-burn control; each move by a burned
  actor and both speed directions across cases; a burned critical, surviving
  burn after Mega Punch miss, attack-phase KO skipping residual, and residual
  KO. Compare genuine native state, requests, event ordering, and draws.
- Add one focused CPU test module and one heterogeneous `jit(vmap(...))` batch
  spanning valid/control/terminal plus pending, invalid, unsupported, dual
  burn, cap, wrong-variant/cross-API and tie lanes. Compare complete numerical
  PyTrees and both recipient views/requests/status conditions, privacy,
  empty slots, exact rejection preservation, and native event ordering.
  Malformed shape/dtype/type/audience inputs fail closed. Avoid a broad
  eager/JIT/vmap matrix.
- Run the pinned new native and focused v7 tests, plus the existing fast
  `test.test_neutral_mega_punch` and `test.test_accuracy_choice` regressions
  once on final shared-helper source. Cap every command at 30 seconds and total
  native/test attempts at 90 seconds; record exact command, result, and elapsed
  time for every attempt. Do not run old long suites, GPU, performance, or
  unrelated checks. Missing required evidence stays pending.
- Inspect the complete final diff and source identities, report exact
  validation, timings, coverage, and unresolved concerns to `/root`. Do not
  self-accept, stage, commit, push, or update accepted count/history before
  primary acceptance.

## Plan and progress

1. Read orchestration/implementer policy, reconcile task count and review
   gates, inspect cached mechanics burn context and the v6 contract, and record
   this task plus both active state fields before source edits.
2. Implement the minimal reusable burn parameter and isolated v7 nested state,
   transition, residual handling, and projection.
3. Add the pinned native oracle, focused parity/privacy tests, and variant
   documentation from actual native observations.
4. Run only the bounded required checks once on final source, inspect the
   complete diff and blobs, and return one frozen evidence package.
5. Retain ownership through parent-requested corrections. Count/clear active
   fields only after `/root` accepts.

The task plan is persisted before source edits. A primary architecture
correction based on direct pinned-source inspection clarifies that fainted
one-Mew Pokémon may retain persistent `brn` status: `checkFainted` assigns
`fnt` only when switching is possible, and this one-Mew boundary cannot switch
(`reference/vendor/pokemon-showdown/sim/battle.ts:2540–2550`). Native
`resolveFaints` calls `clearVolatile(false)` (`battle.ts:2592–2660`), which does
not clear persistent status (`reference/vendor/pokemon-showdown/sim/pokemon.ts:1529–1586`).
Implementation and oracle snapshots therefore preserve/record actual native
status independently of the protocol's `0 fnt` HP condition. The Trail CLI is
absent and no simulator wiring card is cached; the cached burn and
damage-composition cards were consulted, followed by targeted source
inspection. No architecture or native-semantic contradiction remains
unresolved.

## Implementation freeze and validation

The final native fixture has six cases, 11 attempted actions, 9 hits, 2 Mega
Punch misses, one critical by the burned attacker, four residual events, two
terminal outcomes, both initiative directions, and one residual knockout. The
HP-1 residual case ends with HP `[0, 164]`, native status `['brn', '']`, a
`0 fnt` damage condition, and no extra draw. The burned direct-attack KO has
no residual row and skips the slower choice. Each of the four burn-start
fixtures records the actual `brn` status protocol event for both recipients;
trusted status/HP setup and live request creation preserve the native PRNG.

Python/JAX validation used `/tmp/task040-jax-env/bin/python` (Python 3.12.14,
JAX/JAXLIB 0.12.0.1, CPU); Node was v24.22.1. All native/test commands below
were externally limited to 30 seconds per command. Their measured outer real
time totals 37.99 seconds, plus a 0.29-second fixture-inspection diagnostic
and a preliminary 0.40-second Python syntax check, leaving about 51.32 seconds
of the 90-second native/test budget. The failed attempts are retained here rather
than silently discarded.

| Command | Result | Outer real time |
|---|---|---:|
| `time -p timeout --signal=TERM --kill-after=5s 30s node reference/burn-choice-oracle.cjs > /tmp/task052-burn-choice-native.json` | Failed its draw-count assertion because the first oracle draft equated critical draws with critical hits; corrected to count one critical draw per successful hit. | 0.27 s |
| Same native command | Failed because native `Pokemon.sethp(1)` returns the HP delta (`-174`), not a boolean; corrected to assert resulting HP. | 0.27 s |
| Same native command | Passed; emitted all six pinned native cases. | 0.28 s |
| `time -p timeout --signal=TERM --kill-after=5s 30s /tmp/task040-jax-env/bin/python -m unittest test.test_burn_choice -v` | Failed complete-tree comparison because the independent expected-data reader selected an opponent percentage event instead of the target owner's exact HP event; corrected by filtering recipient and target. | 2.14 s |
| `time -p timeout --signal=TERM --kill-after=5s 30s /tmp/task040-jax-env/bin/python -` with fixture inspection of `/tmp/task052-burn-choice-native.json` | Confirmed the mismatched expected values were opponent percentages; used only to localize the fixture-reader issue. | 0.29 s |
| Same focused unittest command | Passed 1/1 after correcting expected target-event selection. | 4.51 s |
| Same focused unittest command | Passed 1/1 after switching expected-state construction to the public v7 constructor and adding v7 projection support checks. | 4.39 s |
| `time -p timeout --signal=TERM --kill-after=5s 30s /tmp/task040-jax-env/bin/python -m unittest test.test_neutral_mega_punch -v` | Passed 2/2; shared v5 default path unchanged. | 2.50 s |
| `time -p timeout --signal=TERM --kill-after=5s 30s /tmp/task040-jax-env/bin/python -m unittest test.test_accuracy_choice -v` | Passed 4/4; public v6 default path unchanged. | 5.84 s |
| `time -p timeout --signal=TERM --kill-after=5s 30s node reference/burn-choice-oracle.cjs > /tmp/task052-burn-choice-native-final.json` | Passed after adding separate setup-status projections for p1 and p2. | 0.29 s |
| Same focused unittest command | Passed 1/1, including both recipients' setup status lines, one heterogeneous JIT/vmap batch, complete transition PyTrees, current/next requests, event order, and residual privacy. | 4.35 s |
| Same focused unittest command | Final focused version passed 1/1 and additionally checks native residual loss against clipped `max(1, floor(baseMaxHP/8))`. | 4.40 s |
| Same focused unittest command | Passed 1/1 after removing one tab-only blank line from the new test; test behavior and expectations are unchanged. | 4.38 s |
| Same focused unittest command | Passed 1/1 after retaining native miss targets, comparing hit/miss `target_player`, validating public and NamedTuple type hints, and checking that a jitted dynamic audience raises `TypeError`. | 4.37 s |
| `python -m py_compile simulator/burn_choice.py simulator/accuracy_choice.py simulator/neutral_mega_punch.py` | Preliminary syntax check passed before later small edits; final source was imported and executed by the final focused test. This check was not part of the native/test budget. | ~0.40 s |

Static state/diff checks: `git diff --check`, `python -m json.tool
docs/ORCHESTRATION_STATE.json`, state invariants, and a ten-path newline/trailing
whitespace scan all passed after cleanup. The first invariant script used the
wrong path `review_state.active_review` and raised `KeyError`; inspection found
the actual fields `active_regular_review` and
`milestone_review.active_review`, and the corrected assertion passed. The
first whitespace scan found a tab-only blank line at
`test/test_burn_choice.py:252`; it was removed, and the final scan passed.
`node --check reference/burn-choice-oracle.cjs` passed, and the native oracle
was also executed after its final edit. The final environment probe confirmed
Python 3.12.14, JAX/JAXLIB 0.12.0.1, CPU, and Node v24.22.1.

### Primary-review correction

The primary review found that the first v7 oracle omitted `-miss` target and
that the focused test lacked static public type-hint and compiled dynamic
audience checks. The oracle now retains both miss actor and target; native
expected rows retain move targets, and both recipients' `target_player` rows
are compared for hits and misses. `make_burn_choice_state.burned` now has a
resolved boolean-vector hint. The test checks public function arguments and
returns, every new NamedTuple field type, and a jitted dynamic-audience
`TypeError`. The variant table now correctly states that P2's Tackle damages
P1 before P1's residual in the critical fixture.

Exact correction check:

`time -p timeout --signal=TERM --kill-after=5s 30s /tmp/task040-jax-env/bin/python -m unittest test.test_burn_choice -v` — passed 1/1 in 4.37 s. The oracle ran inside the test setup; no separate native command was run. Cumulative native/test outer real time is 37.99 s of 90 s, leaving 51.32 s after the recorded diagnostic and syntax-check times. The corrected source blob IDs are recorded in the frozen package to `/root`.

## Final acceptance

Primary `/root` accepted Task052 on 2026-11-05 after reviewing the actual source,
native oracle, focused test, and documentation corrections. Accepted count is
52; baseline 50 and next checkpoint 60 remain fixed. G0 remains the only
accepted milestone. No GPU block or broader-format claim was added.

Frozen source base: `24997e4a7117897e18c6c78b04533ed6d0ddd612`. Relevant final
source/test blobs:

| Path | Blob |
|---|---|
| `simulator/burn_choice.py` | `451570ae807023eec073ac9fdc08fc61ca831fbc` |
| `simulator/neutral_mega_punch.py` | `39ffd192cb02b1d452144f67644fd0e066823356` |
| `simulator/accuracy_choice.py` | `91a87a3492c57635e2ae115dbcf142374eb82034` |
| `simulator/__init__.py` | `e8b5f5f3b4569611f53953594d48c726b9635b02` |
| `reference/burn-choice-oracle.cjs` | `1106438eb99154214a537487d4baf2f338f8ad8c` |
| `test/test_burn_choice.py` | `8ac366ebb0f931e8ba30be9f71a2dae046dacda5` |

Unchanged relevant dependency blobs were verified against the source base:
`mechanics/damage_composition.py` `55bf60a7f9e393967e02435c72ab50256cb7bc7e`,
`mechanics/burn.py` `6993b7528bc30e58e48ef83306cfb87f9ee0e1f2`,
`simulator/neutral_two_move.py` `013dda42c232b47b2e5bd815b7c57a43d817199f`,
`reference/neutral-turn-oracle.cjs` `693c4a8393baf965cee47f615c8f9dbbdbde1239`,
`reference/neutral-mega-punch-oracle.cjs` `698d668f67ba44e55186b0f5adc87760d6c2109e`,
and `test/test_accuracy_choice.py` `f97b747dceae9572af7df65f8612048b027c957e`.
The accepted review corrections and exact final 1/1 focused run are recorded
above. No staging, commit, or push occurred in this task.
