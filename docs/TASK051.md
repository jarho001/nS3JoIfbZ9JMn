# Task051 — complete-battle parity through accuracy-choice v6

**Status:** accepted by `/root` on 2026-11-05
**Owner:** `/root/task051` (Luna/xhigh implementer)
**Primary:** `/root` (architect, reviewer, and commit owner)
**Source base:** `24e9505773cb734959ab5f40f402894e92c1848b`
**Accepted count:** 51 (review baseline 50; next checkpoint 60)
**Milestone / GPU:** G0 remains the only accepted milestone; GPU is not required.

## Deliverable

Extend Task050's exact parity evidence for the unchanged v6 Tackle/Mega Punch
accuracy-choice API across successive native turns and genuine terminal
outcomes. Add an opt-in `--duels` mode to the existing accuracy-choice oracle,
preserve its no-argument fixture behavior and output, add one focused native/JAX
duel test, and update the variant note with observed coverage.

Owned paths are limited to:

- `reference/accuracy-choice-oracle.cjs` (opt-in `--duels` mode only)
- `test/test_accuracy_choice_duel.py` (new)
- `docs/accuracy-choice-variant.md`
- `docs/TASK051.md`
- `docs/TASKS.md`
- `docs/ORCHESTRATION_STATE.json`

The simulator, existing tests, shared oracles, pinned reference/configuration,
and user logs remain unchanged. Do not edit scheduler/configuration files. Do
not add numerical, session, runner, reset, or training APIs; ties, status,
items, weather, Struggle execution, full-format support, GPU/performance claims,
or milestone acceptance.

## Acceptance criteria

- Reuse the existing oracle readiness guard, fixed Gen 5 Dex facts,
  `createBattle`, snapshots, PP setup, request projection, turn capture, and
  original PRNG observer. Keep the Showdown pin at
  `1242c1e2d9763ba5b85d33f13aecb89bae19ae3d`, with direct, unmodified,
  itemless Gen 5 Random Battle Mew sets and generator bypassed.
- The opt-in duel fixture contains four to six bounded battles, with distinct
  speeds in both initiative directions. Use known seeds 0, 1, and 21, and a
  fixture-local deterministic choice schedule that mixes Tackle and Mega
  Punch. Fall back only to an actually remaining legal slot. Every live turn
  uses native `makeChoices` and current/next requests; native PRNG values are
  neither supplied nor changed.
- Record actual choices each turn; full PP/HP/stats/RNG/turn/phase/winner state;
  bounded draw count, bounds, contexts, values, and four-limb pre/post RNG
  states; selected events and move IDs for both split recipients; and final
  terminal state. Limit each battle to 32 turns and fail on cap or unsupported
  all-PP-zero before native termination. Never execute Struggle or call a
  truncated run terminal.
- Include two live PP-depletion battles: final Mega Punch PP (seed 0, with a
  first-Mega-Punch-miss fixture) followed by Tackle in a later turn, and final
  Tackle PP followed by Mega Punch later. Use the existing PP setup/request
  helper and preserve the original RNG. Each depleted slot must be spent by an
  actual selected attempt before the remaining move is selected on a later
  turn; current and next own PP/legal rows must show the disabled slot.
- Across all battles, produce at least four genuine terminals, both initiative
  directions, all four submitted choice pairs actually compared, both moves
  attempted, and native Mega Punch hits, misses, and criticals. Require both
  successful depletion continuations. Cover first-action KO skip and
  terminal-after-second-action when the fixed fixtures permit. Record actual counts for battles,
  turns, attempts, selected moves, hits, misses by actor/slot, criticals, KO
  skips, repeated/double misses, and depletion continuations without claiming
  absent cases.
- The focused test is one small unittest module (one main test is sufficient)
  importing `test.test_accuracy_choice` as helpers so its `TestCase` is not
  collected. Reuse its state/native expected/PyTree helpers and recipient
  assertions. Use one shared module-level `jax.jit(transition_accuracy_choice)`;
  initialize once per battle and feed each actual returned state to the next
  transition. Every turn compares the whole transition tree to the
  independently captured native expectation, actual recipient projections,
  live current requests, actual next requests, PP, legal masks, and unused
  event/draw slots. Do not substitute native post-state as the next JAX input.
- Verify miss hit-only masking, PP deducted only for the attempted actor,
  unchanged skipped-opponent PP on first-action KO, final native winner/HP/PP/
  RNG/turn/phase, and absence of normal legal requests after terminal. Submit
  one rejected post-terminal choice per final state and require full state/RNG
  preservation, empty trace, and terminal status. Native terminal views may
  show the result; do not invent another native winner event. Hand arithmetic
  may check raw overkill only; native HP/events/draws remain authoritative.
- Compare pre-edit and post-edit default oracle JSON semantics. Normalize only
  diagnostic draw stack strings if source location changes; preserve all
  numeric, request, event, and pin fields.
- Keep every native/test attempt within 30 seconds and total Task051 attempts
  within 90 seconds. Do not extend a timeout or repeat slow checks. Record exact
  command, outcome, and elapsed time for every attempt, including failures.
  Existing Task050 focused tests are rerun only if a concrete oracle dependency
  requires it; do not rerun the supplied legacy suites or perform GPU checks.
- Inspect and report the complete final diff including new files, source base,
  changed and unchanged relevant blobs, runtime versions, coverage, remaining
  validation budget, and unresolved concerns. Do not self-accept, stage, commit,
  or push; parent `/root` owns review and final acceptance bookkeeping.

## Plan

1. Read the local task/orchestration contract, reconcile count and review gates,
   inspect the relevant Trail cache, and persist this scope/owner/criteria in
   this handoff plus both active-task state fields before source edits.
2. Capture pre-edit no-argument oracle JSON and inspect only the guarded
   accuracy-choice oracle and Task050 focused helpers needed for the extension.
3. Implement the fixed opt-in native duel fixtures and one threaded-state
   JAX/native parity test, then document actual observed coverage.
4. Run the minimum required bounded checks, compare default oracle semantics,
   inspect the complete diff and blobs, and return one frozen review package.
5. Retain ownership through parent-requested corrections. Do not update accepted
   count/history until primary acceptance.

## Progress

The scoped oracle/test inspection and native duel implementation are complete.
The test feeds each actual JAX result state into the next call and compares all
35 transitions, both recipient views and request boundaries, exact draws/events,
both PP-depletion continuations, terminal outcomes, and post-terminal
rejections. No simulator source or existing test changed. Task050 remains
accepted at count 50; T050Global is accepted; baseline 50/next 60 remain fixed;
no review is due and no GPU block exists. Trail CLI is unavailable and its
cache has no accuracy-choice card; targeted source inspection was used without
claiming a graph pass.

### Native coverage observed

Six battles reached native terminal outcomes in 35 turns and 66 attempted
actions (31 Tackle, 35 Mega Punch): Tackle hit 31 times and had 2 criticals;
Mega Punch hit 28 times, missed 7 times, and had 2 criticals. There were 4
first-action KO skips and 2 terminal turns after both actions. Misses were
5 by p1 with Mega Punch and 2 by p2 with Mega Punch; no Tackle misses, double
miss turns, or same-actor misses on consecutive turns occurred. Submitted
choice-pair counts were Tackle/Tackle 7, Tackle/Mega Punch 11,
Mega Punch/Tackle 10, and Mega Punch/Mega Punch 7. Three of six fixtures
included all four submitted pairs; all four pairs occurred across the fixture
set. One final Mega Punch PP was spent and followed by a later Tackle; one
final Tackle PP was spent and
followed by a later Mega Punch. Per-battle counts and outcomes are in
`accuracy-choice-variant.md`.

### Validation

Environment: Python 3.12.14, JAX/JAXLIB 0.12.0.1 on CPU, Node v24.22.1.
Pinned Showdown commit: `1242c1e2d9763ba5b85d33f13aecb89bae19ae3d`.
Every native/test command was bounded at 30 seconds; outer real times for all
native/test attempts total 7.46 seconds of the 90-second Task051 budget,
leaving 82.54 seconds. Recursive JSON comparison commands took 0.01 seconds
each and are listed separately below.

| Command | Result | Outer real time |
|---|---|---:|
| `time -p timeout --signal=TERM --kill-after=5s 30s node reference/accuracy-choice-oracle.cjs > /tmp/task051-accuracy-choice-default-base.json` | Passed; pre-edit default JSON saved (SHA-256 `dc234685888ff41d67df52e0c3d4af970e59b2c30dc59002b75072872d196592`). | 0.27 s |
| `time -p timeout --signal=TERM --kill-after=5s 30s node reference/accuracy-choice-oracle.cjs --duels > /tmp/task051-accuracy-choice-duels.json` | First attempt failed an oracle assertion comparing move IDs 33/5 to choice values 1/2. Corrected the fixture-local assertion to map choices to move IDs; no simulator mismatch. | 0.25 s |
| `time -p timeout --signal=TERM --kill-after=5s 30s node reference/accuracy-choice-oracle.cjs --duels > /tmp/task051-accuracy-choice-duels.json` | Passed; six native terminal battles and coverage summary generated. | 0.34 s |
| `time -p timeout --signal=TERM --kill-after=5s 30s /tmp/task040-jax-env/bin/python -m unittest test.test_accuracy_choice_duel -v` | First focused revision passed 1/1; unittest 2.529 s. The test invokes the pinned native `--duels` oracle. | 2.92 s |
| `time -p timeout --signal=TERM --kill-after=5s 30s node reference/accuracy-choice-oracle.cjs > /tmp/task051-accuracy-choice-default-post.json` | First post-edit no-argument JSON passed; saved for the initial semantic comparison. | 0.27 s |
| `time -p timeout --signal=TERM --kill-after=5s 30s /tmp/task040-jax-env/bin/python -` with the recursive JSON comparison script | Passed; all default fields matched and zero diagnostic `draw.stack` fields needed normalization. | 0.01 s |
| `time -p timeout --signal=TERM --kill-after=5s 30s node reference/accuracy-choice-oracle.cjs --duels > /tmp/task051-accuracy-choice-duels-final.json` | Passed after adding move-specific hit/critical counts. Six complete terminal battles; Mega Punch recorded 28 hits and 2 criticals. | 0.29 s |
| `time -p timeout --signal=TERM --kill-after=5s 30s /tmp/task040-jax-env/bin/python -m unittest test.test_accuracy_choice_duel -v` | Final focused revision passed 1/1; unittest 2.427 s, comparing all 35 transitions and the added move-specific coverage. | 2.84 s |
| `time -p timeout --signal=TERM --kill-after=5s 30s node reference/accuracy-choice-oracle.cjs > /tmp/task051-accuracy-choice-default-post.json` | Final post-edit no-argument JSON passed. | 0.28 s |
| `time -p timeout --signal=TERM --kill-after=5s 30s /tmp/task040-jax-env/bin/python -` with the same recursive JSON comparison script | Passed again; zero diagnostic `draw.stack` fields needed normalization. | 0.01 s |

Static checks: an initial `python -m py_compile test/test_accuracy_choice_duel.py; node --check reference/accuracy-choice-oracle.cjs` had a Python indentation syntax error before the test was run; the following `node --check` passed, so the shell's final exit status was 0. After correcting the indentation, both checks passed in parallel (wrapper wall time 0.4 s). After the final code revision, capped `python -m py_compile test/test_accuracy_choice_duel.py` and `node --check reference/accuracy-choice-oracle.cjs` passed individually in 0.02 s each. The final `git diff --check`, JSON parse, source blob inventory, and status review are recorded at freeze below.

### Frozen source identity and review status

Source base and `origin/main` at task start: `24e9505773cb734959ab5f40f402894e92c1848b`.
`git diff --check` passed. `jq -e` confirmed accepted count 50, baseline 50,
next checkpoint 60, Task050 at the end of accepted history, both Task051 active
fields, no active regular review, and no GPU block. `git status --short` showed
only the six owned paths. The pre-edit and post-edit no-argument JSON files have
the same SHA-256, `dc234685888ff41d67df52e0c3d4af970e59b2c30dc59002b75072872d196592`.

Unchanged relevant source blobs match HEAD: `simulator/accuracy_choice.py`
`2480a85c574c604a06a705d39ebbf5ab38dbca19`,
`simulator/neutral_mega_punch.py`
`44bac3dae2a4a6ef8fb7b7cc0519412ea5037098`,
`simulator/neutral_two_move.py`
`013dda42c232b47b2e5bd815b7c57a43d817199f`, and
`test/test_accuracy_choice.py`
`f97b747dceae9572af7df65f8612048b027c957e`. The shared native oracle and v5
event helper also match HEAD (`reference/neutral-turn-oracle.cjs`
`693c4a8393baf965cee47f615c8f9dbbdbde1239`,
`reference/neutral-mega-punch-oracle.cjs`
`698d668f67ba44e55186b0f5adc87760d6c2109e`).

At implementation freeze, the accepted count remained 50 and Task051 was still
awaiting primary review. There was no unresolved native/JAX mismatch or
required broader validation. GPU evidence was optional and not run; no long or
legacy suite was deferred as an acceptance dependency.

### Primary-review correction

Primary review requested a direct assertion that actual JAX terminal actor
views expose no legal move. The test now projects the terminal rejection result
for audiences 0 and 1, requires both legal masks to be `[False, False]`, and
checks each current outcome view's terminal flag and winner against the native
final winner. This reads the existing outcome view and does not add another
native winner event. The trailing Markdown hardbreak spaces in this handoff
header were removed.

| Command | Result | Outer real time |
|---|---|---:|
| `time -p timeout --signal=TERM --kill-after=5s 30s /tmp/task040-jax-env/bin/python -m unittest test.test_accuracy_choice_duel -v` | Passed 1/1; unittest 2.434 s. | 2.80 s |

Corrected test blob: `8cf8349cc9412e5ecc051d24e7bab6ed16ed8afa`. Cumulative
native/test command time is 10.26 seconds of 90, leaving 79.74 seconds at the
corrected-review handoff. No engine or oracle file changed in that correction.

## Final acceptance

Primary `/root` reviewed the final oracle, test, documentation, source identity,
and corrected terminal-view evidence, then accepted Task051 on 2026-11-05.
Accepted task count is 51; regular review baseline 50 and next checkpoint 60
remain fixed. G0 remains the only accepted milestone. The two active-task state
fields are cleared and Task051 is appended to completed history. No code,
oracle, or test changes followed the corrected 1/1 focused test.
