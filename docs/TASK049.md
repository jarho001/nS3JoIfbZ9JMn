# Task049 — exact complete-battle parity for Mega Punch Mew

**Status:** accepted complete
**Owner:** `/root/task049` (Luna/xhigh implementer)
**Accepted by:** `/root` on 2026-11-05
**Source base:** `9d4d40111579c65ad777627b3911186bd69c38b5`
**Accepted count:** 49 (review baseline 40; next checkpoint 50)
**Milestone / GPU:** G0 remains the only accepted milestone; GPU is not required.

## Scope

Extend the existing native Mega Punch oracle with an opt-in `--duels` mode and
add one focused test module that validates the unchanged v5 numerical API over
successive real native turns through terminal outcomes. Reuse the existing
guarded Showdown setup, snapshots, observed-draw wrapper, request projection,
event projection, and fixture facts. The oracle's default invocation must keep
its current fixture semantics and output unchanged.

Owned paths are limited to:

- `reference/neutral-mega-punch-oracle.cjs`
- `test/test_neutral_mega_punch_duel.py` (new)
- `docs/neutral-mega-punch-variant.md`
- `docs/TASK049.md`
- `docs/TASKS.md`
- `docs/ORCHESTRATION_STATE.json`

The numerical engine `simulator/neutral_mega_punch.py`, other simulator files,
shared native oracle, and existing tests must remain unchanged. Do not edit
scheduler or configuration files. Do not add a numerical/session API, ties,
Struggle, reset, training, runner, benchmark, GPU, full-format support, or
milestone claim.

## Contract and acceptance criteria

- Pin upstream Showdown commit
  `1242c1e2d9763ba5b85d33f13aecb89bae19ae3d` and the Dex facts for Mega Punch:
  Normal Physical, power 80, accuracy 85, base PP 20 / maximum PP 32, critical
  ratio 1. Use direct one-Mew-per-side sets in the same unmodified
  `gen5randombattle`; do not invoke team generation or forge terminal state.
- Add opt-in `--duels` output while preserving default fixture semantics.
  Implement a bounded fixed fixture set (four to six battles, with a selection
  cap of 32 only if required) using ordinary initial PP 32 and distinct speeds
  in both initiative directions. Actual `battle.makeChoices('move 1',
  'move 1')`, live native requests, and original PRNG return values drive every
  turn. Capture ordered draws with bounds, values, contexts, and four-limb
  before/after RNG state; capture each turn's HP/PP/RNG/turn/winner/phase,
  selected events, and current and next requests for both players.
- Observe actual native `ended`/winner and final snapshots. Declare a
  conservative cap of at most 32 turns per battle and fail if a fixture reaches
  the cap or an unsupported no-PP request before terminal. Never execute
  Struggle or label a truncated battle terminal. If setup needs PP adjustment,
  use live data and `makeRequest`, then assert RNG is unchanged; no adjustment
  is planned.
- Add one focused unittest module, avoiding collection of the prior module's
  `TestCase` by importing it as a helper module. Reuse its state construction,
  native expected result, full PyTree comparison, and native projection helpers
  where appropriate. Use one shared `jax.jit(transition_mega_punch)` across
  all turns and thread each actual result state into the next native turn.
  Compare the complete transition PyTree on every turn and compare actual
  current requests/legal action, both recipient-specific move/miss/critical/
  damage/faint/turn/outcome event projections, and actual next requests at every
  nonterminal boundary. Preserve stable unused event/draw slots and miss
  privacy masks.
- At terminal, compare native winner, HP, PP, RNG, turn and request phase, with
  no legal normal move request. For every finished state, one rejected
  post-terminal submission must preserve the full state and RNG and return an
  empty trace with `STATUS_TERMINAL`. Preserve the genuine winner's current
  view without inventing another native event.
- Across at least four complete native battles, cover both faster players,
  each actor missing, a miss by the second acting slot, a normal hit, and a
  native critical. Select fixtures to cover both first-action KO skipping the
  second action and a second-action terminal outcome when feasible. Report
  exact battles, turns, attempts, hits, misses, criticals, and KO skips. Report
  repeated misses and same-turn double misses honestly if naturally present;
  do not assert uncovered cases.
- Demonstrate that multiple turns consume RNG and feed the returned state into
  the next turn. Do not inject independently generated native per-turn seeds.
  Do not add a broad eager/JIT/vmap matrix; the prior batch remains sufficient.
- Keep focused oracle and test commands within 30 seconds each and 90 seconds
  total. Check default oracle output against base semantics, normalizing only
  diagnostic draw stack paths if a temporary file location changes. Do not run
  the known-slow `test.test_neutral_two_move` suite. Record all exact attempted
  commands and timings, source base and final blobs, environment, and any
  deferred raw command, reason, and `acceptance_dependency: false`.
- No GPU criterion applies. Do not self-accept, stage, commit, or push. The
  primary `/root` owns review and acceptance bookkeeping.

## Plan

1. Persist this bounded contract and owner in both `active_task` state fields
   before implementation; keep accepted count at 48.
2. Inspect the existing oracle/test behavior and pinned runtime; preserve the
   default oracle contract.
3. Implement the native duel trace and focused successive-state parity test
   only in owned paths, then update the variant note with honest coverage.
4. Run only the required bounded commands, inspect the complete diff including
   new files, and report source identities, exact results, and concerns.
5. Retain ownership through any corrections requested by `/root`; primary
   owns acceptance and final task-count bookkeeping.

## Progress

- Kickoff: current source base is clean at `9d4d40111579c65ad777627b3911186bd69c38b5`.
  Accepted count remains 48; both active fields now name Task049. `.trail/cache`
  has no current neutral simulator/oracle cards and the `trail` CLI is absent
  from `PATH`; source fallback is used without a graph-check claim.
- Required runtime is present: Node v24.22.1; Python 3.12.14, JAX/JAXLIB
  0.12.0.1, NumPy 2.5.4 in `/tmp/task040-jax-env`; CPU is the required target.
- Implementation: the oracle now has an opt-in `--duels` mode and keeps its
  no-argument fixture unchanged. Six real direct-Mew battles reached native
  terminal outcomes in 28 turns total, with 53 attempts, 48 hits (43
  noncritical), five misses, five criticals, three first-action KO skips, and
  three terminal second-action outcomes. P1 was first in four battles and P2
  in two; P1 missed three times, P2 twice, including one second-slot miss.
  Three battles had misses; one had multiple misses across turns; no turn had
  two misses. All started at 32 PP and ended within the 32-turn cap.
- Focused CPU validation passed: `time -p timeout --signal=TERM
  --kill-after=5s 30s env JAX_PLATFORMS=cpu
  /tmp/task040-jax-env/bin/python -m unittest
  test.test_neutral_mega_punch_duel -v` — final revision passed one test in
  1.675 seconds (outer command 2.05 seconds). It compares the full result tree on all 28
  successive transitions, threads actual JAX state, checks native draws,
  per-turn current/next requests and both recipient views, terminal snapshots,
  miss privacy, and post-terminal rejection for all six battles.
- Direct duel oracle passed on the final revision:
  `time -p timeout --signal=TERM --kill-after=5s 30s node
  reference/neutral-mega-punch-oracle.cjs --duels` — 0.30 seconds. An earlier
  run exposed only an initiative coverage counter scoped across battles rather
  than per first turn; that counter was corrected, and the following run
  passed. A later coverage-label correction split battles-with-misses from
  repeated-miss battles; the final oracle and focused test were rerun and
  passed.
- Default compatibility: pre-edit
  `time -p timeout --signal=TERM --kill-after=5s 30s node
  reference/neutral-mega-punch-oracle.cjs` passed in 0.28 seconds; post-edit
  invocations passed in 0.26 and 0.27 seconds. Final parsed outputs matched
  exactly, including diagnostic draw stacks, so no normalization was needed.
- All native/test attempts, including the first failed duel counter assertion
  and focused-test reruns, took about 7.78 seconds total; no timeout occurred.
  This excludes three JSON comparisons estimated at about 0.5 seconds each;
  combined native/test and comparison time was about 9.28 seconds. No
  additional validation probe was run. The bounded process
  outputs are `/tmp/task049-mega-default-base.json`,
  `/tmp/task049-mega-default-after.json`, and
  `/tmp/task049-mega-duels-final.json` in this execution workspace.
- Deferred optional regression: raw command
  `JAX_PLATFORMS=cpu /tmp/task040-jax-env/bin/python -m unittest test.test_neutral_two_move -v`.
  This known-slow prior v4 boundary suite is excluded by the Task049 budget and
  scope; the Task049 focused complete-battle parity test passed, so it is not
  an acceptance dependency. The matching state entry records its source base,
  blobs, reason, and `acceptance_dependency: false`. Recommend an explicit
  long-test session for that regression.

## Bounded validation attempt log

All native/test commands below used the 30-second process cap. No timeout
occurred.

1. `time -p timeout --signal=TERM --kill-after=5s 30s node reference/neutral-mega-punch-oracle.cjs > /tmp/task049-mega-default-base.json` — passed, 0.28 seconds.
2. `time -p timeout --signal=TERM --kill-after=5s 30s node reference/neutral-mega-punch-oracle.cjs --duels > /tmp/task049-mega-duels.json` — failed, 0.28 seconds; exposed the initiative-first coverage counter being global across fixtures. The counter was fixed to use each duel's first turn.
3. The same `--duels` command after that fix — passed, 0.29 seconds.
4. `time -p timeout --signal=TERM --kill-after=5s 30s env JAX_PLATFORMS=cpu /tmp/task040-jax-env/bin/python -m unittest test.test_neutral_mega_punch_duel -v` — passed one test in 1.637 seconds (outer command 2.04 seconds).
5. `time -p timeout --signal=TERM --kill-after=5s 30s node reference/neutral-mega-punch-oracle.cjs > /tmp/task049-mega-default-after.json` — passed, 0.26 seconds.
6. Parsed base/after default outputs compared equal after normalizing only diagnostic `draw.stack`; Python comparison passed, about 0.5 seconds.
7. The same `--duels` command after correcting coverage labels — passed, 0.30 seconds; output `/tmp/task049-mega-duels-final.json`.
8. The focused unittest after that coverage revision — passed one test in 1.612 seconds (outer command 2.01 seconds).
9. The focused unittest after removing a tautological initial-state assertion and correcting indentation — passed one test in 1.675 seconds (outer command 2.05 seconds).
10. `time -p timeout --signal=TERM --kill-after=5s 30s node reference/neutral-mega-punch-oracle.cjs > /tmp/task049-mega-default-final.json` — passed, 0.27 seconds.
11. The final base/final comparison with diagnostic-stack normalization passed, about 0.5 seconds.
12. A stricter final JSON equality assertion passed exactly, including all `draw.stack` values, about 0.5 seconds.

The comparison command below was used for attempts 6 and 11, with the stated
`after` input each time. Attempt 12 used this exact JSON comparison:

```sh
python3 - <<'PY'
import json
from pathlib import Path
before = json.loads(Path('/tmp/task049-mega-default-base.json').read_text())
after = json.loads(Path('/tmp/task049-mega-default-final.json').read_text())
assert before == after
PY
```

The normalized comparison used:

```sh
python3 - <<'PY'
import json
from pathlib import Path
before = json.loads(Path('/tmp/task049-mega-default-base.json').read_text())
after = json.loads(Path('/tmp/task049-mega-default-final.json').read_text())
def normalize(value):
    if isinstance(value, dict):
        return {key: ('<diagnostic stack>' if key == 'stack' else normalize(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize(item) for item in value]
    return value
assert normalize(before) == normalize(after)
PY
```

The final routine checks also passed: `python3 -m json.tool
docs/ORCHESTRATION_STATE.json`, focused state invariants for accepted count 48
at implementation review, mirrored Task049 active fields, baseline 40 / checkpoint
50, no pending review corrections or GPU block, and the Task049 deferred
record; `git diff --check`; and trailing-whitespace checks for both new files.
The change allowlist contains exactly the six owned paths. Primary `/root`
accepted Task049 on 2026-11-05. The Trail CLI was unavailable and no graph
check is claimed. Oracle and duel-test blob IDs are recorded in the Task049
`deferred_validation` entry in `ORCHESTRATION_STATE.json`.
