# Task046 complete-battle parity through staged two-move request sessions

## Scope and status

Validate the accepted `gen5-neutral-two-move-mew-1v1-v4` session API from fresh
actor requests through genuine native terminal outcomes. For every simultaneous
turn, route both native choices through separate session submissions and prove
that local episode progression, private pending metadata, recipient current and
prior-event views, and all native battle/RNG effects match the pinned engine.
Emphasize continuation after p1's final Tackle PP is spent and Pound remains
legal. This task adds an opt-in native fixture mode, a focused CPU test, and
bounded documentation only; no production simulator API or numerical code
changes are authorized without a concrete mismatch reported to the parent.

Owner: `/root/task046`. Source base:
`3abd881cd1105abd162c5d190cc28b129926550e`. Status: accepted by primary
`/root` on 2026-11-05. Accepted task count is 46; review baseline 40 and next
checkpoint 50 remain unchanged. G0 is the only accepted milestone. GPU evidence
is not required.

Owned paths: `reference/neutral-two-move-oracle.cjs`, new
`test/test_two_move_session_duel.py`, `docs/TASK046.md`, `docs/TASKS.md`,
`docs/ORCHESTRATION_STATE.json`, and `docs/neutral-two-move-variant.md`.

## Acceptance criteria

- Add an explicit `--session-duels` mode to the native oracle while preserving
  its no-argument, `--duels`, and `--sessions` behavior and semantic fields.
  Produce paired complete battles for the existing four duel scenarios with
  first submitter p1 and p2 (eight genuine native battles).
- On every native turn, derive alternating Tackle/Pound planned choices and
  fall back only to the other currently legal request slot. Use actual
  separate `Battle.choose` commits and accept only native-legal commands. Use
  the last-Tackle-PP setup by changing p1 slot 1 through the live Pokemon,
  calling `battle.makeRequest('move')`, and proving the original RNG limbs are
  unchanged.
- Prove each first accepted commitment preserves the entire native battle,
  both requests, PP, RNG, draw trace, and events. Record the chosen and planned
  commands, all pre/post state and PP, exact second-commit draw bounds,
  contexts, values and limbs, public move order/IDs, recipient events, next
  requests, and true terminal winner/state. Each battle ends via
  `Battle.ended` with p1/p2 winner within 56 turns and both actors retain
  positive PP in at least one slot.
- In every battle both actors execute Tackle and later Pound. In the PP case,
  spend Tackle's final PP, expose the disabled Tackle/enabled Pound request,
  and execute Pound on later turns. Paired same-state/same-choice-policy
  battles resolve to equal terminal state and per-turn semantics regardless of
  first submitter; only diagnostic stack text may be normalized, never RNG or
  phase values.
- The focused CPU test creates one session from each native initial state and
  threads it through every first and second submission without reseeding. Use
  one shared jitted submit helper and static-audience compiled projectors
  across all cases and turns. Compare fresh/current requests and both legal
  masks, private commit bit and same episode, unchanged battle/RNG/draws/events
  and full opponent-view invariance after the first commit; after the second,
  compare full native state, PP, stats, turn/phase/winner, RNG and every draw,
  order and move identity, recipient current requests and previous resolved
  events. Verify both pending slots clear, the episode advances once including
  terminal resolution, and episode tracks native turns.
- At native terminal, verify genuine winner/full state, cleared pending slots,
  episode equal to resolved native turns, all-false legal masks, own bit false,
  and an absorbing next valid submission with `SESSION_TERMINAL`, unchanged
  session/RNG, no events, and no draws. Check the PP-one case distinguishes
  current request projections from previous resolved events and really
  executes Pound later.
- Do not add a simulator/session API, numerical, mechanics, reset, network,
  history-library, driver, policy, PPO, training, tournament, reporting, GPU,
  performance, or milestone claim. Do not edit existing tests or simulator
  files without reporting a concrete mismatch to the parent first.
- Inspect the final owned-file diff and report exact commands/results,
  elapsed times for every attempt, source identity and code hashes, changed
  files, deferred optional checks, and unresolved concerns. Do not self-accept,
  commit, or push.

## Plan and progress

1. Read the scoped orchestration instructions, current state, v4/session
   contract, prior native duel/session seams, and focused test helpers. Trail is
   unavailable at the documented paths and recent cache cards are absent; the
   available graph index was checked before targeted source fallback.
2. Persist this scope, criteria, ownership, and plan in the handoff and compact
   state before implementation.
3. Add only the opt-in native complete-session duel fixture, its focused CPU
   differential test, and concise variant/task-index documentation.
4. Run only the required focused checks under the 30-second-per-command and
   90-second-total budget. Inspect the diff and return a frozen-source evidence
   package to `/root` for acceptance review.

## Evidence

### Implementation

`reference/neutral-two-move-oracle.cjs` adds `--session-duels`; no-argument,
`--duels`, and `--sessions` behavior remains semantically unchanged. The new
mode reuses the four accepted native duel setups in both p1-first and p2-first
orders, yielding eight genuine native battles and 66 turns. Each turn uses two
separate accepted `Battle.choose` calls, alternates planned Tackle/Pound slots,
and falls back only to the other legal current request. The first-commit
capture proves unchanged full state, PP, RNG, requests, events, and draws. The
second capture records full state and PP, exact original draws and limbs, move
order/identity, recipient events, and next requests. The last-Tackle-PP setup
retains its RNG limbs after `battle.makeRequest('move')`, disables Tackle, and
executes Pound on later turns. The fixture requires retained PP for both sides,
a true native winner within 56 turns, and semantic equality across paired
submission orders.

`test/test_two_move_session_duel.py` creates one session per native initial
state and threads each returned session through all first and second commits.
It uses one jitted submit helper and one shared static-audience jitted helper
for each view API. It checks fresh/current requests, legal masks, first-commit
privacy and episode stability, full opponent-view invariance, complete native
state/PP/RNG/draw parity, move order and identity, recipient prior events,
current next requests, episode increments and pending clears, terminal
absorption, and the Tackle-to-Pound PP boundary. The terminal current view
contains its current winner outcome; the last resolved events remain in the
separate prior-event projection.

### Validation

Environment: Python 3.12.14, JAX/JAXLIB 0.12.0.1, CPU backend; pinned native
Showdown fixture. All attempted commands were individually below 30 seconds;
total measured shell real time across syntax/compile, native generation,
mode-compatibility, focused-test attempts, and final static checks was
approximately 12.9 seconds. No timeout occurred.

| Exact command or check | Result and elapsed time |
|---|---|
| `time -p timeout --signal=TERM --kill-after=5s 30s node --check reference/neutral-two-move-oracle.cjs` | Passed; shell real 0.02s. |
| `time -p timeout --signal=TERM --kill-after=5s 30s /tmp/task040-jax-env/bin/python -m py_compile test/test_two_move_session_duel.py` | Initial compile passed before the shared-projector helper adjustment; shell real 0.02s. |
| `time -p timeout --signal=TERM --kill-after=5s 30s /tmp/task040-jax-env/bin/python -m py_compile test/test_two_move_session_duel.py` | Final module compile passed; shell real 0.02s. |
| `time -p timeout --signal=TERM --kill-after=5s 30s node reference/neutral-two-move-oracle.cjs --session-duels > /tmp/task046-native-session-duels.json` | Passed; eight battles, 66 turns. Shell real 0.41s. |
| Extract base oracle with `git show HEAD:reference/neutral-two-move-oracle.cjs`; invoke base and working oracle with no arguments, `--duels`, and `--sessions`; compare parsed JSON after removing only diagnostic `stack` fields | Initial exact-JSON comparison reported only the temporary filename in a draw diagnostic stack (0.58s). A diagnostic comparison located that field (0.56s). The rerun passed all three modes with stack fields alone normalized (1.65s); all RNG, phase, and semantic fields compared exactly. |
| `time -p timeout --signal=TERM --kill-after=5s 30s env JAX_PLATFORMS=cpu /tmp/task040-jax-env/bin/python -m unittest test.test_two_move_session_duel -v` | First attempt failed at 0.77s because the test assumed native terminal turn was one greater than the number of resolved turns; the fixture and existing v4 contract show equality. Second attempt failed at 2.86s because it assumed a terminal current view has no events; the accepted projector includes the current winner outcome. After correcting those assertions, a run passed 1/1 in 2.514s (shell real 2.91s). After sharing the static-audience projector helpers, the final run passed 1/1 in 2.564s (shell real 2.99s). |
| `time -p timeout --signal=TERM --kill-after=5s 30s` state JSON/invariant script | Passed accepted count/history, active owner, review checkpoints, G0-only milestone, null GPU block, and Task046 deferral checks; shell real 0.02s. |
| `time -p timeout --signal=TERM --kill-after=5s 30s` Python wrapper running `git diff --check` and whitespace checks for all six changed files | Passed; shell real 0.02s. A nested-shell wrapper first exited 2 before checks due quoting; this command ran the checks successfully. |

No simulator, session API, numerical engine, or existing test was edited. The
known-slow full v4 boundary module was not run; its exact raw user-session
command, source blobs, reason, and `acceptance_dependency: false` are recorded
in `deferred_validation` in `docs/ORCHESTRATION_STATE.json`.

### Source identity and review boundary

Source base is the full `3abd881cd1105abd162c5d190cc28b129926550e` revision.
Final changed source files:

| File | Git blob | SHA-256 |
|---|---|---|
| `reference/neutral-two-move-oracle.cjs` | `2958da2ecdae8dff7e1f0a5f236208e3741bff84` | `99c46f5af81e67dcd6a0c453432cc4966185e87677dc9e7ba37b7fa536612274` |
| `test/test_two_move_session_duel.py` | `a53da02e9bff63a20ea1cd4ac2b6a11f03f6d1f5` | `e5999bf4c11c3100a79be338eaa8452d02277a69537237aaad7f296ad24f50bb` |

Changed paths are those two source/test files plus this handoff,
`docs/TASKS.md`, `docs/ORCHESTRATION_STATE.json`, and
`docs/neutral-two-move-variant.md`. The accepted count is 46, baseline 40, next
checkpoint 50, G0 remains the only accepted milestone, and GPU block remains
null. Primary `/root` reviewed the frozen oracle and full CPU test, found no
required code correction, and accepted Task046 on 2026-11-05. This does not
claim Astra review or milestone approval. No unresolved native/session semantic
mismatch is known. No commit or push was made.
