# Task047 trusted terminal reset for v4 request sessions

## Scope and status

Add one trusted, pure JAX coordinator API for resetting a terminal
`TwoMoveRequestSession` onto a caller-supplied supported nonterminal v4 state.
Accepted reset copies the supplied state exactly, clears pending choices, and
increments the old local episode once. It never samples RNG or runs a turn.
The v2 reset and the v4 numerical transition remain unchanged.

Owner: `/root/task047` (Luna/xhigh). Source base: `3589103a82e7f08e50577cf78796eb1e4f29a905`.
Status: accepted by primary `/root` on 2026-11-05. Accepted task count is 47;
review baseline 40, next checkpoint 50, G0 only accepted milestone, GPU block
null. This does not claim an Astra review or milestone acceptance.

## Acceptance criteria

- Export a fixed-shape v4 reset result and API from `simulator`, reusing the
  existing reset, not-terminal, stale, unsupported, overflow and episode-limit
  constants.
- Validate session, replacement state and expected-episode shape/dtype before
  runtime status selection. Runtime precedence is stale token; invalid session
  metadata or unsupported replacement; old battle not terminal; episode
  overflow; accepted reset. The old terminal session must have no pending
  choices, and each held action must have positive PP where applicable.
- Preserve the entire input session on rejection. An accepted reset copies the
  supplied battle exactly, clears both choices, increments the old local
  episode once, and does not draw RNG or emit events.
- Use the existing guarded native `--sessions` fixtures. Check reset against
  actual initial/native request states and a real first-action-KO terminal
  snapshot; verify a stale pre-reset submit is rejected, then the new token
  resolves the matching native turn with exact state, PP, RNG, events and draw
  parity.
- Cover one heterogeneous `jit(vmap(reset))` batch with multiple distinct
  accepted native replacements and numeric lane checks, rejections preserving
  each full input session, fresh actor request parity, malformed inputs,
  supported partially depleted replacement, pending terminal metadata, and
  the last safe episode increment.
- Update the v4 API contract. Do not edit the v2 API, numerical engine,
  oracle, fixture modes, training/rollout/history infrastructure, or claim new
  mechanics, RNG initialization, GPU/performance results, or milestone status.

## Plan

1. Read project instructions, compact state, relevant task contracts and
   existing v2/v4 reset/session validation; check the guarded native fixture.
2. Record this bounded contract in the handoff and state before implementation.
3. Add the result/API and public exports; add focused fixture-backed scalar and
   compiled heterogeneous tests; update the v4 API paragraph.
4. Run only the focused reset test and compact static checks within the
   per-command 30-second and total 90-second limits, then report a complete
   source/evidence package for primary review. Do not self-accept or commit.

## Progress

- Start: repository is clean at the stated source base. Trail CLI is absent at
  `PATH` and `/workspace/tools/trail/node_modules/.bin/trail`; the cache has no
  session wiring card. Used targeted source fallback after checking the cache.
- Implementation is complete and accepted by primary `/root` on 2026-11-05.
  The v4 API is exported from `simulator`; it validates both v4 states and the
  expected local episode before status selection, then applies stale,
  unsupported, not-terminal, overflow and reset precedence. Rejection
  preserves the input PyTree.
- The focused test uses the guarded native `--sessions` fixture at upstream
  commit `1242c1e2d9763ba5b85d33f13aecb89bae19ae3d`. It wraps the actual native
  first-action-KO state in a trusted local session; this does not claim native
  reset or native `rqid` lifecycle parity. Accepted reset checks compare the
  exact prepared native snapshot, both requests/legal masks and cleared actor
  bits. The old token is then rejected without events/draws, and the new token
  resolves the matching `tackle-pound-unequal-first-p1` turn with native state,
  PP, RNG, event, draw and next-request parity. The compiled batch has 13 lanes,
  including distinct ordinary, partially depleted and tied native replacement
  states, live-session rejection, stale precedence, unsupported variants and
  terminal replacement, overflow, held Tackle/Pound commitments, corrupt
  metadata, negative episode, and the last safe increment. Rejected lanes are
  compared as complete session PyTrees; accepted lanes are checked numerically
  and against both native actor requests.
- CPU validation used Python 3.12.14, JAX/JAXLIB 0.12.0.1 and Node v24.22.1.
  Final changed Python blob IDs at the reviewed source tree are:

  - `simulator/two_move_session.py`: `c8c4c01ffa31934b17dfe65d56714cba18e27aa5`
  - `simulator/__init__.py`: `2c317bb0084fb0d2774e02d118cfffc321a92e2a`
  - `test/test_two_move_session_reset.py`: `96c525cbe3f88cafa881ea10b6af4c7addda5b76`

  Exact focused command, run four times under the 30-second cap:

  ```sh
  time -p timeout --signal=TERM --kill-after=5s 30s env JAX_PLATFORMS=cpu /tmp/task040-jax-env/bin/python -m unittest test.test_two_move_session_reset -v
  ```

  1. **Failed**, 3.26 seconds elapsed (2.823 seconds unittest time): the
     rejection assertion loop also included the accepted last-safe-episode
     lane. Corrected the test lane range; this was a test assertion error.
  2. **Passed**, 3.29 seconds elapsed (2.864 seconds unittest time), after the
     assertion correction.
  3. **Passed**, 3.26 seconds elapsed (2.823 seconds unittest time), after
     adding held-choice PP and negative-episode cases.
  4. **Passed**, 3.24 seconds elapsed (2.850 seconds unittest time), after
     pinning the fixture upstream commit and removing unused imports. Total
     shell real time across all attempts: 13.05 seconds. All runs completed;
     none timed out.
- `docs/ORCHESTRATION_STATE.json` records the known-slow optional regression as
  deferred with `acceptance_dependency: false`:

  ```sh
  JAX_PLATFORMS=cpu /tmp/task040-jax-env/bin/python -m unittest test.test_neutral_two_move -v
  ```

  Its source base and changed source blobs are in that state entry. The focused
  Task047 acceptance check passed, and Task046 already has full staged-session
  battle parity; the full v4 boundary module is not required for this reset
  slice.
- JSON parsing, local Markdown-link checks and `git diff --check` passed. There
  is no GPU acceptance criterion and no GPU claim. Primary acceptance followed
  focused review of the source, test and documentation against the delegated
  criteria. No implementation concern remains. This work has not been
  committed.
