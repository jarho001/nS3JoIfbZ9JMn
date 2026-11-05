# Task048 — restricted neutral Mega Punch Mew turn

**Status:** accepted complete
**Owner:** `/root/task048` (Luna/xhigh implementer)
**Accepted by:** `/root` on 2026-11-05
**Source base:** `c88334ceaab3e794db615643789050914c4aceb3`
**Accepted count:** 48 (review baseline 40; next checkpoint 50)
**Milestone / GPU:** G0 remains the only accepted milestone; GPU is not required.

## Scope

Add a separately versioned pure-JAX normal turn for two directly initialized
Mew using Mega Punch as each actor's only move. The supported boundary requires
positive, distinct speeds and the tagged v5/Mega Punch state. The implementation
is a restricted development slice and makes no full-format, generator, move-pool,
tie, session, reset, or team-generator support claim.

Owned paths are limited to:

- `simulator/neutral_mega_punch.py`
- `simulator/__init__.py`
- `reference/neutral-mega-punch-oracle.cjs`
- `test/test_neutral_mega_punch.py`
- `docs/neutral-mega-punch-variant.md`
- `docs/TASK048.md`
- `docs/TASKS.md`
- `docs/ORCHESTRATION_STATE.json`

No existing numerical/session/oracle/test files, configuration, or scheduler
files are in scope.

## Contract and acceptance criteria

- Pin original Gen 5 Dex Mega Punch data: Normal, Physical, power 80, accuracy
  85, base PP 20 / max PP 32, critical ratio 1. Use the pinned
  `gen5randombattle` rules with direct Synchronize / Hardy / perfect-IV /
  itemless Mew sets and bypass team generation.
- Export `VARIANT_ID = 5`, variant name
  `gen5-neutral-mega-punch-mew-1v1-v5`, `make_mega_punch_state`,
  `transition_mega_punch`, and `project_mega_punch_view`. A fixed-layout
  `BattleState` tagged with variant 5 and move ID 5 is acceptable. Preserve the
  established two-choice contract: sole action 1, pending 0, negative invalid,
  unknown positive unsupported; prior variants and speed ties fail closed.
- Validate structure before runtime statuses and validate public annotations
  via `typing.get_type_hints`. Supported state uses `_supported_common` for
  move ID 5 / max PP 32, variant 5, and distinct positive speeds.
- First actor is faster. Every attempted move spends one PP and consumes native
  `random(100)`. Rolls 85–99 miss; misses consume no critical or damage draw and
  do no damage. A surviving opponent acts after a miss. Hits consume
  `random(16)` for critical then `random(16)` for damage, using accepted integer
  damage helpers for power-80, non-STAB neutral Normal damage against Psychic.
  A first-action KO skips the second actor's PP and draws. Preserve terminal,
  turn, and winner rules; a final-PP nonterminal result is unsupported and does
  not implement Struggle. Draw traces have six fixed slots with a packed actual
  prefix, explicit bounds/contexts/pre/post limbs, and no holes across a miss.
  Event rows distinguish attempted action and hit/miss, with at most two rows.
- Per-recipient typed views expose only own exact HP/PP/stats/legal action plus
  recipient-visible action/miss/damage/critical/faint/turn/outcome. They contain
  no trusted state, RNG, opponent PP, or concealed exact HP/damage. Preserve
  native `-miss` actor and target in canonical per-recipient event comparison.
  Misses retain action order, move attempt, and target while masking damage,
  target HP, faint, and critical fields.
- The new local native oracle reuses the existing readiness, snapshot, and
  observed-draw instrumentation without replacing the pinned PRNG. It asserts
  upstream commit `1242c1e2d9763ba5b85d33f13aecb89bae19ae3d`, original Gen 5 Dex
  move facts, live initialized snapshots/requests, and actual `makeChoices`.
  Use a small set of deterministic cases for both initiative directions,
  misses by each actor, a normal hit/critical, first-action KO, and a last-PP
  miss/continuation boundary. A last-PP fixture uses a live move slot and
  `battle.makeRequest('move')`, asserting no RNG change. Compare the actual
  ordered PRNG draws (bounds, contexts, values, and four-limb pre/post state),
  full numeric state, requests/legal masks, and each recipient's event view.
- A focused CPU test compiles once and exercises one heterogeneous
  `jit(vmap(transition_mega_punch))` batch containing native hit/miss results
  plus pending, invalid, unsupported, final-PP, and terminal lanes. Compare full
  result trees with expected native lanes and rejection state isolation. Also
  cover malformed structure, audience fail-closed behavior, annotation/layout
  contract, and cross-variant boundaries. Do not run the known-slow
  `test.test_neutral_two_move` suite or broad eager/JIT/vmap cross-products.
- Run only affected checks within 30 seconds per command / 90 seconds total.
  Record exact deferred commands, source identity, reason, and
  `acceptance_dependency: false`; recommend an explicit long-session run.
  Missing required evidence remains unaccepted. No GPU criterion applies.
- Do not self-accept, commit, or push. The primary `/root` reviews the source and
  evidence and owns final acceptance bookkeeping and commit decisions.

## Plan

1. Persist this contract and active ownership in both `active_task` state fields
   before source changes; leave accepted count at 47.
2. Check the pinned Dex facts and current reusable simulator/oracle boundaries.
3. Implement the isolated v5 turn, recipient view, native oracle, focused test,
   and variant note only in the owned paths.
4. Run the focused native differential/JAX test and required syntax/static
   checks within budget, then inspect the complete diff and report exact
   commands, elapsed times, source hashes, and any deferred evidence to `/root`.
5. Keep ownership through any primary-requested corrections. Only `/root`
   performs acceptance and final task-state bookkeeping.

## Progress

- Kickoff: active contract recorded before implementation. Trail CLI is absent
  from `PATH` and its noted workspace path is absent; `.trail/cache/INDEX.md`
  was checked and has no current neutral-simulator wiring card, so source
  fallback is used without claiming a graph check.
- Implementation: the v5 module, package exports, pinned native oracle, focused
  test, and variant note are present in the owned paths. The oracle asserts
  original Gen 5 Mega Punch data and six direct-set native cases. It reuses the
  guarded `assertReady`, `snapshot`, and `withObservedBattleDraws` exports from
  the existing neutral turn oracle. Native traces cover normal hit, critical,
  misses by both first actors, first-action KO, and a last-PP miss followed by
  the opponent's move. The last-PP native request is recorded as Struggle and
  the JAX after-state is normalized only to this declared unsupported phase.
- Final focused CPU command passed both test methods in 2.43 seconds:
  `time -p timeout --signal=TERM --kill-after=5s 30s env JAX_PLATFORMS=cpu /tmp/task040-jax-env/bin/python -m unittest test.test_neutral_mega_punch -v`.
  It compares native state, PP, RNG draws, requests, and both recipient event
  projections, then compares every result leaf in one 17-lane
  `jax.jit(jax.vmap(transition_mega_punch))` batch, including pending choice
  and phase, invalid, unsupported action/state, final-PP, cross-variant, tie,
  and terminal lanes. Recipient projections are derived from the actual
  compiled native lanes. Layout, public type hints, malformed inputs, package
  draw-context exports, and fail-closed audiences also pass.
- The final test result follows these corrected, bounded attempts (all commands
  used the same 30-second cap): initial unittest setup failed because the
  oracle treated a request-side summary as a live move slot (0.66s); direct
  oracle invocation confirmed that boundary assumption (0.28s); the corrected
  direct oracle generation passed with the pinned move facts and six scenarios
  (0.25s for Node, followed by a compact fixture inspection); the next focused
  test exposed the missing conditional state operand and the native
  `mega punch` event spelling (1.17s); the next exposed the PP-Up maximum of 32
  in the fixture (1.63s); the next exposed an overkill event's raw damage
  exceeding remaining HP, so the expected raw hit amount now uses an
  independent integer calculation from native stats and rolls (2.74s). The
  focused test then passed after implementation corrections (2.70s), after
  adding pending-choice/layout assertions (2.82s), after removing an extra
  eager transition call from the public-boundary test (2.41s), after binding
  projections to compiled lanes (2.38s), and on the final revision with
  package draw-context exports (2.43s). No timeout occurred.
- Clarification: the 2.70s run passed both tests (`OK`); the implementer's
  final message incorrectly grouped that passing run with the earlier failures.
  The eleven bounded validation attempts/probes total 19.47 seconds of elapsed
  command time. The known-slow v4 regression is separate and was not run.
- Runtime: Python 3.12.14, JAX/JAXLIB 0.12.0.1, NumPy 2.5.4, CPU device
  `cpu:0`; Node v24.22.1. GPU validation is not an acceptance criterion.
- Deferred optional regression: raw command
  `JAX_PLATFORMS=cpu /tmp/task040-jax-env/bin/python -m unittest test.test_neutral_two_move -v`.
  It is the known-slow prior v4 module explicitly excluded from this task's
  bounded checks; the focused v5 native/JAX parity test passes, so this old
  regression is not an acceptance dependency. See the matching
  `deferred_validation` entry for source base/blob identity and
  `acceptance_dependency: false`. Recommend running it in an explicitly
  requested long-test session.
- `python -m json.tool docs/ORCHESTRATION_STATE.json` and `git diff --check`
  passed after the final durable-state update. Final source blobs are recorded
  in the task's deferred-validation record. Primary `/root` accepted Task048
  on 2026-11-05 after focused actual-code/oracle/test/documentation review.
  Accepted task count is 48; no task is active. No commit or push was made by
  the implementer.
