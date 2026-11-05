# Task042 versioned neutral Pound turn slice

## Scope and status

Implement one additional ordinary-move simulator turn slice: a separately
versioned `gen5-neutral-pound-mew-1v1-v3` variant in which both Mew have only
Pound and each side submits one legal action per turn. Pound is a neutral
Normal-type physical move against Psychic-type Mew, with no STAB. Preserve the
accepted Gen 5 accuracy, critical-hit, damage-roll LCG, event, projection,
action-order, and speed-tie machinery while using Pound's pinned base power.

Owner: `/root/task042`. Source base: `f56f575b9650ae5642bb41e9a49dec2b7852b8d4`.
Status: accepted by primary `/root` on 2026-11-05. Accepted task count is 42;
regular review baseline remains 40 and next checkpoint remains 50. This is
primary task acceptance only; no Astra review or milestone acceptance is
claimed. GPU evidence is not required.

Owned paths: `simulator/neutral_pound.py`, minimal neutral-action and action
callback changes in `simulator/neutral_tackle.py` and
`simulator/neutral_tackle_ties.py`, package exports in `simulator/__init__.py`,
`reference/neutral-pound-oracle.cjs`, guarded helper exports appended to
`reference/neutral-turn-oracle.cjs` if needed, `test/test_neutral_pound.py`,
this handoff, `docs/TASKS.md`, `docs/ORCHESTRATION_STATE.json`, and
`docs/neutral-pound-variant.md`.

## Acceptance criteria

- Verify Pound's pinned Gen 5 move identity, base power, accuracy, PP, and
  default critical ratio from the repository's pinned Showdown data. Give v3 a
  distinct variant ID; enforce v3/move-ID boundaries without changing the
  existing v1/v2 support predicates or their Tackle behavior.
- Expose explicit Pound state, turn, and recipient-view APIs. Both actors have
  only Pound and one legal action. Reuse the fixed battle state, exact native
  arithmetic/order/projection, and the existing tie trace shape and phase
  machinery. The ordinary action path consumes the same accuracy, critical,
  and damage draws as native. Do not relabel Tackle states or outputs as Pound.
- Add a guarded native oracle that initializes actual pinned Gen 5
  `Gen5RandomBattle` battles using direct one-Mew, Synchronize, Hardy,
  perfect-IV, itemless, Pound-only sets. Keep normal engine initialization;
  observe original PRNG values, recipient requests/events, damage and turn
  effects. Do not patch damage or RNG.
- Fixture focused ordinary turns for both unequal-speed actor orders, tied
  speed with trace-verified native draws, a critical hit, and a first-action
  knockout. Compare each relevant state, event, trace, and both recipients'
  requests/projections, including the proven 40-power damage result.
- Exercise compiled Pound transitions and projected views, plus a
  representative heterogeneous `jit(vmap(...))` batch that demonstrates
  supported/unsupported or pending lane isolation. Cover invalid, pending,
  and terminal no-advance behavior. Show v1/v2 continue to reject v3 states
  and v3 rejects Tackle states.
- Run the focused Pound test and one affected Tackle regression within the
  repository's command and total-time budgets. Record exact commands/results,
  source identity, environment, changed files, and any deferred long checks.
  Inspect the full diff and return evidence to the primary; do not self-accept.

## Exclusions

No mixed move sets or move choice, teams/rosters/switches, status, items,
weather, Struggle, v3 reset/session integration, policies/training/reporting,
duel runner, GPU/performance/long-suite work, or milestone claims. Do not
weaken v1/v2 boundaries or parity evidence if an oracle mismatch appears.

## Plan and progress

1. Inspect the accepted turn/tie implementation, guarded native oracle,
   pinned move data, and relevant tests/docs. Trail CLI and simulator wiring
   cards are unavailable in this workspace, so use targeted source-span
   inspection as documented fallback; do not install the toolchain.
2. Implement separately versioned Pound APIs with a minimal shared neutral
   action/callback seam that leaves Tackle defaults and semantics intact.
3. Add a Pound-only pinned native oracle and fixture-backed focused parity
   tests for the acceptance cases, including compiled batch lane isolation.
4. Update the variant documentation and active task bookkeeping, run only
   affected fast checks under the 30-second per-command / 90-second total test
   budget, inspect the complete diff, then report evidence to `/root`.

## Implementation and evidence

Implementation was accepted by primary `/root` on 2026-11-05. The public v3 API uses
variant ID 3, Pound move ID 1, action ID 1, base power 40, and a maximum PP
bound of 56. The pinned native oracle verifies Pound's Gen 5 Dex facts (Normal,
Physical, 40 power, accuracy 100, base PP 35, default critical ratio 1) and
captures five actual normal-request turns: p1-first, p2-first, a tied turn with
the full native 16-draw trace, an actual critical using seed `[9, 2, 3, 4]`,
and a first-action KO. The v3 API validates state structure and preserves the
v1/v2 move/variant rejection boundaries.

Focused CPU evidence used Python 3.12.14, JAX/JAXLIB 0.12.0.1, backend CPU:

```sh
timeout --signal=TERM --kill-after=5s 30s env JAX_PLATFORMS=cpu /tmp/task040-jax-env/bin/python -m unittest test.test_neutral_pound.NeutralPoundTests.test_pinned_pound_turns_match_native_state_rng_events_and_both_views -v
```

1 test passed in 6.921s. It compares the full relevant state, exact native draw
count/bounds/contexts/values/limbs, action order, criticals, damage, both
current recipient views, next requests, and compiled projections. A separate
hand formula verifies Pound damage uses power 40, distinct from Tackle's 50,
and verifies KO overkill.

```sh
timeout --signal=TERM --kill-after=5s 30s env JAX_PLATFORMS=cpu /tmp/task040-jax-env/bin/python -m unittest test.test_neutral_pound.NeutralPoundTests.test_compiled_heterogeneous_lanes_are_independent -v
```

1 test passed in 13.664s. The compiled seven-lane batch covers both unequal
orders, ties, KO, unsupported cross-variant, pending, and terminal lanes;
rejected lanes preserve the complete state and draw/event traces, and a lane
permutation preserves outputs.

```sh
timeout --signal=TERM --kill-after=5s 30s env JAX_PLATFORMS=cpu /tmp/task040-jax-env/bin/python -m unittest test.test_neutral_pound.NeutralPoundTests.test_variant_boundaries_no_advance_and_structure_validation -v
```

1 test passed in 10.244s. Invalid choices, pending, terminal, malformed
structure, and cross-variant calls do not advance. Both existing Tackle
variants reject v3 Pound state, and v3 rejects Tackle state.

```sh
timeout --signal=TERM --kill-after=5s 30s env JAX_PLATFORMS=cpu /tmp/task040-jax-env/bin/python -m unittest test.test_request_session_duel -v
```

1 existing Tackle session-duel regression passed in 2.634s, exercising the
default v2 callback path through native turns to terminal. Total final focused
test execution was 33.463s across four commands.

## Test-attempt accounting

The earlier full-module command was:

```sh
timeout --signal=TERM --kill-after=5s 30s env JAX_PLATFORMS=cpu /tmp/task040-jax-env/bin/python -m unittest test.test_neutral_pound -v
```

It ran 3 tests in 23.715s: the batch and boundary tests passed, while the
native parity test errored before its transition comparisons. The harness
looked for a numeric `damage` field in a recipient-projected semantic event;
that event exposes its HP `condition`. The targeted parity test repeated this
same incorrect assumption once (1 error in 0.276s) before it was corrected.
Its next targeted attempt failed an assertion that the native request starts at
PP 35 (1 failure in 0.288s); the pinned Gen 5 battle initializes the request
at maximum PP 56, so that harness assertion was corrected too. These were test
expectation errors; no native/simulator mismatch was found.

The test module first had a loader-only import failure because draw constants
were imported from the package root. It reported `ImportError: cannot import
name DRAW_ACCURACY from simulator` and 1 loader error in 0.000s; the constants
were then imported from `simulator.neutral_tackle`.

The targeted native parity attempts used this command:

```sh
timeout --signal=TERM --kill-after=5s 30s env JAX_PLATFORMS=cpu /tmp/task040-jax-env/bin/python -m unittest test.test_neutral_pound.NeutralPoundTests.test_pinned_pound_turns_match_native_state_rng_events_and_both_views -v
```

After the two harness assumptions above were corrected, this command passed
once in 6.860s and passed again in 6.921s after adding explicit fixed trace-
shape assertions. Along with the
13.664s batch test, 10.244s boundary test, and 2.634s Tackle regression, all
final focused methods passed.

Summing every unittest-reported duration, including the failed full-module and
targeted attempts, gives 64.602s against the 90s test-execution budget, leaving
25.398s. The initial loader error reported 0.000s. Each executed test command
was capped at 30s and completed below that limit. These are unittest-reported
durations; aggregate shell wall time was not separately recorded. No tests were
rerun to prepare this accounting, and no simulator/oracle mismatch remains.

`node reference/neutral-pound-oracle.cjs` emitted the five-case fixture. These
static checks passed:

```sh
node --check reference/neutral-pound-oracle.cjs
node --check reference/neutral-turn-oracle.cjs
python3 -m py_compile simulator/neutral_tackle.py simulator/neutral_tackle_ties.py simulator/neutral_pound.py simulator/__init__.py test/test_neutral_pound.py
git diff --check
python3 -m json.tool docs/ORCHESTRATION_STATE.json
```

GPU evidence is not required. Trail CLI and simulator wiring cards were
unavailable; targeted source-span inspection followed the documented fallback.

The known-long optional request-session/Tackle regression was not run:

```sh
JAX_PLATFORMS=cpu python -m unittest test.test_request_session test.test_neutral_tackle test.test_neutral_duel test.test_neutral_tackle_ties -v
```

It exceeds the 30-second per-command routine limit and is not an acceptance
dependency. The reviewed source base is `f56f575b9650ae5642bb41e9a49dec2b7852b8d4`.
Final blob identities for the implementation, oracle, focused test, and variant
doc:

| File | Blob |
|---|---|
| `simulator/neutral_tackle.py` | `aecd2641c0e64481d5f54ec550d4a93831fa6fbf` |
| `simulator/neutral_tackle_ties.py` | `efed081d1ca3e1e2bb93eed499155d93f98a9ced` |
| `simulator/neutral_pound.py` | `2f7fc9f9b33afcd2a729ccf2ee25b47c4c754887` |
| `simulator/__init__.py` | `e5ed2f92bff9742afd3275c1c789166e012882b8` |
| `reference/neutral-pound-oracle.cjs` | `b541f9bbf727ecf3967f8e079425d08a0ab81de7` |
| `reference/neutral-turn-oracle.cjs` | `693c4a8393baf965cee47f615c8f9dbbdbde1239` |
| `test/test_neutral_pound.py` | `bd035ad681b94c27f6a14e8503397eb52ed8addd` |
| `docs/neutral-pound-variant.md` | `9cea4d6ca315960376441835e126a37a110734c5` |

The accepted task count is 42; baseline 40 and next checkpoint 50 remain
unchanged. No Astra review or milestone acceptance is claimed. No commit or
push was made.
