# Task041 trusted terminal-session reset

## Scope and status

Implement one pure, JAX-compatible trusted-coordinator API that resets an
already-terminal restricted-v2 `RequestSession` onto a caller-supplied fresh
prepared `BattleState`. The accepted reset clears private commitments and
increments only the local episode exactly once. It does not initialize a
battle, run a transition, or draw RNG.

Owner: `/root/task041`. Source base: `019b601f23cc35777889b24329173adada045545`.
Status: accepted by primary `/root` on 2026-11-05. Accepted task count is 41;
the regular review baseline remains 40 and the next checkpoint remains 50.

## Acceptance criteria

- Export `reset_request_session(session, battle, expected_episode)` and a
  fixed-shape `SessionReset(session, status)` result. Add explicit reset and
  not-terminal statuses while reusing stale, unsupported, and overflow
  statuses where applicable.
- Validate existing session and replacement battle shapes/dtypes and the
  expected local episode scalar at the API boundary. A replacement must be a
  supported, nonterminal restricted-v2 request state.
- Apply status precedence: stale expected episode; invalid existing metadata
  or unsupported replacement; old session not terminal; max-episode overflow;
  accepted reset. Accepted output uses the supplied battle exactly, zeros both
  commitments, and sets old episode plus one. The old terminal session must
  already have both commitments clear; terminal plus any pending choice is
  invalid metadata (`SESSION_UNSUPPORTED`). A valid live session with a pending
  choice remains `SESSION_NOT_TERMINAL`. Rejection preserves the complete input
  session and performs no transition or RNG draw.
- Add a focused CPU test using guarded native `--session-duels` terminal
  snapshots and initial requests. Verify the prepared state/RNG, projected
  request, monotonic episode, cleared commitments and stale pre-reset submit.
- Cover mixed `jit(vmap(reset_request_session))` lanes for reset, live,
  pending/corrupt metadata, stale token, unsupported replacement and overflow;
  compare complete session PyTrees on every rejection and verify lane isolation.
- Cover malformed shape/dtype and the last safe episode increment with focused
  boundary cases. Do not add a native oracle mode or synthetic native outcome.
- Update the public API paragraph and keep task/state bookkeeping compact and
  truthful. No new mechanics, variants, RNG initialization algorithm, teams,
  policies/training/rollout infrastructure, performance/GPU/long tests or
  milestone acceptance are in scope.

## Plan and progress

1. Inspect the session and validator contracts, native fixture helpers, docs,
   task index and current orchestration state.
2. Record this active task before implementation.
3. Implement the reset result/API and package exports; document its boundary.
4. Add one fixture-backed scalar/API test and one heterogeneous compiled batch
   test with malformed and episode-boundary cases.
5. Run only affected checks under the 30-second per-command / 90-second total
   budget, inspect the complete diff, and report evidence to the primary for
   acceptance. Do not commit or push.

Implementation is complete. The public API is exported from `simulator`,
replacement validation uses the existing restricted-v2 request predicate, and
the focused reset tests cover native initial and terminal fixture snapshots,
stale submission rejection, heterogeneous compiled lanes, malformed shapes
and dtypes, and the last safe episode increment. The primary review correction
requires terminal commitments already be clear: a terminal `[ACTION_TACKLE, 0]`
session now returns `SESSION_UNSUPPORTED` unchanged, while a live pending
session still returns `SESSION_NOT_TERMINAL`. The batch also checks an already
terminal replacement is unsupported and retains stale-token precedence. The
test wraps the guarded oracle's terminal state in a local `RequestSession` with
the fixture turn count; it does not claim native session lifecycle or native
`rqid` parity.

Focused validation passed on CPU:

```sh
timeout --signal=TERM --kill-after=5s 30s env JAX_PLATFORMS=cpu /tmp/task040-jax-env/bin/python -m unittest test.test_request_session_reset -v
```

Final result after correction: 3 tests passed in 2.659s (command elapsed 3.009s), using Python
3.12.14, JAX/JAXLIB 0.12.0.1. No GPU evidence was required; the long regression
remains deferred below.

Primary acceptance followed a full diff/evidence review and the requested
bounded correction. The corrected implementation requires terminal commitments
already be clear; the compiled batch proves terminal `[ACTION_TACKLE, 0]` is
`SESSION_UNSUPPORTED` and unchanged, live pending remains
`SESSION_NOT_TERMINAL`, terminal replacement is unsupported, and stale-token
precedence is preserved. Accepted source blobs are
`simulator/request_session.py` `ffdae56f4c8a0495b64324ff7a0e10631c97dcc2`,
`simulator/__init__.py` `563ad9ac7ce6f346798b2f2757bc177267ed5249`, and
`test/test_request_session_reset.py` `549d164b0548db229a3f69251db9a79a750f6780`.
The primary review is not an independent Astra review; no GPU or milestone
acceptance is claimed.

The Trail CLI is absent from `PATH` and `/workspace/tools/trail/node_modules/.bin`,
and the cache has no simulator/request-session wiring card. I used the current
source spans as the documented fallback; no Trail graph refresh was possible
without setting up the missing tool.

The known-long request-session/mechanics regression remains deferred and is
not an acceptance dependency:

```sh
JAX_PLATFORMS=cpu python -m unittest test.test_request_session test.test_neutral_tackle test.test_neutral_duel test.test_neutral_tackle_ties -v
```

It exceeds the 30-second per-command task budget. Relevant changed blobs at
review are `simulator/request_session.py` `ffdae56f4c8a0495b64324ff7a0e10631c97dcc2`,
`simulator/__init__.py` `563ad9ac7ce6f346798b2f2757bc177267ed5249`, and
`test/test_request_session_reset.py` `549d164b0548db229a3f69251db9a79a750f6780`.
