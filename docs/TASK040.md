# Task040 — heterogeneous batched staged request sessions

Status: **Accepted** — 2026-11-05. Owner: `/root/task040`; accepted by
`/root` after the prospective Task040 checkpoint review.

## Scope

Add one focused CPU parity test for heterogeneous request-session lanes using a
single `jit(vmap(submit_choice))` compiled function. Each lane must retain its
own private commitments, RNG, requests, outcomes, and terminal state. Cover the
complete native `--session-duels` fixture lanes through real terminal states,
including tied and unequal speeds and both submission orders. Exercise
asynchronous waiting and resolution in a heterogeneous batch that also includes
an already-terminal lane or rejected duplicate/stale operations.

The test will reuse existing duel/session helpers where sound, compare resolved
lanes with their native fixture state, RNG trace, recipient requests and events,
and show that completion or rejection in one lane does not advance another.
Actor inputs remain audience projections only. This validates batch parity; it
does not add a mechanic or variant and makes no GPU, performance, or milestone
claim.

## Exclusions

- Simulator and oracle source changes, unless a concrete defect is first
  reported to the parent with reproducing evidence.
- GPU, performance, or throughput claims.
- Running known-long session/mechanics suites or broad regression matrices.
- Changing accepted task count, task history, review checkpoints, or approval
  records before parent acceptance.

## Acceptance criteria

1. A focused CPU test calls one shared compiled `jit(vmap(submit_choice))`
   function across heterogeneous independent sessions.
2. Complete native `--session-duels` fixture lanes cover tied and unequal
   speeds in both submission orders and reach their real terminal states.
3. The batch schedule includes a first-commit `WAITING` lane and a
   `RESOLVED` lane alongside a terminal lane or rejected duplicate/stale
   operation. Waiting lanes retain exact battle/RNG and update only their own
   commitment. Rejected/terminal lanes preserve the full session. These lanes
   emit no transition events or draws.
4. Each resolved lane matches its native fixture's state, exact RNG trace,
   recipient requests, and events. Completion or rejection in one lane cannot
   advance another; terminal lanes remain absorbing; commit bits and episode
   tokens are correct per lane.
5. Actor inputs are audience projections only. No simulator or oracle source
   changes are made without first reporting a concrete defect to the parent.
6. Run only meaningful fast checks under the per-command 30-second and
   per-task 90-second test budget. Record exact commands, counts, elapsed time,
   deferred optional long-session command/source identity, and any unresolved
   concern. Keep Task040 unaccepted and count 39 before parent acceptance.

## Plan

1. Check the context graph for request-session/test coverage; because the
   current graph may omit these files, use only exact relevant source spans as a
   documented fallback.
2. Inspect the existing duel parity test, request-view test, request-session
   API, and fixture shape before designing the batch schedule.
3. Add the focused batched parity test and update the API-contract paragraph
   documenting heterogeneous batched staged sessions.
4. Run the smallest meaningful CPU test command within the strict timeout,
   inspect the diff, and report evidence and open concerns to the parent.

## Validation and progress

- Source base: `b30bd77dff9e304d154988e5610838858e277356`; clean worktree at
  kickoff. No simulator or oracle source file changed.
- Trail could not be used: the graph index had no request-session or staged
  duel test cards, `trail` was absent from `PATH`, and
  `tools/trail/node_modules/.bin/trail` was unavailable. Exact-span source
  inspection was the justified fallback. I inspected the request-session
  NamedTuples and `submit_choice`/projection APIs, existing compiled scalar
  duel helpers, the existing simple `jit(vmap(submit_choice))` test, request
  view assertions, and the `generateStagedSessionDuelFixture` shape.
- Native fixture command passed:
  `timeout --signal=TERM --kill-after=5s 30s node reference/neutral-turn-oracle.cjs --session-duels > /tmp/task040-session-duels.json`.
  It completed in about 0.7s and generated four actual terminal lanes: tied
  7-turn battles and unequal 8-turn battles, each with p1 and p2 submitting
  first; p2 won each fixture.
- The prescribed `/opt/jax-env/bin/python` interpreter was absent. The
  initial focused-test attempt exited 127 with `env: '/opt/jax-env/bin/python':
  No such file or directory`:

  ```sh
  timeout --signal=TERM --kill-after=5s 30s env JAX_PLATFORMS=cpu /opt/jax-env/bin/python -m unittest test.test_request_session_batch -v
  ```

  The available interpreter probe `python3 -c 'import jax, jaxlib'` also
  failed with `ModuleNotFoundError: No module named 'jax'` under Python
  3.12.14 (the import stopped before checking `jaxlib`). The repository
  declares CPU JAX in `requirements-jax-cpu.txt`, so one temporary CPU
  environment was installed under `/tmp` with `uv` under a 30-second timeout.
  Setup exited 0 with Python 3.12.14 and JAX/JAXLIB 0.12.0.1; no repo
  dependency pins or configuration changed.

  Exact recovery command:

  ```sh
  env UV_CACHE_DIR=/tmp/task040-uv-cache timeout --signal=TERM --kill-after=5s 30s bash -c 'uv venv --python /opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python3 /tmp/task040-jax-env && uv pip install --python /tmp/task040-jax-env/bin/python -r requirements-jax-cpu.txt'
  ```
- Focused CPU command passed, 1 test in 6.725s:

  ```sh
  timeout --signal=TERM --kill-after=5s 30s env JAX_PLATFORMS=cpu /tmp/task040-jax-env/bin/python -m unittest test.test_request_session_batch -v
  ```
- Python AST and orchestration JSON parsing passed. `git diff --check` passed.
- Source blob identities at the focused test run: base commit
  `b30bd77dff9e304d154988e5610838858e277356`; batch test
  `9dcec6ae2c0b110e449aba0bce8c69f6338f829e`; request-session API
  `4a94e7a4eaec59d574dab5a5274a7875c4ba1e0e`; duel parity helpers
  `fa3cb5bb17742b42c95f22dccbf919d7aa7e4270`; base session helpers
  `b9b9eb65816229b3bf40a55ef3fecd326b06c9cd`; request-view test
  `a6611019117f52f1306bd92ad85d1c5811d2e96b`; native oracle
  `da3a9c5f0166e006da8bbf1cf676eedabb234700`.
- Optional long-session regression remains deferred. These suites exceed the
  per-command 30-second cap; exact source blobs and non-blocking status are
  recorded in `deferred_validation` in the state:

  ```sh
  JAX_PLATFORMS=cpu python -m unittest test.test_request_session test.test_neutral_tackle test.test_neutral_duel test.test_neutral_tackle_ties -v
  ```
- GPU and performance checks were not required or run. No simulator/oracle
  source changed and no milestone claim was made. The accepted count is 40,
  baseline 40, and next checkpoint 50. The optional long-session regression
  above remains deferred and is not an acceptance dependency.

## Acceptance

Primary `/root` accepted Task040 on 2026-11-05 against the complete delegated
criteria and implementation/evidence package. The prospective checkpoint
review was independently accepted by `/root/t040_global`; report:
[T040-global.md](reviews/T040-global.md). The report records no required
corrections or blockers, independently inspected the candidate and supporting
source, and confirms the limits of this restricted v2 parity evidence. Task040
is accepted as numbered task 40; the regular review baseline is 40 and next
checkpoint is 50. G0 remains the only accepted milestone; no GPU, performance,
training, or full-format support claim is made.

Reviewed source base: `b30bd77dff9e304d154988e5610838858e277356`. The final
commit hash and candidate tree are not recorded because no task commit exists
yet. The candidate test and dependency source blob identities are listed
above; the full independent review evidence is preserved in
[`reviews/T040-global.md`](reviews/T040-global.md).
