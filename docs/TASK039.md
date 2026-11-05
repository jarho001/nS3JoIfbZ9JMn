# Task039 — complete staged-session battle parity

Status: **Accepted** — 2026-11-04. Owner `/root/task039`; primary `/root`.
Source base: `f88f2d7c24adb23628d9b316c2d9d2e1db1d026d`.

## Scope and acceptance

Validate complete restricted v2 battles through the staged request-session API against pinned native Showdown, extending the existing two-turn evidence to terminal outcomes.

Exclusions: new battle mechanics/variants, actor history storage, policy/training/tournaments, GPU/performance/long suites and milestone acceptance. Existing battle kernels and session submission behavior should remain unchanged unless a concrete parity failure requires an authorized correction.

- Add an explicit guarded complete-session oracle mode using actual separate Battle.choose submissions and the unchanged pinned PRNG. Cover tied and unequal-speed battles in both first-submitter orders, reaching real native terminal outcomes within a declared bound. Preserve existing fixture modes.
- A focused CPU differential uses the actual session API from initial request to terminal, reusing one compiled submission function. Compare each initial/current actor request and legal action, local episode/own commitment, first-commit freeze/no draws and opponent-view invariance, second-commit state/HP/PP/turn/phase/winner and exact RNG draw bounds/context/values/limbs, recipient events, next requests and terminal no-action behavior. Check paired submission-order invariance and final pending-choice clearing; never expose trusted traces to actor views.
- Run relevant fast checks only: <=30s per command and <=90s total tests. Do not execute known-long suites/probes or retry timed-out checks. Missing required parity evidence remains pending. Record useful optional long regression with exact command/source blobs.
- Update API docs and compact handoff/state; preserve history/review counters until primary acceptance. Refresh/check Trail, inspect diff and return one complete review package. No milestone or GPU acceptance claim.

## Progress

Primary accepted the complete implementation and focused native parity evidence. `reference/neutral-turn-oracle.cjs --session-duels` adds a guarded full-battle fixture that uses separate native `Battle.choose` calls and unchanged PRNG values. It covers tied and unequal speeds with p1 and p2 each submitting first, records every turn through the real native terminal winner, and preserves the existing modes. `test/test_request_session_duel.py` compares the compiled session API from initial request through terminal, including partial-commit freeze, current requests, local episode and own-commit projection, state, exact draw trace, recipient events, next requests, order invariance, cleared commitments, terminal no-action masks, and terminal rejection.

The native sample uses seed `[1, 2, 3, 4]`; tied battles end on turn 7 and unequal-speed battles on turn 8, with p2 winning each pair. Continuing turns consume 16 tied-speed draws or 6 unequal-speed draws. The one-action terminal turns consume 7 and 3 draws respectively. These counts come from the live pinned oracle; no RNG values are supplied or replaced.

### Validation

- `timeout --signal=TERM --kill-after=5s 30s node reference/neutral-turn-oracle.cjs --session-duels > /tmp/task039-session-duels.json` — passed in 2.5s; generated four complete native battles, each under the 56-turn bound.
- `timeout --signal=TERM --kill-after=5s 30s env JAX_PLATFORMS=cpu /opt/jax-env/bin/python -m unittest test.test_request_session_duel -v` — passed, 1 focused test in 4.61s (7.80s outer command).
- `timeout --signal=TERM --kill-after=5s 30s node --check reference/neutral-turn-oracle.cjs && timeout --signal=TERM --kill-after=5s 30s node --test test/neutral-turn-oracle.test.cjs` — syntax passed; existing guarded oracle tests passed 3/3 in 6.81s (7.10s outer command).
- A HEAD-source baseline comparison of `--sessions` passed after normalizing only the temporary script filename; all fixture fields, raw draw stack line numbers and existing mode output matched. It completed in 4.44s.

The known-long regression `JAX_PLATFORMS=cpu python -m unittest test.test_request_session test.test_neutral_tackle test.test_neutral_duel test.test_neutral_tackle_ties -v` is deferred to an explicit user-run test session (includes known-long suites beyond this task's per-command cap). This is optional regression evidence; the focused Task039 acceptance test passed. No GPU or performance run was required.

Primary `/root` reviewed the complete oracle/test/API diff and exact validation evidence. The live differential covers four complete battles (30 total turns), all declared request/commit/state/draw/event/terminal criteria and both submission orders. All simulator source and existing test helpers remain unchanged. No required evidence is missing; the broad optional regression is deferred. Accepted at count39 with baseline30/next40, no active task, GPU blocker or milestone acceptance. No additional tests ran during acceptance.

Trail graph refreshed (974 nodes); owner and primary inspected the learning check. Lexical conflicts in the native capture match stopped-lane and RNG-observation warnings: the oracle stops on actual Battle.ended and observes PRNG.prototype.random through the unchanged helper. Documentation matches concern simulator-priority, mutable-team and output-path terms; no learning/tournament work, shared mutable battles or output writer was added. The commit body records the final learning-check rationale. No new durable learning was warranted.

Implementation source blobs at validation freeze:

- `reference/neutral-turn-oracle.cjs`: `da3a9c5f0166e006da8bbf1cf676eedabb234700`
- `test/test_request_session_duel.py`: `fa3cb5bb17742b42c95f22dccbf919d7aa7e4270`
- `docs/neutral-tackle-variant.md`: `89f86c3dfcda039902cdb8a4b0d48d1f4a4386a5`
