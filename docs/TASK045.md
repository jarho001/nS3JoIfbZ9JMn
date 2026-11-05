# Task045 private staged submissions for two-move Tackle/Pound turns

## Scope and status

Add a private staged request-session adapter for the accepted
`gen5-neutral-two-move-mew-1v1-v4` turn API. The first valid move commitment
changes only trusted pending metadata. The second commitment calls the existing
`transition_two_move` once, clears both commitments, and advances a bounded
local request episode once. Player projections expose only the current
recipient request, their own committed bit, and a separate recipient-filtered
projection of the previous resolved events.

Owner: `/root/task045`. Source base:
`f0b159242756f6362e03cbd43d38971fd2e415f2`. Status: accepted by primary
`/root` on 2026-11-05. Accepted task count is 45; review baseline 40 and next
checkpoint 50 remain unchanged. G0 is the only accepted milestone. GPU
evidence is not required.

Owned paths: new `simulator/two_move_session.py`, `simulator/__init__.py`,
`reference/neutral-two-move-oracle.cjs`, new `test/test_two_move_session.py`,
`docs/TASK045.md`, `docs/TASKS.md`, `docs/ORCHESTRATION_STATE.json`, and
`docs/neutral-two-move-variant.md`. Do not modify existing numerical engines,
the existing `request_session.py` API, or old tests unless a concrete mismatch
is reported to the parent and authorized.

## Acceptance criteria

- Export typed constructor, submission, current-request view, and
  submission-view APIs: `make_two_move_request_session`,
  `submit_two_move_choice`, `project_two_move_request_view`, and
  `project_two_move_submission_view`. Every public result/view annotation must
  resolve with `typing.get_type_hints`.
- A first valid commitment changes only the submitting actor's private
  pending-choice slot. Battle fields, PP, RNG, events/draws, and the opposing
  actor's view stay unchanged. It returns `WAITING` at the same local episode.
- The second accepted commitment calls the unchanged v4 transition exactly
  once, clears both slots, increments the bounded local episode once, and
  returns `RESOLVED` only on a successful turn. A stopped transition preserves
  the full input session and reports `STOPPED`.
- Initial request projection works before any transition. Current request and
  prior resolved-event projections are distinct; a view cannot reveal the full
  session, pending vector, opponent commitment, hidden PP/RNG, or unexecuted
  choices.
- Preserve runtime precedence: stale episode, invalid player, invalid action,
  terminal, unsupported state/metadata, episode overflow, duplicate own
  commitment, unavailable selected PP, then accepted new commitment. Invalid
  or unavailable choices preserve the full session and empty transition
  traces; a valid terminal request reports terminal status before PP checks.
- Pending entries are int32 0/1/2. A held positive action with no own PP is
  unsupported metadata. Reuse `SESSION_INVALID_ACTION` for selected zero-PP
  moves. Keep existing v2 request-session APIs unchanged.
- Add an opt-in native `--sessions` fixture mode with actual separate
  `Battle.choose` submissions. Cover all four choice pairs, both first-submitter
  orders, tied and unequal speeds, a first-action KO, and a depleted Tackle
  slot choosing legal Pound. First submissions must prove native acceptance
  without battle/PP/RNG/draw/event changes and without request changes. Second
  submissions must capture full state, PP, original RNG/draws, phase, order,
  move identities, recipient events, and next requests. Paired orderings with
  the same initial state/choices must resolve identically. Preserve default and
  `--duels` behavior; do not issue invalid native commands.
- Focused CPU tests use one shared compiled submit kernel and static-audience
  compiled request/submission projectors across value cases. Cover first-submit
  privacy, native resolution parity and cleared commitments, local episode
  advancement, rejection preservation, terminal absorption, malformed inputs,
  public type-hint resolution, and representative heterogeneous
  `jit(vmap(submit))` lanes for waiting/resolved/stale-or-duplicate/unavailable/
  terminal results.
- Inspect the final diff and report exact source hashes, commands, results,
  elapsed times, deferred checks, changed paths, and unresolved concerns. Do
  not self-accept, commit, or push.

## Exclusions

No new battle mechanics, state transitions, reset or networking layer,
policies/training/reporting/rollouts/history storage, full-format support,
GPU/performance claims, long suites, or milestone claims. No refactor of the
v2 session module or changes to the accepted numerical engines.

## Plan and progress

1. Check task count/state, available graph context, v4/session APIs, and native
   oracle seams. The Trail CLI and recent simulator cards are unavailable, so
   use the available cache/index then targeted source fallback.
2. Persist this scope, criteria, ownership, and plan in this handoff and the
   compact orchestration state before implementation.
3. Implement the isolated adapter, native opt-in fixture mode, one focused
   CPU test module, and concise variant/task-index documentation updates.
4. Run only minimal distinct focused commands within 30 seconds each and 90
   seconds total, inspect the owned-file diff, and return one source-frozen
   evidence package to `/root` for acceptance review.

## Evidence

### Implementation

`simulator/two_move_session.py` adds `TwoMoveRequestSession`, typed
submission/request/submission-view results, the bounded constructor, and the
four requested public APIs. It reuses v2 session status constants and scalar
validation, validates v4 structure and held PP, and reuses the existing idle
event/draw helpers. A first commitment only updates one pending slot. The
resolve branch calls `transition_two_move` once; success clears both slots and
increments the episode, while an unexpected non-success reports `STOPPED` and
preserves the input session. `simulator/__init__.py` exports the new API.

The opt-in native `--sessions` fixture adds eight cases for the four move pairs
in both first-submitter orders, with tied and unequal speeds, plus a first-action
KO and a p1 Tackle-zero-PP/Pound-legal case. It observes separate native
`Battle.choose` calls, confirms the first is accepted without changing battle,
PP, RNG, events, draws, or requests, then records the resolved state, exact
draw values/phases/limbs, action and move order, recipient events, and next
requests. It compares paired resolution across submission orders. The depleted
case sets p1 slot 1 PP through the live Pokemon, calls
`battle.makeRequest('move')`, confirms RNG limbs stay fixed, and chooses legal
move 2.

The focused CPU test shares one jitted submission for scalar cases and static
audience request/submission projectors. It compares fresh native requests,
first-commit privacy/freeze, all ten resolved native cases, PP/RNG/state/draws,
move identity/order, both recipient event views, next requests, local episode
and cleared bits. It also checks stale/duplicate/invalid player and action,
unavailable PP, unsupported battle/session metadata, overflow, terminal
absorption, malformed scalar/shape/dtype inputs, type-hint resolution, and
representative heterogeneous `jit(vmap(submit))` lanes.

### Validation

Final source base is `f0b159242756f6362e03cbd43d38971fd2e415f2` and final code
tree is uncommitted. The final focused test and native fixture passed on CPU
with Python 3.12.14 and JAX/JAXLIB 0.12.0.1. Total measured shell real time for
all test and native-fixture attempts below, including the primary's evidence
correction, was about 49 seconds, within the
90-second task budget; no command reached its timeout.

| Command | Result; shell real time |
|---|---|
| `time -p timeout --signal=TERM --kill-after=5s 30s node --check reference/neutral-two-move-oracle.cjs` | Final run passed; 0.03s. |
| `time -p timeout --signal=TERM --kill-after=5s 30s /tmp/task040-jax-env/bin/python -m py_compile simulator/two_move_session.py simulator/__init__.py test/test_two_move_session.py` | Final run passed; 0.02s. |
| `time -p timeout --signal=TERM --kill-after=5s 30s node reference/neutral-two-move-oracle.cjs --sessions > /tmp/task045-native-sessions.json` | Passed; 0.27s. Generates all ten native cases and pair-order assertions. |
| `time -p timeout --signal=TERM --kill-after=5s 30s env JAX_PLATFORMS=cpu /tmp/task040-jax-env/bin/python -m unittest test.test_two_move_session -v` | Primary-requested correction run passed: 1 test in 10.228s; shell real 10.69s. It includes the strengthened native resolving vmap lane, waiting/rejection lane assertions, and stale/player/action precedence case. The pre-correction run passed in 10.221s; shell real 10.67s. An earlier run after terminal/metadata coverage passed in 9.941s; shell real 10.40s. |
| `time -p timeout --signal=TERM --kill-after=5s 30s git diff --check` | Passed after the final documentation update; 0.00s. |
| State/link assertion script (`python3 - <<'PY' ...`) | Pre-acceptance run passed; 0.02s. Checked JSON, count 44, baseline 40, checkpoint 50, active Task045 owner/status, unchanged completed history and direction milestones, G0-only review status, no accepted Task045 index row, and touched Markdown links. |

Development attempts that failed and were corrected:

- The first unittest import failed because the test imported
  `PHASE_UNSUPPORTED` from the root package; it was changed to import from
  `simulator.neutral_tackle`. Shell real 0.39s.
- The first native `--sessions` attempt expected only p1 Pound in the depleted
  case; the native turn also executes p2 Tackle. The assertion now checks p1's
  actor-specific executed move. Shell real 0.27s. The test fixture setup also
  failed on this same assertion before correction; shell real 0.64s.
- A focused test run rejected a scalar-shape `ValueError` because the test
  expected only `TypeError`. The assertion now accepts the existing scalar
  validator's `ValueError`. Shell real 4.15s.
- The first state assertion used the direction's `milestones_accepted` field as
  the G0 approval ledger; that field is empty in the unchanged source baseline.
  The corrected assertion compares that field with the baseline and checks
  `review_state.milestone_review`, where G0 is the only recorded approval.
  Shell real 0.02s; corrected run passed in 0.02s.
- The first acceptance-bookkeeping assertion used literal text matches that
  missed wrapped Markdown wording. A corrected whitespace-normalized JSON,
  history, hash, and link check passed; this was a check-script mismatch, not a
  repository-state discrepancy.

### Primary-requested evidence correction

The mixed vmap resolving lane now starts from native scenario
`tackle-pound-unequal-first-p1`, holds p1's Tackle choice, then submits p2's
Pound. It checks the input `before` state, every resolved native state field,
the exact native draw prefix, cleared choices, and episode 1. The waiting lane
checks unchanged full battle and RNG, pending `[1, 0]`, episode 0, and empty
events/draws. Each stale/duplicate/unavailable/terminal lane checks unchanged
session, before state, transition state, and empty traces. The stale precedence
case now submits invalid player 8 and action 0 with expected episode 1 and
asserts `SESSION_STALE_EPISODE`.

The one requested rerun used the focused unittest command in the table above
and passed (1 test in 10.228s; shell real 10.69s). No simulator, oracle, or
additional test module changed.

Trail was unavailable on PATH and at `/workspace/tools/trail/node_modules/.bin/trail`;
the cache contains no cards for the recent session/two-move source. The
available index was checked and targeted source fallback used. No GPU run is
required. The old `test.test_neutral_two_move_duel` module was not rerun; it
remained an unaffected optional check.
Only the full boundary module `test.test_neutral_two_move` is known-slow and
deferred; its exact raw command and source identity are recorded in
`deferred_validation` in `docs/ORCHESTRATION_STATE.json`. Task044 measured the
focused `test.test_neutral_two_move_duel` module at 2.732s; it is not classified
as slow and is not an acceptance dependency.

```sh
JAX_PLATFORMS=cpu /tmp/task040-jax-env/bin/python -m unittest test.test_neutral_two_move -v
```

The full boundary module's exact source blobs and
`acceptance_dependency: false` are recorded with this raw command in
`deferred_validation` in the orchestration state.

Changed source hashes at the final checked revision:

| File | Git blob | SHA-256 |
|---|---|---|
| `simulator/two_move_session.py` | `8ab75411f1ddc8cdf9ff45e482d56067f8801c2d` | `57cfe82437086a1dd41da6a45af4e8525b2e67db18a238906c47af4ac234fbea` |
| `simulator/__init__.py` | `ff58a8f6967f1dfb9be4b2f2bb759e42aab9fa1c` | `a8da379ec038603b4ab58160b3f28da45396bf440d24e62db745113869a72fd2` |
| `reference/neutral-two-move-oracle.cjs` | `2ee13e18f0d5643c3974b5e66c12e2df26ccec8a` | `e2598fb9ceb8617e58517dd51b3dfbde435e5b89849b6b3a6ebc999c915d1e99` |
| `test/test_two_move_session.py` | `c6bc7f7f29cae92fbcbab185ea47c38b45a09551` | `64dd57c3f7f1818eb7235e213e3e7b5b860eadf8c3f9e62196321ba6bd535693` |

### Acceptance bookkeeping

On 2026-11-05, the compact completed-task history was extended by appending
Task045 after the exact history prefix from the source-base state. Count 45,
full source base `f0b159242756f6362e03cbd43d38971fd2e415f2`, cleared active slots,
baseline 40, checkpoint 50, G0-only milestone record, and null GPU block were
confirmed. The Task045 deferred entry contains the raw optional full-boundary
command, source blobs, and `acceptance_dependency: false`. JSON, history/count,
links, source-blob hashes, and `git diff --check` passed. No tests were rerun
during this bookkeeping sync.

### Review boundary

Changed paths are the eight owned files listed above. The accepted numerical
engines and `simulator/request_session.py` are unchanged. Primary `/root`
accepted the task on 2026-11-05. The accepted count is 45; review baseline 40
and checkpoint 50 remain fixed, and G0 remains the only accepted milestone.
No Astra review, GPU evidence, or performance claim is made. No unresolved
semantic mismatch is known. No commit or push was made.
