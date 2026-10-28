# Project task handoff

## Task 001 — pinned Showdown reference

**Status:** Accepted and complete — 2026-10-28.

**Scope:** Pin the upstream reference; export resolved `gen5randombattle` metadata; generate two seeded six-member teams; initialize a battle and capture both initial player requests. This is oracle infrastructure only. The request payloads contain private information and must not be treated as player-view examples.

**Acceptance checklist:**

- [x] Exact full Showdown commit and upstream origin in `reference-lock.json`.
- [x] Repeatable setup verifies commit and clean state before install/build and never overwrites a checkout.
- [x] Versioned resolved format manifest with declared/effective rules, inherited mod chain, generation, game type, generator, restrictions, and preview status.
- [x] Separate four-limb Gen 5 team and battle seeds; reject malformed and out-of-range limbs; fixture includes both six-member teams and each player's initial request.
- [x] Initial protocol facts are checked from Showdown output; requests must match submitted team identities; time-only protocol lines are excluded.
- [x] Raw `update` packets are labeled as omniscient trusted data because they include split markers/private lines; no player-view projection is claimed.
- [x] Build/runtime provenance is a separate command from semantic fixtures.
- [x] Tests guard wrong commit/origin, tracked edits, and untracked files using a temporary local Git repo without mutating the real checkout.
- [x] Future work remains explicit; this task does not claim G0/G1 completion.

**Commands:**

```sh
node reference/setup.cjs
node reference/oracle.cjs format
node reference/oracle.cjs smoke
node reference/oracle.cjs provenance
node --test test/reference.test.cjs
```

**Artifacts:** `reference-lock.json`, `reference/setup.cjs`, `reference/oracle.cjs`, `reference/fixtures/format-v1.json`, `reference/fixtures/smoke-v1.json`.

**Acceptance rationale:** Architect review accepted the pinned upstream guard, runtime-derived Gen 5 format manifest, reproducible seeded initialization artifact, and the test coverage for provenance guards, protocol facts, request/team matching, seed validation, and stream cleanup. This acceptance applies to Task 001 only; the full project and G0/G1 remain in progress.

**Verification run:** On 2026-10-28, a clean bootstrap from an absent checkout and an idempotent setup rerun both passed at Node `v24.22.1` / npm `11.19.1`, using Showdown commit `1242c1e2d9763ba5b85d33f13aecb89bae19ae3d`. The Node test suite passed all 7 tests, covering a clean local Git fixture, wrong commit/origin, tracked edits, untracked source, live format facts, fixture parity across two fresh seeded initializations, team/request matching, malformed seeds, and stream error/end/timeout cleanup. Two separate smoke CLI processes produced byte-identical output for identical seeds. Custom team/battle seeds also passed a smoke run. The provenance command reported Showdown's package-lock SHA-256 separately from fixture SHA-256 values. `trail build` passed. `trail check` was attempted with process-scoped `safe.directory` and returned `✗ not a git repository`; this workspace's Git repository has an unborn `master` and no commits. Repeat verification with the commands above after any changes.

## Task 002 — player-view projection and visibility fixtures (G0/G1 portion)

**Status:** Accepted and complete — 2026-10-28. Architect review accepted the single-recipient protocol projector and its enforced split, request, clock, terminal, and fail-closed boundaries. Resume with `trail map`, then `rg --files environment test` and `node --test test/*.test.cjs`; protocol source is the pinned ignored checkout under `reference/vendor/pokemon-showdown`.

**Scope:** Produce one scoped view (`p1`, `p2`, or `spectator`) from each complete Showdown object message. Route each `SECRET`/`PUBLIC` split for its audience before exposing events; raw message bodies are never actor-visible. This is a protocol view only; it does not implement feature extraction, belief tracking, action masking, a full battle runner, a learner, or the JAX simulator. Keep the raw oracle data from Task 001 trusted and separate.

**Implementation plan:**

1. Inspect the pinned object-packet routing behavior and agree a small versioned output record schema.
2. Implement a pure CommonJS projector for an ordered array of complete object packets and one explicit recipient, with fail-closed handling for malformed, unsupported, error, and unknown channels.
3. Add a hand-authored transcript fixture and tests for all three audiences, split visibility, addressed requests, terminal/error handling, ordering, and paired-history non-leakage.
4. Cross-check representative routing against Showdown's API and a seeded Gen 5 initialization, then document the module and run the targeted and prior test suites.

**Acceptance criteria:**

- [x] Correctly route `|split|PLAYER` pairs; consume exactly the next two lines even if a branch is empty; reject unsupported IDs, malformed/nested markers, and truncated records without echoing secrets.
- [x] Return records for only the requested `p1`, `p2`, or spectator. A recipient sees its own `SECRET` branch, others see `PUBLIC`; no raw split marker or unselected branch is returned.
- [x] Route `sideupdate` only to its addressed player. Never expose private requests to the opponent or spectator, and do not disclose other private packet counts.
- [x] Exclude clock-only `|t:|` events and structural blanks after split parsing, while preserving visible event order.
- [x] Drop trusted `end` metadata; preserve public `|win|`/`|tie|`. Document and fail closed on `error` and unknown channels.
- [x] Use a hand-reviewed, non-self-generated transcript fixture for exact projection assertions, including empty split branches, exact versus rounded HP, separate requests, clock lines, and terminal metadata.
- [x] Show paired histories that vary only in concealed opponent data produce identical views, while the owner receives allowed private changes.
- [x] Cross-check routing against the pinned Showdown API and a seeded initialization; keep all prior Task 001 tests passing.
- [x] Document the schema and test commands; Task 002 ends awaiting architect review, with Task 003 identified as the complete request-to-request baseline runner.

**Verification run:** On 2026-10-28, `node --test test/protocol.test.cjs` passed 7/7, including hand-fixture parity with pinned routing and seeded initialization. `node --test test/*.test.cjs` passed all 14 tests (7 Task 001 and 7 Task 002). `git diff --check` and `trail build` passed; scoped `trail check` reported no conflicts.

**Resume/check commands:**

```sh
trail map
node --test test/*.test.cjs
node reference/oracle.cjs smoke
trail build
GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0=/workspace trail check
git -c safe.directory=/workspace diff --check
```

## Task 003 — request-to-request baseline battles

**Status:** Accepted and complete — 2026-10-28. Task002/003 source, tests, docs, and the request-episode learning are committed together. Resume with `trail map`, then `node --test test/battle-runner.test.cjs` and `node --test test/*.test.cjs`.

**Scope:** One deterministic random legal-choice policy per side and a complete request-driven Gen 5 Random Battle runner. Each policy receives only its own projected protocol history, current addressed request, and its own policy RNG. This task adds no tournament, strategy heuristics, JAX simulator, or learning.

**Implementation plan:**

1. Inspect pinned request, choice, rejection/reissue, and wait behavior; use the accepted projector as the sole policy visibility boundary.
2. Define separate team, battle, and policy seeds plus request-derived legal command candidates.
3. Implement a per-side random policy and request coordinator that submits choices through BattleStream while Showdown holds simultaneous choices.
4. Bound battle time, packets, decisions, and retries; distinguish submitted/rejected/accepted choices and verify public terminal output against trusted end metadata.
5. Add fake-stream and live full-battle tests; document the CLI, trusted transcript, and handoff limits.

**Acceptance criteria:**

- [x] Policy callbacks receive a frozen `{ history, request, random }` object containing only their own projected history/request and own RNG. Fake-stream spies vary opponent requests and private terminal metadata and observe identical own inputs; policies never receive teams, stream, transcript, opponent view, or pending commands.
- [x] Candidate moves/switches come from the request, including forced replacement and forced move rows. Honor disclosed `trapped`; do not resolve `maybeTrapped`. Explicit wait has no policy action; malformed actionable requests without legal candidates fail.
- [x] Independently seeded policies sample uniformly over request-derived candidate commands. Choices are sent to BattleStream on each player's own request; opponent pending choices are never policy inputs.
- [x] Handle duplicate request payloads only while their previous submission is pending, corrected unavailable-choice requests, same-episode retry bounds/exclusions, one-sided waits/replacements, and unexpected invalid choices as failures.
- [x] Canonical transcript records all five seeds, pinned SHA, requests, submitted/rejected/accepted choices, normalized complete packets, and verified outcome. Fresh CLI processes with identical seeds produce equal transcript JSON.
- [x] Enforce a 60-second overall deadline, packet/decision/retry bounds, cleanup on success/failure, and explicit errors for premature end, stream error, missing/conflicting terminal evidence; no synthetic outcomes.
- [x] Complete real seeded Gen 5 battles, verify public winner against end metadata, and exercise forced replacements.
- [x] Fake streams cover repeated request episodes, simultaneous requests, one-sided forced switch plus wait, retries, terminal metadata, and cleanup; controlled teams against real BattleStream confirm an actual Gen 5 unavailable-choice error and corrected request.
- [x] README and task handoff describe commands, policy input boundary, transcript sensitivity, and remaining scope; Task 001/002 tests pass.

**Resume/check commands:**

```sh
trail map
node --test test/battle-runner.test.cjs
node --test test/*.test.cjs
node environment/run-random.cjs --help
node environment/run-random.cjs --team-seed-p1=1,2,3,4 --team-seed-p2=5,6,7,8 --battle-seed=9,10,11,12 --policy-seed-p1=13,14,15,16 --policy-seed-p2=17,18,19,20 --output trusted-battle.json
trail build
GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0=/workspace trail check
git -c safe.directory=/workspace diff --check
```

**Verification run:** On 2026-10-28, `node --test test/battle-runner.test.cjs` passed 15/15 and `node --test test/*.test.cjs` passed all 29 tests (15 Task 003, 7 Task 002, 7 Task 001). The default seeded game ended on turn 102; two fresh CLI processes produced identical trusted transcript JSON. Tests also exercised an actual Shadow Tag unavailable-choice retry, forced replacements, repeated request payloads, opponent-private retry noninterference, retry budgets, and cleanup. `trail build` and `git diff --check` passed. Scoped `trail check` reports one reviewed false positive at `docs/TASKS.md:56`: “brings back battlestream battle stream,” referencing Task002's accepted criterion to preserve public `|win|`/`|tie|` events. That criterion remains accurate: terminal events are preserved only through recipient-specific projection; trusted raw update splits and `end` metadata are not actor input. README passes the learning check. The request-episode implementation learning is `.trail/learnings/2026-10-28-track-choices-by-request-episode-developer.md` (uncommitted).

## Task 004 — reference protocol trace and replay harness

**Status:** Accepted and complete — 2026-10-28. Architect review accepted the trusted trace boundary, direct input replay, and 34/34 test evidence. Resume with `trail map`, then `node --test test/trace.test.cjs` and the capture/replay commands below.

**Scope:** Capture the complete input/output event sequence of the pinned Showdown BattleStream and replay recorded inputs directly into a fresh simulator. Replay does not rerun policies. Trace files contain packed teams, private requests, split update branches, and trusted end metadata; they are trusted oracle artifacts, not actor observations. This adds no policy, learning, or JAX mechanics.

**Implementation plan:**

1. Wrap the accepted Task003 runner's BattleStream boundary to record every exact input before forwarding it and each complete output packet when read.
2. Normalize only clock-only protocol lines without moving split branch positions; version the trace with pinned SHA, format, five seeds, events, and outcome.
3. Validate the trace schema and allow only known start/player/choice inputs. Replay each recorded input directly and compare each output in order, reporting the first event mismatch.
4. Bound replay time/events, require public terminal and trusted end evidence, check for unexpected trailing outputs, close the stream after successful replay, and destroy it on failures.
5. Test fixtures and a full live capture/replay, update README and this handoff, then run the full suite and Trail/diff checks.

**Acceptance criteria:**

- [x] Trace schema records version, exact upstream SHA, format, all five seeds, ordered input/output events, and public/trusted terminal result.
- [x] Capture records each input before submission and every complete output packet as read; packed teams are retained so replay is independent of team generation and policy execution.
- [x] Replay feeds only validated start/player/move/switch commands directly to a fresh pinned `BattleStream`; rejects unknown/eval commands, wrong schema/pin/format, truncation, and malformed terminal evidence.
- [x] Compare every replay output against the trace after removing only clock messages with split positions preserved; report the first mismatching event/channel or structural field; detect changed requests/damage/commands and trailing outputs.
- [x] Keep traces trusted/private. Do not pass trace packets to policies; any player-view assertion must use `projectPackets`.
- [x] Hand-authored tests cover normalization including empty split branches, mismatch cases, pin/schema validation, truncation/trailing output, and bounded cleanup. A real complete captured seeded battle replays successfully without invoking policy code.
- [x] Document CLI/schema and limitations; the full suite passes all 34 tests. Task004 is accepted and complete.

**Resume/check commands:**

```sh
trail map
node --test test/trace.test.cjs
node reference/trace.cjs capture --output /tmp/gen5-random-trace.json
node reference/trace.cjs replay /tmp/gen5-random-trace.json
node --test test/*.test.cjs
trail build
GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0=/workspace trail check
git -c safe.directory=/workspace diff --check
```

**Verification run:** On 2026-10-28, `node --test test/trace.test.cjs` passed 5/5 and `node --test test/*.test.cjs` passed all 34 tests (5 trace, 15 Task003, 7 Task002, 7 Task001). The live test captured and replayed a complete seeded battle in a fresh CLI process without team generation or policy execution during replay. CLI capture/replay also reproduced the default 102-turn result across 553 events. Tests exercised clock normalization with split blanks, input-order/schema/pin validation, request/damage/command mismatches, extra output, bounded timeout cleanup, and policy-free replay. `trail build` and `git diff --check` passed. Scoped `trail check` reports the reviewed false positive at `docs/TASKS.md:120`, where the trace is explicitly labeled trusted/private; README notes the same trusted boundary and is not flagged as a conflict. The learning's restriction is satisfied because trace inputs never reach an actor and player views still use `projectPackets`.

## Task 005 — first JAX arithmetic slice

**Status:** Planned — next bounded task after Task004 acceptance.

**Acceptance outline:** Implement one explicitly selected Gen 5 arithmetic mechanic in JAX and compare it against pinned Showdown traces for exact edge cases, batching, and deterministic inputs. Keep the slice independent from a full battle simulator or learner; define the mechanic and parity boundary in a reviewed task plan before implementation.

Task 002 establishes protocol projection only. It does not establish full-game player memory, feature extraction, learning, or JAX mechanics parity.
