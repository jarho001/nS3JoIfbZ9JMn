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

**Status:** Accepted and complete — 2026-10-28. Resume with `trail map`, then inspect `reference/stat-stage-oracle.cjs`, `mechanics/stat_stage.py`, and `test/test_stat_stage.py`.

**Scope:** Implement the integer stat-stage scaling substep used by the pinned Gen 5 `Pokemon.calculateStat`, after `ModifyBoost` and field stat selection and before stat modifiers. The current scaler's caller precondition is `[0, 65535]`; Task007 includes the zero-valued result reachable through Gen 5 nature truncation. The bounds are not checked dynamically. This is not the complete `calculateStat` pipeline and does not handle natures, EVs, Wonder Room selection, event hooks, or stat modifiers.

**Implementation plan:**

1. Inspect the exact pinned `Pokemon.calculateStat` path and available direct Battle/Pokemon API; prove the oracle has no `ModifyBoost` handler, no Wonder Room, and modifier 1.
2. Generate a small versioned fixture from actual pinned Gen 5 `Pokemon.calculateStat` calls using controlled stored stat values; record source commit, paths, scope, and input domain.
3. Add a pure JAX int32 kernel with stage clamp `[-6, 6]`, integer numerator/denominator, exact floor division, and broadcast-compatible inputs.
4. Pin/document an isolated Python CPU JAX environment without modifying the existing GPU environment.
5. Test independent anchor values, fixture reproduction, all 1..1024 stored stats at all 13 stages plus boundary/clamp cases, eager/JIT/vmap/broadcast behavior, and the existing Node oracle suite.

**Acceptance criteria:**

- [x] Kernel implements the exact post-`ModifyBoost` stage ratio: for `s >= 0`, `floor(stat * (2+s) / 2)`; for `s < 0`, `floor(stat * 2 / (2-s))`; `s` is clamped to `[-6, 6]`.
- [x] Documented domain is non-HP stored stat in `1..65535` and signed int32 stage; the stored-stat range is a caller precondition, not a dynamic check. Numerator arithmetic is int32-safe; no floating-point operations or silent float conversion.
- [x] JAX kernel is pure and works eagerly, under `jax.jit`, `jax.vmap`, and normal broadcasting with int32 outputs.
- [x] Versioned fixture is generated by the pinned Showdown `Pokemon.calculateStat` in a neutral Gen 5 battle. It records the upstream SHA and exact source paths; the harness does not duplicate the formula in JavaScript.
- [x] Live oracle reproduction and differential tests cover representative odd/boundary fixtures, every stage for stats 1..1024, domain edges, and clamp extremes.
- [x] Independent hand-computed anchors and eager/JIT/vmap/broadcast tests pass under the pinned CPU JAX setup; existing Node tests still pass.
- [x] README and handoff state the exact scope and environment verification. Task005 is accepted and complete.

**Resume/check commands:**

```sh
trail map
node reference/stat-stage-oracle.cjs fixture
JAX_PLATFORMS=cpu python -m unittest discover -s test -p 'test_*.py' -v
node --test test/*.test.cjs
trail build
GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0=/workspace trail check
git -c safe.directory=/workspace diff --check
```

**Verification run:** On 2026-10-28, an isolated Python 3.13 environment installed `jax==0.12.0.1` and `jaxlib==0.12.0.1`; `JAX_PLATFORMS=cpu` reported CPU. The six Python tests passed in 5.1 seconds and compare 13,363 live pinned-Showdown outputs through eager, JIT, and JIT-vmap paths. The existing `/opt/jax-env` suite also passed 6/6 on its GPU backend in 8.4 seconds and was not modified. `node --test test/*.test.cjs` passed 34/34. The stored-stat range `[1, 65535]` is a documented input precondition, not dynamically validated.

## Task 006 — stored-stat derivation slice

**Status:** Accepted and complete — 2026-10-28. Architect review accepted the target-format nature semantics, joint-axis broadcasting, and 13/13 CPU/GPU test evidence. Resume with `trail map`, then inspect `mechanics/stats.py`, `reference/stats-oracle.cjs`, and `test/test_stats.py`.

**Scope:** Compute six stored stats from base stats, level, IVs, EVs, and per-stat nature multipliers. Axis order is `hp, atk, def, spa, spd, spe`. Target the pinned `gen5randombattle` stat path, including Gen 5 unsigned 16-bit truncation during nature modification. The effective format has no `overflowstatmod` rule, so do not apply its 595/728 cap branch. Shedinja has an explicit fixed-1 HP override; do not infer the override from base HP 1. This task does not apply stat stages, forms, event hooks, or team legality.

**Implementation plan:**

1. Confirm target format rule-table facts and exact `Pokemon`/`Battle` stat paths from the pinned source.
2. Add a pure JAX int32 API with final-axis-six arrays, level and nature broadcasting, exact HP/non-HP floor operations, explicit nature multipliers `[9, 10, 11]`, and a separate Shedinja mask.
3. Generate a small versioned fixture by constructing real pinned Gen 5 Pokemon from sets and reading `maxhp` and `storedStats`, without overriding their calculations.
4. Cover all 25 natures over levels `1, 17, 50, 99, 100`, rotate IV and EV boundary patterns across each stat, and vary real species/base-stat profiles including Shedinja. Add separate real `battle.statModify` synthetic cases at pre-nature Attack 595, 596, and 609 for plus/minus/neutral nature paths.
5. Test hand anchors, fixture reproduction, eager/JIT/JIT-vmap parity, dtype and broadcast contracts, CPU and available GPU; run the prior Python and Node suites.

**Acceptance criteria:**

- [x] Pure int32 JAX API has final axis order `hp, atk, def, spa, spd, spe` and supports documented batch broadcasting; domains are base stats `1..255`, level `1..100`, IV `0..31`, EV `0..255`, and nature multipliers `9/10/11`. Value ranges are caller preconditions, not dynamic checks.
- [x] HP and non-HP stat equations match pinned Gen 5 `Battle.statModify` floor behavior; nature products preserve unsigned low-16-bit truncation. Do not apply `overflowstatmod` caps because the target format rule is absent.
- [x] Explicit Shedinja input forces max HP 1; a different species/profile with base HP 1 uses the ordinary formula.
- [x] Versioned fixture is generated from actual pinned Pokemon sets and `Pokemon.maxhp`/`storedStats`; includes pinned SHA, format rule facts, and exact source paths. No JavaScript formula clone supplies expected fixture outputs.
- [x] Differential grid covers all 25 natures, levels `1/17/50/99/100`, IV/EV boundaries `0/3/4/31` and `0/3/4/252/255`, varied real species, Shedinja, and real-`statModify` synthetic nature paths at the 595/596 wrap threshold and 609 upper base-stat case.
- [x] Independent anchors plus eager/JIT/JIT-vmap and broadcast checks pass under pinned CPU JAX and available GPU; Task005 Python tests and all Node tests pass.
- [x] README and handoff document scope, pinned environment, range preconditions, test evidence, and Task007. Task006 ends awaiting architect review.

**Resume/check commands:**

```sh
trail map
node reference/stats-oracle.cjs fixture
JAX_PLATFORMS=cpu python -m unittest discover -s test -p 'test_*.py' -v
node --test test/*.test.cjs
trail build
GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0=/workspace trail check
git -c safe.directory=/workspace diff --check
```

**Verification run:** On 2026-10-28, the pinned fixture regenerated identically and confirmed Gen 5 `gen5randombattle` has `overflowstatmod: false`. The fixture contains 125 real Pokemon cases across all 25 natures and five levels, plus nine real `Battle.statModify` synthetic cases: pre-nature Attack 595/596 and base-stat upper case 609 under Adamant, Modest, and Hardy. At 595, `595*110=65450` does not wrap and the final stored Attack is 654; at 596, `596*110=65560` wraps to 24 and the final stored Attack is 0. Python tests passed 13/13 on isolated pinned CPU JAX in 9.8 seconds and 13/13 on the existing GPU environment in 14.0 seconds. `node --test test/*.test.cjs` passed 34/34. Both existing environments remain unchanged.

## Task 007 — compose Attack stored-stat derivation and stage scaling

**Status:** Accepted and complete — 2026-10-28. Architect review accepted the Attack-only composition, zero-inclusive stage domain, and guarded pinned oracle. Resume with `trail map`, then inspect `mechanics/attack.py`, `reference/attack-oracle.cjs`, `test/test_attack.py`, and the two stat-slice fixtures.

**Scope:** Compose the existing stored-stat derivation and stage-scaling kernels for Attack only, matching `Pokemon.calculateStat('atk', stage, 1)` on pinned format id `gen5randombattle` with neutral `ModifyBoost`, no Wonder Room, and modifier 1. This does not implement ability/item/status effects, dynamic field selection, other stat selectors, a general event engine, or a full battle calculator.

**Implementation plan:**

1. Extend the stage scaler's caller precondition from positive to nonnegative stored stats and add zero cases to its live Showdown fixture/grid; keep the formula unchanged.
2. Add a thin `derive_attack` composition that calls `derive_stored_stats`, selects final-axis Attack, then calls `scale_stat_stage`. Let stage inputs add broadcast axes.
3. Generate a guarded, compact pinned fixture from ordinary real Pokemon sets across natures, levels, IV/EV profiles, and all stages including clamp extremes. Read stored values from actual Pokemon instances and expected values from the real `calculateStat` method; check neutral event and field behavior without monkeypatching.
4. Add one labeled synthetic-zero oracle row: obtain zero from real `Battle.statModify` at the nature-wrap edge, then explicitly inject that stored Attack into a real Pokemon only for the arithmetic-only `calculateStat` call.
5. Test hand-computed noncommuting floor anchors and every stage value in the synthetic-zero fixture vector, plus fixture reproduction, eager/JIT/JIT-vmap, shared-table and multidimensional broadcasting, and prior CPU/GPU Python plus Node suites.

**Acceptance criteria:**

- [x] `scale_stat_stage` documents and supports `stored_stat` in `[0, 65535]`; its integer formula is unchanged and zero remains zero for all stages.
- [x] `derive_attack(base_stats, level, ivs, evs, nature_multipliers, stage)` is a thin pure composition: no duplicated equations, final-stat selector, modifier, or policy behavior.
- [x] Attack inputs preserve Task006 int32/domain and nature semantics; stage and all batch axes broadcast, including shared `(6,)` stat tables with batched levels/stages.
- [x] Guarded oracle fixture records pin, source paths, conditions, and both ordinary real-Pokemon cases and the separately labeled synthetic-zero case. Ordinary expected values come directly from real `Pokemon.calculateStat('atk', stage, 1)` without overwriting stored stats.
- [x] Oracle fixture covers all 25 natures, varied levels/IVs/EVs/species, all 13 stages and out-of-range clamp inputs; live fixture regeneration is deterministic.
- [x] Tests cover the synthetic zero across all stages, hand anchors demonstrating nature then stage rounding, eager/JIT/JIT-vmap/broadcast parity, strict dtype, CPU and available GPU; Task005/006 and all Node tests pass.
- [x] README/task handoff documents the neutral Attack-only boundary and Task008. Task007 ends awaiting architect review.

**Resume/check commands:**

```sh
trail map
node reference/attack-oracle.cjs fixture
JAX_PLATFORMS=cpu python -m unittest discover -s test -p 'test_*.py' -v
node --test test/*.test.cjs
trail build
GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0=/workspace trail check
git -c safe.directory=/workspace diff --check
```

**Verification run:** On 2026-10-28, `node reference/attack-oracle.cjs fixture` produced identical bytes in separate runs, and matched `reference/fixtures/attack-v1.json`; the zero-inclusive stat-stage fixture also regenerated identically. The Attack fixture has 25 initialized Pokemon cases (all 25 natures), each checked at 17 stage inputs: all 13 stages and four out-of-range clamp inputs. It also has one synthetic zero case: real `Battle.statModify` derives zero from pre-nature Attack 596; zero is injected only into a real Pokemon for `calculateStat` arithmetic. The 20-test Python suite passed on CPU in 13.2 seconds and the available GPU environment in 17.6 seconds. `node --test test/*.test.cjs` passed 34/34. Both oracle scripts assert the pin before requiring Showdown. `trail build` passed. Scoped `trail check` reports one reviewed false positive at `docs/TASKS.md:235`: it interprets the mention of pinned format id `gen5randombattle` as applying to every Gen 5 format. The Task007 oracle and implementation are scoped to that exact format id. Other rule matches are non-blocking advisories. `git diff --check` passed.

## Task 008 — first damage arithmetic primitive

**Status:** Accepted and complete — 2026-10-28. Architect review accepted the uint32 random-roll kernel, guarded controlled/live oracle, and test coverage. Resume with `trail map`, then inspect `mechanics/damage_roll.py`, `reference/damage-roll-oracle.cjs`, `test/test_damage_roll.py`, and `reference/fixtures/damage-roll-v1.json`.

**Scope:** Mirror the pinned Showdown `Battle.randomizer(baseDamage)` operation with an explicit supplied roll. Match uint32 truncation of the multiplication followed by floor division by 100. The kernel stops at the value returned by `Battle.randomizer`; it does not generate the roll or implement later damage steps.

**Implementation plan:**

1. Verify the pinned `Battle.randomizer` and its `Battle.random(16)` contract, exact `tr`/unsigned-width behavior, and call position in the damage pipeline.
2. Add a pure JAX int32 API `apply_damage_roll(base_damage, roll)`, with caller preconditions `base_damage` in `[0, 2147483647]` and roll in `[0, 15]`; use uint32 multiplication for explicit wrap, then floor division by 100 and int32 output.
3. Generate a guarded pinned oracle fixture using real `Battle.randomizer` calls. Control only the `Battle.random` result for the exhaustive fixture and restore it in `finally`; separately record live seeded draws, including battle seed `[101, 202, 303, 404]`, by wrapping the original random method after Battle initialization.
4. Cover all rolls for damage `0..1024`, wrap boundaries and high values through int32 max, plus small independent anchors; clearly distinguish controlled arithmetic rows from real seeded draw rows.
5. Test eager/JIT/JIT-vmap, broadcasting, dtype, fixture regeneration, CPU/GPU parity, and the prior Python/Node suites. Update README and handoff, then refresh/check Trail.

**Acceptance criteria:**

- [x] The pure kernel returns `floor(uint32(base_damage * (100-roll)) mod 2^32 / 100)` as int32; all inputs are strict int32 arrays/scalars and broadcast normally.
- [x] The documented input domain is base damage `[0, 2147483647]`, roll `[0,15]`; caller preconditions are not dynamically checked. Output fits int32.
- [x] Guarded pinned oracle records source SHA, format, source paths, domains, and draw contract. Controlled oracle calls actual `Battle.randomizer` and asserts exactly one `random(16)` invocation; restore the original method even on failure.
- [x] A separate live seeded oracle records battle seed `[101, 202, 303, 404]`, wraps the original random method after initialization, records each draw, and matches the real `randomizer` outputs. It does not claim these are the PRNG's first draws. Fixture generation is deterministic.
- [x] Differential coverage includes every roll for base damage 0..1024, multiplication wrap boundaries and high values including int32 max. Anchors include zero, floor-to-zero, 100 at roll 15, and verified uint32 overflow.
- [x] Eager/JIT/JIT-vmap/broadcast, strict dtype, CPU and GPU tests pass; prior 20 Python and 34 Node tests remain passing.
- [x] README documents the arithmetic-only boundary and Task009. Task008 ends accepted and complete.

**Resume/check commands:**

```sh
trail map
node reference/damage-roll-oracle.cjs fixture
JAX_PLATFORMS=cpu python -m unittest discover -s test -p 'test_*.py' -v
node --test test/*.test.cjs
trail build
GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0=/workspace trail check
git -c safe.directory=/workspace diff --check
```

**Verification run:** On 2026-10-28, the pinned fixture reproduced byte-identically in two independent CLI processes. Its controlled matrix covers 1,036 base-damage values (0..1024 plus selected wrap/high boundaries) by all 16 rolls; it includes a separate seven-case live seeded sequence with battle seed `[101, 202, 303, 404]`, recorded after Battle initialization. Python tests passed 27/27 on CPU in 15.4 seconds and 27/27 on the available GPU in 19.1 seconds. `node --test test/*.test.cjs` passed 34/34. `trail build` and staged `git diff --check` passed. Scoped `trail check` reports one reviewed false positive at this section's function name `apply_damage_roll`; the matcher interprets it as a broad format claim, though this API is a pure arithmetic kernel with no format selection. The commit body records the same limitation.

## Task 009 — base-damage integer core

**Status:** Accepted and complete — 2026-10-28. Task009 was accepted after review of the kernel, oracle, and tests. Resume with `trail map`, then inspect `mechanics/base_damage.py`, `reference/base-damage-oracle.cjs`, `test/test_base_damage.py`, and `reference/fixtures/base-damage-v1.json`.

**Scope:** Implement only the integer core inside pinned `BattleActions.getDamage`: `tr(tr(tr(tr(2*level/5+2)*basePower*attack)/defense)/50)`. Inputs are the effective level, base power, Attack, and Defense already selected by the simulator. The slice excludes stat derivation/selection, events, critical hits, `modifyDamage`, the subsequent `+2`, random rolls, and later modifiers. A zero Defense matches pinned unsigned conversion of Infinity/NaN to 0 and must use safe operands before division in JAX.

**Implementation plan:**

1. Confirm pinned `getDamage`, the exact target move-call shape, `tr` width, and placement before `modifyDamage`/`+2`.
2. Add pure int32 `base_damage_core(level, base_power, attack, defense)` with caller domains level `[1,100]`, effective base power `[1,65535]`, Attack/Defense `[0,65535]`, and broadcast-compatible shapes. Preserve uint32 truncation of the product before defense division; select a positive safe divisor before dividing, then return 0 for defense 0.
3. Generate a guarded oracle from real `BattleActions.getDamage` in pinned `gen5randombattle`. Temporarily intercept only `actions.modifyDamage` to capture and return the core value; assert it is called once and restore it in `finally`. Use actual initialized Pokemon and verify the supplied level/effective stats; keep synthetic arithmetic cases separately labeled.
4. Cover boundary levels, powers, zero/one Attack and Defense, integer-floor transitions, uint32 wrap adjacent values, and int16-scale power; compare actual pinned outputs, not a JavaScript formula clone.
5. Test eager/JIT/JIT-vmap, broadcasting, strict dtype, CPU/GPU and prior suites. Document the neutral formula boundary, next Task010 composition, and any safe zero-denominator learning.

**Acceptance criteria:**

- [x] `base_damage_core(level, base_power, attack, defense)` returns the pinned nested-truncation core as int32, with explicit uint32 product wrap and correct zero-Defense result.
- [x] All four inputs are strict int32 and broadcast normally. Domains are documented caller preconditions, not dynamically checked; output remains int32.
- [x] Guarded fixture records pin, format, source paths and input domains. Ordinary Pokemon cases use actual initialized sets without overwriting stored stats; synthetic numeric rows are labeled.
- [x] Oracle invokes actual `getDamage` with a numeric move power and neutral `willCrit=false` path, captures the unmodified core before downstream modifiers, verifies actual level/effective Attack/Defense, and restores the interceptor on every exit.
- [x] Cases cover levels 1/2/4/5/17/50/99/100, low/high powers, zeros/ones, floor boundaries, wrap-adjacent products, and an int16-scale power. Fixture generation is deterministic.
- [x] Hand anchors and differential grid match eager/JIT/JIT-vmap/broadcast on CPU and GPU; prior 27 Python and 34 Node tests pass.
- [x] README documents the precise boundary; Task010 composes the core, later `+2`, the neutral fixed-point event, and explicit damage roll, not a full damage engine. Task009 ends accepted and complete.

**Resume/check commands:**

```sh
trail map
node reference/base-damage-oracle.cjs fixture
JAX_PLATFORMS=cpu python -m unittest discover -s test -p 'test_*.py' -v
node --test test/*.test.cjs
trail build
GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0=/workspace trail check
git -c safe.directory=/workspace diff --check
```

**Verification run:** On 2026-10-28, the guarded fixture regenerated byte-identically in independent processes. It includes eight initialized-Pokemon cases with stored stats untouched and 1,218 explicitly synthetic cases from real `getDamage` calls, including zero Defense (both zero/nonzero numerator), floor transitions, and uint32 wrap-adjacent values. All 34 Python tests passed on CPU in 20.1 seconds and GPU in 25.3 seconds; `node --test test/*.test.cjs` passed 34/34. `trail build` and diff checks passed. Scoped `trail check` reports one reviewed false positive at the statement that inputs are effective stats already selected by the simulator: this bounded kernel consumes those inputs but claims no format-wide stat rule or selection behavior. The other format-claim matches concern the neighboring `apply_damage_roll` pure arithmetic API, as documented in the preceding Task008 verification.

## Task 010 — neutral base-damage composition

**Status:** Accepted and complete — 2026-10-28. Resume with `trail map`, then inspect `mechanics/neutral_damage.py`, `reference/neutral-damage-oracle.cjs`, and `test/test_neutral_damage.py`.

**Scope:** Add `neutral_damage_after_roll(level, base_power, attack, defense, roll)` as a pure composition of the accepted core and explicit roll kernels. It returns the integer intermediate after `Battle.randomizer`, not final damage. Preserve pinned order: compute core, `modifyDamage` adds `2`, the neutral `WeatherModifyDamage` event applies `Battle.modify(value, 1)`, then the caller-supplied roll. No duplicated core/roll arithmetic, PRNG, full modifier engine, critical/spread/STAB/type/burn/minimum-one/final-16-bit behavior.

**Implementation plan:**

1. Use transparent wrappers around the real `modifyDamage`, `WeatherModifyDamage` priority event, and `Battle.randomizer`; delegate each to its original implementation. Control only the underlying `Battle.random` roll, assert exactly one draw in `[0,16)`, and restore every method in `finally`.
2. Check neutral Gen 5 conditions and events, including base power and effective Attack/Defense. Verify the no-handler WeatherModifyDamage result equals real `Battle.modify(input, 1)`, including its uint32 intermediate wrap (equivalent to masking to 20 bits). Record returned final damage separately from the target intermediate. Use ordinary initialized Pokemon without stat overrides plus a compact, explicitly labeled synthetic arithmetic grid.
3. Cover all 16 rolls over selected ordinary and synthetic cases, including zero Attack/Defense, uint32 core wrap, the 20-bit neutral event boundary, and examples where the post-roll intermediate differs from eventual final damage. Task008 separately covers randomizer uint32 product wrap; the neutral event bounds this composition's roll input. Record pinned SHA, format, seeds, source boundary, scope, and fixture schema.
4. Implement a thin JAX function calling `base_damage_core`, adding int32 `2`, applying the proven low-20-bit neutral-event identity, and calling `apply_damage_roll`; preserve strict int32, broadcast, JIT, and vmap behavior.
5. Add anchors for operation order and boundaries, differential tests against the fixture, CPU/GPU and prior Python/Node regressions. Update README, then build/check Trail.

**Acceptance criteria:**

- [x] Composition reuses existing kernels exactly, includes the pinned neutral fixed-point event step, and preserves the Task009 input domains and roll `[0,15]`; strict int32 and broadcasting include a roll axis.
- [x] Oracle invokes real `getDamage`, delegates original `modifyDamage`, the neutral event, and `Battle.randomizer`; controls only the random draw and restores wrappers even on failure.
- [x] Fixture distinguishes the captured post-roll intermediate from `getDamage`'s returned final value; eight ordinary and thirteen synthetic cases cover all rolls; generation is deterministic and guarded by the pin.
- [x] Tests cover core-wrap and zero edges, the neutral event's 20-bit boundary, `+2` before roll, a result demonstrating the minimum-one boundary, eager/JIT/JIT-vmap, broadcasting, and dtype.
- [x] Full Python suite passes on CPU and GPU; Node suite passes; README and this handoff record evidence and a bounded Task011 proposal.

**Resume/check commands:**

```sh
trail map
node reference/neutral-damage-oracle.cjs fixture
JAX_PLATFORMS=cpu python -m unittest discover -s test -p 'test_*.py' -v
node --test test/*.test.cjs
trail build
GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0=/workspace trail check
git -c safe.directory=/workspace diff --check
```

**Verification run:** On 2026-10-28, the guarded fixture regenerated byte-identically in independent CLI processes. It contains 8 initialized-Pokemon and 13 synthetic cases × all 16 rolls, and records the true final `getDamage` result separately from the post-randomizer intermediate. A synthetic case demonstrates intermediate 0 followed by final Gen 5 damage 1. The neutral `Battle.modify(value, 1)` fixture boundaries are 1,048,575→1,048,575, 1,048,576→0, and 1,048,577→1. All 41 Python tests passed on CPU in 24.6 seconds and GPU in 30.2 seconds; `node --test test/*.test.cjs` passed 34/34. Trail build and diff checks passed. Scoped Trail check reports one reviewed false positive at this scope paragraph: it matches the API's effective stat input names to a prior learning about applying stat caps across every Gen 5 format; this task only consumes selected inputs and makes no format-wide stat-cap claim.

**Boundary note:** A neutral Showdown event is not identity for arbitrarily large integers. `runEvent` applies `Battle.modify(relayVar, 1)` even when there are no event handlers; its int32 truncation makes the result `relayVar mod 2^20`. Task010 models only this actual neutral fixed-point step, not arbitrary modifier composition. Consequently this path's randomizer input is at most `2^20-1`; Task008 remains the independent oracle for randomizer uint32-product wrap. See the new Trail learning `neutral-showdown-event-identity-can-wrap-developer.md`.

**Next bounded task:** Task011 completed the fixed-point primitive. Task012 can implement one pinned damage-finalization primitive under explicit inputs; keep the scope below a full damage engine.

## Task 011 — fixed-point modifier primitive

**Status:** Accepted and complete — 2026-10-28. Resume with `trail map`, then inspect `mechanics/fixedpoint.py`, `reference/fixedpoint-oracle.cjs`, and `test/test_fixedpoint.py`.

**Scope:** Add strict-int32 `apply_fixedpoint_modifier(value, modifier)` for an already encoded Gen 5 modifier. Domain preconditions: `value` in `[0, 2147483647]`, encoded modifier in `[0, 65535]`; neither is dynamically checked. Reproduce `Battle.modify(value, modifier, 4096)` with uint32 product wrap, then its JavaScript-number `+2047` and floor division by 4096. Do not implement fraction-to-encoded conversion, chained modifiers, event ordering, or a generic modifier engine.

**Implementation plan:**

1. Use uint32 multiplication to preserve product wrap. Compute quotient and remainder by 4096 and round up only when remainder is greater than 2048; exact half ties round down. Avoid uint32-wrapping the post-product `+2047`, which Showdown performs as a JavaScript number.
2. Extend the neutral damage helper to call `apply_fixedpoint_modifier(value, 4096)` without changing Task010 outputs. This verifies the identity modifier path remains a 20-bit wrap.
3. Add a guarded pinned oracle calling the real `Battle.modify(value, modifier, 4096)` with no overrides. Store a compact fixture grid covering values 0..1024 over representative modifiers and targeted rounding/product-wrap pairs, with pin/source/domain provenance.
4. Test eager/JIT/JIT-vmap, multi-axis broadcasting, strict dtype, identity and rounding anchors, UINT32_MAX-product carry, and neutral-composition parity. Preserve prior Python/Node suites on CPU/GPU.
5. Update README and this handoff with evidence; suggest Task012 as a bounded damage finalization primitive or other well-motivated next slice, not a full engine.

**Acceptance criteria:**

- [x] Kernel exactly matches pinned `Battle.modify(value, modifier, 4096)` for its documented domains and broadcasting behavior.
- [x] Product is uint32-wrapped before rounding; the `+2047` rounding step does not wrap. Ties at remainder 2048 round down; 2049 rounds up.
- [x] Guarded fixture is generated from actual pinned Showdown with no formula clone or arithmetic override, records source pin/path/domain, and reproduces deterministically.
- [x] Existing Task010 neutral results are unchanged after routing through the new primitive.
- [x] Eager/JIT/JIT-vmap, broadcast, dtype, CPU/GPU and prior Node/Python regressions pass; docs record the accepted scope and evidence.

**Resume/check commands:**

```sh
trail map
node reference/fixedpoint-oracle.cjs fixture
JAX_PLATFORMS=cpu python -m unittest discover -s test -p 'test_*.py' -v
node --test test/*.test.cjs
trail build
GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0=/workspace trail check
git -c safe.directory=/workspace diff --check
```

**Verification run:** On 2026-10-28, the pinned fixture reproduced byte-identically in independent CLI runs. Its real `Battle.modify` grid covers 1,025 values × 13 encoded modifiers (13,325 outputs) plus targeted half-rounding, identity-wrap, int32-maximum, and UINT32_MAX-product carry cases. The full Python suite passed 47/47 on CPU in 28.1 seconds and GPU in 34.2 seconds; `node --test test/*.test.cjs` passed 34/34. Task010's neutral-event tests remain in that suite and pass unchanged. Trail build and staged diff checks passed. Scoped Trail check emitted two reviewed false positives: it interpreted this pinned arithmetic-oracle fixture as player-view evidence, and the modifier input domain as an earlier stored-stat domain. These are unrelated to the contents or information boundary of this task.

**Boundary note:** The modifier input is already encoded as an integer and is not a fraction. The uint32 product wraps before rounding, but Showdown's `+2047` occurs afterward as a JavaScript number. Therefore a wrapped product of `2^32-1` rounds to `2^20`, rather than wrapping again or being masked to 20 bits. See `keep-fixed-point-rounding-carry-outside-developer.md`.

**Next bounded task:** Task012 can implement one pinned damage-finalization primitive under explicit inputs; do not expand this into fraction encoding, modifier chains, or a complete damage engine.

Task 002 establishes protocol projection only. It does not establish full-game player memory, feature extraction, learning, or JAX mechanics parity.


## Task 012 — Gen 5 damage-finalization tail

**Status:** Accepted and complete — 2026-10-29.

**Scope:** `finalize_damage(damage, modifier)` implements only the pinned tail after type effectiveness and burn: replace zero with one, apply one already encoded final modifier using Task011, and truncate to 16 bits. Strict int32 inputs broadcast, with caller-precondition domains damage `[0,2147483647]` and modifier `[0,65535]`. It excludes upstream damage calculation, event dispatch, modifier chains, bypassProtect, and runtime range checks. Zero can be returned after the modifier or final truncation.

**Implementation plan:**

1. Inspect the pinned `BattleActions.modifyDamage` tail and retain the Gen 5 order.
2. Reuse `apply_fixedpoint_modifier`; do not duplicate its wrap/rounding arithmetic.
3. Generate a pinned oracle fixture from actual `getDamage`: ordinary initialized Pokemon and Life Orb, plus explicitly synthetic pre-final damage inputs injected at the randomizer in a neutral path. Register a real event calling `chainModify(modifier,4096)` and delegate the actual finalization arithmetic.
4. Check eager/JIT/JIT-vmap, broadcast, strict dtype, order and wrap anchors, ordinary item behavior, and composition against the existing Task010 fixture. Check oracle wrapper/event restoration on success and forced failure.
5. Run regressions, refresh Trail, and record environmental validation limits.

**Acceptance evidence:**

- [x] Kernel preserves zero-to-one before modifier and uint16 truncation afterward, including possible final zero.
- [x] Oracle fixture records the pin, source spans, input domains, seed, and synthetic interception boundary. Eight ordinary and 430 synthetic cases reproduce byte-identically in independent CLI processes.
- [x] Real ordinary event modifiers are asserted as 4096 and 5324; synthetic modifier zero is honored. Oracle restores method/event descriptors even after failure.
- [x] All 54 Python tests pass on CPU, including seven new differential/composition tests.
- [x] All 35 Node tests pass with subprocess execution permitted.
- [x] Final review accepted the bounded implementation and validation evidence.

**Deferred validation:** GPU validation is unavailable in this CPU-only environment. Task012 acceptance covers CPU parity; it does not claim GPU parity. Rerun the Python suite on a CUDA-capable environment when available.

**Verification run:** On 2026-10-29, JAX/JAXLIB 0.12.0.1 CPU ran all 54 Python tests successfully in 16.7 seconds. The fixture matched byte-for-byte across independent CLI runs. The runtime has no available CUDA backend; `JAX_PLATFORMS=cuda` fails backend initialization. Node v24.19.0 passes 34/35 tests with `--test-isolation=none`; the sole failure is the unchanged trace fresh-process test. A minimal `child_process.spawnSync('node', ...)` probe returns `EPERM` and empty stdout in this sandbox, which explains its JSON parse error. Direct shell-invoked trace capture and replay succeed for the same 553-event, 102-turn battle and outcome. The same complete Node suite was then rerun with sandbox escalation allowing the existing subprocess test and passed 35/35 in 3.0 seconds. No existing trace source or test was changed. Trail build and staged diff checks pass. Scoped Trail check reports two reviewed lexical false positives: it matches final-damage input prose to a stored-stat positivity warning, and trusted oracle evidence to a player-observation warning. This kernel makes no stored-stat claim and exposes no oracle data to an actor.

**Resume/check commands:**

```sh
node reference/finalize-damage-oracle.cjs fixture > reference/fixtures/finalize-damage-v1.json
JAX_PLATFORMS=cpu .venv-jax/bin/python -m unittest discover -s test -p 'test_*.py' -v
node --test test/*.test.cjs
trail build
trail check --base HEAD
git diff --check
```

**Next bounded task:** Task013 can implement the pinned type-effectiveness arithmetic with an explicit clamped exponent, using actual Showdown differential evidence. Keep immunity decisions, type lookup, STAB, burn, modifier chains, and the complete damage engine out of that slice. Resolve its precise input domain and overflow behavior from the pinned code before implementation.

## Task 013 — Gen 5 type-effectiveness arithmetic

**Status:** Accepted and complete — 2026-10-29. Scheduled run 1. The architect accepted the implementation after reviewing the diff and validation evidence.

**Scope:** `apply_type_effectiveness(damage, exponent)` applies only the pinned type exponent arithmetic. Inputs are strict int32 and broadcast; damage has caller-precondition domain `[0, 33554431]`, exponent is any signed int32 and is clamped to `[-6, 6]`. The output remains nonnegative int32, with the largest positive result `33554431 * 64 = 2147483584`. Positive exponents repeatedly multiply by two without truncation; negative exponents repeatedly truncate `damage / 2` toward zero. This excludes type lookup, immunity, STAB, burn, modifier chains, and the complete damage engine; ranges are not dynamically checked.

**Implementation plan:**

1. Use the actual pinned `BattleActions.getDamage` path with a guarded fixture. Ordinary cases use real type effectiveness. Synthetic cases replace only the randomizer output and type-effectiveness lookup; a temporary wrapper captures the value at the post-type burn boundary and delegates Showdown arithmetic. Restore every patched descriptor in `finally`.
2. Add a pure JAX int32 kernel, with exponent clamp and broadcast behavior, then differential tests for ordinary and synthetic oracle records, clamp extremes, odd/floor cases, and the maximum domain boundary.
3. Run targeted CPU validation, report GPU availability accurately, refresh the Trail graph, run `trail check`, and record exact evidence before requesting architect review.

**Acceptance evidence:**

- [x] Strict int32 inputs, broadcast behavior, explicit exponent clamp, and the documented nonnegative damage precondition are implemented. The domain maximum times `2**6` is `2147483584`, so positive doubling does not overflow int32.
- [x] The guarded fixture records four ordinary initialized-Pokemon cases with real neutral, resisted, and super-effective type lookup, plus a 65-case synthetic grid over all 13 clamped exponents for damage values 0, 1, 63, 65, and 33554431. Three labeled synthetic anchors exercise signed-int32 clamp extremes. The synthetic boundary observes the value before burn and Gen 5 minimum-one; the fixture separately records downstream results so those later steps cannot be mistaken for this kernel.
- [x] Differential checks cover odd repeated flooring, zero, all clamped exponents, both signed-int32 clamp extremes, exponent endpoints, and the maximum damage boundary. Eager, JIT, vmap, and 2D broadcasting agree with the live pinned oracle.
- [x] Oracle wrappers and temporary source status restore after both successful capture and an injected failure.
- [x] `JAX_PLATFORMS=cpu .venv-jax/bin/python -m unittest test.test_type_effectiveness -v` passes 4/4. `node --test test/type-effectiveness-oracle.test.cjs` passes 1/1. `trail build` and `git diff --check` pass.

**Deferred validation:** This environment exposes only `[CpuDevice(id=0)]`; no GPU validation was available. The installed JAX and JAXLIB reported 0.12.0.1 during this run; the tests do not assert those exact versions.

**Trail review:** `trail check --base HEAD` reports two lexical conflicts in the new task section, matching the unrelated stored-stat cap/zero-domain learning and trusted-oracle-versus-player-observation learning. This task neither applies stat caps nor exposes its oracle fixture to an actor.

**Review:** Architect diff and evidence review accepted Task013. The implementer ran 4/4 focused CPU Python tests, 1/1 Node restoration test, Trail build, and diff checks. The parent will push the local commit to `main`.

**Next bounded task:** Task014 can add the Gen 5 physical-burn reduction as an explicit arithmetic slice between type effectiveness and the Task012 finalization tail. Keep burn eligibility inputs explicit and defer event dispatch.
