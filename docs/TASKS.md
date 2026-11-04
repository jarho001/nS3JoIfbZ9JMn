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

## Task 014 — burn-reduction kernel

**Status:** Accepted and complete — 2026-10-29. Scheduled run 2.

**Scope:** Implement `apply_burn_reduction(damage, burned, physical, has_guts)` as a pure arithmetic slice after type-effectiveness and before finalization. It uses explicit strict-boolean eligibility inputs and delegates the eligible `0.5` modifier to the existing fixed-point kernel. The pinned `BattleActions.getDamage` path in `gen5randombattle` reduces burned physical moves unless Guts is active; Facade receives the same reduction. This task does not resolve runtime status, move category, or ability, dispatch events, or include any other damage steps.

**Bounded implementation plan:**

1. Use the pinned Gen 5 `modifyDamage` condition as the guarded oracle, with ordinary real moves for physical/special, burned/unburned, Guts, and Facade checks. For arithmetic boundaries, control only the randomizer output, capture immediately before burn through a transparent `Battle.modify` wrapper, and delegate to the original implementation.
2. Implement strict int32 damage and strict bool eligibility inputs with broadcast semantics. Apply the existing `apply_fixedpoint_modifier(damage, 2048)` only when `burned & physical & ~has_guts` is true; otherwise return damage unchanged.
3. Cover real-oracle fixtures, synthetic low/odd/even/zero/int32-max/2**21-wrap cases, the zero intermediate before Gen 5 minimum-one, dtype/predicate/broadcast behavior, and restoration of every temporary descriptor and status after success or failure.
4. Run the focused Python and Node tests once, record CPU/GPU availability, refresh Trail, check the diff against Trail learnings, then update README and this handoff with exact evidence.

**Acceptance criteria:**

- [x] Eligible damage matches pinned `Battle.modify(damage, 0.5)`; ineligible damage is unchanged. The predicate is exactly `burned & physical & ~has_guts`, with no Gen 5 Facade exemption.
- [x] Inputs are strict int32 damage in `[0, 2147483647]` and strict bool eligibility flags; values broadcast normally and outputs are int32. Eligibility ranges are represented by booleans, not resolved from simulator state.
- [x] Guarded fixture has ordinary real Gen 5 examples for burn, physical/special, Guts, Facade, and unburned conditions. Synthetic cases inject only randomizer output and use a transparent modifying wrapper that delegates to Showdown; they cover zero, low odd/even, int32 maximum, and 2**21 fixed-point wrap boundaries, including a pre-minimum zero result.
- [x] Oracle monkeypatch descriptors and temporary status restore after successful capture and injected failure.
- [x] Focused Python eager/JIT/vmap/broadcast/dtype/predicate tests and Node fixture/restoration test pass. Record available CPU/GPU devices without dependency-version assertions.
- [x] README documents this slice's boundaries. Trail build and diff checks pass; Trail check's one blocking hit is a lexical false positive in the scope description against the unrelated stat-cap learning. The other printed hits are explicitly non-blocking and unrelated to this arithmetic slice. Task015 is proposed below.

**Verification evidence:** On 2026-10-29, `JAX_PLATFORMS=cpu .venv-jax/bin/python -m unittest test.test_burn -v` passed 4/4. `node test/burn-oracle.test.cjs` passed 2/2, including pinned fixture reproduction and descriptor/status restoration after success and injected failure. The equivalent `node --test test/burn-oracle.test.cjs` worker launch was attempted with a narrow sandbox escalation, but automatic approval did not finish before its deadline; direct `node:test` execution passed. JAX reported `[CpuDevice(id=0)]` with backend `cpu`; no GPU is available. `trail build`, `trail build --check`, and `git diff --check` passed. `trail check --base HEAD` reports one blocking lexical false positive at this task's scope line: it matches the unrelated stat-cap learning, although this task neither changes nor models stat caps. Other printed hits are non-blocking matches to unrelated randomizer and stat-stage learnings. No conflicting implementation behavior was identified.

**Review:** Architect reviewed and accepted the implementation and evidence. The implementer ran the focused Python and Node tests above; the parent reviewed the diff and ordinary differential evidence. A local Task014 commit is being created; the parent handles pushing it to `main`.

**Next bounded task proposal:** Task015 can implement the explicit Gen 5 STAB multiplier as one arithmetic slice before type effectiveness, with STAB and any applicable ability state supplied explicitly. Resolve its exact pinned multiplier/rounding boundary first; keep type lookup, burn, finalization, event dispatch, and the full damage calculator outside that task.

## Task 015 — Gen 5 STAB arithmetic

**Status:** Accepted and complete — scheduled run 3.

**Scope:** Implement `apply_stab(damage, stab_eligible, adaptability, typeless)` as a pure arithmetic slice immediately after the randomizer and before type effectiveness. Inputs are strict int32 damage in `[0, 2147483647]` and strict boolean flags, broadcast normally. `typeless=True` bypasses STAB entirely and returns damage unchanged, even when the eligibility flags conflict. Otherwise use the pinned modifier: 1.5 (`6144`) for eligible non-Adaptability, 2 (`8192`) for eligible Adaptability, and identity (`4096`) for typed non-STAB. Typed identity still invokes fixed-point arithmetic and can wrap; reuse `apply_fixedpoint_modifier` rather than cloning it. The explicit eligibility flag already resolves native/force STAB, and Adaptability is supplied as effective state. Exclude type and ability lookup, event dispatch, Tera, other generations, type effectiveness, burn, finalization, and the full damage calculator.

**Implementation plan:**

1. Confirm the pinned `BattleActions.getDamage` STAB block and Adaptability handler; record any disagreement before coding. Generate the fixture through guarded actual `getDamage` calls, observing real modifying steps. Ordinary cases use real moves, including native STAB, non-STAB, Adaptability, and forceSTAB where feasible. Synthetic cases replace only randomizer output and inject explicit move/ability state as needed; a transparent wrapper records and delegates each applicable `Battle.modify` call. Assert that `???` skips the modifying call. Restore patched descriptors and temporary move fields on success and forced failure; source status and ability remain unchanged.
2. Implement the strict-dtype JAX kernel using the existing fixed-point modifier for all typed cases and an unchanged branch for typeless damage.
3. Cover actual oracle parity plus all eligibility/Adaptability/typeless combinations, broadcasting, eager/JIT/vmap, dtype rejection, synthetic outputs 0/1/2/3, fixed-point product-wrap boundaries for each of the three typed modifiers, and int32 maximum.
4. Add a deterministic pinned fixture reproduction check; run focused CPU tests, report device availability, refresh/check Trail, and record exact validation evidence.

**Acceptance criteria:**

- [x] Pinned `sim/battle-actions.ts:1809-1847` confirms typed non-STAB executes the identity modifier and `???` skips the entire STAB block. `data/abilities.ts:43-54` confirms Adaptability resolves STAB to 2; ordinary STAB resolves to 1.5.
- [x] Strict int32 damage and strict bool flags broadcast normally and return int32. `typeless` bypasses all typed modifier paths even with conflicting flags. Eligibility and effective Adaptability are explicit inputs; no simulator lookup or event logic is included.
- [x] Reuse `apply_fixedpoint_modifier` for all three typed modifiers. Preserve its uint32 product wrap and half-down rounding behavior, including at int32 maximum.
- [x] Guarded fixture uses the real pinned `getDamage`/`ModifySTAB`/Adaptability path; synthetic input control does not clone the formula. Five ordinary cases and 47 synthetic cases match eager/JIT/vmap tests.
- [x] Oracle wrapper/state restoration passes on success and injected failures. The fixture reproduces deterministically with Node.
- [x] Focused CPU tests, Trail build/check, and diff check pass. The JAX runtime reports only CPU; no GPU coverage is claimed.

**Verification evidence:** On 2026-10-29, `JAX_PLATFORMS=cpu .venv-jax/bin/python -m unittest test.test_stab -v` passed 4/4; `node test/stab-oracle.test.cjs` passed 2/2. Two independent `node reference/stab-oracle.cjs fixture` runs produced byte-identical output. The fixture covers native STAB, typed non-STAB, Adaptability with and without STAB, forceSTAB, explicit synthetic damage 0/1/2/3, values immediately adjacent to `floor(2^32 / modifier)` for modifiers 4096/6144/8192, int32 maximum, and typeless conflicting flags. Tests cover eager/JIT/JIT-vmap, all eight eligibility/Adaptability/typeless combinations, 2D broadcasting, and strict dtype rejection. The oracle restores randomizer/modify/runEvent/target.runEffectiveness descriptors and move fields after successful and forced-failure captures. JAX reports `[CpuDevice(id=0)]`, backend `cpu`; `nvidia-smi` is unavailable. Trail build, Trail graph freshness check, and `git diff --check` pass. Scoped Trail check's remaining hit, if any, is recorded below after final run.

**Trail review:** `trail build` and `trail build --check` pass. `trail check --base HEAD` reports no conflicts; its seven “worth a look” matches are existing unrelated guidance matches elsewhere in README/TASKS. `git diff --check` passes.

**Review:** Architect reviewed the corrected native/forced eligibility distinction, live STAB capture boundary, oracle state coverage, and focused test evidence; accepted Task015. Task014 was pushed as `2b9b00e` before this scheduled run. This implementation is ready for the authorized local commit; the parent handles pushing.

**Next bounded task:** Task016 should compose the already implemented randomizer, STAB, type-effectiveness, burn, and finalization slices only after Task015 review. Keep its scope to a bounded arithmetic composition with explicit intermediate inputs; do not add runtime type/ability lookup, event dispatch, or a full damage engine.

## Task 016 — bounded post-core damage composition

**Status:** Accepted — scheduled run 4.

**Scope:** Add `damage_from_preroll(pre_roll_damage, roll, stab_eligible, adaptability, typeless, type_exponent, burned, physical, has_guts, final_modifier)`.
Compose the existing arithmetic slices in order: roll, STAB, type effectiveness, burn reduction, finalization.
The arithmetic starts just before randomizer arithmetic and ends after finalization. Int32 fields and boolean flags use ordinary JAX broadcasting. Caller preconditions: `pre_roll_damage` `[0, 33554431]`, `roll` `[0, 15]`, signed-int32 `type_exponent` (clamped by the type kernel), `final_modifier` `[0, 65535]`. The API does not check ranges.
The bounds keep the randomizer product below `2**32`, its result within `[0, 33554431]`, and the positive type result at most `2147483584`. Inputs beyond the preroll bound do not claim the standalone randomizer's uint32-wrap behavior.
The slice starts after the core and earlier modifiers. It excludes `+2`, weather, critical, spread, RNG generation, event dispatch, bypassProtect, and other damage stages.

**Implementation plan:**

1. Capture through pinned `BattleActions.getDamage` and return its actual final result.
Ordinary cases use real moves and initialized ability/status/type state. The original randomizer remains active; only its damage `random(16)` draw is controlled.
Synthetic cases replace the input passed to the original randomizer, then delegate to that real randomizer. They explicitly control `target.runEffectiveness` and a registered final modifier event where needed.
Record only observed intermediate operations and assert `bypassProtect` is false.
Restore patched descriptors, temporary event registry, and status after success or injected failure.
2. Add the pure JAX composition using the five existing primitives, with no duplicated arithmetic.
3. Differentially cover ordinary and synthetic behavior, every roll 0–15, order-sensitive anchors, zero/odd low values, type exponent clamp extremes, typeless and typed STAB, burn eligibility, zero final modifier and uint16 truncation. Verify eager, JIT, vmap, broadcasting, and strict dtypes.
4. Run focused Python and direct Node tests plus relevant constituent regression tests once, record current CPU/GPU availability, refresh and check Trail, inspect the diff, update README and this handoff, and report exact evidence for architect review.

**Acceptance criteria:**

- [x] Composition delegates to the five existing primitives in the required order and keeps every field strict int32/bool with broadcasting; range bounds are documented preconditions, not dynamically checked.
- [x] Pinned oracle calls actual `getDamage` and returns its actual final result. Ordinary calls retain real move/type/status/ability and original randomizer behavior with a controlled damage draw. Synthetic calls inject only via the original randomizer input and controlled effectiveness/final event as declared; assert actual flags, event arguments, and captured stage values. Assert `bypassProtect` is false.
- [x] Fixtures include all 16 rolls, high bounded preroll, type clamp boundaries, typed/typeless, physical burn and Guts bypass, zero-before-finalization and zero final modifier, plus anchors that distinguish the required order from reversed steps and expose final uint16 zero truncation.
- [x] Oracle monkeypatches, temporary event state, and statuses restore after success and forced failure.
- [x] Eager/JIT/vmap/broadcast/dtype checks and focused constituent regression tests pass. JAX reports CPU only; no GPU is available.
- [x] README documents the composition boundaries and bounded-domain limitation. Trail build/check and `git diff --check` complete; Task017 proposal builds on this slice.

**Verification evidence:** Before the final oracle assertions, the relevant Python constituent set passed 30/30 with `JAX_PLATFORMS=cpu .venv-jax/bin/python -m unittest test.test_damage_composition test.test_damage_roll test.test_stab test.test_type_effectiveness test.test_burn test.test_finalize_damage -v`. The direct Node constituent oracle tests passed: STAB 2/2, type effectiveness 1/1, burn 2/2, and finalization 1/1. After the final oracle/test edits, the composition-only rerun passed `JAX_PLATFORMS=cpu .venv-jax/bin/python -m unittest test.test_damage_composition -v` 4/4 and `node test/damage-composition-oracle.test.cjs` 2/2. The Node test regenerates the pinned fixture deterministically and checks temporary descriptor/event/status restoration after success and an injected failure. The fixture contains five ordinary cases spanning real native STAB, typed non-STAB, burned physical, burned physical with Guts, Adaptability, and Life Orb, plus 25 synthetic cases. Synthetic calls pass the selected preroll into the original `Battle.randomizer`, whose sole `random(16)` draw is controlled; captured stage outputs and actual final returns match the JAX composition. Device discovery reports `[CpuDevice(id=0)]`; `nvidia-smi` is unavailable. No GPU validation is claimed.

**Trail review:** `trail build` and `trail build --check` pass. `trail check --base HEAD` reports two lexical false positives in this task section: the `apply_type_effectiveness` kernel name is matched to a warning about optional stat caps, although this task does not derive or modify stats; the type/arithmetic oracle is matched to a warning about player observation, although this fixture is trusted engineering evidence and is never exposed to a player. Other reported matches are marked non-blocking by Trail. `git diff --check` passes.

**Review:** Architect reviewed and accepted the diff and evidence. Task015 was pushed as `0bb350e`; the parent handles pushing Task016. The authorized local Task016 commit is created after final checks.

**Next bounded task proposal:** Task017 can compose a pre-roll prefix from `base_damage_core`, Gen 5 `+2`, and Task010's neutral `WeatherModifyDamage` fixed-point identity step, then feed that preroll into `damage_from_preroll` once. Effective Attack/Defense, resolved flags, and roll stay explicit. Capture the real `modifyDamage` tail and verify the event occurs before randomization; do not apply a second roll. End at final damage and continue to exclude runtime event/type/ability lookup and RNG generation.

## Task 017 — neutral full damage composition

**Status:** Accepted and complete — scheduled run 5.

**Scope:** Add `neutral_damage_before_roll(level, base_power, effective_attack, effective_defense)` by extracting the existing Task010 prefix: `base_damage_core`, int32 `+2`, then the existing neutral `WeatherModifyDamage` identity fixed-point operation. Keep `neutral_damage_after_roll` behavior unchanged by calling the new prefix and applying its explicit roll once. Add a clearly named composition from effective stats to final damage that calls this prefix and then the existing Task016 `damage_from_preroll` exactly once, passing all flags, exponent, roll, and final modifier explicitly. Inputs use the established domains: level `[1, 100]`, base power `[1, 65535]`, effective stats `[0, 65535]`, roll `[0,15]`, signed-int32 type exponent, final modifier `[0,65535]`; no dynamic range checks. Core plus two is safe in int32, and neutral weather constrains pre-roll to `[0,1048575]`, within Task016's accepted preroll bound.

**Implementation plan:**

1. Extend the guarded pinned `getDamage` capture to observe actual effective level, `BasePower`, category-appropriate `ModifyAtk`/`ModifySpA` and `ModifyDef`/`ModifySpD` outputs, and the core result. Record the actual source/target weather event with zero handlers, then delegate through the original randomizer and final tail. Ordinary initialized cases cover STAB/non-STAB, Adaptability, burn/Guts, and Life Orb. Synthetic cases mutate only level, base power, and stored Attack/Defense for core wrap, zero Defense, and event inputs 1,048,575/1,048,576/1,048,577. Force and assert `willCrit=false`, assert one core call, one weather event, and one `random(16)` draw; restore temporary descriptors/status/stats/events in `finally` after success and failure. Do not fabricate unobserved intermediates or infer them from final damage.
2. Extract `neutral_damage_before_roll`, preserve the old API through it, and add `damage_from_effective_stats_neutral` by composing the prefix with `damage_from_preroll`; add no duplicated formulas or stat derivation.
3. Differentially cover all 16 rolls over compact ordinary and selected synthetic cases, plus order anchors distinguishing `+2`/weather from roll and proving no duplicate roll. Exercise strict dtype and eager/JIT/vmap/broadcast behavior; verify fixture reproduction and state restoration.
4. Run focused Python and direct Node oracle checks, inspect CPU/GPU availability, refresh Trail, check the diff, and record exact evidence below. Then propose the next bounded architecture slice.

**Acceptance criteria:**

- [x] The extracted prefix returns the pinned `base_damage_core -> +2 -> WeatherModifyDamage` result, including uint32 core wrap, zero-defense behavior, and neutral weather's 20-bit wrap.
- [x] `neutral_damage_after_roll` is numerically unchanged and applies exactly one explicit roll after the extracted prefix.
- [x] The new full composition routes effective inputs through prefix and `damage_from_preroll` exactly once; no formulas or stats are derived locally. Inputs remain strict dtypes and broadcast as constituent kernels do.
- [x] The guarded oracle uses actual pinned `getDamage`, confirms core and weather call order, original randomizer and downstream finalizer, one controlled `random(16)`, and actual effective inputs. Ordinary and synthetic cases exercise the required categories and edge domains without fabricating intermediates.
- [x] Temporary oracle state is restored after successful and failed captures. The fixture reproduces deterministically; focused Python/Node checks, Trail build/check, and diff check pass. CPU/GPU availability is reported accurately.

**Verification evidence:** `JAX_PLATFORMS=cpu .venv-jax/bin/python -m unittest test.test_neutral_damage test.test_base_damage test.test_damage_composition test.test_neutral_damage_composition -v` passed 21/21; after the final fixture witness update, `JAX_PLATFORMS=cpu .venv-jax/bin/python -m unittest test.test_neutral_damage_composition -v` passed 3/3. Direct `node test/neutral-full-damage-oracle.test.cjs` and `node test/damage-composition-oracle.test.cjs` each passed 2/2, covering the new full-path fixture/restoration and prior Task016 fixture regression. The captured full fixture contains five ordinary cases and six synthetic cases, each covering rolls 0–15. It observes core inputs from actual pre-core event results, checks core/weather counts, actual event source/target, zero weather handlers, and neutral wrap values at 1,048,575/1,048,576/1,048,577. A post-roll witness distinguishes one roll from a duplicate. The prior Task016 fixture is unchanged byte-for-byte. JAX reports `[CpuDevice(id=0)]` with backend `cpu`; no GPU device is available. `trail build`, `trail build --check`, and `git diff --check` pass. `trail check --base HEAD` reports three lexical conflicts: the Task017 scope states the established effective-stat domain `[0,65535]` (the zero value is specifically valid), the test plan mentions full oracle evidence (trusted engineering fixture data, never a player observation), and `neutral_damage_before_roll` names the neutral core/event composition. They do not contradict their matched learnings: the stats are explicit effective values, fixture evidence stays in the trusted reference harness, and the prefix delegates to the existing neutral fixed-point event after core plus two. Remaining printed matches are non-blocking guidance hits.

**Review:** Architect reviewed and accepted the diff and evidence. Task016 was pushed as `9590e91`; the parent handles pushing Task017.

**Progress:** Scheduled run 5 is finalized. No unresolved behavior difference found.

**Next bounded task proposal:** Task018 completed the G0 mechanics coverage ledger below. Task019 should address the remaining G0 reuse decision by auditing the named candidate projects; its bounded scope and evidence requirements are recorded at the end of Task018.

## Task 018 — G0 mechanics coverage ledger

**Status:** Accepted and complete — 2026-10-30. Architect review accepted the coverage inventory, evidence links, and G0 boundary. Task017 base commit `b0635e8` was confirmed pushed.

**Scope:** Documentation audit only. Create `docs/MECHANICS_COVERAGE.md` enumerating each public function in `mechanics/` exactly once, with source symbol, supported input/dtype/broadcast contract, observed operation, pinned oracle, fixture, Python test, evidence type, and explicit exclusions. Distinguish atomic kernels from compositions; ordinary initialized Showdown cases from controlled synthetic probes; historical GPU evidence for Tasks005–011 from CPU-only or deferred evidence for later tasks. Include the audit base commit and pinned upstream revision. Keep format metadata, protocol projection, random-policy baseline, and trusted trace infrastructure separate from JAX mechanics. Update README and this handoff before final audit evidence.

**Acceptance criteria:**

- [x] Every public mechanics function is inventoried once (or grouped only where the grouping is explicit), with exact linked code/evidence paths and accurate caller bounds.
- [x] The ledger states the presently supported neutral full-damage path, including its limits, and does not imply arbitrary damage, complete battle support, or an inflated coverage percentage.
- [x] Ordinary and synthetic evidence are labeled distinctly; current CPU evidence and historical GPU evidence are reported accurately.
- [x] Reference-backed infrastructure is presented outside JAX mechanics coverage. The missing battle state, scheduler, coupled PRNG, rules/team-generation data, player-view/full-game transfer, and learning benchmarks are named as gaps.
- [x] G0 is explicitly called partial, not complete; README and task progress link the ledger.
- [x] Scoped link/inventory/reference checks, `trail check`, and `git diff --check` are reviewed. No mechanics source edits, full test suite, or committed validation tooling.

**Plan:**

1. Read the repo instructions and Trail graph, then use the public function skeletons and task evidence to enumerate the mechanics API.
2. Cross-check each row against the pinned oracle's recorded source boundary, fixture schema/cases, and focused Python test; inspect linked source for uncertain bounds or sequencing.
3. Write the mechanics ledger, then update README and this task handoff. Keep percentages and unsupported full-engine claims out.
4. Run scratch link/inventory/reference checks plus Trail and diff checks. Review false positives as needed; record exact results below.

**Progress:** Public inventory identified as 14 functions across stat arithmetic and damage arithmetic/compositions; underscored helpers are private and excluded. The ledger, README summary, and task plan are complete. No source or test files were changed.

**Validation evidence:** A scratch Python AST inventory found exactly 14 public mechanics functions and 14 ledger rows; every source line anchor resolves to the named function definition, with no duplicate or missing functions. A scratch relative-link check found 69 Markdown links and all linked files exist. All 13 fixture JSON files with recorded upstream commits match `reference-lock.json`. `trail map`, `trail ask ... --source`, and `trail skeleton` informed the audit. `trail check --base HEAD` reports two blocking lexical false positives on the Task018 scope line: broad stat/input language matches an optional stat-cap learning, and oracle/test-evidence language matches a player-view learning. Architect review confirmed these do not describe the audit's arithmetic contracts or trusted fixture boundary; additional new-file hits are non-blocking lexical matches. `git diff --check` passes. No mechanics tests or current GPU probe were run because this task changes documentation only.

**Next bounded task proposal:** The largest explicit remaining G0 item is the reuse decision in `PROJECT_PLAN.md`. Task019 should audit the named simulator/integration candidates (Showdown, Jaxcalibur, PokeJAX research and Gen4 repository, `poke-env`, Metamon, and Pgx) against source availability, license, Gen 5 mechanics fit, observation fidelity, tests, maintainability, and throughput evidence. Report measured throughput only with workload, date, and source provenance; otherwise mark it unknown. Do not add benchmarking to this documentation audit, implement an engine, or start training. This follows the ledger because it separates the small reusable arithmetic evidence from the much larger engine, data, and interface gaps.

## Task 019 — simulator and integration reuse audit

**Status:** Accepted and complete — 2026-10-30.

**Scope:** Documentation-only audit of the seven projects named in `PROJECT_PLAN.md`: Pokémon Showdown, Jaxcalibur, the PokeJAX research paper, the separate `JerJer2465/pokejax` Gen 4 repository, `poke-env`, Metamon, and Pgx. Assess source availability, license evidence, Gen 5 target fit, player-view/observation fidelity, tests, maintenance signals, and throughput evidence. Create `docs/REUSE_AUDIT.md`; update README and this handoff. No dependency adoption, source copying, benchmarking, engine work, or training.

**Acceptance criteria:**

- [x] Each named candidate has primary-source links and an immutable revision where a source repository exists; non-repository sources are dated and labeled as author reports.
- [x] Availability, license evidence, Gen 5 fit, observation fidelity, tests/maintenance, and throughput are addressed, with unknowns distinguished from negative findings.
- [x] Any throughput figure names its workload, hardware, date/source, and limits; absent evidence is marked unknown. No candidate throughput is presented as this project's result.
- [x] Decision retains the pinned Showdown oracle and provisionally continues the custom, incremental JAX engine path rather than adopting an inspected external port. G0 remains partial and a single bounded next task is proposed.
- [x] README and this task handoff link the audit and accurately state its status. Scoped documentation checks and diff checks are complete; no source/tests are changed.

**Plan:**

1. Read repository instructions and ask Trail for project architecture, the named candidate set, and current G0 boundary.
2. Inspect the pinned local Showdown source/lock. Fetch candidate repository metadata, README/license/source/test paths and immutable commit references from GitHub; inspect the Jaxcalibur author write-up and PokeJAX paper directly.
3. Write the audit with primary evidence, separate self-reported results from verified source facts, record unknowns, and make a bounded reuse decision.
4. Update README and this task entry, then check links, table fields, revision references, Trail, and the diff. Do not benchmark or run full tests.

**Progress:** The pinned local Showdown checkout is clean at `1242c1e2d9763ba5b85d33f13aecb89bae19ae3d`; upstream had advanced to `ccfaa20f32af363569cd21a9d501b2bc1ded26b1`. GitHub source/metadata was inspected for Showdown, `poke-env`, Metamon, Pgx, and the separate Gen 4 engine. The Jaxcalibur author page and arXiv 2603.12145v2 were inspected. Evidence and bounded decision are recorded in [`REUSE_AUDIT.md`](REUSE_AUDIT.md). No dependencies were installed and candidate tests/benchmarks were not run. This scheduled invocation completed Task019; no Task019 deliverable remains. The next invocation begins Task020. The run ordinal is not inferred here, and the existing 30-run/five-day cap remains unchanged.

**Validation evidence:** Implementer-scoped Python checks found no missing local Markdown targets across `REUSE_AUDIT.md`, README, and this handoff. The audit matrix has seven candidate rows; static checks found the required throughput metadata, Gen 5 uncertainty, G0 boundary, next-task proposal, and repository SHAs. `git diff --check` passes. `trail check --base HEAD` reports one lexical conflict within the Task019 reuse-audit entry (not the Task020 proposal), matching a learning about not treating complete Showdown updates as player observations. The audit explicitly requires player-view projection and does not classify raw updates as observations, so the warning is lexical rather than a semantic conflict. No candidate test suite or benchmark was run, as scoped.

**Review:** Architect independently reviewed the audit and diff, spot-checked the PokeJAX Table 2 throughput and Jaxcalibur release statement, and accepted the documentation. The architect independently reran `git diff --check` (pass) and `trail check --base HEAD`; Trail reported one lexical conflict at `docs/TASKS.md:651` within the Task019 reuse-audit entry, matching the player-observation learning while the audit explicitly requires player-view projection. Other Trail matches were informational. The architect will handle commit and push.

**Next bounded task proposal:** Task020 should draft the benchmark charter and source/version manifest. Freeze fields supported by current evidence, such as the scored format revision and known observation/action contract. Specify the workload, seed procedure, throughput counters/warm-up handling, and data/team-generator provenance where evidence permits. Label unresolved hardware/resource budgets and other unsupported fields explicitly; do not benchmark or train.

## Task 020 — benchmark charter and source/version manifest

**Status:** Accepted and complete — 2026-11-02.

**Scope:** Documentation-only. Create `docs/BENCHMARK_CHARTER.md` to define a versioned, evidence-backed Showdown reference workload and current observation/action contract, with source/team-generator provenance and a proposed reproducibility protocol for later measurements. Update README and this handoff. Separate implemented facts, proposed protocol, and unresolved budgets; retain the current G0/G1 boundary. No source/dependency edits, tests, benchmarks, or training.

**Acceptance criteria:**

- [x] The pinned rule identity, upstream revision, effective-rule fixture, team-generator provenance, and source paths point to local evidence and immutable upstream files.
- [x] The request/action contract distinguishes each player's private request and projected history from trusted combined traces; supported singles request shapes and retry behavior follow local evidence.
- [x] Proposed workload, seed hierarchy, counters, timing/warm-up/device synchronization, and reproducibility record are concrete enough for a future benchmark implementation without claiming current benchmark results.
- [x] Policy invocations, sent and resolved choices, terminal turn indexes, canonical log records, terminal games, retries, failures, and deadlines have distinct definitions and report treatment; existing counters are labeled according to what they actually count.
- [x] Existing implementation, proposed protocol, unresolved hardware/resource/statistical budgets, and unavailable full JAX match-engine/learning workloads are explicit. No invented speed, throughput, or completed G0 claim.
- [x] README and task handoff link the charter, preserve the original 30-run/five-day cap and avoid fabricating a scheduled-run ordinal. Only documentation files change.
- [x] Scratch link/manifest/source-consistency checks, Trail diff check, and `git diff --check` are recorded. Stage documentation, then run Trail check on the staged/new-file diff. Do not commit or push.

**Plan:**

1. Read repo instructions and Trail context; inspect the Task019 audit, source lock, format fixture, and runner/protocol evidence needed for the contract.
2. Record this bounded plan in the handoff before drafting the charter.
3. Draft the charter with an embedded manifest and visibly label implemented facts, proposed benchmark protocol, and unresolved decisions.
4. Update README and the handoff, then run scratch Markdown-link, lock/fixture, and source-contract checks plus diff review. Stage only the documentation files and run `trail check` against the staged diff; report exact evidence and any lexical matches without treating them as source validation.

**Progress:** The bounded plan was recorded before drafting. [`BENCHMARK_CHARTER.md`](BENCHMARK_CHARTER.md) now records the pinned format/source manifest, current per-player projection and own-request contract, proposed complete-battle legal-random reference workload with explicit team/policy seat pairing, separate decision/submission/resolution/turn-index/event/battle/failure counters, timing rules, workload tiers, and future result fields. The generator dispatch and Gen 5 implementation are linked to verified files in the pinned local checkout. Implemented evidence, proposed protocol, and unresolved budgets are labeled separately. The single default seed case is identified as a smoke sample; no speed claim, benchmark result, complete-G0 claim, or full JAX battle/learning capability is asserted. Original run ordinal is not inferred and the 30-run/five-day cap remains unchanged.

**Validation evidence:** A scratch check found all local Markdown links resolve across the charter, README, and handoff; the fixture's upstream commit and immutable GitHub source links match `reference-lock.json`; format fields, five runner seed tuples, runner limits, and JAX version match their source files. Upstream `config/formats.ts`, `sim/teams.ts`, and `data/random-battles/gen5/teams.ts` external links were checked against the same files in the clean pinned local checkout. `git diff --check` passes. Staged `trail check --base HEAD` reports lexical matches on the handoff and charter about audience-specific split data, format-wide stat caps, and unrelated damage arithmetic. The charter explicitly limits policy inputs to each player's own projected history/request, distinguishes top-level stream errors from a player's retained private rejection feedback, and takes effective rules from the pinned fixture; no match indicates a semantic conflict. No tests, benchmarks, training, or source edits were made.

**Review:** Primary review accepted the source manifest, workload tiers, concrete paired seed procedure, per-player protocol description, turn-index definition, and current-versus-proposed capability boundary. One correction clarified that top-level stream errors fail closed while addressed private sideupdate errors remain player feedback classified by the runner. No deliverable remains unfinished.

**Next bounded task proposal:** Implement versioned per-battle result records and separate counters/timing/failure fields from the charter in the Showdown runner, preserving its smoke path. Keep suite size, hardware allocation, and measured throughput outside that instrumentation task.

## Task 021 — versioned per-battle runner result records

**Status:** Accepted and complete — 2026-11-02. Primary review accepted Task021 after the documented corrections; accepted count is 21.

**Scope:** Add a versioned per-battle result record to the pinned Showdown runner. Record pin, format, five seeds, configured limits, explicit completed/failed status, validated outcome or null, monotonic elapsed milliseconds, and separate counters for policy calls, successfully sent choices, choice resolutions, retry requests/submissions/exhaustion, moves/switches, public events, and latest public turn. Categorize failures at typed call sites, retain partial counters on failure, and make the record available through a returned value or attached error. Add focused fake-stream coverage, preserve the existing summary and trusted trace transcript, and document the schema and optional `--result PATH` output. Update this handoff, README, and charter language so they distinguish these implemented fields from proposed aggregate benchmarking rules.

**Exclusions:** No suite-size/statistical choice, benchmark aggregation/orchestration, hardware allocation, throughput claims, mechanics/JAX work, or G0 closure. No changes to policy visibility or the trusted-transcript boundary. No commit or push by the implementation owner.

**Acceptance criteria:**

- [x] A schema-versioned record identifies the pinned revision, format, all five seeds, effective configured limits, status, validated outcome or null, and finite nonnegative monotonic `elapsedMs` spanning team generation through terminal validation or failure. Timing and new counters stay out of canonical transcript and existing summary.
- [x] Actual policy invocations are counted by player and total, separately from successfully sent submissions. Accepted, rejected, and pending sent submissions reconcile to total submissions. Retry requests, successful retry submissions, retry-limit exhaustion, moves/switches, public projected events, latest public turn, and terminal metadata turn index are distinct. Wait/duplicate pending requests do not invoke policy; guards do not count nonexistent invocations.
- [x] Failure records retain partial counters and explicitly categorize initialization/configuration, protocol/terminal validation, policy/invalid choice, runtime/infrastructure, packet/decision/retry limits, and timeout. Result records omit raw private packet/error text. Pre-stream initialization, stream, and cleanup failures are covered as applicable while thrown errors and cleanup behavior remain compatible.
- [x] Focused fake-stream tests cover success, retries, waits/duplicates, split public-event counting, invalid policy, limit exhaustion, timeout, premature end, contradictory terminal metadata, and early initialization failure. Existing pinned seeded smoke and trace regression remain deterministic.
- [x] README and charter document schema, units, failure access, CLI output, and that elapsed time is instrumentation rather than a throughput claim. Paired-history visibility boundaries remain explicit.
- [x] Targeted validation and diff inspection are recorded. Trail CLI availability is reported honestly; primary acceptance is recorded, the accepted count is 21, and the review checkpoint remains due at 30.

**Plan:**

1. Read the orchestration policy and Trail cards for the runner, CLI, protocol projection, and tests; inspect only the cited source spans. Confirm the accepted task count and clean base tree.
2. Implement a separate result-record path with monotonic timing and typed categorized failures, preserving the existing summary/transcript shape and projection boundary.
3. Add exact fake-stream assertions for counters and failure snapshots, then update README, charter, and this task evidence.
4. Run the focused runner tests, pinned seeded smoke/trace regression, `git diff --check`, and available Trail checks. Review the diff and report changed files, exact results, and unresolved concerns to the parent without accepting the task or committing.

**Progress:** Plan persisted before implementation. `runBattle` returns a separate version 1 result record on success and attaches it to failures. The record uses branded internal failures, safe categories/codes, a monotonic elapsed scope, reconciled sent-choice resolutions, and spectator-projected public event/turn counters; summary and trusted transcript fields remain stable. `--result PATH` writes success and failure records, including early CLI initialization errors. README and charter state that result records are trusted engineering metadata and never policy inputs, and elapsed time is not throughput. Trail CLI is unavailable in this environment (`command -v trail` returned no executable); the checked-in Trail index and exact runner/protocol/test cards were read first. The base tree was clean at `eca7a3e221d23ccf25fa5cedcba69bd2bf0be771`; the accepted count was 20 at task start and became 21 after review.

**Validation evidence:** `node --check environment/battle-runner.cjs` passes. `node --test test/battle-runner.test.cjs` passes 21/21, including exact successful/retry/wait/duplicate counters, split public projection, invalid policy, choice-write failure, packet/decision/retry limits, timeout, premature/contradictory terminal data, initialization failure, cleanup failure, malicious stream error category/code, successful CLI `--result`, failed CLI `--result`, summary stability, and fresh-process transcript reproducibility. `node --test test/trace.test.cjs test/protocol.test.cjs` passed 12/12, including fresh-process trace replay and pinned projection. Those trace/protocol tests were run before the final error-brand safety correction; the correction changed only runner failure metadata handling, and the runner suite passed afterward. `git diff --check` and JSON parsing of `docs/ORCHESTRATION_STATE.json` pass. The Trail CLI and `trail check` are unavailable; no Trail result is claimed. No GPU is required for this task.

**Review:** Primary `/root` accepted Task021 on 2026-11-02. The focused review first required three corrections: brand internal failure metadata so external stream errors cannot inject categories/codes; preserve runner elapsed time and label the pre-run CLI timing boundary; and identify result records as trusted engineering metadata excluded from policy input. It also requested a concrete paired-seed follow-up proposal and an indentation fix. These corrections were applied, including a malicious stream-error regression. Final `node --test test/battle-runner.test.cjs` passed 21/21; `node --test test/trace.test.cjs test/protocol.test.cjs` passed 12/12 before the final failure-brand correction. The runner suite passed after that correction. `git diff --check` and orchestration-state JSON parsing passed. Trail was unavailable and no GPU evidence was needed. No commit or push was made by the implementer.

**Next bounded task proposal:** Task022 should implement a deterministic paired-seed manifest generator that emits each five-stream tuple and its seat-swapped partner with stable pair identifiers, then verify both runs reuse the same generated teams. Leave suite size, aggregation, throughput, and hardware allocation unresolved. Task022 has not started.

## Task 022 — deterministic paired-seed manifests

**Status:** Accepted and complete — 2026-11-02. Primary `/root` accepted the implementation after focused diff and evidence review.

**Scope:** Add a deterministic command-line manifest generator for explicitly requested paired battle seeds in the pinned `gen5randombattle` format. For each pair, emit initial and reversed seat arrangements as runner-compatible five-stream seed configurations, stable pair/run identifiers, seat identities, root seed, derivation metadata, and pinned format identity. Add strict input handling, a frozen known-vector test, prefix and fresh-process reproducibility checks, and a pinned Showdown integration check that observes packed team writes across both seat arrangements, confirming the same generated teams occupy exchanged seats while policy seeds follow their policy identities. Update README and this handoff with exact usage and boundaries.

**Exclusions:** No suite-size or statistical-power decision, default scored manifest, batch runner/aggregation, throughput claims or benchmark results, hardware allocation, mechanics/JAX work, player visibility changes, or G0/G1 closure. The current runner continues generating teams independently for each battle; this task verifies that the two team seed streams regenerate the same packed teams across the paired runs and does not add team injection or a generate-once runner path. No commit or push by the implementation owner.

**Acceptance criteria:**

- [x] Generation requires a four-limb uint16 root seed and explicit positive pair count; there is no implicit suite size. Reject malformed CLI, root, and count input with clear errors.
- [x] A documented versioned derivation hashes a frozen JSON tuple of domain, root seed, pair index, and stream name with SHA-256; the first eight digest bytes become four big-endian uint16 limbs. Team A, team B, battle, policy A, and policy B use separate named streams. Pair/run IDs are stable for the same root/index and independent of requested count.
- [x] Every pair contains two expanded runner-compatible seed configs. The second swaps both team streams and policy streams, and keeps the battle seed. Stream arrays do not alias between runs. Manifest output includes schema/derivation versions, upstream commit, format, root, stable identifiers, and policy/team seat assignments.
- [x] A frozen independently checked known vector, determinism, CLI fresh-process byte equality, stable-prefix identifiers, exact tuple exchange, strict invalid inputs, and upstream metadata checks pass.
- [x] At least one paired real pinned-Showdown runner test validates both outcomes and observes each packed generated team in both seats across the two runs. It confirms policy seeds stay with their identities. Test data and transcripts remain trusted engineering/oracle evidence and never enter policy input.
- [x] README, charter, and task handoff describe implemented manifest generation and exact team-regeneration verification separately from proposed workload rules. They do not imply teams are injected/generated once or claim benchmark results.
- [x] Focused and relevant regression validation, Trail build, state JSON parsing, and diff inspection pass. Trail check was run; its unrelated lexical learning conflict is explained in the commit message per repository policy. Primary acceptance advances the count to 22; the review checkpoint remains 30. No GPU run is required.

**Plan:**

1. Read Trail cards for runner seed handling, pinned format identity, and trusted transcripts; persist this task scope, criteria, owner, and in-progress state before implementation.
2. Implement the separate manifest derivation and CLI without changing the existing battle runner path.
3. Add strict and deterministic unit/CLI checks plus one real pinned paired-battle test that records the runner's actual packed team writes.
4. Update README, charter, and this handoff; run focused runner/trace/protocol regressions, Trail build/check, and inspect the final diff. Report evidence to the parent without accepting, committing, or pushing.

**Progress:** The plan and active state were recorded before implementation. `environment/paired-seeds.cjs` emits deterministic version 1 manifests, requiring an explicit four-limb root and positive count. Each pair has five independently domain-separated SHA-256 streams, stable IDs, explicit team/policy identities, and two deep-copied runner seed configurations. The pinned integration test observes the actual packed-team writes and confirms each generated team returns in the opposite seat on the paired run. This is deterministic regeneration; the runner still creates teams independently per battle. Task022 is accepted and complete.

**Primary review correction (addressed and accepted):** Focused primary review reproduced that `validateSeed` accepted sparse arrays because `Array.some` skips holes. `validateSeed` now inspects each indexed limb. Direct `generateManifest` coverage includes sparse/malformed roots, undefined/invalid/out-of-range counts, and valid boundary limbs `[0, 65535, 0, 65535]`. The charter's stale “Task022 has not started” text was removed; it now distinguishes manifest generation and team-regeneration evidence from proposed generate-once pair execution. README wording makes clear the manifest envelope changes with `pairCount`, while prior pair records and IDs stay stable.

**Validation evidence:** On 2026-11-02, `node --check environment/paired-seeds.cjs`, `node --check test/paired-seeds.test.cjs`, and `node --test test/paired-seeds.test.cjs test/battle-runner.test.cjs test/trace.test.cjs test/protocol.test.cjs` passed; the combined suite passed 38/38, including two completed pinned paired battles with exchanged packed team writes. The known vector and pair ID were computed independently with Python's standard `hashlib`/JSON encoder and frozen in the test. `python3 -m json.tool docs/ORCHESTRATION_STATE.json` and `git diff --check` pass. `trail build` passed. `trail check --base HEAD` exits 1 with one lexical conflict at this scope description against Developer's nature-overflow learning; that learning concerns mechanics stat arithmetic and Task022 changes only seed derivation plus existing runner integration. The remaining match is unrelated to the changed behavior; record this rationale in the commit message as repository policy requires. Other Trail learning matches were informational. No GPU evidence is required. No learning was added.

After the primary correction, `node --test test/paired-seeds.test.cjs` passed 5/5, including the sparse-array regression and live paired battles; `trail build` passed again. With both new files included via intent-to-add, `trail check --base HEAD` examined all six changed files and still exits 1 on one lexical conflict at `docs/TASKS.md:745`: the description of seed manifests and pinned-runner team verification overlaps a learning about optional stat caps across formats. That learning is exclusively about mechanics stat rules; Task022 adds no stat limits or mechanics behavior. Explain this false positive in the task commit message per repository policy. Trail's remaining matches are informational.

**Review:** Primary `/root` accepted Task022 on 2026-11-02 after independently reviewing the focused diff and validation evidence, including the sparse-array correction. The accepted count is 22, the regular review checkpoint remains 30, and no milestone closure was recorded. The primary will include the unrelated Trail lexical-warning rationale in the commit message. The implementation owner did not commit or push.

## Task 023 — reuse seed-bound prepared teams

**Status:** Accepted and complete — 2026-11-02. Primary `/root` accepted the implementation after focused diff and evidence review.

**Scope:** Add an opaque prepared-team handle that generates and packs one pinned format team once for an explicit four-limb seed. Allow `runBattle` to consume two valid handles, bound to its Showdown API and matching `teamP1`/`teamP2` seeds, so a caller can reuse the exact teams across swapped seats. Mark prepared team provenance in the per-battle result record. Add fake and real pinned tests, and document usage and timing boundaries.

**Exclusions:** No pair CLI or orchestration, default seed/root/count selection, batch execution or aggregation, scored suite, benchmark or throughput claims, hardware allocation, mechanics/JAX work, visibility changes, or G0/G1 closure. Do not change the existing generated-team default or commit/push as implementation owner.

**Acceptance criteria:**

- [x] Preparation accepts four indexed uint16 seed limbs, copies seed data, and calls the pinned generator and packer once. Its immutable opaque handle has internal WeakMap data binding API identity, format, pin, seed, and packed team. Only handles issued by the helper are accepted.
- [x] `runBattle` requires both player handles when prepared teams are supplied and checks handle authenticity, shared Showdown API, format, pin, and exact player team seeds before creating a stream. Error records contain only safe categories/codes, not packed teams or raw/private data.
- [x] Valid handles bypass generation/packing, can be reused and swapped, and carry `teamSource: "prepared"` in result records. Existing calls without prepared teams retain generated behavior, summary/transcript equality, and policy input boundary.
- [x] Fake tests cover malformed/sparse seeds, immutability, forged/cloned/API-mismatched/seed-mismatched or missing-seat handles, rejection before stream creation, and reusable handles.
- [x] Real paired pinned battles prepare A/B once, exercise both seat arrangements, count exactly two team generations and pack operations, compare captured packed player writes, validate both outcomes, and match normal-path same-seed summaries/transcripts excluding timing.
- [x] README, charter, and handoff explain prepared reuse and per-battle timing scope; automatic pair execution, aggregation, suite sizing, and performance remain proposed/unresolved.
- [x] Focused runner, paired-seed, protocol and trace regressions, Trail build/check, syntax checks, state JSON parsing, and diff review pass. No GPU run is required.

**Plan:**

1. Read Trail context for runner initialization/result records and persist this scope, owner, criteria, and active state before source edits.
2. Implement opaque seed-bound preparation and integrate validation/use into the runner before stream initialization.
3. Add focused fake cases and real pinned paired-run verification, including deterministic compatibility and exact generation/packing counts.
4. Update README, charter, and this handoff; run focused validation, refresh/check Trail, inspect the diff, and report to the parent without accepting or committing.

**Progress:** Plan and active state persisted before source changes. `environment/prepared-teams.cjs` creates immutable handles containing a seed snapshot and packed-team digest while keeping the Showdown object and packed string in a module-local `WeakMap`. `runBattle` authenticates both handles against API identity, pinned format/revision, and canonical player team seeds before stream creation. Prepared runs bypass both team generation and packing, add `resultRecord.teamSource: "prepared"`, and retain the original default generation and packing call sites. Handles and hashes do not enter the policy callback input.

**Validation evidence:** `node --check environment/prepared-teams.cjs`, `node --check environment/battle-runner.cjs`, and `node --check test/prepared-teams.test.cjs` passed. `node --test test/prepared-teams.test.cjs test/battle-runner.test.cjs test/paired-seeds.test.cjs test/protocol.test.cjs test/trace.test.cjs` passed 41/41. The real pinned test prepares exactly two teams (two generation and two pack calls total), reuses them across both seat assignments, verifies packed-team hashes in actual player writes and both completed outcomes, checks recipient policy input keys, and matches the generated-path summary/trusted transcript for identical seeds. Fake tests reject sparse/out-of-range seeds, invalid packed output, forged/cloned/API-mismatched/seed-mismatched/missing-seat handles before stream creation. CPU only; no GPU requirement.

`trail build` passed and refreshed the wiring graph. `trail check --base HEAD` inspected seven source/docs files and returned no conflicts; remaining “worth a look” notices were informational. The eighth changed file is the new repo learning `l-e9a39748`, which records the seed-provenance rationale. `python3 -m json.tool docs/ORCHESTRATION_STATE.json` and `git diff --check` passed.

**Review:** Primary `/root` accepted Task023 on 2026-11-02 after independently reviewing the focused diff and validation evidence. The accepted count is 23, the regular review checkpoint remains 30, and no milestone closure was recorded. The owner did not commit or push.

## Workflow configuration update — not a project task

The repository orchestration rules, role configuration, scheduled prompt, and
durable state are documented in [`ORCHESTRATION.md`](ORCHESTRATION.md),
[`ORCHESTRATION_STATE.json`](ORCHESTRATION_STATE.json), and
[`SCHEDULED_PROMPT.md`](SCHEDULED_PROMPT.md). This workflow-only update does not
create Task021 or change the accepted project-task count (20 at setup). The
bootstrap Astra review at task 20 is due and pending at setup; after it is
accepted, the next regular review is due at task 30. Reconcile future counts
with numbered task history and preserve newer durable state. Existing task
history and the declared 30-run/five-day scheduler cap are preserved; actual
cap start/count remain unknown and are not inferred.

**Bootstrap review handoff:** The first T020 bootstrap review returned
`NEEDS_CHANGES`. Root authorized a bounded correction handoff to the sole owner
`/root/global_reviewer/t020_corrections`; parent-owned overlapping
implementation is excluded. This initial `NEEDS_CHANGES`/pending state is now
historical: the independent parent review accepted the corrections on
2026-11-02. The bootstrap review baseline is 20 and the next regular checkpoint
is 30; T020 remains accepted, the count remains 20, no milestone acceptance is
recorded, and this handoff does not create Task021. The explicit Luna/max child
route and requested Astra/medium review route were accepted, but there is no
independent backend identity or automatic scheduler-role-loading proof. See
[`reviews/T020-bootstrap.md`](reviews/T020-bootstrap.md) for findings and
validation evidence.

## Reusable task entry point — workflow update, not a project task

[`SCHEDULED_PROMPT.md`](SCHEDULED_PROMPT.md) is now a short entry point for
manual and scheduled work. Cadence, concurrency, and limits remain the
invoking scheduler's responsibility; earlier 30-run/five-day references above
are historical records, not repository execution gates. This update does not
change the accepted task count or review history.


## Task 024 — execute one paired seed case

**Status:** Accepted and complete — 2026-11-03. Primary `/root` accepted the deliverable after focused source, diff, and evidence review. The accepted project-task count is 24; the next regular review remains due at 30.

**Scope:** Add validated direct selection of one deterministic seed pair by explicit root and zero-based pair index. Implement a library function and CLI that prepare both teams once, then run the base and swapped seat configurations sequentially through existing `runBattle`. Return a versioned trusted pair result with both nested battle records and bounded failure details. Update README and benchmark charter.

**Exclusions:** No multi-pair/full-manifest execution, concurrency, aggregation, win-rate summary, implicit root/index/count, scored suite, benchmark/throughput claim, hardware allocation, mechanics/JAX, visibility change, or G0/G1 closure. No commit or push by the implementation owner.

**Acceptance criteria:**

- [x] A direct `generatePair(rootSeed, pairIndex)` validates a dense four-limb uint16 seed and index 0..99,999, copies the root, and yields exactly the existing manifest pair without building earlier pairs. Existing `generateManifest` and CLI output remain unchanged.
- [x] A run succeeds only after both base and swapped battles reach existing validated terminal records. It prepares team A/B once and preserves team seeds, shared battle seed, and each policy identity/seed across seat exchange.
- [x] The schema-versioned pair record carries upstream/format identity, copied root/index/pair ID, team identities and hashes, run IDs/seats/policy identities/statuses, nested safe per-battle result records, and finite nonnegative monotonic elapsed milliseconds spanning preparation and both battle cleanups. No raw exception, packet transcript, packed team, or handle enters pair result or policy input.
- [x] Preparation or battle failure stops later work, records `failed` and `not_attempted` honestly, retains a prior completed run where applicable and the actual safe runner failure record, and never invents an outcome or trusts metadata attached by a foreign preparation exception.
- [x] CLI requires explicit root and pair index, emits JSON on success and nonzero safe error on failure, and optional `--result` writes structured success and preparation/runner-failure output. Output file write errors are surfaced. Parse/load/write limitations are documented and covered as appropriate.
- [x] Controlled tests exercise preparation, first-run, and second-run failure paths, no launch after failure, and safe result fields. Pinned real execution verifies two generator/packer calls total, actual packed team seat exchange, correct policy identities, both terminal records, and unchanged actor input boundary.
- [x] README/charter/handoff distinguish one-pair execution from proposed manifest execution and aggregation. The combined regressions passed 50/50 before focused CLI corrections, and the affected pair-runner suite passed 10/10 after the final correction. Syntax, JSON, and diff checks passed. Trail build passed; Trail check ran and its six lexical conflicts were reviewed and documented for the commit rationale. GPU is not required.

**Plan:**

1. Inspect seed manifest, prepared-team, runner and CLI Trail cards; persist this plan and `active_task` before implementation.
2. Add O(1) direct seed-pair selection without changing the existing full manifest.
3. Implement safe sequential paired execution and explicit root/index CLI.
4. Add selection, failure, CLI, and pinned real-pair coverage.
5. Update README, charter, and this handoff; run focused combined validation and Trail checks; report the diff/evidence for parent review.

**Implementation progress and correction history (2026-11-03):** The direct selector, single-pair executor, CLI, README/charter updates, and focused tests are implemented and accepted. Parent review identified four corrections across the initial implementation and two focused correction rounds; the same owner applied them before acceptance:

- The library root is an array while `parseRootSeed` accepts CLI comma-separated text. `runPair` now validates/copies the API array with `validateSeed`; CLI text continues through its parser.
- A concrete bootstrap reproducer patched `oracle.assertReady` to throw an `Error` whose `pairResult` claimed `kind: "gen5-paired-battle-result"`, `status: "completed"`, and included injected code/data. The CLI previously trusted that foreign record. It now accepts a pair error only when it has the module-private brand and only after the actual `runPair` stage starts; the regression asserts nonzero exit and a safe failed initialization record with no injected fields.
- Invalid-root/parse/bootstrap failures previously reported `elapsedScope: "paired_runner"` with zero elapsed time. CLI initialization now has its own monotonic timer and `elapsedScope: "cli_initialization"`; runner errors keep the pair runner's elapsed record.
- A second concrete reproducer made the `limits` option getter throw an Error carrying `resultRecord: {status:"completed", failure:{category:"INJECTED_CATEGORY", code:"INJECTED_CODE"}, foreignData:"INJECTED_RECORD"}`. Because option reads and runBattle argument construction shared one catch, the foreign result was copied as if the actual runner had failed. `runPair` now snapshots root/index/Showdown API/limits/streamFactory in a configuration boundary before preparation, makes a plain copy of enumerable limits values, and builds each plain `runBattle` argument object before a catch that contains only the actual invocation. Configuration getter failures produce a safe `configuration/invalid_configuration` record with two null `not_attempted` runs. A second regression exercises a throwing `streamFactory` option getter through the same path.

Validation evidence: the original combined pair, seed, prepared-team, battle-runner, protocol, and trace command passed 50/50 before the CLI boundary corrections. After the CLI corrections, `node --test test/pair-runner.test.cjs` passed 9/9. After the final configuration-provenance correction, the same focused command passed 10/10. The four final `node --check` commands, `python3 -m json.tool docs/ORCHESTRATION_STATE.json`, `git diff --check`, and `trail build` passed. `trail check --base HEAD` exited 1 with six lexical false positives: README/charter/state/task passages describe opaque prepared-team handles or explicitly exclude raw packed-team strings, not acceptance of arbitrary strings; `TASKS.md:856` describes safe pair result metadata and matched an unrelated format-specific stat rule, while `TASKS.md:857` preserves an actual runner record and matched an unrelated player-observation rule. The primary reviewed these conflicts and authorized their explanation in the commit rationale. Trail learning `l-d998d91c` records the CLI metadata provenance finding. Primary `/root` accepted Task024 on 2026-11-03; the accepted-task count is 24 and the next regular review remains due at 30.


## Task 025 — sequential paired-manifest runner

**Status:** Accepted and complete — 2026-11-03. Primary `/root` accepted after independent source, diff, and evidence review. Accepted project-task count is 25; the next regular review remains due at 30. Owner `/root/task025`; the implementation owner did not commit or push.

**Scope:** Add a library and explicit CLI that execute every pair in an explicitly generated paired-seed manifest sequentially through the existing `runPair` API. Write the exact generated manifest, one safe result line per attempted pair, and a final honest summary into a newly created output directory. Export only the existing `ownsPairError` predicate needed to recognize real `runPair`-owned errors. Add controlled and pinned integration coverage and update the handoff and benchmark charter.

**Exclusions:** Concurrency, resume, retry, crash-recovery/fsync guarantees, scored suite or sample-size recommendation, outcome aggregation/rates/confidence/win summaries, benchmark or throughput claims, hardware allocation, battle/prepared-team/seed/policy/runner/visibility redesign, mechanics/JAX, G0/G1 closure, and commit or push by the implementation owner. GPU is not required.

**Acceptance criteria:**

- [x] Library and CLI require an explicit four-limb uint16 root, pair count in the existing valid range, and output directory; snapshot and validate configuration before artifact creation or battle execution.
- [x] Create only a new non-recursive output directory and exclusive owned artifacts. An existing destination remains unchanged and no battle starts.
- [x] `manifest.json` exactly matches `generateManifest(root,count)`. `results.jsonl` has one safe actual pair result per attempted pair in stable order, fully appended before the next pair starts. `summary.json` version 1 reports identity, status, honest execution/recording counts, safe failure details, and finite `manifest_runner` elapsed time.
- [x] Stop at the first failure. Journal a real owned partial pair result when possible, preserve earlier lines, and never fabricate unattempted pair records. Ignore foreign `pairResult`/`summary`-shaped metadata from configuration, bootstrap, and I/O failures.
- [x] Pinned integration executes at least two real pairs sequentially, validates terminal records and policy seed identity through seat exchange, confirms packed-team writes/hashes exchange seats, observes two generation and packing calls per pair, and confirms stream cleanup.
- [x] Controlled tests cover success, existing destination, invalid config, later-pair and first-pair failures, append/summary initialization/write errors, journal visibility before next pair, CLI safe nonzero errors, foreign metadata injection from config/bootstrap/I/O, and fail-fast behavior.
- [x] README/charter/task docs describe schemas, new-directory requirement, failure preservation, and timing boundaries. Full-manifest execution is implemented; scored suite design, aggregation, concurrency, and resume remain future. No benchmark claim or milestone closure is added.
- [x] Run manifest and pair tests plus paired-seed, prepared-team, runner, protocol, and trace regressions; record syntax, state JSON, diff, Trail build/check.

**Plan:** Read instructions/state/Task024/charter and Trail context; persist this plan; implement library and CLI over `generateManifest` and real `runPair`; add controlled and pinned integration coverage; update README/charter/handoff; refresh Trail; run and report the focused checks for parent review.

**Progress:** Scope, exclusions, criteria, owner, and plan were recorded in `docs/ORCHESTRATION_STATE.json` before source edits. Baseline was clean `main` at `5a9eb01`; Task024 was accepted at count 24, with the next regular review due at 30. Parent source review identified and resolved the owned failure-category mismatch, including the retry-limit follow-up. The combined required regressions and strengthened manifest suite passed. Primary acceptance is recorded below; no Task026 was started.

**Review correction:** Parent focused source review found that manifest summary category mapping collapsed owned `packet_limit`/protocol/policy/timeout/retry failures into `initialization`. The concrete `limits: { maxPackets: 1 }` reproducer had `packet_limit` in the journal and `initialization` in the summary. The owner now preserves only an allowlisted category from the actual branded `pairResult`, falls back safely for unknown categories, and never reads an unowned error category. Controlled checks assert `packet_limit` and `retry_limit` categories and codes match the journal. Parent follow-up identified that the first allowlist omitted the supported `retry_limit` category (at `battle-runner.cjs#runBattle`); the list and test now include it. Combined regression validation passed 65/65 across manifest, pair-runner, paired-seed, prepared-team, battle-runner, protocol, and trace tests, including packet-limit and retry-limit summary reconciliation. Trail learning `l-ec89ac92` records this distinct wrapper-category loss cause; Trail build passed with 65 indexed source/test files, 548 nodes, and 1166 edges. After `git add -N` for the new module, test, and learning, Trail check examined seven changed files against 12 learnings and exited 1 with four reviewed lexical false positives at README.md:99, docs/BENCHMARK_CHARTER.md:36, docs/ORCHESTRATION_STATE.json:219, and docs/TASKS.md:884. Those passages describe opaque seed-bound prepared-team handles or the existing runPair API; none accepts arbitrary packed-team strings. New runner/test matches about Teams.generate and BattleStream were advisory only: each manifest pair is created once through runPair and raw packets are not supplied to policies. No blocking finding applied to the new implementation. Final diff inspection is complete. Primary `/root` accepted Task025 on 2026-11-03; count advances from 24 to 25, with the next regular review due at 30.


**Acceptance review:** Primary `/root` confirmed exclusive output-directory and artifact creation, exact generated manifest/hash, real sequential `runPair` composition, owned error boundaries, journaling before next-pair preparation, honest execution-versus-recording counts, category preservation, fail-fast partial results, and explicit timing boundaries. Validation was 65/65 across the combined regressions and 13/13 for the strengthened manifest suite; syntax, JSON, and diff checks passed. Trail build passed. Trail check exited 1 with four reviewed lexical warnings explained above; it included the new runner and test and found no blocking implementation conflict. No Astra checkpoint or milestone gate is due. The accepted count is 25, the next checkpoint remains 30, and no Task026 was created. The implementation owner did not commit or push.


## Task 026 — offline recorded-pair outcome accounting

**Status:** Accepted and complete — 2026-11-03. Primary `/root` accepted after independent source, diff, and evidence review. Accepted project-task count is 26; the next regular review remains due at 30.

**Scope:** Add a read-only CommonJS library and explicit CLI that validate Task025's canonical manifest and a contiguous recorded prefix of its pair-result JSONL journal, then report recorded run outcomes by seat and policy identity. Report generated metadata and accounting only.

**Exclusions:** Battle simulation or runner changes, Task025 summary reconciliation or authenticity, verification of `packedTeamSha256` values, counters, and timing, completeness claims about execution, outcome rates/confidence intervals, performance or scored-benchmark claims, suite-size/statistical-power choices, resume/concurrency, GPU, and G0/G1 closure.

**Acceptance criteria:**

- [x] Manifest root/count and all generated schema-v1 metadata/pairs exactly match `generateManifest`; reject modified, unsupported, or foreign manifests.
- [x] Accept only contiguous pair-result prefixes in manifest order. Validate pinned format/upstream and expected pair/run IDs, seat mappings, policy identities, seed mappings, and completed/failed/not-attempted outcome topology. Handle genuine team-preparation failures with two not-attempted runs and genuine partial battle failures with optional nested failure result records.
- [x] Emit a versioned report with canonical manifest identity, recorded/unrecorded pairs, pair status counts, and completed/failed/not-attempted/unrecorded run slots. Report valid wins/losses/ties by seat and policy identity. Never call an absent journal suffix unattempted; failed or missing runs are not outcomes. Empty prefixes are valid. Emit only derived counts and identities; omit full records, artifact digests, counters, rates, timing, and confidence.
- [x] The pure API leaves caller data unchanged; invalid files or command arguments exit nonzero with a fixed diagnostic and no record data.
- [x] Test hand-calculated multi-pair wins, losses, ties, policy seat exchange, and exact summaries; empty/full/partial prefixes; preparation and battle failures; schema, identity, ordering, status, and outcome tampering; CLI success, malformed/truncated/blank JSONL, manifest errors, and I/O errors; and actual pinned runner records plus a genuine partial failure. Verify offline CLI operation and unchanged input artifacts.
- [x] Update README and benchmark charter to distinguish recorded-outcome accounting from future counter/rate aggregation and scored reporting, state that missing records may hide attempted work, and make no milestone closure claim.
- [x] Run the focused new suite and affected runner regressions, syntax/state JSON/diff checks, Trail build/check; report exact evidence for primary acceptance. GPU is not required.

**Plan:** Inspect Trail cards and relevant existing manifest/pair/battle contracts; persist this handoff and state; implement the offline validator/accountant and explicit JSON/JSONL CLI; add controlled and pinned tests; update README/charter; run focused checks, refresh/check Trail, inspect the complete diff, and report to the primary without accepting or committing.

**Progress:** Implemented the pure offline report and read-only CLI, controlled and pinned regression coverage, documentation, and durable handoff. Primary `/root` accepted Task026 on 2026-11-03. The report validates serialized identity/shape but cannot restore runtime private brands or prove that input JSON was emitted by the real runner. The accepted-task count advances from 25 to 26; the regular review baseline remains 20 and next checkpoint remains 30. No GPU evidence or milestone closure was required.

**Validation evidence:** The final `node --test test/summarize-results.test.cjs` passed 10/10, including pinned success, packet-limit failure, early initialization failure without `teamSource`, safe offline CLI, and unchanged input bytes. The combined requested regression command (`test/summarize-results.test.cjs`, `test/manifest-runner.test.cjs`, `test/pair-runner.test.cjs`, `test/paired-seeds.test.cjs`, `test/prepared-teams.test.cjs`, `test/battle-runner.test.cjs`, `test/protocol.test.cjs`, `test/trace.test.cjs`) passed 73/73 before the final early-failure compatibility relaxation; the affected summarize suite passed 10/10 afterward. Syntax, state JSON, and `git diff --check` passed. `trail build` passed with 570 nodes and 1225 edges. `trail check --base HEAD` exited 1 on four reviewed lexical conflicts; the commit rationale is recorded in the acceptance review below. Trail learning `l-86fd14c5` is recorded at `.trail/learnings/2026-11-03-offline-outcome-reports-cannot-attest-who-jarho001.md`.

**Acceptance review:** Primary `/root` independently reviewed the source, diff, and validation evidence against all delegated criteria and accepted Task026 on 2026-11-03. Review covered canonical manifest equality; contiguous pair identity, seeds, and seat mappings; sequential completed/failed/not-attempted topology and rejection of records following a failure; independent hand-calculated outcome tallies by seat and policy identity; safe CLI errors and report allowlisting; pinned success, packet-limit partial failure, and early initialization failure before `teamSource`; and unchanged offline input artifacts. No blocking findings remain. No Astra review or milestone gate was due; the count advances from 25 to 26, with baseline 20 and next checkpoint 30. GPU was not required. The four Trail conflicts are lexical: the implementation compares serialized seed arrays with the canonical generated manifest and does not accept packed-team strings or handles; the `packedTeamSha256` wording is an explicit exclusion; “all generated metadata” refers only to this manifest's schema-v1 fields, not all game formats. Include this rationale in the task commit body as Trail requests.

## Task 027 — offline recorded-work counter accounting

**Status:** Accepted and complete — 2026-11-03. Primary `/root` accepted after independent focused source, diff, and test review. Accepted project-task count is 27; the next regular review remains due at 30. Owner `/root/task027`.

**Scope:** Add a separate read-only offline CommonJS library and explicit CLI that compose Task026's canonical manifest/contiguous-prefix validator with checked additive counters from present nested battle `resultRecord` values. Emit an allowlisted report for completed runs, failed recorded attempts, and all recorded attempts, with totals by player seat and policy identity.

**Exclusions:** Changes to the existing outcome-only reporter, battle runner, pair runner, or generated record shape; execution authenticity or Task025 `summary.json` reconciliation; treating absent slots as attempts or missing counters as zeros; summing turn indices, terminal metadata, timing, limits, or seeds; rates, uncertainty, scored-suite choice, performance claims, and G0/G1 closure; GPU; commit or push by the implementation owner.

**Acceptance criteria:**

- [x] Reuse `summarizeResults` to validate the complete canonical manifest and actual contiguous recorded pair prefix. Preserve missing suffix slots as unrecorded and `not_attempted` slots as separate. Make no execution-authenticity claim and do not read/reconcile `summary.json`.
- [x] Require the full additive counter shape on every present nested `resultRecord`, whether completed or failed. Validate every selected numeric value as a nonnegative safe integer, per-player `byPlayer` sum against supplied total, safe-integer arithmetic on per-record sums and cross-record aggregation, and the documented per-player resolution/submission/selection invariants. Preserve valid partial failed-run counters. A failed run with null `resultRecord` is valid and counted as missing-counter coverage, never zero work. Exclude absent suffix and `not_attempted` slots from work totals.
- [x] Aggregate only `objectPackets`, `publicEvents`, policy invocations, choice submissions/resolutions, retry request/submission/exhaustion counters, and selected/submitted move/switch actions. Keep globals, seats, and policy identities explicit; derive policyA/policyB totals from validated `run.policySeats`. Keep selected distinct from submitted, and rejections/retries distinct from invocations/submissions. Never sum `latestPublicTurn` or `terminalMetadataTurns`.
- [x] Emit schema-versioned `gen5-paired-work-accounting` output with three explicit cohorts (completed, failed-recorded, all-recorded), run counts, with-counter/missing-counter record counts, and allowlisted `observedTotals`. Coverage must make totals visibly incomplete when a failed recorded attempt lacks counters. Empty and preparation-only prefixes are valid with zero observed totals and honest coverage. No raw records, nested counters, hashes, transcripts, or unknown input fields escape.
- [x] Test hand-built exact cohort/report sums across multiple seat-swapped runs and failed partial counters; empty/prefix/preparation-only and null-result failure coverage; all invalid numeric, shape, mismatch, arithmetic-overflow, and invariant cases; valid selected>submitted and invocations>selected cases; throwing getters; input immutability and output allowlisting. Cross-check explicit sums from pinned real successful and genuine failed `runPair` records.
- [x] Test CLI success, malformed/truncated/blank input, I/O and argument failures, unchanged artifact bytes, generic safe errors, and offline operation without oracle bootstrap. Existing outcome reporter and its output remain unchanged under its tests.
- [x] Update README and benchmark charter to document schema/cohorts, numeric invariants, missing coverage, ignored indices/timing, and the distinction from future rates, uncertainty, or timing reports. Do not add scored-suite choices, performance claims, or G0/G1 closure; keep Task026's historical scope accurate.
- [x] Run the focused suite and `summarize-results`, `manifest-runner`, `pair-runner`, `paired-seeds`, `prepared-teams`, `battle-runner`, `protocol`, and `trace` regressions as appropriate, plus syntax/state JSON/diff checks, `trail build`, and `trail check --base HEAD`. Inspect the diff and report exact evidence and concerns to the primary. GPU is not required.

**Plan:** Read Task026, the counter charter and applicable Trail cards; persist this plan and active state; implement `environment/summarize-work.cjs` as a compositional pure API plus explicit CLI; add independent aggregate/invariant/pinned-run tests; update README, charter and handoff; run focused and affected validation, refresh/check Trail, inspect the diff, and report for primary acceptance without self-acceptance or commit.

**Progress:** Scope, exclusions, criteria and plan persisted before implementation. Primary `/root` accepted Task027 on 2026-11-03 after focused source, diff, and test review. The report snapshots finite JSON data, rejects accessors/custom prototypes, composes Task026 validation, validates full additive counters and safe invariants, and allowlists its cohort output. The accepted-task count advances from 26 to 27; review baseline 20 and checkpoint 30 remain unchanged. No GPU evidence or new Trail learning was needed.

**Validation evidence:** `node --check environment/summarize-work.cjs && node --check test/summarize-work.test.cjs` passed. The requested combined regression command (`node --test test/summarize-work.test.cjs test/summarize-results.test.cjs test/manifest-runner.test.cjs test/pair-runner.test.cjs test/paired-seeds.test.cjs test/prepared-teams.test.cjs test/battle-runner.test.cjs test/protocol.test.cjs test/trace.test.cjs`) passed 84/84 before the final test-only additions for a nonempty partial prefix, switch inequality, getter non-invocation, and custom prototypes; after those additions `node --test test/summarize-work.test.cjs` passed 9/9. Final syntax passed again. State JSON and `git diff --check` passed. `trail build` passed with 604 nodes and 1311 edges (69 indexed cards). `trail check --base HEAD` exited 1 with nine reviewed lexical conflicts and no source contradiction: README.md:122, docs/BENCHMARK_CHARTER.md:48, docs/ORCHESTRATION_STATE.json:331, and docs/TASKS.md:939 discuss serialized result records for structural accounting but explicitly disclaim execution attestation; README.md:128, state:336, and TASKS.md:941 say seeds are excluded from additive totals while still retained in canonical identity, not that an arbitrary packed team is accepted; TASKS.md:945 validates the canonical manifest and does not claim complete execution; TASKS.md:946 refers to nested runner result records and not complete Showdown oracle evidence. The offline report never requires runner-private brands, accepts arbitrary packed-team strings, or treats serialized metadata as proof of execution. No learning was added. GPU is not required.

**Acceptance review:** Primary `/root` accepted on 2026-11-03 after independently reviewing the detached finite-JSON snapshot, canonical manifest/prefix/status composition, complete additive counter validation, per-player invariants and safe arithmetic (including cross-cohort overflow), global versus seat/policy mapping, exact cohort tables, partial/null failure coverage, absent-slot exclusions, offline CLI safety and artifact immutability, pinned success and genuine failure counters, excluded metadata/timing, and unchanged outcome reporter. Validation was 84/84 across the combined regressions before the final test-only additions, followed by focused 9/9; syntax, JSON, diff, and Trail build passed. The nine Trail lexical findings are documented above and do not contradict the implementation. No Astra checkpoint or milestone review was due; GPU was not required. No commit or push is recorded in this task handoff.

## G0 specification milestone review (unnumbered review unit)

**Status:** Accepted — G0 specification closure only, 2026-11-03, by `/root/g0_reviewer`; this is an unnumbered milestone review, not a project task. See the [independent G0 review report](reviews/G0-specification.md). Final documentation/state synchronization and review were accepted on 2026-11-03 by independent Astra `/root/g0_reviewer` and primary `/root`; see the final focused-review entry in the [G0 review report](reviews/G0-specification.md). Accepted numbered-task count remains 27; review baseline remains 20 and next regular review remains due at 30.

**Reviewed source identity:** `b4fed94964ff8185c0fc8d9a67c4d77c2f5f4cf2` (clean synchronized `main` at plan start).

**Scope:** Independently review the existing G0 specification-closure boundary at the G0 row of `PROJECT_PLAN.md`, the G0 section of `docs/MECHANICS_COVERAGE.md`, the G0 boundary of `docs/REUSE_AUDIT.md`, the closing G0 statements of `docs/BENCHMARK_CHARTER.md`, and applicable bootstrap-review required findings. Assess whether the pinned target, rules/data authority and source manifest, reuse decision, mechanics inventory, and known gaps are explicit and consistent. The review must identify any remaining G0 decision that changes the target or rules/data scope, or explicitly bound operational benchmark questions to later measurements.

**Exclusions:** G1 integration or full-engine acceptance; new engine features; policy learning or end-to-end learning benchmarks; GPU acceptance; candidate-project re-audit or newest external research; operational budgets that do not change the game target; a numbered development task; and any claim that G0 is accepted before an independent Astra disposition.

**Astra review criteria:**

- [x] Regenerate or verify the pinned resolved `gen5randombattle` fixture identity and its rule, mod, and generator facts from the pinned local source/runtime; check the data authority and source-manifest commit.
- [x] Check that the accepted reuse decision and mechanics ledger distinguish implemented arithmetic from known target, rules/data, visibility, PRNG, event, full-engine, and learning gaps. Confirm there is no implied coverage percentage or downstream milestone acceptance.
- [x] Explicitly identify remaining G0 target/scope decisions, if any, or bound operational benchmark questions to future measurements. G0 does not require later implementations.
- [x] Provide an independent report naming the reviewed source identity, actual evidence and check commands, disposition (`ACCEPTED`, `NEEDS_CHANGES`, or `BLOCKED`), project direction, and scope limitations.
- [x] After the genuine `ACCEPTED` disposition, update current G0 status consistently while retaining G1-and-later gaps and historical task language. Preserve accepted-task count 27 and regular checkpoint 30; create no numbered task.
- [x] Run focused documentation, JSON, link, pin, diff, and Trail checks as appropriate. Do not claim GPU, benchmark, or new-mechanic validation.

**Plan:** Persist this scope and state, then stop for independent Astra review. The reviewer examines the pinned source/fixture and existing specification artifacts without implementing features or broadening the target. If the reviewer requests corrections, `/root/task027` remains the sole documentation/correction owner for at most two rounds (initial review and one follow-up); optional suggestions are not auto-implemented. After a genuine acceptance only, make the bounded G0 status updates, retain all later-gate limitations, run the affected documentation/state checks, and report the accepted disposition to the primary. No commit or push is authorized for this review unit.

**Acceptance record:** The independent report accepted the specification boundary against the complete pinned Showdown `gen5randombattle` target at commit `1242c1e2d9763ba5b85d33f13aecb89bae19ae3d` (generation 5 singles, random teams, inherited rules/data, and the fixture's complete effective rules). G0 acceptance records exact target and known gaps; the mechanics ledger remains a partial arithmetic inventory, with G1 request/visibility/action integration, G3 state/event/PRNG/representative-mechanics work, G4 complete reachable rules/data and full-format differential transfer, and later learning/evaluation open. Suite size, seed publication, precision, hardware, concurrency, warm-up, repetitions, resource limits, and measurement-environment locks remain future benchmark protocol choices. This review did not accept benchmark validity, engine implementation, G1+, or GPU evidence. The reviewed workspace source base is `b4fed94964ff8185c0fc8d9a67c4d77c2f5f4cf2`; no new numbered task was created.

**Status-sync validation:** `python3 -m json.tool docs/ORCHESTRATION_STATE.json` and focused Python assertions for unchanged task history/counts, approval metadata, reviewed source identity, and pinned format passed. A local Markdown-link scan across the changed documents and `docs/reviews/G0-specification.md` passed; `git diff --check` passed. `trail build --check` reports the wiring graph is current (604/604 nodes). The latest `trail check --base HEAD --json` run included seven changed files and exited 1 with six lexical conflicts: `README.md:99` has two matches for seed-bound prepared-team and offline-report wording, while the text preserves seed provenance and makes no execution-authenticity claim; `ORCHESTRATION_STATE.json:60` and `TASKS.md:978` repeat the satisfied plan condition that current status updates follow Astra acceptance; `TASKS.md:964` matches the accepted status/report link to unrelated audience-specific evidence; `TASKS.md:983` matches the pinned game target and future seed-publication protocol to the packed-team seed learning. These statements concern different boundaries and do not contradict the cited learnings. Other results were non-conflicting `lead` or `follows` findings. No source tests, benchmark, GPU run, or mechanic validation were repeated; the independent report records its own pinned source checks and test results. No new Trail learning was added.

**Final documentation/state review:** Accepted on 2026-11-03 by independent Astra `/root/g0_reviewer`; primary `/root` accepted the complete G0Review unit and final status synchronization. The report's final review entry records 119 resolved local links, preserved task history/count 27/baseline 20/checkpoint 30, and no required corrections or blockers. See the [G0 review report](reviews/G0-specification.md). This acceptance does not create a numbered task or set `last_review_commit`.

## Task 028 — request-aware eleven-slot singles action adapter

**Status:** Accepted and complete — 2026-11-03. Owner `/root/task028`; primary `/root`. Synchronized starting revision `cbf6c8fe736d437c878cac9796f7cf6f772f8f70`. Primary-authorized bounded scope adjustment: fix the exact pinned sole-Struggle request parser shape, because the runner validates candidate actions before the policy callback. No coordinator or default-selection changes.

**Scope:** Add a pure CommonJS request-to-action-space adapter and checked command decoder for the `PROJECT_PLAN.md` Gen 5 singles eleven-slot representation, composing the existing `candidateActions` request/retry rules. Include the exact pinned sole-Struggle parser correction authorized below. Verify actual pinned Showdown request and choice behavior, plus a seeded pinned battle through an adapter-backed policy.

**Exclusions:** runner coordination, retry-episode tracking, stream tuple derivation, and default random-selection algorithm changes; the only random-policy.cjs change is a parser branch for the exact pinned sole-Struggle row (`id: struggle`, `move: Struggle`, `disabled: false`, omitted `pp`); policy feature encoding or trajectory collection; learning or learned policies; new battle mechanics; timing/rates/uncertainty reports; scored-suite selection; G1 or later milestone closure; GPU runs; implementation-owner commits or pushes.

**Acceptance criteria:**

- [x] Export a documented version-1 pure adapter with an eleven-element boolean mask and command mapping: zero-based slots 0..3 are current request move rows 1..4, 4..9 are own current roster positions 1..6, and 10 is explicit Struggle when offered as the sole struggle request row (command `move 1`). Reduced locked/recharge requests map their current row to slot 0; do not infer original move indices.
- [x] Compose candidateActions(request,excluded) so only own request-visible permission and current-episode rejected commands determine available actions. Preserve maybeTrapped switches, suppress trapped switches, exclude active/fainted replacements, handle forced switching, and expose non-actionable wait/forceSwitch:[false] states with no learnable WAIT slot. An actionable all-masked request throws an error; unavailable choices remain retryable in the existing coordinator.
- [x] Validate singles shape and finite slot bounds, reject unsupported over-four move rows or over-six rosters and malformed requests without mutating inputs. Decode only enabled integer slots against the supplied current request and exclusion list; fixed generic diagnostics exclude caller-supplied values. Output no roster/private metadata beyond own command mask and mapping.
- [x] Apply only the primary-authorized parser correction for the exact pinned sole-Struggle request `{move: "Struggle", id: "struggle", target: "randomNormal", disabled: false}` with omitted PP. Preserve all other request validation and selection behavior; make no runner or default selection algorithm change.
- [x] Test independent expected masks/commands and enabled-slot round trips for normal/disabled/zero-PP moves, replacements, fainted/active slots, trapped/maybeTrapped, forced replacement/wait, locked/recharge/Struggle, exclusions and exhaustion, malformed Struggle variants, unique non-sole row mapping, invalid slot/shape errors, safe diagnostics, and input immutability.
- [x] Obtain actual requests from the guarded pinned Showdown runtime for ordinary moves, forced replacement, wait, locked/recharge and Struggle as feasible through controlled local battle fixtures; demonstrate locked/recharge/Struggle command acceptance using the pinned Battle choice handler. Also verify the generated Struggle row passes `candidateActions` pre-validation. Exercise hidden trap request/rejection semantics with the adapter through the pinned runner. Run a seeded reference battle with adapter-backed policies through terminal validation and assert recipient-scoped request history.
- [x] Document action indices, reduced-request and Struggle mapping, non-actionable masks, retry episode responsibility, request-relative decoder semantics and provisional versioned scope in README. Include the authorized Struggle scope adjustment and source evidence here and in durable state; preserve historical tasks and all milestone limitations.
- [x] Run targeted adapter and affected runner/protocol/trace regressions plus syntax, state JSON, diff, Trail build and Trail check. Inspect diff and report changed files, exact commands/results and unresolved concerns to primary before acceptance. GPU is not required.
- [x] Normalize all foreign thrown values from request or exclusion access to a fixed diagnostic without reading properties from the thrown value; preserve safe fixed diagnostics for adapter-owned errors. Cover Proxy objects and primitive throws.
- [x] Reject top-level combinations of wait, active, and forceSwitch fields that do not occur in pinned request shapes, including active plus forceSwitch:[false]; preserve standalone wait:true and forceSwitch:[false] behavior.

**Authorized scope adjustment and source evidence:** The pinned `Pokemon.getMoveRequestData()` synthesizes Struggle at `reference/vendor/pokemon-showdown/sim/pokemon.ts:1125` as `{move: "Struggle", id: "struggle", target: "randomNormal", disabled: false}` without a `pp` field. Existing `candidateActions()` at `environment/random-policy.cjs:80-89` requires finite `pp` whenever `disabled` is present. Counterexample before correction: the request parser rejected that exact pinned row with its generic malformed move-slot diagnostic. `runBattle.submit()` calls `candidateActions()` before invoking a policy (`environment/battle-runner.cjs:303-325`), so adapter-only normalization cannot support this request. Primary `/root` authorized the smallest correction: recognize only this exact sole-Struggle row as `move 1`; retain all other parsing behavior, slot 10 mapping, and all coordinator/retry/default selection behavior. Acceptance includes malformed-variant rejection and pinned Battle choice acceptance.

**Parent review correction round 1 — `/root` disposition `NEEDS_CHANGES`:**

- Finding: a request containing both `active` and `forceSwitch`, even `forceSwitch: [false]`, was interpreted as a forced-switch request and could expose a switch mask. Correction: incompatible top-level wait/active/forced combinations now fail with the fixed malformed-request diagnostic; standalone `wait: true` and `forceSwitch: [false]` remain valid.
- Finding: the `SAFE_ERROR` Symbol check accessed properties on a thrown Proxy, leaking its trap diagnostic. Correction: use a private WeakSet for adapter-owned errors and avoid property access on foreign thrown values; Proxy and primitive throws normalize to the fixed error.
- `node --test test/action-space.test.cjs` — 7/7 passed after both corrections. The unchanged dependency suites from the prior 97/97 run were not repeated per primary direction.

**Owner implementation validation and primary acceptance:**

- Changed files: `README.md`, `docs/TASKS.md`, `docs/ORCHESTRATION_STATE.json`, `environment/action-space.cjs`, `environment/random-policy.cjs`, `test/action-space.test.cjs`, `test/battle-runner.test.cjs`, and `.trail/learnings/2026-11-03-handle-the-pinned-struggle-request-without-jarho001.md`.
- `node --test test/action-space.test.cjs test/battle-runner.test.cjs test/prepared-teams.test.cjs test/pair-runner.test.cjs test/manifest-runner.test.cjs test/paired-seeds.test.cjs test/summarize-results.test.cjs test/summarize-work.test.cjs test/protocol.test.cjs test/trace.test.cjs test/reference.test.cjs` — 97/97 passed before the final module/test-only review corrections; the affected adapter suite then passed 7/7. This includes adapter-driven regular battles, pinned generated locked/recharge/Struggle request rows accepted by `Side.choose`, parser pre-validation of the generated Struggle row, recipient-scoped callback histories, and the pinned Shadow Tag unavailable-switch correction/retry.
- `node --check environment/action-space.cjs && node --check test/action-space.test.cjs`; `python3 -m json.tool docs/ORCHESTRATION_STATE.json >/dev/null`; and `git diff --check` — passed after source/test corrections; final acceptance metadata receives focused JSON/diff checks.
- `trail build` — passed with 624 nodes, 1,355 edges, and 71 indexed cards after the source/test correction. Owner final `trail check --base HEAD` exited 1 with five lexical conflicts in documentation; the final staged check and rationale follow below.
- CPU only; no GPU run or benchmark claim. No full `runBattle` Struggle battle is claimed: the exact generated request is validated through the shared parser and its decoded `move 1` is accepted by pinned `Side.choose`; the separate full `runBattle` integration exercises ordinary, forced-switch and wait requests through terminal validation.
- No blocking implementation concern remains. Primary `/root` retains independent acceptance and commit/push ownership.

**Plan:**

- Confirm tools and pinned reference readiness; read Trail context, exact relevant spans, prior learning and orchestration guidance.
- Implement the separate action adapter and checked decoder; make only the authorized sole-Struggle request-parser correction while retaining runner coordination, retry, seed and default-selection behavior.
- Add independent contract cases and pinned request/command acceptance fixtures, then run a seeded reference battle with adapter-backed policies.
- Update README and Task028 handoff; refresh Trail, run focused checks and inspect the final diff.
- Return evidence to primary for independent acceptance; keep ownership through requested corrections and do not commit or push.

**Startup evidence:** Planned on clean synchronized main. Required tools git, trail 0.28.0, Node v24.22.1, npm 11.19.1 and Python 3.13.18 are available; trail map/ask/callers/skeleton and pinned oracle provenance work. GPU is not required; nvidia-smi reports RTX 3060 Ti. No review gate is due (count 27, checkpoint 30, G0 accepted); no pending corrections or GPU block. Runtime status tool and /etc/codex/network-policy.json are unavailable, so no managed readiness claim is made; authenticated git fetch succeeded. The requested repository is mounted at `/workspace`, confirmed by its origin `git@github.com:jarho001/nS3JoIfbZ9JMn.git`; the named child directory does not exist. All 27 numbered handoff entries are accepted and their commits are already on `origin/main`. G0Review was delivered in `cbf6c8f`; no delivery remains pending.

**Primary acceptance:** Primary /root independently inspected the adapter, parser correction, tests and diff. Accepted fixed slot mappings, request-relative decoding, request-only trapping masks, episode exclusions, non-actionable states, shape/bound checks, immutable inputs and fixed diagnostics. Pinned live locked/recharge/Struggle requests and Side.choose acceptance, shared-parser Struggle prevalidation, adapter-backed complete battle, own-side request histories and actual Shadow Tag rejection/retry evidence satisfy the declared scope. One follow-up correction resolved incompatible top-level fields and foreign-thrown-value normalization. The affected suite passed 97/97 before the module/test-only correction; the complete affected adapter suite then passed 7/7. Syntax, JSON, diff and Trail build passed. No blocking finding remains. No full runBattle Struggle battle, GPU, benchmark result or milestone closure is claimed. No Astra gate is due: count advances to 28, baseline stays 20 and checkpoint stays 30.

**Trail review rationale:** The source changes implement current-request action mapping and one pinned Struggle parser exception. They do not change arithmetic/stat domains, prepared-team provenance, mutable battle-state separation, player visibility, or offline execution-attestation boundaries. Documentation of acceptance and test results is validation evidence for this unit, not authentication of serialized battle journals. Mentions of the prepared-team regression suite do not introduce arbitrary packed-team input. Lexical findings on these phrases do not contradict the cited learnings; include this explanation in the commit body.

**Final staging review:** `trail check --base HEAD --json` exited 1 after checking 7 files against 14 learnings, with 8 lexical documentation conflicts and no overlaps. Primary reviewed each finding against the source diff: README current-history synopsis (two matches), state scope exclusions and validation summary, and handoff scope, action criteria, validation criteria and regression-suite listing. The rationale above applies to every finding and is retained in the commit body. No blocking source conflict remains. `trail build --check` and focused final state/history/diff checks follow acceptance metadata only; source tests are not repeated.

## Unnumbered optional GPU validation — preliminary JAX mechanics correctness

**Status:** Accepted complete by primary `/root` on 2026-11-03. This unnumbered optional check does not change the accepted-task count. It ran the existing Python mechanics suite against CPU then CUDA, without changing kernels, fixtures, or tests. The run verifies backend parity and actual device placement; it does not measure performance or close a milestone.

**Source identity:** Commit `6bc48b3edd7ea0a96cdbf4f877d1e889c13e3289`, tree `00a02cdea11018026fefc8c20b3fb690ce0f1260`, captured before the validation record was added. The tracked worktree was clean. The validation uses the existing `/opt/jax-env` installation without dependency changes.

**Exact suite commands and results:**

```sh
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu /opt/jax-env/bin/python -m unittest discover -s test -p 'test_*.py' -v
# Ran 73 tests in 36.376s — OK
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda /opt/jax-env/bin/python -m unittest discover -s test -p 'test_*.py' -v
# Ran 73 tests in 43.251s — OK
```

Both runs used Python 3.13.18, JAX 0.12.0.1, and JAXLIB 0.12.0.1 from the existing `/opt/jax-env`. CUDA plugin packages were `jax-cuda13-pjrt` and `jax-cuda13-plugin` 0.12.0.1; `nvidia-cuda-runtime` was 13.4.92. `nvidia-smi` reported an NVIDIA GeForce RTX 3060 Ti with 8192 MiB, utility 610.57.01, KMD/driver 610.88, and CUDA UMD 13.3. The selected CUDA backend was `gpu`, device `[CudaDevice(id=0)]`; eager input and compiled output were each placed on that device. CPU selection used `CpuDevice(id=0)`. The expected eager/JIT probe result was `[1, 2, 5, 10, 17, 26, 37, 50]` for both.

The readiness probe was:

```sh
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda /opt/jax-env/bin/python - <<'PY'
import jax
import jaxlib
import numpy as np
import jax.numpy as jnp
print('python/imported_jax', __import__('sys').version.split()[0], jax.__version__, jaxlib.__version__)
print('backend', jax.default_backend())
print('devices', jax.devices())
x = jnp.arange(8, dtype=jnp.int32)
y = jax.jit(lambda value: value * value + 1)(x)
y.block_until_ready()
print('array_device', x.devices())
print('jit_device', y.devices())
print('jit_result', np.asarray(y).tolist())
assert jax.default_backend() == 'gpu', f'expected GPU backend, got {jax.default_backend()}'
assert any(device.platform == 'gpu' for device in x.devices()), f'input array placed on {x.devices()}'
assert any(device.platform == 'gpu' for device in y.devices()), f'compiled result placed on {y.devices()}'
PY
```

Trail exhaustive searches over indexed code returned no hits for `jax.devices`, `platform`, `JAX_PLATFORMS`, or `jax_platform_name` in the test suite, so tests do not explicitly force CPU placement. Both verbose logs remain at `/tmp/gpu-validation-cpu.log` and `/tmp/gpu-validation-cuda.log`. Source commit/tree are the identities above; only documentation/state files changed after that snapshot. This current optional run supplements but does not rewrite historical task-scoped GPU statements. No failures or unresolved concerns; there is no performance claim, full-battle GPU claim, milestone acceptance, or accepted-task count change.

**Primary acceptance:** `/root` reviewed all three documentation/state diffs and both verbose logs and accepted the 73/73 CPU run followed by the 73/73 CUDA run, exercised mechanics/composition edge cases, explicit CUDA placement, unchanged source/tests/dependencies, and the recorded source identity and historical boundaries. No required Astra gate or milestone is due. Final JSON/state invariants, `git diff --check`, and `trail check --base HEAD` pass after metadata synchronization; test suites were not rerun. No failure or unresolved concern remains.

## Task 029 — completed-battle player decision trajectory collector

**Status:** Accepted and complete — 2026-11-03. Owner `/root/task029`; primary `/root`. Delivered at `b0f996104f028306886f0326ba241dc27718ce70`.

**Headline reconciliation (2026-11-03):** The prior status read, “In progress — source accepted by primary after the follow-up test-only correction; final task acceptance and delivery metadata remain primary-owned, 2026-11-03.” Task029's primary acceptance and delivery are recorded at `b0f9961`; this corrects the stale headline as documentation bookkeeping and records no new approval.

**Scope:** Add a separate CommonJS completed-battle collector composing runBattle, its existing policy callback API, and the Task028 action adapter. Record actor-scoped policy invocation samples with append-only own-history deltas, current requests, masks, selected commands/slots and request-episode IDs; add fixed-player terminal reward targets only after validated completion. This is a bounded G1 trajectory-schema step.

**Exclusions:** changes to existing runner scheduling, retries, policies, protocol projection, action mappings or record contracts; partial failed-battle datasets or resume; submission/acceptance/execution labels for selected actions; next-state transition or complete post-final-decision history claims; behavior probabilities, PPO or learner-ready claims; observation feature encoders, recurrent models or learning; CLI, artifact persistence, manifests, tournament or benchmark implementation; GPU, new mechanics or milestone acceptance; implementation-owner commits or pushes.

**Acceptance criteria:**

- [x] Expose one documented version-1 library API for a single completed battle. Compose the actual existing runBattle and keep its explicit ps/seeds/limits/preparedTeams/streamFactory contract where supported; default policies remain the existing randomPolicy and optional callbacks retain exactly the existing history/request/random input API. No existing source module changes were needed.
- [x] Emit a serializable allowlisted artifact with pinned format/upstream identity, validated public terminal outcome and two separate player trajectories. Each per-player sample has contiguous decision index, positive current request-episode ID, request kind, own current request snapshot, eleven-slot mask, selected slot/command and an own-history delta captured before that policy selects an action. Do not store RNG functions, seeds, packed teams, raw packets, trusted transcript/end metadata, arbitrary result/config fields or opponent-private data in decision observations.
- [x] Use Task028 actionSpace/decodeAction and only current-episode rejection evidence when recording masks and validating the chosen command. Sampling must consume no additional policy RNG draws. Return only valid actionable decisions: wait/forceSwitch false and duplicate unresolved requests yield no extra samples. Identical request payloads in later episodes remain separate samples; retries remain distinguishable in their original episode.
- [x] Capture immutable or detached snapshots before user-policy invocation; in-memory per-player history deltas reconstruct exactly the callback-visible history prefix at each successive decision, including non-actionable requests/events/resolutions received between callbacks. Deltas avoid copying the complete history at every step. Standard JSON serialization omits optional `undefined` fields; parsed deltas reconstruct the JSON projection of the callback prefix. A final selected command has no invented submission/resolution status, and post-final-decision observations are explicitly outside this initial schema.
- [x] Store terminal reward targets separately per fixed seat: winner +1, loser -1, tie 0. No alternating-player sign convention and no reward/outcome/future-opponent field enters policy callback input or pre-action observations. Behavior probability is null because policy likelihoods are unavailable.
- [x] Fail safely without returning a completed or partial dataset when runBattle, configuration, policy invocation or action validation fails. Fixed diagnostics avoid copying arbitrary exception properties/result-shaped metadata. No collector snapshot accessor is exposed.
- [x] Add independent scripted-stream tests for exact snapshots/history reconstruction, interleaved and consecutive own decisions, forced replacement, wait/no-choice, duplicate pending vs repeated later requests, unavailable-choice retry/exclusion reset, all three terminal reward cases, invalid selection/config/policy/runner failures, detached data and strict output allowlisting. Tests assert concrete expected fields and full reconstruction.
- [x] Enforce non-leakage with paired controlled streams that keep one actor callback history/current request identical while varying opponent private requests/split branches and pending choices or hidden end metadata. The recorded actor observations/masks/actions match under the same policy RNG; post-game targets are separate. A guarded pinned battle verifies masks/commands, own-side request histories and terminal target mapping, and matches the unwrapped command sequences under the same seeds.
- [x] Document API, JSON schema, privacy/target boundary, episode/retry responsibilities, delta reconstruction, unavailable likelihoods and completion-only/final-tail limitations in README. Task028, optional GPU history, task count/review baseline and the open G1 boundary are preserved.
- [x] Run focused collector tests and affected action-space/battle-runner/protocol/trace/reference regressions; syntax, JSON/diff, Trail build/check and inspected diff evidence are recorded below. GPU was not required. Primary independently accepted the task; implementation owner did not commit or accept its own work.

**Plan:**

- [x] Read the scoped handoff, project-plan G1/input-target boundaries, applicable instructions, Trail graph and relevant learnings before source edits.
- [x] Implement only a separate completed-battle wrapper/collector over existing callback and action APIs, with per-player observation deltas and separately finalized targets.
- [x] Add exact controlled-stream, failure, privacy-equivalence and guarded pinned integration tests; verify unchanged random-policy command sequences.
- [x] Update README and Task029 handoff/current durable record, refresh/check Trail, validate focused changes and inspect diff.
- [x] Stop edits and return evidence to primary for independent acceptance and delivery; retain ownership through any authorized corrections.

**Startup evidence:** Planned on clean main synchronized with origin/main at 1382890. Task028 and optional CPU/GPU validation are accepted and delivered. Count28/baseline20/checkpoint30; no pending review, active milestone review or GPU block. git, Trail0.28.0 map/ask/callers/skeleton, Node24.22.1, npm11.19.1 and Python3.13.18 are functional; guarded pinned reference provenance passes. GPU reports RTX3060Ti/driver610.88 but is not required. The requested remote repository is mounted at /workspace; the named child directory is absent. Cloud status tool/policy snapshot remain unavailable; no managed readiness claim is made. Git fetch succeeds with inherited settings.

**Implementation evidence:** Changed paths owned by this task are `environment/decision-trajectories.cjs`, `test/decision-trajectories.test.cjs`, `README.md`, this Task029 handoff, `docs/ORCHESTRATION_STATE.json`, and the two Trail-managed note files `.trail/learnings/2026-11-03-capture-trajectory-observations-from-policy-callbacks-developer.md` (retained as the replaced note record) and `.trail/learnings/2026-11-03-capture-trajectory-observations-from-policy-callbacks-developer-2.md` (current note, ID `l-b9bd63dc`). The current note was added with `trail note --replaces` to record the JSON-wire normalization clarification from primary review; Trail retains the predecessor and links it through the replacement metadata. No existing runner, protocol, action-space, Task028, or GPU-history source was changed. The module is the only new source API; its pre-selection snapshots/deltas and terminal targets follow the callback boundary. It preserves prepared-team handle identity and existing callback arguments.

**Primary review correction:** `/root` found the implementation sound and requested one test-only follow-up: prove detached snapshots against actual callback references, strict nested allowlists and JSON roundtrip behavior. The test now retains callback inputs, checks non-aliasing, mutates nested returned request/history data and verifies the callback inputs remain unchanged, asserts exact artifact/player/step/observation/action/target keys, and checks parsed request/mask/action fields plus history reconstruction against independently JSON-normalized callback inputs. The first raw object equality probe showed that a runner-owned `detail: undefined` value is omitted by ordinary JSON encoding; no collector or runner source change was made. README and the current Trail learning explain that in-memory deltas are exact and parsed histories match the JSON projection. `/root` reviewed the strengthened correction and accepted the source with this JSON-projection distinction; primary has accepted the task and retains delivery ownership.

**Validation:** `node --check environment/decision-trajectories.cjs && node --check test/decision-trajectories.test.cjs` passed after the correction. `node --test test/decision-trajectories.test.cjs` passed 5/5 after the correction. The affected command `node --test test/decision-trajectories.test.cjs test/action-space.test.cjs test/battle-runner.test.cjs test/prepared-teams.test.cjs test/protocol.test.cjs test/trace.test.cjs test/reference.test.cjs` passed 55/55 before the test-only correction and was not repeated, as authorized by primary. Trail was rebuilt after the correction with 651 nodes, 1,412 edges and 73 cards. `python3 -m json.tool docs/ORCHESTRATION_STATE.json >/dev/null` and `git diff --check` passed after final documentation updates.

`trail check --base HEAD` exited 1 with 11 lexical conflicts (README:82 three, state:549 two and state:551 one, TASKS:1091 two, TASKS:1093 two, TASKS:1097 one). None is a semantic contradiction: README/state/task scope reuses Task028's action-space adapter but never normalizes protocol packets in a policy; duplicate unresolved requests and later identical requests retain current runner behavior; `preparedTeams` are validated opaque seed-bound handles, whereas packed teams are excluded from observations; schema/version fields describe data shape and no authenticity claim is made; “complete” deltas mean exact own-view prefixes at decisions, not omniscient records or post-final tails; and this API targets pinned `gen5randombattle`, not every Gen 5 format. Additional leads were explicitly marked non-failing by Trail. The complete diff was inspected; no unresolved implementation concern remains. Primary re-review and acceptance are complete; primary owns delivery; no G1 closure, GPU claim, learning likelihood, or performance claim is made.

**Primary acceptance:** Primary /root accepted Task029 after independently inspecting the new collector, complete focused tests, documentation, learning and validation evidence. The wrapper uses only live own-view callbacks, detached pre-selection request/history deltas and Task028 masks; it forwards opaque handles and preserves policy RNG/selection behavior. Selected actions carry no fabricated resolution or likelihood; completed public outcomes produce separate fixed-seat targets. Exact scripted histories cover waits, repeats, consecutive decisions and episode-scoped retries; paired hidden-data tests preserve actor observations/actions, and a guarded real battle matches unwrapped default command sequences with prepared-team forwarding. One test-only follow-up added actual-reference mutation isolation, strict nested keys and JSON-projection reconstruction. All criteria are satisfied by 55/55 affected tests before the test-only strengthening and 5/5 focused tests afterward, plus syntax/JSON/diff/Trail evidence. No source correction or further task was needed. No blocker remains. G1 and later milestones remain open; GPU, likelihoods, full final tails and partial failed datasets are outside this accepted unit. Count advances 28 to 29, baseline remains 20 and next checkpoint remains 30.

**Final delivery checks:** The primary preserves all prior task, milestone and optional GPU evidence, including the original optional-evidence serialization. Final staged scope is restricted to this unit and its two CLI-managed learning records. Trail lexical findings are reviewed against the unchanged runner, opaque prepared-team handles, own-view observation and post-game target boundaries. New schema fields and completion metadata describe the artifact and do not authenticate serialized execution. The replacement learning distinguishes exact live JavaScript history from its standard JSON projection while retaining its predecessor. The task commit body records final Trail rationale.

**Final staged validation:** Primary JSON/history/scope assertions, byte-preservation of optional GPU evidence, linked learning-replacement assertions, staged diff check and graph-current check passed. The staged `trail check --base HEAD --json` exited 1 with 11 reviewed lexical documentation conflicts across 5 checked files and 15 active learnings, no overlaps; the distinctions above remain valid and are retained in the commit body. Accepted task count is 29; review baseline 20 and checkpoint 30 are unchanged, with no active task or GPU block.

## Task 030 — versioned move-first reference baseline policy

**Status:** Accepted and complete — 2026-11-03. Owner `/root/task030`; primary `/root`. Independent checkpoint `T030Global` accepted by `/root/t030_global`; source base `b0f996104f028306886f0326ba241dc27718ce70`.

**Scope:** Add a stateless move-first-v1 reference baseline callback composing Task028 action masks: uniformly choose among available move commands when any exist, otherwise uniformly among available switches. Honor current-request-episode rejections and supplied policy RNG. Verify through completed paired-seat pinned battles and trajectory observations; document the baseline without changing existing defaults. Reconcile the stale Task029 handoff headline with its recorded acceptance and delivered b0f9961 commit.

**Exclusions:** existing runner, protocol, action-space, default policy, pair/manifest execution or output-schema changes; damage/type/dex lookups, concealed state access or strength claims; tournament execution, policy registry or statistical/scored-suite design; observation features, training, learned models or likelihood labels; CLI, persistence or new mechanics; GPU or G1 milestone acceptance; implementation-owner commit/push.

**Acceptance criteria:**

- [x] Add a separate CommonJS module exposing a documented stable policy identity move-first-v1 and a synchronous callback compatible with existing {history,request,random}. It is stateless across calls and uses only current own request, own episode-scoped rejected commands and supplied RNG. Existing source modules/defaults remain unchanged unless a concrete mismatch is approved by primary.
- [x] Use the existing actionSpace/decodeAction contract to prefer every enabled move-row slot or explicit Struggle slot over switches. Uniformly select within the preferred set by one supplied RNG draw for each actionable call, including sole-choice calls; use enabled switches only when no move command remains. Non-actionable wait/forceSwitch false returns null without a draw; an actionable exhausted/invalid mask fails with a fixed diagnostic rather than a fallback.
- [x] Keep retry exclusions local to the latest requestEpisode in own history, so earlier rejections stop suppressing a command after progress. Preserve current-request reduced locked/recharge rows, Struggle, disabled/zero-PP moves, active/fainted replacement rules, trapped/maybeTrapped visibility and forced replacement. Reject invalid RNG values outside finite [0,1) safely; do not mutate callback input or leak private exception/request data through diagnostics.
- [x] Add independent exact-choice unit cases for move preference despite switches, random bucket endpoints, forced-switch fallback, sole-choice RNG use, wait/no-choice zero draws, episode rejection and reset, all-move rejection fallback, malformed/exhausted requests, reduced rows/Struggle and invalid RNG/input failures. Use concrete expected commands; prove inputs unchanged and seeded reproducibility without module-global state.
- [x] Complete two actual pinned battles in exchanged team seats against existing randomPolicy, keeping team/policy identities and policy seeds mapped through an existing generated pair and opaque prepared handles. Use actual Task029 callback trajectories to verify every baseline selection has an enabled slot and is a move whenever its recorded mask offers one; otherwise it is a switch. Verify terminal outcomes/reward mapping and cleanup without making scored or strength claims.
- [x] Document move-first-v1 semantics, request-only permissions, retry responsibility, RNG contract and direct runner/collector usage in README. Correct only the stale Task029 headline using its existing primary acceptance and actual delivery commit; retain its original stale wording as explicitly historical context or a dated reconciliation note. Preserve other task/milestone/GPU history.
- [x] Run focused baseline plus affected action-space, collector, runner, prepared-team, paired-seed, protocol, trace and reference checks as justified; syntax, JSON/diff, Trail build/check and inspected diff evidence are required. CPU suffices. Report exact commands/results/changed paths/concerns; stop edits for primary review and retain ownership through requested corrections.
- [x] Required independent Astra Task030 global checkpoint review covers accepted Tasks021-029 and the Task030 candidate, actual code/tests/pinned evidence, G0 and open downstream milestones, direction toward the reference environment and full JAX goal, and honest benchmark/GPU boundaries. No task count or review baseline advances until primary acceptance and the genuine independent gate ACCEPTED disposition; retain checkpoint30 while pending. After acceptance record count30/baseline30/next40 and stop, with no Task031.

**Plan:**

- [x] Read task handoff/state, applicable AGENTS/orchestration, project-plan G1 baseline goals and scoped Trail/learning context; verify tools/reference readiness.
- [x] Implement only the separate versioned baseline and focused contract tests, then paired-seat pinned collector integration.
- [x] Update README, Task030 evidence and the explicitly reconciled Task029 headline; preserve history and add a Trail learning only for a justified new decision/finding.
- [x] Run relevant checks once per meaningful revision, refresh/check Trail, inspect diff, and stop edits for primary acceptance review.
- Primary records the candidate and invokes independent Astra checkpoint review before final counting/delivery; the same implementer owns any required bounded corrections; no further deliverable is started.

**Startup evidence:** Planned on clean synchronized main at b0f9961. Count29/baseline20/checkpoint30, no active task, pending corrections or GPU block. Git, Trail0.28.0 lookups, Node24.22.1, npm11.19.1, Python3.13.18 and guarded pinned reference provenance are functional. GPU reports RTX3060Ti but no GPU criterion applies. Cloud status tool and policy snapshot unavailable; inherited authenticated Git fetch succeeds. Task029 delivery is verified at b0f9961; its stale headline is documentation bookkeeping, not a new acceptance decision.

**Required checkpoint:** Independent Astra `T030Global` is ACCEPTED; see [report](reviews/T030-global.md). Candidate count29/checkpoint30 were retained throughout review. Final acceptance sets count30/baseline30/next40. G1 remains open.

**Implementation evidence:** The only new source and test files are `environment/move-first-policy.cjs` and `test/move-first-policy.test.cjs`. README documents the stable identity, request-only mask behavior, episode-local retries, RNG contract and direct runner/collector use. The Task029 status headline was corrected with its dated reconciliation note above; its recorded acceptance and delivery remain `b0f9961`. No existing runtime/default APIs or other task/milestone/GPU records changed. No new Trail learning was warranted: the implementation follows the existing request-episode and explicit-Struggle learnings.

**Validation:** `node --check environment/move-first-policy.cjs && node --check test/move-first-policy.test.cjs` passed. The final `node --test test/move-first-policy.test.cjs test/action-space.test.cjs test/decision-trajectories.test.cjs test/battle-runner.test.cjs test/prepared-teams.test.cjs test/paired-seeds.test.cjs test/protocol.test.cjs test/trace.test.cjs test/reference.test.cjs` passed 71/71, including both generated-pair pinned runs, policy/team identity and seed mapping, selected-mask priority, terminal reward mapping, and tracked stream destruction. An initial 70/71 run failed only because the new test expected `switch 2` at slot 4; the existing action-space mapping uses slot 5, and the corrected test passed in the final run. GPU was not required.

`trail build` completed with 667 nodes, 1,442 edges and 75 cards. `trail check --base HEAD --json` exited 1 with 10 lexical conflicts and additional non-failing leads/follows across the checked files; there were no overlaps. The conflicts are false positives: the README describes explicit Struggle through the existing action mask/parser and policy-seed usage without changing parser behavior or accepting arbitrary packed teams; state/task exclusions and acceptance text refer to preserving existing runner/result boundaries; “complete” describes completed pinned battle runs and never claims a complete trusted transcript; the live trajectory collector, not a replayed transcript, verifies actual callback masks; the phrase about a fixed failure within the policy is unrelated to the stat-arithmetic learning. No production or test contradiction was reported. `python3 -m json.tool docs/ORCHESTRATION_STATE.json >/dev/null && git diff --check` passed.

**Review boundary:** Candidate evidence is ready for primary acceptance review and the required independent `T030Global` gate. Task030 is still unaccepted; accepted task count remains 29, baseline 20 and checkpoint 30. The implementer made no commit or push.

**Primary candidate review:** `/root` inspected source, tests and focused diff and found the implementation criteria satisfied, with no requested source correction. Combined validation passed 71/71 after the recorded wrong-slot test expectation was corrected. Implementation owner `/root/task030` stopped edits; it retains correction ownership. No task count or checkpoint advance is recorded before the independent gate.

**T030Global review request:** Independent Astra `/root/t030_global` reviews accepted Tasks021–029 and the Task030 candidate, actual source/tests/pinned evidence, direction and milestone/measurement boundaries. Reviewed base is `b0f996104f028306886f0326ba241dc27718ce70`; candidate staged tree `0c01e302feee05d40e9ae92356473a6309f698f2` was captured before this request metadata. Count stays 29, review baseline 20, checkpoint 30; no G1 closure is requested. Report: `docs/reviews/T030-global.md`.

**Final primary acceptance:** Primary /root accepted Task030 after its focused source/test review and independent Astra T030Global ACCEPTED disposition. The separate stateless move-first-v1 callback composes current masks/decoder, prioritizes enabled moves including Struggle, handles waits with zero RNG draws and actionable calls with one, scopes rejections to the current episode, and emits fixed diagnostics. Exact unit expectations and two real exchanged-seat pinned battles establish seed/identity mapping, mask priority, terminal targets and cleanup. The only implementation iteration corrected a wrong new test expectation (switch2 maps to slot5); final affected validation passed71/71. Independent checkpoint source sampling and tests passed57/57 plus23/23 pair/manifest. No required finding or blocker remained. Existing product APIs/defaults are unchanged. Task029 headline was reconciled to its established b0f9961 acceptance/delivery with original wording retained historically. G0-only milestone acceptance, optional historical GPU evidence and open G1+ boundaries remain. Count advances29 to30; accepted checkpoint sets baseline30 and next40. No Task031 starts.

**Checkpoint disposition:** Independent `/root/t030_global` wrote [T030-global.md](reviews/T030-global.md) with ACCEPTED and no substantive correction or blocker. It independently passed 57/57 tests and the corrected pair/manifest suites23/23, pinned provenance and diff checks. The report preserves the initially mistyped test filenames and explains the corrected coverage. Historical CPU/GPU evidence remains attributed to its source and no new GPU run is claimed. Its optional charter wording observation is explicitly deferred; no optional change is implemented. The gate covers Tasks021–030 only and closes no later milestone.

**Final coordination:** All existing task/milestone/GPU history is preserved; Task029 has only the documented headline reconciliation. New source/tests remain identical to candidate tree `0c01e302feee05d40e9ae92356473a6309f698f2`. The primary performs final state/history/scope/diff/graph checks and records reviewed lexical Trail findings in the commit body before delivery. No Task031 is started.

**Final delivery validation:** Primary state/history/count/checkpoint, source identity, staged scope, optional GPU byte-preservation, diff and graph-current checks passed. All numbered task headlines now match the recorded accepted history. The final staged Trail lexical findings were reviewed against actual code and the preserved request, visibility, provenance, arithmetic and serialized-data boundaries; no blocking contradiction remains. Exact findings and semantic rationale are recorded in the task commit body.

## Task 031 — named baseline paired matchup execution

**Status:** Accepted and complete — 2026-11-03. Owner `/root/task031`; primary `/root`. Synchronized source base `3c7286492c4e4e4c199dcedd000cb60a7cc30359`.

**Scope:** Add a separate CommonJS API for one completed paired matchup using caller-selected allowlisted baseline IDs legal-random-v1 and move-first-v1. Compose generatePair, prepareTeam and actual runBattle calls; prepare each team once, run both seat assignments sequentially with policy identity and RNG seed mapped together, and emit a distinct versioned engineering artifact with canonical configuration and safe per-battle results. This promotes the Task030 paired integration procedure into a reproducible named-matchup building block for G1.

**Exclusions:** changes to existing runner, default pair/manifest executors, callback/action/protocol contracts or existing journal/report schemas; arbitrary policy callbacks or caller-declared custom version names; CLI, file persistence, journal/resume or tournament aggregation; partial failed-matchup artifacts or fabricated unattempted results; scored suite selection, rates, uncertainty, performance or policy-strength claims; trajectory collection in the new output, training or new mechanics; GPU or milestone acceptance; implementation-owner commit/push; optional charter cleanup deferred by T030Global.

**Acceptance criteria:**

- [x] Expose a documented version-1 completed-matchup library API with required rootSeed, zero-based pairIndex, policyA and policyB IDs and explicit pinned ps API; forward supported limits and streamFactory. Accept only legal-random-v1 (the current unchanged randomPolicy) and move-first-v1 (the existing versioned policy). Both roles may use the same ID for self-play. Snapshot/validate selection and IDs before any team generation or stream creation; unknown IDs/config getters fail safely, never silently default or inspect arbitrary exception fields.
- [x] Select the requested pair directly using generatePair without materializing earlier indices. Prepare team A/B once each with existing opaque seed/API-bound handles, reuse them in two separate sequential actual runBattle calls, and exchange team seats exactly as the generated pair specifies. Map callbacks through each run.policySeats so a policy identity retains its policy RNG seed across seats; no extra RNG draws or regenerated teams.
- [x] Emit a distinct allowlisted schema/kind for named baseline matchups, separate from Task024/025 legacy pair/journal shapes. Include pinned format/upstream, root/pair identity and exact canonical paired run metadata/seeds, explicit policy IDs by role and by run seat, prepared-team digests, and the two actual completed nested safe resultRecord objects. No raw transcript, packets, packed-team strings/handles, policy input/history, raw errors, arbitrary config fields or forged foreign result-shaped metadata leave the API. Artifact metadata does not authenticate execution after serialization.
- [x] Return an artifact only after both battles complete with validated terminal outcomes and cleanup. Fail with a fixed safe diagnostic and no partial artifact on malformed configuration, team/packing, first/second battle or cleanup failure. If the first battle fails no second stream starts; each attempted stream receives existing runner cleanup. Do not infer or claim zero/unattempted work from a failed call with no artifact. Existing legacy pair/manifest behavior and schemas remain unchanged.
- [x] Add independent controlled tests for exact policy identity/seed/seat/team-digest metadata, both policy orderings and same-ID self-play, current request-based command behavior, two preparations/packings, exclusive sequential fresh streams, safe output allowlisting and detached configuration/metadata. Guard invalid roots/indices/IDs, hostile getters and thrown foreign result metadata before work; preparation and first/second/cleanup failures; and no artifact after failure. Compare concrete expected commands and counters rather than merely calling implementation helpers for expectations.
- [x] Execute a real guarded pinned mixed-baseline pair, observe actual packed-team writes exchange seats and hashes recur, verify each baseline is attached to its intended seat/policy seed, both terminal result records have prepared source and match canonical run seeds/outcomes, and both streams are destroyed. Use trusted stream instrumentation only in tests; policy inputs stay the existing recipient-scoped API. No strength estimate or performance result is reported.
- [x] Document named baseline version semantics, required configuration, schema distinction, generate-once/reuse and identity-seed mapping, completion-only/error boundary, safe metadata and timing/attestation limits in README. Update only Task031 handoff/current state; preserve Task001-030, G0/T030Global and optional GPU evidence. Existing default policy APIs and legacy output schemas stay unchanged; G1 remains open.
- [x] Run focused matchup tests and justified affected baseline, runner, prepared-team, paired-seed, legacy pair/manifest, trajectory, protocol and reference tests once per meaningful revision; syntax, JSON/diff, Trail build/check and own diff review are required. GPU is not required. Report exact changed paths/checks/results/concerns to primary, stop edits for acceptance and retain ownership through bounded corrections. No Astra gate is due at count30/checkpoint40 unless a real architectural contradiction arises.

**Plan:**

- Read current task/state/orchestration, project-plan G1 pairing/tournament goal and Trail/learning context; verify tools and reference pin.
- Implement a separate strict named completed-pair wrapper over existing pairing, opaque team preparation and battle execution primitives, leaving legacy executors unchanged.
- Add controlled validation/failure/metadata/sequence tests and a real pinned mixed-baseline seat-exchange check.
- Update README and Task031 evidence, refresh/check Trail, run affected checks and inspect the diff; add a learning only for a justified non-obvious decision/finding using trail note.
- Stop for independent primary criteria review; primary records acceptance/count31 and delivers only this unit without advancing review checkpoint40 or starting Task032.

**Startup evidence:** Planned on clean synchronized main at 3c72864. Task030 and independent T030Global are accepted/delivered; count30/baseline30/checkpoint40, no pending correction, active review or GPU block. Git, Trail0.28.0, Node24.22.1, npm11.19.1, Python3.13.18 and guarded pinned reference provenance work. GPU RTX3060Ti is present but not required. Cloud status tool/policy snapshot unavailable; inherited Git fetch succeeds. Requested origin is mounted at /workspace. No accepted but undelivered work remains.

**Implementation evidence (accepted after primary review):** Added `environment/baseline-matchup.cjs` and `test/baseline-matchup.test.cjs`, documented the API in README, and created the scoped Trail learning, now `l-ed326324`, which replaces and preserves same-unit predecessor `l-30da8511` through the CLI. The API validates a fixed option/ID allowlist and supported limits before team generation; selects with direct `generatePair`; prepares each opaque team once; and invokes the existing runner twice sequentially with seat, policy-role and policy-seed mappings from each generated run. Its version 1 artifact kind is `gen5-named-baseline-paired-matchup` and contains only canonical pairing metadata, prepared-team digests, policy IDs and copied completed safe result records. It does not return partial artifacts, runner transcripts or caller-supplied metadata. The existing runner, default pair/manifest executors, legacy reporter, policy callbacks, and schemas were not changed. A controlled boundary test confirms `summarizeResults` rejects the new kind even with the matching root and pair manifest.

Controlled tests cover both mixed policy orders, same-ID self-play with distinct role seed streams, concrete command and counter expectations, team generation/packing counts, fresh sequential streams, detached configuration, invalid and hostile options, first/second battle and cleanup failures, and legacy schema rejection. The guarded pinned mixed run recorded exchanged packed-team writes and matching digests, canonical role seeds/outcomes, prepared source and destruction of both streams. This is integration evidence, not a strength or performance result. Initial focused attempts found only test assertion issues (checking cleanup on the raw Showdown stream and comparing null-prototype JSON counters with plain-object prototypes); the final focused run passed after correcting the assertions.

**Validation:** `node --check environment/baseline-matchup.cjs && node --check test/baseline-matchup.test.cjs` passed. `node --test test/baseline-matchup.test.cjs` passed 7/7, including the guarded pinned mixed run. The affected combined command `node --test test/baseline-matchup.test.cjs test/move-first-policy.test.cjs test/action-space.test.cjs test/battle-runner.test.cjs test/prepared-teams.test.cjs test/paired-seeds.test.cjs test/pair-runner.test.cjs test/manifest-runner.test.cjs test/summarize-results.test.cjs test/summarize-work.test.cjs test/decision-trajectories.test.cjs test/protocol.test.cjs test/reference.test.cjs` passed 115/115. Trail was refreshed once at the frozen source/test revision and indexed 710 nodes, 1,533 edges, and 77 cards. `trail check`, JSON validation and final diff review are recorded in the active state after completion.

**Trail check review:** `trail check --base HEAD --json` reported 8 lexical conflicts, 9 leads, and no overlaps. Inspection found no source or architectural contradiction. The packed-team matches refer to explicitly excluded serialized team strings and the digests of opaque prepared handles; the kind/schema and runner-boundary matches are accompanied by explicit statements that serialized metadata is not attestation and legacy error ownership is unchanged; the format match is the pinned `gen5randombattle` identifier; and the transcript matches are exclusions of transcripts from this artifact. Other leads match preserved Task history or mechanics wording outside this deliverable. These findings remain documented as lexical results, not waived acceptance criteria.

**Review boundary:** Primary accepted Task031 after the owner stopped edits. Count is31; review baseline30/checkpoint40 and open G1 status are unchanged. No GPU criterion applies. Implementation owner did not commit or push; primary owns delivery.

**Primary acceptance:** Primary /root independently reviewed the complete new module, controlled and guarded pinned test source, README contract and exact validation evidence, and accepted Task031 with no source/test correction. Strict data/ID/limit snapshots occur before work; the requested canonical pair is selected directly; each team is prepared/packed once and opaque handles reused in sequential fresh battles. Policy versions, roles, seats and RNG streams remain mapped together, including reverse order and self-play. A distinct allowlisted completion-only artifact carries copied actual safe records and no transcripts, packed teams or exception metadata. Failure stages return a fixed error without a partial artifact, and the legacy summarizer rejects the new envelope. Handwritten command/seed/counter expectations, detachment, no-work-invalid-config cases, cleanup and real packed seat/hash exchange support the criteria. Focused7/7 and affected115/115 passed after test-only instrumentation/prototype expectation corrections; syntax, JSON/diff and Trail evidence passed or was semantically reviewed. Existing product modules/defaults and all earlier task/milestone/GPU history are unchanged. Count advances30 to31; review baseline30 and checkpoint40 stay fixed. No GPU, performance/strength or G1 acceptance claim; no Task032 starts.

**Durable learning:** [Keep named matchup semantics separate from legacy pair records](../.trail/learnings/2026-11-03-keep-named-matchup-semantics-separate-from-jarho001.md), `l-ed326324`, replaces the preserved same-unit predecessor `l-30da8511`. The replacement records the same design choice with the required learning sections; no earlier committed learning was edited.

**Final coordination:** Historical state serialization, including optional GPU evidence, is preserved. Final staged Trail lexical findings are reviewed against the actual safe wrapper and unchanged legacy contracts; the commit body retains the semantic rationale. Final JSON/history/count/scope, source identity, staged diff and graph-current checks precede delivery. No Task032 is started.

**Final delivery validation:** State/history/count/scope/source identity, learning-replacement linkage, exact optional GPU evidence preservation, staged diff and graph-current checks passed. Final staged Trail matches were reviewed as lexical documentation findings, with semantic distinctions recorded in the task commit body. Count31/baseline30/checkpoint40 and null active-task/GPU-block state are confirmed; no Task032 was started.

## Task 032 — single named baseline matchup CLI

**Status:** Accepted and complete — 2026-11-03. Owner `/root/task032`; primary `/root`. Source base `b4ffdaa917c903654a5e30f0aaafc0a746e8f2ac`.

**Scope:** Add a separate CLI for exactly one named completed baseline pair, composing the accepted Task031 library and exclusively publishing its safe artifact to an explicit new file. This makes the G1 named evaluation building block runnable without custom JavaScript.

**Exclusions:** changes to existing runtime/library schemas or policy behavior; batch/tournament aggregation, scoring, confidence intervals or strength/performance claims; resume, partial results or failure journals; new mechanics, training or GPU validation; G1 acceptance; unrelated optional documentation cleanup; owner commit/push.

**Acceptance criteria:**

- [x] Require explicit --root-seed, zero-based --pair-index, --policy-a, --policy-b and --output; support strict flag/value and flag=value forms, sole help, reject missing/duplicate/unknown/malformed values and IDs before bootstrap or execution. No implicit seeds/IDs/pair index.
- [x] Use pinned oracle.assertReady and actual runBaselineMatchup; preserve its exact completed artifact and limits/defaults, opaque teams, role/seed mapping and cleanup contract. Leave existing library and legacy executors/schemas unchanged.
- [x] Create the explicit new output file exclusively before battle execution so existing files/symlinks or missing parents fail without a battle. Publish only after successful complete execution; safely remove the owned reservation on pre-publication failure. Never overwrite/delete unrelated paths. Document crash/write/close/cleanup failure limitations honestly, and never emit fabricated partial outcomes.
- [x] Print no raw exceptions, paths from exception text, packets, transcripts or failed result-shaped metadata. Fixed safe diagnostics and nonzero failure exit. Report success only after completed output write/close; no failed invocation claims zero attempted work.
- [x] Add controlled tests for parser, bootstrap/execution/order, exclusive-file preservation, invalid/no-work inputs, safe errors, owned reservation cleanup and write/close failures as practical. Run a subprocess real pinned mixed-policy pair and validate saved kind/config/pin/canonical seed/seat mapping and both completed prepared-team result records.
- [x] Document usage and success/failure/file/timing/attestation boundaries. Run focused and justified affected regressions, syntax, JSON/diff and Trail build/check; inspect own diff, report exact evidence and stop for primary acceptance. Preserve earlier history, count31/baseline30/checkpoint40 until primary acceptance; no GPU or Astra gate required.

**Plan:**

- Read applicable Trail and prior learnings and trace affected symbol dependencies before implementation.
- Implement separate CLI and targeted tests, leaving accepted Task031 library unchanged.
- Run meaningful focused and affected checks including a real pinned CLI subprocess; document boundaries, update only Task032 progress, refresh/check graph and inspect diff.
- Stop edits for primary review; primary alone accepts/counts/delivers, then stop at this task boundary.

**Startup evidence:** Startup verified clean matching local and remote default main; SSH unavailable in new developer runtime, repaired Git safe.directory and synchronized via anonymous HTTPS with GitHub connector branch verification. Connector has repository push permission for delivery. Required Trail0.28.0, git2.39.5, Node24.22.1, npm11.19.1, Python3.13.18 and pinned oracle provenance functional; RTX3060Ti present but no GPU needed. No due review, active task or required external evidence.

**Implementation evidence:** Added `environment/run-baseline-matchup.cjs`, `test/run-baseline-matchup.test.cjs`, and the README CLI contract. Parsing requires explicit seed, pair index, policy IDs and output path; validation precedes pinned reference setup. The CLI exclusively reserves the destination before the pair starts, retains its descriptor, writes and flushes only a returned completed artifact, closes it before reporting success, and uses fixed safe diagnostics. It checks destination inode identity before writes and cleanup; concurrent path replacement is unsupported. Abrupt process or filesystem failures may leave an empty or partial reservation, and the containing directory is not fsynced. No journal, resume, aggregate or partial result is created.

**Validation:** `node --check environment/run-baseline-matchup.cjs && node --check test/run-baseline-matchup.test.cjs && node --test test/run-baseline-matchup.test.cjs` passed initially 10/10. The affected command `node --test test/run-baseline-matchup.test.cjs test/baseline-matchup.test.cjs test/paired-seeds.test.cjs test/manifest-runner.test.cjs` passed 36/36 before the primary-requested follow-up assertions. The already-running `node --test test/*.test.cjs` also completed 140/140 before those assertions. Primary review requested malformed-root and successful-bootstrap/missing-parent no-work checks, plus comparison of real pair/run metadata to `generatePair([1,2,3,4],0)`; all were added. First follow-up run was 9/10 because the expected diagnostic count was not updated for the second malformed input; after correcting that test expectation, `node --check test/run-baseline-matchup.test.cjs && node --test test/run-baseline-matchup.test.cjs` passed 10/10. No source correction was needed. Trail graph refreshed after final source/test edits to 738 nodes, 1,574 edges and 79 cards. JSON/history/diff and graph-current checks pass. Trail lexical check reports 8 documentation-only false-positive conflicts, no source conflict/overlap; their semantic rationale is in the primary delivery body. No GPU evidence is required. Primary accepted count32 after the owner stopped edits; review baseline30/checkpoint40 and prior GPU/history remain unchanged.

**Durable learning:** [Reserve output files before long-running battles](../.trail/learnings/2026-11-03-reserve-output-files-before-long-running-jarho001-2.md), `l-e0423be3`, replaces the same-unit preliminary note `l-7f54bde1`, which remains preserved. Both new note files are included in this delivery; no earlier learning was edited.

**Primary acceptance:** Primary /root independently reviewed the complete CLI, corrected tests, README contract and exact owner validation evidence. Accepted Task032 after the owner stopped edits. Explicit arguments are validated before pinned bootstrap; output is exclusively reserved before actual runBaselineMatchup, and only its completed artifact is written through the retained descriptor, flushed and closed before success. Existing files/symlinks and missing parents stop before battle work; fixed diagnostics expose no foreign exception metadata. Ordinary failures attempt identity-checked reservation cleanup; documented crashes/filesystem failures can leave empty or partial files, and concurrent output replacement is unsupported. Controlled no-work, cleanup/write/sync/close/replacement tests and a real pinned subprocess with canonical pair/run metadata support all criteria. Primary requested only test evidence additions and corrected learning spans via CLI replacement; no runtime change or criterion waiver. Corrected focused10/10 passes after a test-only diagnostic-count fix; earlier affected36/36 and already-running full140/140 passed at the identical source revision. Existing Task031 library, legacy executors, policies and schemas remain unchanged. Count31 becomes32, baseline30/checkpoint40 stay fixed; G1 remains open and no GPU or strength/performance claim is made. Stop at Task032; no Task033 starts.

**Final coordination:** Final staged scope, JSON/history/source identity, exact optional GPU evidence preservation, learning replacement, diff and graph-current checks precede delivery. Trail lexical findings are reviewed against actual code and recorded with semantic rationale in the commit body. Delivery uses the authorized GitHub connector and a non-forced default-branch update because this runtime has no working SSH transport; HTTPS fetch synchronizes local Git afterward. No Task033 starts.

**Final delivery validation:** Primary JSON/history/count/review/GPU preservation, exact Task001–031 handoff prefix, source-blob identity, seven-file staged scope, learning replacement, diff and graph-current checks passed. Final staged Trail check exits1 with8 lexical documentation findings across5 checked files/17 active learnings, no source conflict or overlap; exact findings and semantic rationale are retained in the commit body. Count32/baseline30/checkpoint40 and null active-task/GPU block are confirmed. No Task033 starts.

## Task 033 — offline named baseline outcome report

**Status:** Accepted and complete — 2026-11-03. Owner `/root/task033`; primary `/root`. Source base `a671f15266bb49e634bce2677008963c1284c938`.

**Scope:** Add a separate pure CommonJS outcome-report API for a contiguous recorded prefix of Task031/032 named completed paired matchups, bound to explicit root seed, expected pair count, policy IDs and complete expected limits. Validate canonical identities and outcome topology, then count completed outcomes by seat and policy role. This provides named-baseline accounting for the G1 tournament path without changing legacy accounting.

**Exclusions:** execution, Showdown bootstrap or new battle run orchestration in the report API; CLI, filesystem I/O, persistence, journal/resume or failed/partial-matchup accounting; changing accepted named/legacy library, executor, policy or report schemas; rates, confidence intervals, suite/power/stopping-rule selection, strength/performance claims; authentication or execution attestation from serialized metadata; packed-team regeneration or hash attestation; new mechanics, training, GPU or milestone acceptance; unrelated optional documentation cleanup and owner commit/push.

**Acceptance criteria:**

- [x] Expose a documented version1 pure synchronous named report API with required rootSeed, positive bounded pairCount, policyA, policyB, complete expected limits and matchups array. Only legal-random-v1 and move-first-v1 are accepted, including same-ID self-play. Strictly snapshot own plain JSON data, reject unknown config/envelope/nested schema keys, accessors, sparse/non-JSON/cyclic inputs and malformed values with one fixed safe diagnostic; never read foreign exception fields or mutate inputs.
- [x] Validate each artifact as the exact Task031 named completed kind/schema/pin/format, rootSeed, pairIndex equal its array index, canonical pairId, policy-role IDs, complete expected limits, policy seeds, team seeds and lowercase SHA256-shaped digests. Require both canonical runs in order with generated run IDs/seats/policySeats/seed tuples and matching IDs by seat. Use generatePair directly for recorded indices, not earlier/whole configured manifests. Reject duplicate, out-of-order, skipped, oversized or legacy-artifact prefixes and mixed policies/limits/pins.
- [x] Validate both nested safe battle-result envelopes: exact required keys, schema/kind/pin/format/seeds/limits, completed status, prepared team source, null failure/cleanupFailure, finite nonnegative battle_runner elapsed time, valid win/p1-or-p2 or tie/null outcome and nonnegative safe turns. Counters remain finite JSON data as accepted by the named producer; do not claim validation of their mechanics/arithmetic or re-emit them. No raw transcripts, packed teams, handles, callback inputs or foreign error/result metadata enter the report.
- [x] Return a distinct allowlisted report kind/schema with detached pinned configuration and coverage counting recorded completed pairs/battles and unrecorded suffix pairs/run slots. Empty/short/full prefixes are supported. Missing suffix artifacts disclose no execution, failure or attempted-work status; do not label them failed, unattempted, successful workload completion or zero work. Serialized structural validation never attests who ran the battles.
- [x] Count wins/losses/ties by p1/p2 seat and policyA/policyB role, each role carrying its explicit version ID and separate by-seat tally. Keep roles distinct for same-ID self-play. Include nonduplicated overall win-by-seat/tie totals and completed battle denominator; test handwritten all-win/loss/tie and swapped-seat cases and arithmetic consistency. No rate/strength statistic.
- [x] Add independent controlled valid artifacts and mutation/hostile-input rejection tests for all identity/config/topology boundaries, exact output allowlisting, no input mutation, output detachment, empty/partial/full prefixes, same-ID role accounting and missing-record semantics. Obtain real guarded pinned named mixed-policy artifacts, JSON-roundtrip and report them with expected outcomes/counts independently checked; keep test-only execution outside report API. Existing legacy reporter rejects named artifacts and new reporter rejects legacy ones.
- [x] Document API/schema and prefix, role, limits, digest/attestation and omitted-work boundaries. Run focused and justified affected named/report/paired-seed/reference checks once per meaningful revision; syntax/JSON/diff, Trail refresh/check and own diff inspection. Preserve all prior task/review/GPU history and count32/baseline30/checkpoint40 until primary acceptance; no GPU or Astra gate required absent a concrete contradiction. Report exact evidence and stop edits for primary review.

**Plan:**

- Read applicable instructions, relevant Trail learnings and scoped dependency/source context.
- Implement a separate strict pure named accounting module and independent controlled tests; leave accepted modules unchanged.
- Run real pinned roundtrip integration and affected tests, document boundaries and update Task033 progress; refresh/check Trail and inspect diff.
- Freeze for primary criteria review; primary alone accepts/counts/delivers and stops after Task033.

**Startup evidence:** Clean synchronized main at a671f15 with32 accepted tasks and no active/pending review or GPU block. Default main and connector push permission verified; HTTPS fetch/fast-forward works while configured SSH remains unavailable from prior runtime diagnosis. Trail0.28.0, git2.39.5, Node24.22.1, npm11.19.1, Python3.13.18 and pinned oracle provenance functional. RTX3060Ti present but not required. Cloud status tool/policy snapshot unavailable; no runtime readiness claim. Requested repository remains mounted at /workspace.

**Implementation evidence:** Added `environment/summarize-baseline-matchups.cjs` and `test/summarize-baseline-matchups.test.cjs`, and documented the pure API in README. The report snapshots own plain data without invoking accessors, validates each zero-based prefix member against `generatePair` and the pinned Task031 schema, requires matching role IDs and all explicit battle limits, checks safe completed nested records, and counts by seat and role. The returned schema contains configuration, artifact coverage and integer tallies only. It omits counters, transcripts, packed data, attempts/failures and rates; structural artifact identity is not an execution attestation. Existing named and legacy modules were not modified.

**Earlier validation before primary correction:** The initial focused command `node --check environment/summarize-baseline-matchups.cjs && node --check test/summarize-baseline-matchups.test.cjs && node --test test/summarize-baseline-matchups.test.cjs` passed6/6, including controlled W/L/T vectors, coverage prefixes, same-ID role mapping, strict invalid/hostile input cases, detachment, and actual guarded pinned JSON-roundtrip pair output. The affected command `node --check environment/summarize-baseline-matchups.cjs && node --check test/summarize-baseline-matchups.test.cjs && node --test test/summarize-baseline-matchups.test.cjs test/baseline-matchup.test.cjs test/run-baseline-matchup.test.cjs test/paired-seeds.test.cjs test/summarize-results.test.cjs test/reference.test.cjs` passed46/46. Earlier initial failures and their exact causes/fixes are preserved in active_task progress. `trail build` and `trail build --check` pass at765 nodes/1653 edges/81 cards. JSON, syntax and diff checks pass. No GPU required.

**Trail review:** Candidate `trail check --base HEAD --json` reports10 lexical conflicts, no overlaps, and no source conflict. Documentation findings refer to explicit reproducibility/seed-bound/packed-team boundaries and safe runner records; the report validates metadata shape but never authenticates execution or reruns packed teams. Other matches are task wording and are unrelated to positive stat mechanics, full oracle player observations, or changing the runner private error boundary. A final check and primary semantic review remain.

**Owner status:** Owner froze implementation before primary acceptance and made no commit or push. Primary correction revision1 is recorded below; owner claimed no acceptance or GPU evidence. Historical Task001–032 prefix and all state/review/milestone/optional GPU evidence remain preserved.


**Initial debugging evidence:** Two source issues were corrected before primary review. Valid matchup fixtures initially failed because `copyArray` did not pass the run index into `validateRun`; passing it selected the proper generated run. That change then produced `TypeError: ancestors.has is not a function` because `copyJson` was used directly as the mapper and received the index in its second parameter; wrapping the call in an item-only callback corrected it. Separate test-only issues were the stale unused `ROOT` identifier after import cleanup (`ReferenceError: path is not defined`), a p1 expected count of2 for a vector with one win and one tie (corrected to1), and a same-ID vector that assumed policyA occupied p1 in the swapped run (outcomes now derive from the canonical role/seat mapping; expected totals stay hand-counted). No criteria were waived.

**Primary correction revision 1:** Primary found direct `JSON.stringify` equality rejected otherwise-valid artifacts when object key order differed. Replaced it with recursive object-key-order normalization; arrays and the two canonical run positions remain ordered. A JSON-roundtrip/reversed-object-key fixture confirms the report is unchanged, while reordered runs remain invalid. The same correction revision expands the independent test table across upstream, format, root/team/run seeds, seats/roles/policy IDs, nested result pin/status/limits/failures/tie/turns and unknown keys; checks the exact report root allowlist, same-ID per-seat tallies, exact fixed error diagnostic and zero accessor reads; and throws a foreign error from a proxy ownKeys trap while verifying the input proxy getter stays unread. The nested upstream accessor is also covered. Final corrected affected suite passes47/47; no change was made to the producer or legacy schema.

**Final owner report:** Files: `README.md`, `docs/TASKS.md`, `docs/ORCHESTRATION_STATE.json`, `environment/summarize-baseline-matchups.cjs`, `test/summarize-baseline-matchups.test.cjs`. Final checks: focused seven tests7/7; affected combined suite47/47; syntax and JSON checks pass; `git diff --check` passes; Trail graph769 nodes/1662 edges/81 cards and current. Final lexical check result/rationale is recorded after handoff finalization. Source SHA-1 Git blob `ef564a4e5ed3c1c8bf4109b83ffe2da034a8cca4`; test SHA-1 Git blob `0423e240bb16dab1ded41d82fb449ecac826ff9e`. No new durable learning was added because existing offline-report and named-matchup learnings cover the decisions. GPU not required. At owner freeze Task033 remained unaccepted and count32. Primary acceptance below advances count to33; review baseline30/checkpoint40, prior histories and exact optional GPU evidence stay unchanged.


**Owner-stage Trail findings:** `trail check --base HEAD --json` exits1 with10 lexical conflicts across5 checked files and17 active learnings, zero overlaps. The matches are documentary phrase collisions: README:231 says the report compares canonical artifact schema and seed/seat metadata while explicitly checking digest shape only; it accepts no packed-team strings and claims no attestation. State:920 scope describes a runnable report workflow, not executing battles or path-based writes; state:933 validates the producer's safe completed result shape while omitting exception metadata and making no execution-authentication claim. Task criteria at1274 describe generated seeds, run topology and result fields, not accepting arbitrary packed teams; at1276 they refer to nested runner result fields for accounting without attestation; at1282 complete battle result shapes mean required safe nested records, not full unprojected player oracle evidence. No file changes player visibility, exposes transcripts/packed teams, alters mechanics, or changes runner ownership. One source-line lexical lead repeats the explicit exclusion of team repacking/digest attestation; it is a non-conflict and matches implemented shape-only validation.

**Primary acceptance:** Primary /root independently reviewed the full report module, corrected tests, README API contract and owner validation evidence. Accepted Task033 after the owner froze all edits. Strict data snapshots precede canonical prefix/config/run/result validation; role/seat tallies count only structurally valid completed named artifacts and retain distinct roles for same-ID self-play. Explicit complete expected limits prevent mixed-budget pooling. Output is detached configuration, recorded/unrecorded coverage and integer outcomes only, with no raw data, counters, execution/attempt claims or attestation. Primary correction revision1 fixed object-key-order sensitivity while preserving array/run ordering and strengthened identity/topology, exact-output, same-ID by-seat and zero-read accessor tests. Corrected focused7/7 and affected47/47 pass, including real guarded pinned JSON roundtrip. The reflection test throws a plain Error from an input proxy ownKeys trap and verifies the input get trap is unread; independently, source catch has no exception binding and never inspects thrown fields. Initial copyArray mapper/index defects and test-vector mistakes are retained as history, not hidden. Source/test blobs match the frozen owner revision, and existing producer/CLI/legacy/policy schemas are unchanged. No criteria waived, new GPU run, strength/performance claim or G1 closure. Count32 becomes33; review baseline30/checkpoint40 unchanged; no Task034 starts.

**Final coordination:** Primary reconciled stale pre-correction summaries as historical6/6 and46/46 evidence; final corrected counts are7/7 and47/47 at graph769 nodes/1662 edges/81 cards. Both implementation hashes are SHA-1 Git blob IDs. No new learning was needed; existing named-schema and offline-attestation learnings apply. Final staged JSON/history/count/scope/source/diff/graph checks and Trail semantic review precede delivery through the authorized GitHub connector with a non-forced default-branch update and local HTTPS fast-forward. Stop at Task033.

**Final delivery validation:** Primary JSON/history/count/review/GPU preservation, exact Task001–032 prefix, source/test blob identity, five-file staged scope, diff and graph-current checks passed. Primary final staged Trail check reports10 lexical documentation findings across5 checked files/17 active notes and no overlaps or source conflict; exact current findings and semantic rationale are retained in the delivery commit body. No source or test changed after passing47/47. Count33/baseline30/checkpoint40 and null active-task/GPU block are confirmed. No Task034 starts.

## Project direction decision — simulator-first

**Status:** Adopted by the primary on 2026-11-03 as an unnumbered planning decision. This does not change accepted task count33, review baseline30, next checkpoint40, or any milestone status. No Task034 starts as part of this decision.

**Decision:** Prioritize an integrated, increasingly playable JAX battle simulator and its parity validation against pinned Showdown. G3 simulator work may proceed while G1 reference-environment milestone acceptance and G2 learning proof remain open. Neither G1 nor G2 is waived or accepted. The complete pinned `gen5randombattle` target and fidelity requirement remain unchanged.

**Rationale:** The immediate project objective is simulator replacement, while the existing Showdown runner, action adapter, player-view boundary, and replayable validation artifacts already provide enough reference context to start simulator work. Raw Showdown transcripts remain trusted oracle data; projected player views define actor-visible inputs. More policy, tournament, reporting, or trajectory infrastructure does not reduce the main uncertainty unless it directly supports simulator semantics or parity checks. A minimal policy-facing contract is still essential because observations depend on legal requests, simultaneous commitments, visible events/history, and terminal outcomes; neural feature encoding and learning can follow the simulator slice.

**Retain:** Showdown remains the semantic oracle, reference-data and team source, and differential harness. Validate request/action behavior and player-visible histories along with RNG, internal state, and emitted events. Cover reachable interactions and full traces as implementation expands; report unsupported effects explicitly. Compile and batch representative slices early and measure JIT/vmap behavior and throughput without making training necessary. Restricted development variants must name their supported scope and cannot imply full-format support. Validate on CPU first; require GPU evidence only when a task criterion or device-specific property needs it, not as a full performance suite for every slice.

**Defer:** PPO, tournament/scoring work, expanded trajectory collection, and additional baseline/reporting infrastructure unless they directly enable simulator semantics or parity validation. Do not silently drop G1/G2 acceptance criteria or claim downstream milestone closure.

**Next bounded-work candidate:** An integrated runnable JAX turn slice for an explicitly named restricted development variant covering declared ordinary move cases, with exact RNG, state, and request parity against Showdown. Reassess scope and feasibility when selecting the next numbered task. The slice does not claim full-format support, is not active or accepted, and does not require GPU validation unless the chosen criterion requires it; current `active_task` remains null. Measure progress by parity-tested playable battles rather than the number of task entries.

**Validation and coordination:** Planning-only change; no source tests or GPU evidence are required. Preserve task history and all review/GPU records. The primary reviews this decision and docs diff; there is no new milestone or global review acceptance implied.

## Task 034 — restricted seeded JAX neutral Tackle turn

**Status:** Accepted and complete — 2026-11-03. Owner `/root/task034`; primary `/root`. Source base `19616d6c0a5f0401ceca28bf2928f0dcc06f22b2`.

**Scope:** Implement one integrated runnable simultaneous-choice turn for a named restricted Gen5 one-versus-one neutral Tackle development variant, with fixed typed JAX battle/request state, exact pinned Gen5 PRNG evolution, ordinary hit/critical/KO handling and differential state/request/event projections against an actual pinned Showdown turn. This advances simulator-first G3 work while G1 and all implementation milestones remain open.

**Exclusions:** full gen5randombattle support or G1/G3 acceptance; random teams, multiple roster members, switching/replacements, speed ties, non-Tackle moves or Struggle; items, weather, statuses, stat boosts, hazards, relevant abilities or other passive/secondary effects; full raw Showdown protocol byte equality, full observation feature encoder or existing runner/schema changes; learning/PPO, tournament/reporting expansion, throughput claims or GPU acceptance; weakening required parity evidence or replacing actual execution with a mocked damage calculator; owner commit/push.

**Acceptance criteria:**

- [x] Declare a concrete named restricted variant and validated boundary before implementation: one active Pokemon per side, Tackle only, neutral non-STAB physical damage, distinct effective speeds, positive HP/PP and ordinary bounded level/stats, no effects beyond supported Tackle accuracy/critical/damage/faint mechanics. Use pinned Showdown sets/data for oracle initialization; document any custom-format/rule overrides and initialization boundary. Reject unsupported inputs explicitly without state/RNG progression; do not claim exact full-format support. Stop/report concrete feasibility or oracle contradictions rather than silently narrow these criteria.
- [x] Expose a fixed-shape typed JAX PyTree state and pure compiled-capable simultaneous-choice transition; both sides commit from the same request boundary before execution. Include HP/max HP, PP, ordinary effective stats/speeds, turn/request phase, terminal winner/status and exact Gen5 four-limb RNG state. Account for move order, Tackle PP use, accuracy/critical/damage RNG consumption, HP clamp, KO and skipping a fainted actor; terminal inputs absorb without draws. Pending/invalid/unsupported choices produce explicit diagnostics and no concealed-choice disclosure or partial advance.
- [x] Implement exact pinned Gen5 LCG/upper32/bounded sampling using verified integer limb arithmetic compatible with default JAX32, not JAX-native randomness or prerecorded result replay. Validate known vectors and multiple high/low/carry seeds against live pinned PRNG; instrument actual turn random calls including bounds/context/state before/after so accidental extra/missing/reordered calls fail. Scope draw operations to concrete bounded calls needed by the variant and declare their supported range.
- [x] Build a guarded pinned reference oracle that executes actual submitted Tackle choices from a captured normal-request boundary (not isolated getDamage calls or replacement RNG values). Compare exact pre/post RNG limbs and draw consumption, internal HP/PP, action order, next turn, terminal outcome and declared minimal request/legal-action plus both-recipient visible event/history projections. Projections may normalize protocol into an explicit semantic schema but cannot erase relevant critical/damage/faint/turn or recipient HP visibility differences. Keep trusted internal state and unprojected packets out of actor-facing views.
- [x] Cover both speed orderings, multiple seeds including a genuine critical hit, normal surviving exchanges, first-action KO with skipped second move and terminal absorption, plus unsupported/invalid input no-advance and simultaneous-choice boundaries. Retain multiple-turn continuations through at least a short supported sequence to establish state/RNG handoff, without claiming full battle coverage. Use independently computed/handwritten invariants alongside actual oracle outputs; no self-generated expected-output identity tests.
- [x] Run CPU-first eager/JIT exact parity and JIT/vmap batch lane equivalence with heterogeneous supported states/seeds and invalid/terminal lanes; demonstrate no cross-lane state/RNG coupling. Record one synchronized representative compile/warmed execution probe (configuration and measured durations only, no speedup/strength claim). GPU is optional and no GPU criterion is imposed.
- [x] Document APIs, variant limitations, state/request/history projection, initialization/RNG boundaries, actual checks and measured CPU probe. Update relevant mechanics coverage/current handoff and state while preserving all historical task/review/GPU data and simulator-first decision. Run focused and justified affected tests, syntax/JSON/diff, Trail build/check and own diff inspection; freeze/report exact commands, results, changed files and concerns for primary acceptance. Count33/baseline30/checkpoint40 remain until primary accepts; no milestone/global review is due absent a concrete contradiction.

**Plan:**

- Read current simulator-first plan/learning and scoped Trail/source context; perform bounded oracle feasibility inspection before source implementation and report contradictions.
- Implement the explicitly restricted fixed-state JAX turn and exact PRNG alongside actual pinned turn oracle/fixtures and focused differential tests.
- Validate CPU eager/JIT/vmap and short continuations, record a synchronized probe, document strict exclusions and update task evidence; refresh/check graph and inspect diff.
- Freeze for primary review; primary accepts/counts/commits/pushes only this unit or records a concrete blocker and stops.

**Startup evidence:** Clean main19616d6 synchronized with default main; planning-only simulator-first decision is delivered and count33 reconciles with accepted history. No active correction/review/GPU block; baseline30,next40. Trail 0.28.0/git2.39.5/Node24.22.1/npm11.19.1/Python 3.13.18 and pinned oracle provenance functional. JAX initially absent: python -m pip install -r requirements-jax-cpu.txt succeeded, then JAX_PLATFORMS=cpu verified jax/jaxlib0.12.0.1, CpuDevice and JIT2+1=3. RTX3060Ti present but not required. Cloud status tool/policy unavailable; no runtime readiness claim. HTTPS Git and authorized GitHub connector delivery available; requested origin is mounted at /workspace.

**In-progress evidence:** The restricted boundary and implementation are complete, pending primary acceptance. The pinned unmodified `gen5randombattle` Battle receives two directly supplied one-Mew sets, then captures `battle.p1.activeRequest` and `battle.p2.activeRequest` plus each side's own `getRequestData()` projection before actual `Battle.makeChoices('move 1', 'move 1')`. The named variant is `gen5-neutral-tackle-mew-1v1-v1`; Mew has Synchronize, no item, Hardy nature, perfect IVs, and Tackle only, with only declared speed-EV or level differences. No format or move override is used. Runtime Tackle has default critical ratio1; each executed action consumes native accuracy `random(100)`, critical `random(16)`, and damage `random(16)`. The oracle instruments `PRNG.prototype.random` after the normal request boundary, records actual context and four-limb before/after state, and returns original values. Seed `[9,2,3,4]` yields a real critical event. In the L1-versus-L100 first-action KO, p2 acts first, p1 ends at0 HP, p2 remains at341 HP, p1 PP remains56, p2 PP becomes55, exactly three draws occur, winner is p2, and turn remains1. The JAX continuation carries its prior output into the next input and checks the next native before/request phase; strict dtype, audience, hidden-damage, terminal, last-PP/Struggle, and int32 turn-overflow boundaries are explicit. Exact theoretical damage stays trusted; opponent numeric damage is hidden, and owner-visible loss is clamped to HP removed.

**Final validation and probe:** `node --check reference/neutral-turn-oracle.cjs && node --check test/neutral-turn-oracle.test.cjs && node --test test/neutral-turn-oracle.test.cjs` passed 3/3. `JAX_PLATFORMS=cpu python -m py_compile simulator/__init__.py simulator/gen5_rng.py simulator/neutral_tackle.py test/test_neutral_tackle.py && JAX_PLATFORMS=cpu python -m unittest test.test_neutral_tackle -v` passed 6/6, including live PRNG vectors, actual turns/views, both speed orders, crit, KO skip, continuation, eager/JIT/vmap privacy and lane isolation, and invalid/pending/unsupported/terminal/turn-overflow no-advance. The last two Python modules had only trailing-blank-line cleanup; executable behavior was unchanged. A synchronized CPU `jax.jit(jax.vmap(transition))` probe used eight lanes and seeds `[1,2,3,4]`, `[9,2,3,4]`, `[31,41,59,26]`, `[65535,0,65535,0]`, `[0,65535,0,65535]`, `[65535,65535,65535,65535]`, `[4,3,2,1]`, `[101,202,303,404]`. Each lane had maxHP `(175,175)`, HP `(175-3i,175-2i)`, PP `(56-i,56-i)`, level `(50,50)`, Attack/Defense `(120,120)`, alternating Speed `(140,120)`/`(120,140)`, and choices `(1,1)`. Python 3.13.18, JAX/JAXLIB 0.12.0.1, CPU `cpu:0`; first synchronized call0.536254s, warmed 0.000157 s; all statuses0. The exact script is in `docs/neutral-tackle-variant.md`; these are configuration-specific timings, not a performance claim. Source base `19616d6c0a5f0401ceca28bf2928f0dcc06f22b2`. Frozen Git blobs: `simulator/__init__.py` `4e196b53297262368ce50e0c8e26b5449de2547f`; `simulator/gen5_rng.py` `ed8c8cddd8f001d2c7da63ff2bd2ef3c4cf060d7`; `simulator/neutral_tackle.py` `67b707d8b089fc7e2173d17497cbf1a62b10bf2f`; `reference/neutral-turn-oracle.cjs` `e87d8f433252380f46436746b40f146e4266d55a`; `test/neutral-turn-oracle.test.cjs` `6fd98adb28d0837f307bf43e2195849bd5b45273`; `test/test_neutral_tackle.py` `183f82977a3956b41cfe8d4e3ceaec97dee81c87`; `docs/neutral-tackle-variant.md` `78ec29e064432a2e7a90b0d20868246338461073`.

**Initial failures and fixes:** JAX was initially absent; installation from `requirements-jax-cpu.txt` succeeded and CPU/JIT0.12.0.1 was verified. The first import probe raised `ModuleNotFoundError: simulator.neutral_tackle` because package import preceded creation of that module; it was a construction-order probe and passed after the file existed. An initial RNG harness advanced one native PRNG across several bounds while each JAX comparison started from the same seed, causing a bound2 mismatch for all-zero state; the harness now resets native PRNG independently for each bound and vectors match. `_seed_bytes_le` initially called `.astype` on a Python tuple (`AttributeError: 'tuple' object has no attribute 'astype'`); conversion now follows `jnp.stack`. The transition initially passed a `uint32` draw into an existing `int32` arithmetic kernel; it now casts at the new simulator boundary, with existing arithmetic modules unchanged. One temporary test view parser attempted `int("0 fnt")`; the semantic condition parser was corrected. The first Node test run was2/3 because the expected regex disagreed with the oracle's correct “record and return” wording; the assertion was corrected and the rerun passed 3/3. Wrapping `Battle.random` alone missed `randomChance` accuracy/critical calls, so attribution moved to `PRNG.prototype.random`. An early inference from move metadata was corrected by pinned source/runtime evidence: the move constructor defaults critical ratio to1, so Tackle remains unmodified. The final `git diff --check` initially flagged extra blank lines at EOF in the two Python source files; they were removed, then focused Python checks and the documented probe passed on the final bytes.

**Trail and owner status:** Trail 0.28.0 is installed and functional. `trail build` refreshed the graph to 833 nodes, 1812 edges, 87 cards; `trail build --check` passes. After `git add -N` on the exact new implementation, test, variant, and learning paths, `trail check --base HEAD --json` reported9 lexical conflicts and11 leads over 12 changed files/19 notes, with no overlaps. Manual review found documentation/history-only matches: reused one-team vs prepared-team wording, unrelated prior runner/report and learning/tournament histories, inherited stat-format wording, and the new simulator candidate aligned with the current simulator-first decision; no implementation source conflict was found. `docs/ORCHESTRATION_STATE.json` parses, `git diff --check` is clean, and the final diff is reviewed. Task034 stays in progress/unaccepted; accepted count33, review baseline30, next checkpoint40, prior milestone and GPU history are unchanged. No GPU is required or claimed. No commit or push was made; primary acceptance is pending.

**Primary acceptance:** Primary /root independently reviewed the complete JAX PRNG/turn/view source, guarded native oracle, focused tests and variant contract; accepted Task034 after the owner stopped edits. The explicitly restricted one-Mew-per-side neutral Tackle variant uses initialized pinned values, unequal speeds and no relevant passive effects. Exact JAX32 limb RNG, simultaneous commitments, order/PP/HP/critical/KO/skip/terminal behavior match eight actual native turns, including JAX-output-to-JAX-input continuations and both recipient semantic request/history projections. Primary corrections prevent raw opponent/overkill damage disclosure, reject invalid/static audience and nonintegral input types, prove chained request/state handoff, and guard turn overflow. Existing arithmetic modules remain unchanged. Final Node oracle3/3 and CPU JAX6/6 pass, with live six-seed/five-bound RNG vectors, independent damage arithmetic, eager/JIT and heterogeneous vmap lane equivalence. A CPU eight-lane synchronized probe recorded first0.536254s/warm0.000157s with exact reproducer and source blobs; no throughput, GPU or full-format claim. Native Tackle critical ratio1 and PRNG-level instrumentation resolved an initial metadata-reading mistake without changing the move or format. Unrelated suites were not rerun; focused tests exercise all new integration paths and unchanged arithmetic dependencies against actual Showdown. All accepted histories, simulator-first decision and exact optional GPU evidence remain. Count33 becomes34; review baseline30/checkpoint40 and open G1/G3 remain. No Task035 starts.

**Final coordination:** Earlier in_progress/count33 owner evidence above is historical. Final focused evidence is Node3/3 and CPU JAX6/6; the synchronized eight-lane probe is0.536254s first-call and0.000157s warmed. Frozen source/test Git blobs are recorded in state. Final learning `l-a0f44bbc` replaces `l-3c4d8fd1` with explicit required code-reference section; all three same-unit predecessors are preserved and no earlier committed learning is edited. Primary checks exact staged scope, prior task/review/decision/GPU preservation, source identity, JSON/diff/graph and Trail semantics before delivery. Count34/baseline30/checkpoint40; no GPU, full-format or milestone acceptance; no Task035 starts.

**Final delivery validation:** Primary checks passed: JSON/history/count/review/decision/GPU preservation, exact Task001–033 handoff prefix, frozen source/test blob IDs, sixteen-file staged scope, learning replacement, diff and graph-current checks. Unauthorized mechanical spacing changes to older task sections were restored before delivery. Final primary staged Trail check has11 lexical documentation conflicts across12 checked files/19 active learnings, no overlaps or source conflict; exact findings and semantic rationale are retained in the commit body. All executable code is identical to the tested/probed final owner revision. Count34/baseline30/checkpoint40 and null active-task/GPU block are confirmed. No Task035 starts.

## Task 035 — bounded compiled restricted Tackle duels

**Status:** Accepted and complete — 2026-11-03. Owner `/root/task035`; primary `/root`. Source base `5d55d7266456c25c3fd2c88b88326f0b7655263b`.

**Scope:** Add a pure bounded JAX duel executor composing the accepted Task034 neutral Tackle transition entirely inside compiled control flow, and validate complete native restricted battles rather than short turn prefixes. Preserve the named gen5-neutral-tackle-mew-1v1-v1 mechanics and initial normal-request boundary. A finite execution budget yields a resumable state with honest diagnostics; no synthetic battle result.

**Exclusions:** new mechanics, speed ties, switching/rosters, items/status/weather/passives, other moves or Struggle; changing accepted Task034 turn/PRNG/player-view semantics or default oracle output; random teams/stat construction, full gen5randombattle or G1/G3 acceptance; policy training, tournaments/scoring/reporting infrastructure, CLI persistence/journals/resume storage; raw protocol byte equality, actor access to privileged full traces; GPU requirement, throughput/speedup/strength claims or owner commit/push.

**Acceptance criteria:**

- [x] Expose a documented pure fixed-PyTree run_duel-like API taking the existing typed BattleState and an explicit static positive bounded max_turns capacity. Use actual accepted transition with both sole legal Tackle choices committed each step, entirely within JAX compiled control flow with no host callbacks or Python battle loop. Keep existing transition/PRNG/view APIs and gen5-neutral-tackle-mew-1v1-v1 support boundary unchanged. Structural bad capacities/types reject before simulation; record the concrete capacity bound and trace shapes.
- [x] Return detached trusted final state, execution disposition, successful-turn count, final diagnostic/cause and a fixed-capacity trace with an explicit successful-prefix validity mask. Only successfully executed turns occupy valid rows. Padding after terminal/stop/budget never counts turns or emits fake draws, events, requests or outcomes. Initial terminal state completes with zero executed turns/draws; unsupported/pending initial state stops without advance and preserves its original diagnostic. Reaching capacity while nonterminal is budget exhaustion with a continuing state/winner, never a draw, loss or completed battle. Completion at the exact last slot takes precedence over budget exhaustion.
- [x] Support resuming the pure executor from a budget-stopped final state without resetting HP, PP, RNG, turn or request phase. Compare concatenated valid traces and final state/outcome with a single larger-capacity run and native full battle. If execution reaches the unchanged turn kernel unsupported boundary, stop honestly at that state; never add an unsupported fallback or claim completion.
- [x] Extend the guarded native oracle with an explicit separate full-duel mode/helper that starts at actual normal requests, repeatedly submits simultaneous native move1/move1 choices and records every turn until real termination. Reuse native PRNG/context/recipient projection capture with unchanged random returns; preserve the original default eight-turn fixture byte-for-byte semantically. Bound the oracle loop and fail if the selected native scenario fails to complete within it or leaves the supported variant; do not truncate and label complete. Capture initial boundary, all turn state/draw/request/views and actual terminal outcome.
- [x] Validate at least three actual complete native restricted duels with distinct seeds, both speed orderings, a genuine critical and terminal KO/skip; include a nontrivial surviving multi-turn duel. Feed each JAX output into the next transition/executor state. Compare every valid trace row and final native HP/PP/order/turn/phase/winner, all draw bounds/context/values/pre/post limbs, and both recipient semantic request/event histories using existing project_view. Retain independent known invariants, trace/count consistency and no hidden opponent/overkill data in actor projections.
- [x] Test empty/terminal, pending, unsupported, small-budget stop/resume, exact-capacity completion and invalid static-capacity boundaries, valid-prefix/padding behavior and no input mutation. Run CPU eager/JIT parity and JIT/vmap mixed lanes with different episode lengths, diagnostic states and budgets (common static capacity), including lane permutation/independence; inactive lanes preserve final state/RNG and padding is never executed experience. Record one small synchronized compile/warm probe with exact reproducible configuration and source identity, measurement only. No GPU acceptance is imposed.
- [x] Document API/control-flow, capacity/mask/stop/resume/outcome/visibility boundaries and complete restricted-native evidence in the existing variant doc/README and focused coverage/project-status sections. Preserve old Task001-034 history and all review/direction/GPU records; count34/baseline30/checkpoint40 until primary acceptance. Use Trail context/callers before affected existing symbols, focused new and justified affected Task034/PRNG/reference checks once per meaningful revision, syntax/JSON/diff and Trail refresh/check; inspect diff and freeze with exact commands/results/source blobs for primary review. No Astra gate is due absent a concrete contradiction.

**Plan:**

- Inspect accepted kernel dependencies and native capture boundaries via Trail; persist scope and feasible complete-duel scenarios before implementation.
- Implement a separate bounded compiled duel module and explicit native full-duel oracle mode while preserving current default fixtures/APIs.
- Add full-trajectory/state/RNG/view differential, budget/resumption and heterogeneous compiled-lane tests; record one reproducible synchronized CPU probe.
- Update only current relevant docs/task evidence, refresh/check Trail and inspect diff; freeze for primary criteria review and one-unit delivery or concrete blocker.

**Startup evidence:** Clean synchronized main5d55d72 with34 accepted tasks, no undelivered work, pending correction/review or GPU block. Regular baseline30/next40; simulator-first direction retained. Trail0.28.0, git2.39.5, Node24.22.1, npm11.19.1, Python3.13.18 and JAX/JAXLIB0.12.0.1 CPU JIT functional; guarded pinned provenance matches1242c1e2d9763ba5b85d33f13aecb89bae19ae3d. RTX3060Ti present but optional. Runtime status tool/policy snapshot unavailable; no readiness claim. HTTPS Git/authorized GitHub connector delivery works; origin is mounted at /workspace.

**Progress:** Task scope/exclusions/criteria reconciled with handoff; Trail transition, state, recipient projection and native capture boundaries inspected. Parent approved max capacity56, successful-prefix scan design, separate full-duel oracle mode, and PP=1 capacity/unsupported cases. Preserve the static `max_turns` trace capacity and use optional scalar int32 `turn_budget` (defaults to capacity) for heterogeneous vmapped lanes; valid zero budget returns unchanged state as budget exhaustion, and runtime values outside `[0, capacity]` return an explicit invalid-budget disposition with no advance and empty trace. Structural type/shape errors reject before simulation. Valid terminal states complete with zero turns even at zero budget. Default oracle output saved pre-edit at `/tmp/task035-neutral-turn-default-before.json` (124744 bytes, sha256 `d1a219a9483d8d1c0859338089a8d980372e79987c796f155e5c210395828157`). First native generation produced four complete duels (8/8/8/1 turns), both first-action orders, critical draws, and terminal winners; the full JAX trace matches them row-by-row.

**Owner evidence (unaccepted; primary review pending):** `node --test test/neutral-duel-oracle.test.cjs test/neutral-turn-oracle.test.cjs` passed 5/5; `JAX_PLATFORMS=cpu python -m unittest discover -s test -p test_neutral_duel.py` passed 4/4 before three additional mixed-lane diagnostic cases were added; the affected final test `PYTHONPATH=test JAX_PLATFORMS=cpu python -m unittest test_neutral_duel.NeutralDuelTests.test_jit_vmap_mixed_lengths_diagnostics_budgets_and_lane_order` passed 1/1 afterward. `JAX_PLATFORMS=cpu python -m unittest discover -s test -p test_neutral_tackle.py` passed 6/6. The no-argument oracle output compares byte-for-byte to its saved pre-edit output (124744 bytes, SHA-256 `d1a219a9483d8d1c0859338089a8d980372e79987c796f155e5c210395828157`); generated full-duel JSON SHA-256 is `baeb3b3bd3f1eccf35f59f2d4fa4e9f23f933ec8cefb4fd7f6f60b2ead2eaf51` (temporary `/tmp/task035-full-duels.json`). One synchronized CPU `jit(vmap(run_duel))` probe with four captured initial states, capacity8 and budgets `[8,8,8,8]` returned counts `[8,8,8,1]`; first synchronized call 1.121906s, warm 0.000532s. This is a configuration-specific measurement only. Probe source blobs: `neutral_duel.py` `81d290fbd60179a559667df9d2ad84463f90dbc8`, unchanged `neutral_tackle.py` `67b707d8b089fc7e2173d17497cbf1a62b10bf2f`, and `gen5_rng.py` `ed8c8cddd8f001d2c7da63ff2bd2ef3c4cf060d7`. Full implementation blob IDs: `reference/neutral-turn-oracle.cjs` `39494d2a9e667906fc3762cfcca5cc849a48b5c4`, `simulator/__init__.py` `01cbbd6ae7e712807674fb7d97e306eb34b55a25`, `test/test_neutral_duel.py` `935109104f6e382d19e210a862ef0d92738afd32`, and `test/neutral-duel-oracle.test.cjs` `ca16656de2aff403eb0886f91e09d878aa9bef04`. Accepted count34, review baseline30/checkpoint40, no GPU block/requirement, no milestone acceptance; parent owns final acceptance and delivery.

**Primary acceptance:** Primary /root independently reviewed the complete bounded executor, native full-duel oracle diff, new tests, API documentation and exact evidence, and accepted Task035 after implementation freeze. Static capacity1..56 and optional signed-int32 per-lane budgets drive pure scan/conditional execution of unchanged Task034 transitions. Only successful turns occupy the masked prefix; terminal, stopped, invalid-budget and budget-exhausted dispositions preserve honest causes and state. Four complete native battles across four seeds (8/8/8/1 turns,25 total) match every state, order/event, draw bound/context/value/pre/post limb, recipient request/history projection and actual outcome, with independent damage and privacy assertions. Budget-stop resumption matches uninterrupted traces; PP exhaustion is diagnosed only when a subsequent kernel step is attempted. Primary corrected eager/JIT invalid-budget consistency and expanded mixed-lane evidence to terminal/pending/invalid-budget cases. New duel suite4/4 passed before that test-only extension, then its affected eight-lane method1/1 passed; new/default Node tests5/5 and unchanged Task034 JAX6/6 passed. The original no-argument oracle remains byte-identical124744 bytes with SHA256 d1a219a9483d8d1c0859338089a8d980372e79987c796f155e5c210395828157. CPU four-lane capacity8 probe measured first1.121906s/warm0.000532s, counts8/8/8/1, with reproducer and measured blob recorded; one final EOF newline removal is verified as the sole difference from tested/probed executor bytes, so tests were not repeated for formatting. Existing turn/PRNG mechanics are unchanged. Earlier histories/reviews/direction/GPU evidence remain; count34 becomes35, baseline30/next40 unchanged. No GPU, throughput/strength, full-format or G1/G3 acceptance claim; no Task036 starts.

**Final coordination:** Owner-stage count34 and pending-review wording above is historical. Primary records count35/baseline30/checkpoint40 and stops at Task035. The measured executor blob81d290fbd60179a559667df9d2ad84463f90dbc8 differs from final38efee781f8dc711931669eb0e202c90fca4e870 solely by one removed EOF newline; adding it back reproduces the measured hash. Default oracle before/after SHA and all earlier history/direction/GPU evidence are preserved. New learning l-582fd216 is included. Final staged scope/source/JSON/diff/graph and Trail semantic checks precede delivery. No Task036 starts.

**Final delivery validation:** Primary JSON/history/count/review/direction/GPU-byte preservation, exact Task001–034 prefix, unchanged prior turn/PRNG mechanics, frozen code/test identities, twelve-file staged scope, EOF-only measured-blob proof, learning sections and diff checks passed. The EOF cleanup invalidated the cached file identity; deterministic Trail rebuild/check now passes at862 nodes/1897 edges/90 cards. Final staged Trail check has13 lexical conflicts/7 leads across11 checked files/20 active notes, including one critical-scenario source hit and no overlaps. All findings were reviewed against actual native PRNG observation, pure execution and visibility boundaries; exact rationale is in the commit body. Count35/baseline30/checkpoint40, null active task/GPU block, and open G1/G3 are confirmed. No Task036 starts.


## Task 036 — seeded speed ties in restricted JAX Tackle duels

**Status:** Accepted and complete — 2026-11-04. Owner `/root/task036`; primary `/root`. Source base `3fdf97eb491c45a4b7dc9737ae2b125e1ffcfa35`.

**Scope:** Add native seeded speed-tie resolution to a separately named restricted one-Mew-per-side neutral Tackle variant, including bounded compiled complete duels and exact state/RNG/recipient parity, while preserving accepted v1 public APIs and fixtures.

**Exclusions:** switching or rosters; other moves or Struggle; items, status, weather, boosts or passive effects; random teams or stat construction; changes to accepted v1 public API shapes or default oracle fixtures; training, tournament or reporting expansion; full-format, G1 or G3 acceptance; required GPU execution or performance claims; owner commit/push.

**Acceptance criteria:**

- [x] Establish the actual pinned native speed-tie draw sequence from normal request boundaries without replacing RNG values or overriding mechanics. Observe every relevant shuffle/phase draw, including draws with no visible event; record context, bounds and pre/post limbs. Propose and justify fixed trace capacity and ordering before implementation; stop if external evidence or an unresolved architectural contradiction is required.
- [x] Expose a documented separately named/id restricted tie-capable variant with fixed-shape pure JAX transition and bounded resumable duel API. Preserve accepted v1 public APIs, trace shapes and deterministic fixture bytes; any internal refactor must preserve their semantics. Reuse accepted RNG/arithmetic/view semantics rather than duplicating unsupported effects. Detect unsupported states and diagnostic/pending/terminal/budget boundaries without RNG advance or fabricated experience.
- [x] Compare at least three complete native tied-speed duels with distinct seeds and both realized initiative outcomes, a surviving multi-turn case and a terminal first-action KO/skip. Chain actual JAX outputs. Compare every state, event/order, draw context/bounds/value/pre/post limb, both recipient requests/history and actual terminal result; independently assert tie-to-order mapping and privacy/HP/PP invariants. Include an unequal-speed control in the new variant.
- [x] Validate CPU eager/JIT equivalence and heterogeneous JIT/vmap lanes for ties, unequal speeds, completion, stop/budget/resume and no-advance diagnostics, including lane permutation and valid-prefix/padding semantics. Preserve terminal precedence, PP unsupported boundary and no actor access to privileged RNG or combined views. Record a small reproducible synchronized compile/warm probe with exact configuration and source identities, measurement only.
- [x] Update focused variant/API/coverage/project status documentation and append Task036 handoff. Preserve Task001-035 history, prior review/direction/GPU records and count35/baseline30/checkpoint40 until primary acceptance. Use Trail context/callers, focused affected validation once per meaningful revision, diff/syntax/JSON checks and graph refresh/check. Save any non-obvious design learning using trail note. Inspect diff and freeze with exact changed files/commands/results/source blobs for independent primary acceptance.

**Native evidence and approved design:** Four actual pinned equal-Speed120/120 duels from normal activeRequest boundaries completed in seven turns. A surviving tied turn consumes16 random calls: one `BattleQueue.sort` speedSort shuffle; two BeforeTurn/Update sort draws; for each Tackle, `random(100)`, `random(16)` critical, `random(16)` damage; then two move-hit Update sort draws and one runAction Update sort draw; after the second move a residual Update sort draw. All scheduler draws are native `PRNG.random(0,2)` calls reached through `PRNG.shuffle`/`Battle.speedSort`; values0/1 retain/swap list order. If first attack ends battle, final turn stops after7 calls: three initial scheduler draws, one attack triplet and one post-hit Update; if second attack ends battle, it stops after13 calls. Seed `[1,2,3,4]` demonstrates a p2-first KO that skips p1; `[31,41,59,26]` demonstrates p2-first at first turn, while other turns/seeds yield both orders. Native observer leaves RNG returns untouched. Parent approved distinct variant ID2,16-slot context-tagged trace, narrow shared internal move/view/duel helpers, and unchanged v1 public signatures/shapes/default and `--duels` fixture bytes. Keep scheduler draws interleaved; do not preconsume a tape. Native evidence scripts are temporary `/tmp/task036-probe.js`, `/tmp/task036-fullprobe.js`; no implementation started before the design approval.

**Plan:**

- Capture pinned tied-speed native draw/phase evidence and propose fixed-shape integration while preserving v1.
- Implement exactly the approved tie-capable variant and bounded executor, plus native fixtures.
- Validate complete differential traces and compiled heterogeneous/resumable execution, record small CPU probe.
- Update focused docs/handoff/state, refresh/check Trail, inspect diff and freeze for primary review; stop at acceptance/delivery or concrete blocker.

**Startup evidence:** Startup verified clean synchronized default main, 35 accepted numbered tasks, no pending review/corrections/GPU block. Native speedSort uses PRNG.shuffle; owner must establish full per-turn draw sequence and capacity before implementation. Required tools and pinned provenance functional. Trail0.28.0, Git2.39.5, Node24.22.1, npm11.19.1, Python3.13.18, JAX/JAXLIB0.12.0.1 CPU/JIT work. RTX3060Ti present; no GPU requirement. Runtime status tool and network snapshot unavailable; no readiness claim. Verified origin checkout is /workspace; requested child path is absent. HTTPS fetch and authorized GitHub connector are functional.

**Owner implementation and validation (2026-11-04; primary acceptance pending):** Added the separate ID2 `gen5-neutral-tackle-speed-ties-mew-1v1-v2` with 16-slot context-tagged traces, tie-only native `random(0,2)` scheduler draws, unchanged v1 public API/shapes, shared action/view/duel internals, and bounded resumable duel support. Native `--ties` fixture covers three distinct seeds and both realized initial initiative orders. Surviving tied turns consume 16 calls; terminal first-action and second-action KO turns stop at 7 and 13. Oracle phase context uses observed native `runAction` choice and `eachEvent` event ID; wrappers delegate and restore original methods. Effective JAX bound 2 corresponds to raw native range `(0,2)`. Scheduler calls are interleaved, not preconsumed.

Focused validation: `node --test test/neutral-tie-duel-oracle.test.cjs test/neutral-duel-oracle.test.cjs test/neutral-turn-oracle.test.cjs` passed 6/6; `JAX_PLATFORMS=cpu python -m unittest discover -s test -p test_neutral_tackle_ties.py` passed 3/3; the existing neutral duel suite passed 4/4 and v1 neutral Tackle suite 6/6. `python -m py_compile simulator/neutral_tackle.py simulator/neutral_duel.py simulator/neutral_tackle_ties.py test/test_neutral_tackle_ties.py; node --check reference/neutral-turn-oracle.cjs; node --check test/neutral-tie-duel-oracle.test.cjs` passed. `git diff --check` passed. The no-argument output remains byte-identical to saved pre-Task036 SHA-256 `d1a219a9483d8d1c0859338089a8d980372e79987c796f155e5c210395828157`; `--duels` remains byte-identical to `baeb3b3bd3f1eccf35f59f2d4fa4e9f23f933ec8cefb4fd7f6f60b2ead2eaf51`.

One synchronized CPU probe used three native tied initial states and the existing p2-first unequal-speed full-duel control, `jax.jit(jax.vmap(run_tie_duel))`, four lanes, capacity 8, and per-lane budgets 8. Python 3.13.18, JAX/JAXLIB 0.12.0.1, backend CPU/device cpu:0; output counts `[7,7,7,8]`, compile plus synchronized execution 3.185057 s, warm 0.000741 s. A first probe harness assertion expected `[7,7,7,7]` and failed only because the unequal-speed native control takes 8 turns; the observed `[7,7,7,8]` led to correcting the harness, not simulator code. The corrected probe passed once. No source/test changed afterward. Reproducer and measurements are in `docs/neutral-tackle-variant.md`.

Frozen blobs: `reference/neutral-turn-oracle.cjs` `b9b9277a366db29d73024dafebc11d0dfd4c9eb4`; `simulator/__init__.py` `17153d391343768b0964147ef80e625f786aae85`; `simulator/neutral_duel.py` `a358dc7fa3901566c893254c9118931abb3e76f2`; `simulator/neutral_tackle.py` `71a7186b4f2572618a8f08c91fd8fa8aa32d7704`; `simulator/neutral_tackle_ties.py` `a7c1aae754e1b45ea92078421753d623342b1e7e`; `test/neutral-tie-duel-oracle.test.cjs` `7781f5dc598d3f077d0b7f14b69a88b9e1ebbdaf`; `test/test_neutral_tackle_ties.py` `d6c5847c0f9efa0c6fa5e88625736bed00d162f3`. Documentation, state, learning, and Trail files changed after the source freeze. Trail note `l-0172a92e` records the native phase/interleaving and carry-type learning. `trail build` refreshed the 915-node/2035-edge graph; `trail build --check` passed. `trail check --base HEAD --json` saw all 13 changed/new files and 21 notes, with 12 lexical conflicts, 8 leads, and zero overlaps; findings were reviewed as unrelated historical conflicts or lexical matches to existing exclusions/wording, with no new-learning contradiction. No GPU required. Accepted count 35, review baseline 30, next checkpoint 40; owner did not commit or push.


**Primary acceptance:** Primary /root independently reviewed the native phase evidence, all new simulator/oracle/test code, shared v1 refactors, focused documentation and validation. Accepted separate variantID2 with sixteen chronological draw slots and exact native queue/BeforeTurn/Update/residual phase sampling. Three distinct-seed complete tied duels (21 turns) and one complete native unequal-speed control (8 turns) match every state, event/order, draw bound/context/value/pre/post limb, both recipient views and terminal result. Shared v1 action/view/choice and bounded-executor refactors preserve public APIs, six-slot traces and original oracle bytes. The owner corrected a typed JAX cond carry and test-import/probe-expectation harness mistakes. Primary initial test review required returned diagnostic-state freeze, independent tie initiative mapping, native unequal control, individual-vmap equivalence, inactive padding, exact-capacity and PP1 follow-up checks; all are now covered. Final tie suite3/3, affected v1 duel4/4, v1 Tackle6/6 and Node6/6 pass; source bytes are frozen and identified. The four-lane CPU probe is measurement only (3.185057s first synchronized/warm0.000741s, counts7/7/7/8). No GPU, throughput, full-format, G1/G3 or implementation-milestone acceptance. No pending findings or external-evidence block; owner stopped. Count35 becomes36; review baseline30/next40 remain. No Task037 starts.

**Final coordination:** Owner pending-acceptance/count35 statements above are historical. Count36/baseline30/checkpoint40, null active_task/GPU block, and open G1/G3 are current. Prior Task001–035 prefix and all old review/direction/GPU evidence are preserved. Parent reviewed final learning findings, including oracle scenario labels and audience0 projection arguments; they do not alter native RNG observation, actor privacy or inactive-lane state. Exact final findings and rationale accompany delivery. Stop at this accepted unit; no Task037 starts.
