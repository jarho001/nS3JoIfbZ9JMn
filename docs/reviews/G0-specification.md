# G0 specification milestone review

**Disposition: ACCEPTED — 2026-11-03.** Reviewer: `/root/g0_reviewer`, requested independent Astra global-review route. This accepts the specification boundary in `PROJECT_PLAN.md:267–283`, whose G0 criterion is “Exact target and known coverage gaps are explicit.” It does not accept G1 or later gates.

Reviewed source base: `b4fed94964ff8185c0fc8d9a67c4d77c2f5f4cf2`. At review start, HEAD matched this identity and the only working changes were the G0 review plan in `docs/TASKS.md` and `docs/ORCHESTRATION_STATE.json`. The review is unnumbered: accepted task count remains 27, regular review baseline 20, and next regular checkpoint 30. No source or test changes, narrowed tests, or waived requirements were part of this unit.

## Findings and decision

The frozen target is the complete pinned Showdown `gen5randombattle` game, not a selected arithmetic subset. `reference-lock.json` identifies `smogon/pokemon-showdown` at `1242c1e2d9763ba5b85d33f13aecb89bae19ae3d`. The live resolved manifest exactly matches `reference/fixtures/format-v1.json`: generation 5, singles, random teams, mod chain `gen5 → gen6 → gen7 → gen8 → base`, no Team Preview, and the declared Obtainable, Sleep Clause Mod, HP Percentage Mod, Cancel Mod, and Illusion Level Mod rules. Equality covers the complete effective rule/value/ban lists, including `evlimit: Auto`, rather than just these selected assertions.

Rules and data authority are the entire pinned Showdown source and its inherited behavior. The fixture is descriptive metadata, not an executable rules export. Direct source inspection confirms format dispatch to the Gen 5 generator, its Gen 6 superclass and local `sets.json`, and Dex parent-data inheritance/overrides. The charter's source/version manifest identifies this authority; no alternative data source or unresolved choice changes the scored game. Future JAX exports must preserve the reference behavior, including inherited rules and reachable species/move/item/ability/team-generation data.

The accepted Task019 reuse decision is explicit and bounded: retain Showdown as oracle; provisionally continue custom incremental JAX; do not adopt an inspected external engine. Candidate availability, generation, license-identification and unexecuted-test limits are recorded in the audit. This review accepts that bounded decision artifact, without re-auditing current external candidates or independently reproducing their published claims.

The mechanics ledger accurately distinguishes pure arithmetic and compositions from executable battle support. Sampling the largest neutral composition confirms explicit effective inputs, supplied roll/flags/modifier, and no battle state, scheduler, lookup or RNG implementation. Its oracle and tests distinguish ordinary cases from synthetic boundary controls. Neither composition counts nor fixtures establish a percentage of game coverage. State transitions, event/turn scheduling, coupled PRNG consumption, complete reachable data/mechanics, full request/visibility/action-path parity, learned policies and training/evaluation remain later work.

**No remaining target, rules or data-scope decision blocks G0.** Suite size, seed publication/root selection, statistical precision, hardware, concurrency, warm-up, repetitions, resource limits and complete measurement-environment locks must be chosen before applicable scored measurements. They do not change the frozen game. The draft status of the measurement protocol therefore does not prevent this specification acceptance.

Direction remains consistent with the durable goal: build a faithful research platform toward complete JAX training while retaining independent Showdown transfer evidence. Existing reference orchestration and offline accounting are useful infrastructure, not evidence of a full JAX battle engine, measured speedup or learning strength. G1 trajectory/baseline integration and broader differential coverage remain unaccepted here; G3/G4 simulator and G5+ training/evaluation requirements remain intact.

## Evidence sampled

- Governance and history: `AGENTS.md`, `docs/ORCHESTRATION.md`, current orchestration state, the unnumbered G0 handoff at the end of `docs/TASKS.md`, and `docs/reviews/T020-bootstrap.md`. The earlier bootstrap findings distinguish specification review from downstream implementation and operational budgets; this review now supplies that missing specification disposition.
- Specification: `PROJECT_PLAN.md:267–283`; `docs/MECHANICS_COVERAGE.md:1–56`; `docs/REUSE_AUDIT.md:1–49`; `docs/BENCHMARK_CHARTER.md:7–24,34–44,66–111`. Existing pending-G0 wording was accurate before this review and is to be updated by the sole documentation owner after acceptance.
- Reference implementation: `reference/oracle.cjs:28–81,234–251`; `test/reference.test.cjs:15–105`; complete resolved format fixture and lock.
- Pinned upstream source, under `reference/vendor/pokemon-showdown/`: `config/formats.ts:4608–4615`; `sim/teams.ts:628–652`; `data/random-battles/gen5/teams.ts:1–4,58–62`; `sim/dex.ts:683–750`. Vendor code is outside the Trail index; exact searches followed the Trail lookup.
- Arithmetic evidence: `mechanics/neutral_damage_composition.py:7–47`; `reference/neutral-full-damage-oracle.cjs:28–94`; `test/neutral-full-damage-oracle.test.cjs:16–70`; `test/test_neutral_damage_composition.py:40–93`. Python test source includes eager, JIT and vmapped comparisons; those Python tests were inspected, not rerun in this review.
- Trail orientation, symbol/skeleton results and returned learnings were consulted before source inspection. Relevant boundaries included neutral-event wrapping, request-episode choices, prepared-team seed provenance, and the inability of offline records to attest execution. Visible savings counters total at least approximately 162,635 tokens; one additional counter was lost to output truncation, so this is a lower bound, not an exact total.

## Checks run by this reviewer

- `git rev-parse HEAD` matched the reviewed source base; `git status --short` showed only the two plan edits before this report.
- `node --test test/reference.test.cjs` — **7/7 passed**. Includes clean/origin/commit guards with real Git mutation cases, exact live format-fixture equality, seeded initial teams/requests, seed validation and stream cleanup/error handling.
- `node --test test/neutral-full-damage-oracle.test.cjs` — **2/2 passed**. Regenerated the pinned five ordinary/six synthetic case fixture and checked core/weather/single-roll/final-stage evidence plus instrumentation restoration.
- `node reference/oracle.cjs provenance` — passed the reference readiness guard and reported Node `v24.22.1`, npm `11.19.1`, upstream package-lock SHA-256 `10960416dfb6af611bcc7f6bd503e3b69d3ef7df0647fa9ce9e9ecf9ac857f82`, format-fixture SHA-256 `87b303476a12f0a14559c659c986fd13eb5112d26b2cac8371f393f84b498b71`, and smoke-fixture SHA-256 `aeb4b897cb55136da088e487d6484c8b7e8d842828888bfabb5ca04a1ffa9b09`.
- Python standard-library JSON assertions verified count 27, baseline 20, next checkpoint 30, and null GPU block. `git diff --check` passed.

## Corrections, blockers and limits

Required substantive corrections: **none**. Blockers: **none**. Sole owner `/root/task027` may perform the already planned acceptance-status synchronization and focused documentation/state checks; the parent remains final coordinator and commit owner. No implementation handover or correction round is required.

GPU execution is not a G0 acceptance requirement. This reviewer ran no GPU workload and makes no new device, benchmark or accelerator-performance claim. Historical CPU/GPU evidence remains attributed to its original tasks. Scheduler caps/history are unchanged. Requested reviewer routing and repository configuration do not independently establish backend identity or automatic scheduler-role loading.

Optional observation only: the opening of `PROJECT_PLAN.md` retains its original pre-implementation workspace description. The current ledger/charter and explicit dated audit context establish present scope; this stale opening is not a target-definition blocker and no unrelated edit is requested.

Disposition history: initial independent G0 specification review **ACCEPTED**, with no preceding G0 rejection or implied downstream approval.

## Final documentation and state review

**ACCEPTED — 2026-11-03.** After `/root/task027` completed and stopped the status synchronization, the reviewer inspected this revision's focused diff once. README, mechanics ledger, reuse audit and charter consistently record the dated G0 specification approval and link this report. They preserve the full pinned target, partial arithmetic boundary, provisional reuse direction, open G1-and-later implementation gates, and draft measurement protocol/budgets. No new coverage percentage, performance or GPU result is claimed.

Reviewer checks: `git diff --stat`, focused `git diff --` for the six owner-edited documents/state files, `git status --short`, and `git diff --check`; all were consistent with the bounded update. A Python standard-library check against `git show b4fed94964ff8185c0fc8d9a67c4d77c2f5f4cf2:<path>` verified that numbered task history is byte-preserved before the appended unnumbered handoff, completed structured task history and non-milestone state are unchanged, and `PROJECT_PLAN.md` is unchanged. It asserted count 27, baseline 20, checkpoint 30, null active task/GPU block/active review/delivery commit, no pending corrections, exactly one G0 approval, and retained plan/criteria in milestone review history. It also verified the reviewed source identity and lock/fixture pin, resolved **119 local Markdown links**, and restricted changed paths to the seven expected documentation/state/report files. Source, tests and configuration are unchanged; no source tests were repeated. Owner-reported Trail lexical findings do not warrant changing the accepted semantics; final staging checks remain the parent's delivery responsibility.

No required corrections or blockers remain. This closes the planned status synchronization within the existing G0 approval, without a new numbered task, correction round, later milestone approval, or commit claim. The parent owns final delivery.
