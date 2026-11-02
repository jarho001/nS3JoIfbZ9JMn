# Gen 5 Random Battles benchmark charter

**Charter version:** 0.1-draft

**Status:** Draft protocol for future implementation. This document freezes the reference identity and current harness contract supported by repository evidence. Workload details marked **Proposed** are measurement rules, not implemented benchmark features or results. Decisions marked **Unresolved** need evidence, hardware access, or an explicit resource budget.

## Scored reference and source manifest

The scored reference is the pinned Pokémon Showdown `gen5randombattle` format at commit [`1242c1e2d9763ba5b85d33f13aecb89bae19ae3d`](../reference-lock.json). The machine-readable resolved-format snapshot is [`format-v1.json`](../reference/fixtures/format-v1.json); the fixture is metadata from that exact revision, not an executable rules export. It identifies Gen 5 singles, the random team generator, inherited mod chain `gen5 → gen6 → gen7 → gen8 → base`, effective rule entries, and no Team Preview. The declared format rules include Obtainable, Sleep Clause Mod, HP Percentage Mod, Cancel Mod, and Illusion Level Mod. The scored reference must retain the locked revision until a separately reviewed version change regenerates and audits this manifest and affected evidence.

| Manifest field | Frozen value or evidence | Status |
|---|---|---|
| Upstream repository | [`smogon/pokemon-showdown`](https://github.com/smogon/pokemon-showdown/tree/1242c1e2d9763ba5b85d33f13aecb89bae19ae3d) | Implemented lock |
| Upstream commit | `1242c1e2d9763ba5b85d33f13aecb89bae19ae3d`, [`reference-lock.json`](../reference-lock.json) | Implemented lock |
| Format and resolved rules | [`gen5randombattle`, format-v1.json](../reference/fixtures/format-v1.json); generated from `Dex.formats.get` and its effective rule table by [`reference/oracle.cjs`](../reference/oracle.cjs) | Implemented metadata fixture |
| Format source | [`config/formats.ts` at the pin](https://github.com/smogon/pokemon-showdown/blob/1242c1e2d9763ba5b85d33f13aecb89bae19ae3d/config/formats.ts) | Immutable upstream source |
| Team-generator dispatch | [`sim/teams.ts` at the pin](https://github.com/smogon/pokemon-showdown/blob/1242c1e2d9763ba5b85d33f13aecb89bae19ae3d/sim/teams.ts) resolves the format's `gen5` mod to `data/random-battles/gen5/teams`. | Pinned source; verified against local checkout |
| Gen 5 team generator | [`data/random-battles/gen5/teams.ts` at the pin](https://github.com/smogon/pokemon-showdown/blob/1242c1e2d9763ba5b85d33f13aecb89bae19ae3d/data/random-battles/gen5/teams.ts) defines `RandomGen5Teams`, extending `RandomGen6Teams`, and loads this mod's `sets.json`. | Pinned source; verified against local checkout |
| Workspace source snapshot | Base code snapshot `6f26c5b58e859609cd413bbf8fca4a4d49b88ad7`; current documentation audit starts from this tree. Source file links in this charter point to the audited workspace. | Snapshot identity |
| Fixture identity | `schemaVersion: 1`, `formatId: gen5randombattle`, `upstreamCommit` as above; Task020's local consistency check confirmed the fixture commit matches the lock. | Implemented metadata |
| Runtime/dependency provenance | [`reference/oracle.cjs` `provenance()`](../reference/oracle.cjs) records Node, npm, the pinned checkout's `package-lock.json` SHA-256, and semantic fixture hashes. `reference/setup.cjs` runs `npm ci` and builds that checkout. | Implemented reference-harness provenance |
| JAX dependency identity | [`requirements-jax-cpu.txt`](../requirements-jax-cpu.txt) pins JAX and JAXLIB to `0.12.0.1`; this identifies the present CPU dependency file, not a full benchmark environment lock. | Existing pin; benchmark environment incomplete |

These pins identify the game and oracle. A future benchmark result must additionally record the workspace revision, dirty-tree status, runtime and dependency lock hashes, command/configuration, and device/driver/runtime details. It must not silently float Showdown to upstream `master`.

## Current player and action contract

The current reference harness runs a complete Showdown `BattleStream` battle for the format above. It generates two six-member teams with separately seeded calls to `Teams.generate`, initializes the battle with its own seed, and gives each player a policy RNG seed. The runner's current default seed configuration has five independent four-limb seeds (`teamP1`, `teamP2`, `battle`, `policyP1`, `policyP2`); each limb is an integer from 0 through 65,535. The default is a deterministic smoke sample, not a statistical benchmark suite. Current limits are 60,000 ms, 20,000 object packets, 4,000 decision attempts, and five unavailable-choice retries per request.

The policy receives its own player-scoped history and its current private request. [`environment/protocol.cjs`](../environment/protocol.cjs) projects split `update` packets by audience: each player sees its own secret branch and the public branch for the opponent. A `sideupdate` request is delivered only to its named player; spectator projection omits private side updates and uses public split branches. The projection drops `end` metadata from player views and rejects an unsupported top-level stream `error` channel. An addressed private `sideupdate` error is retained as that player's `privateEvent`; the runner classifies unavailable-choice and invalid-choice feedback and handles it as retry or failure. The policy does not receive the opponent's request, the complete packed teams, combined omniscient records, or the raw trace packet stream. Those remain trusted oracle evidence. [`reference/trace.cjs`](../reference/trace.cjs) stores trusted inputs and outputs for deterministic reference replay.

The currently supported request parsing is Gen 5 singles: a move request with one active slot and legal move slots, including the locked/recharging one-move form; a forced-switch request with one active slot and available replacements; and a non-actionable wait request. Legal choices are represented as `move N` or `switch N`. A Showdown unavailable-choice rejection invalidates that command for the same request episode, produces a retry against the updated request, and is counted separately from a newly attempted policy decision. See [`candidateActions`](../environment/random-policy.cjs) and [`runBattle`](../environment/battle-runner.cjs). This is the implemented harness contract; it does not claim a JAX environment or complete future battle API.

## Reference workload and seed procedure

**Proposed reference workload:** run complete battles through the pinned Showdown runner with the deterministic legal-random policy in both seats, preserving team generation, request projection, choice submission, retries, and terminal validation. For every base seed case, generate team A and team B once; run the base case with A/P1 and B/P2, then the paired case with the same generated teams assigned B/P1 and A/P2, using the same battle seed. Keep each policy identity's policy RNG seed with that policy when swapping seats. Record both expanded five-stream tuples and the two team identities; verify the reused teams match exactly. Report the pair and each battle. Keep fixture-generation/mechanics-oracle jobs separate from full-battle throughput: they answer different questions. Never substitute one seed for another stream or reuse policy RNG state across battles.

The current default five-stream seed tuple may be used as a single smoke case and for trace reproduction. A future scored throughput suite should enumerate and publish a dedicated seed manifest. Seed tuples should be reproducibly derived from a recorded root seed with documented, domain-separated streams for team A, team B, battle RNG, policy A, and policy B; record the derivation algorithm/version and the expanded tuples. A base tuple `[teamA, teamB, battle, policyA, policyB]` pairs with `[teamB, teamA, battle, policyB, policyA]`. Paired runs hold the generated teams and battle seed fixed, as defined above.

**Unresolved:** suite size, seed roots, and statistical power/resource budgets are not set. Choose them before a scored measurement, publish the manifest, and justify the number of independent battles from the desired interval precision and available compute. Do not choose or report a target speed from the PokeJAX or Jaxcalibur figures in [`REUSE_AUDIT.md`](REUSE_AUDIT.md); their workloads and hardware are not comparable.

## Counters and result accounting

Report counters separately and retain both completed-battle and attempted-work denominators:

| Counter | Definition for the future report | Current implementation evidence |
|---|---|---|
| Decision attempts | One invocation of the policy for one actionable player request. Count each player separately and in total. | Runner `decisionCount` increments once before `choose`; wait requests are not decisions. It is an attempted decision count, not accepted choices. |
| Choice submissions | Each command sent to Showdown for a request, including a later retry. | Runner keeps individual `attempts` and request resolution state. |
| Accepted choices | Commands resolved as accepted by the harness after subsequent battle progress. Report accepted and rejected submissions separately. | Runner infers acceptance from progress evidence and records it in trusted records; there is no explicit Showdown acceptance counter. Its `actions.move/switch` summary increments at policy selection before acceptance, so these action counts are policy-selected submissions, not accepted choices. |
| Turn index | Terminal `turns` value and latest numbered public `|turn|` marker. Treat this as the simulator's reported turn index; do not relabel it as completed policy decisions or infer decisions from it. | The terminal value is present at `summary.outcome.turns`. The runner also tracks the latest public `|turn|` marker internally as `turnCount`, but does not currently export that marker as a summary field. |
| Events | Count each canonical public `event` record once from the spectator projection. Exclude packet envelopes, `|split|` markers, timestamps/heartbeats, requests, and private side updates. | Runner retains trusted records and packets; an event counter is not currently in the summary. |
| Completed battles | Battles that reach validated public terminal output and agree with trusted end metadata. Report wins, losses, draws, and seat separately. | Runner returns outcome only after terminal checks. |
| Rejections and retries | Report rejected submissions, corrected submissions, and exhausted retry limits independently. | Per-request retry bounds are implemented; a retry does not itself count as a new policy decision unless the policy is invoked again. |
| Failures/timeouts | Report separately from wins/losses/draws: initialization/protocol errors, invalid policy choices, infrastructure/runtime errors, configured limit exhaustion, and wall-clock timeout. Include partial counters and elapsed time. | Packet, decision, retry, and timeout limits exist, but failures throw and are not written as aggregate result rows. The runner summary has no elapsed-time field. Infrastructure failure must never be scored as a battle loss. |

Use completed validated battles per wall-clock second as an end-to-end reference rate, with decisions/s and events/s as supplementary rates. Also report raw elapsed time, completed/attempted counts, retries, and failures. Do not divide by accepted choices alone or hide timeouts/retries. These are proposed reporting rules; the current harness does not yet emit every counter in a stable benchmark record.

## Timing protocol

**Proposed:** define and report two timing boundaries. End-to-end wall time starts before per-battle team generation and includes initialization, policy calls, packet handling, retries, and terminal validation. A separately labeled steady-state step measurement may exclude one-time process/setup and compilation only after an explicit warm-up phase; it must still include all work represented by the named unit. Report team-generation and startup time separately when excluded from a secondary rate.

For JAX, report compilation/first-call time separately from warmed execution. Synchronize asynchronous work before starting and after ending each timed interval (for example, block on returned arrays); say which buffers/results are synchronized. For CPU reference runs, state process reuse and concurrency. Record monotonic wall clock, device utilization where available, and peak host/device memory where available. Do not compare a JAX device rate with Showdown unless batch size, policy, transition boundary, concurrency, initialization, warm-up, and synchronization rules are all stated.

**Unresolved:** benchmark implementation, batch size, worker/process count, concurrency, warm-up count, repetitions, confidence interval method, hardware allocation, memory cap, and maximum elapsed-time budget. No timing run or measured performance claim is part of Task020.

## Reproducibility record and version changes

A future result record should include:

- charter version, date, command, source workspace revision, clean/dirty status, configuration, and benchmark harness revision;
- Showdown repository and commit, resolved-format fixture hash, generator/source identity, and five expanded seed streams per battle;
- Node/JAX/Python and package versions or lockfile hashes, compiler/runtime versions, OS, CPU model, accelerator model, driver/runtime versions, and device count;
- workload definition, batch/concurrency settings, warm-up and compilation treatment, timing boundaries, synchronization rule, and resource limits;
- decision/submission/accepted-choice, turn, event, completed-battle, retry, rejection, timeout, failure, elapsed-time, and memory counters with their definitions;
- result aggregation, uncertainty interval, excluded runs and reasons, and the complete per-run data needed to recompute summaries.

Changing the Showdown revision, format rules, team generator, player projection, policy, or measurement boundary creates a new benchmark identity. Regenerate and review the format manifest and relevant oracle/trace fixtures; do not splice results across identities. The current repository has reference fixtures and replayable seeded traces, but it has no scored benchmark suite, no statistical seed manifest, and no stable aggregate timing record.

## Workload tiers and remaining gaps

Report three workloads separately; they answer different efficiency questions and their rates must not be combined:

- **Simulator-only:** batched reset/step throughput with scripted or preselected actions and no learned policy. A JAX batched full-battle simulator is not implemented. The current Showdown runner is an orchestration reference, not an accelerator simulator-only result.
- **Rollout plus policy:** complete battle collection including player projection, observation encoding, policy inference/action selection, legal-action handling, and retries. The current Showdown runner with legal-random policies is a reference workload for this tier, not a JAX implementation or measured throughput result.
- **Full training:** rollout plus trajectory storage and learner update work, including recurrent state, optimization, and evaluation hooks. This workload is not implemented for full battles.

## Current scope and remaining gaps

Implemented today: pinned Showdown reference, format metadata export, seeded team/battle/policy streams, a deterministic legal-random reference runner with limits/retry accounting, player-view projection, and trusted trace replay. Narrow JAX mechanics kernels and their fixtures are listed in [`MECHANICS_COVERAGE.md`](MECHANICS_COVERAGE.md).

Proposed here: workload pairing, published seed suites, benchmark counters/report schema, timing boundaries, device synchronization, warm-up treatment, resource/result reporting, and version-change procedure. These protocol choices require a later implementation and review.

Unresolved for future measurements: scored sample size and statistical budget; hardware, concurrency, memory and timeout budgets; exact warm-up/repetition settings; package/runtime/device lock completeness; and long-term seed-manifest policy. These operational choices must be settled before a scored measurement where they apply, but do not automatically gate G0.

G0 specification closure remains pending review. The necessary G0 review is to confirm that the pinned `gen5randombattle` target and rules/data source basis, reuse decision, source/version manifest, and known mechanics/coverage gaps are explicit and consistent, and to resolve or explicitly bound any open decision that would change the target or its rules/data scope. The draft charter, format fixture, reuse audit, and mechanics ledger are evidence for that review; this document does not record G0 acceptance.

Later implementation gaps remain at their project gates: G1 reference-environment integration and full request/visibility/action-path differential coverage; G3 battle/request state and event/turn scheduling, coupled PRNG consumption, and representative mechanics; G4 complete reachable rules/data and full-format differential transfer. The project also lacks a full JAX battle state/transition simulator, batched complete-battle workload, trajectory collection, recurrent policy learner, and training/evaluation benchmark. The arithmetic composition is not a full battle engine. G0 does not require those later implementations, and this charter does not claim G0 or G1 completion.

**Next bounded deliverable proposal:** Implement a versioned per-battle result record in the Showdown harness for elapsed time, distinct decision/submission/resolution/event/turn counters, terminal outcomes, retries, and failures. Preserve the existing smoke command; do not select a statistical suite size, allocate benchmark hardware, or claim throughput until those budgets are reviewed.
