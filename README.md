# Gen 5 Random Battle research foundation

This repository is beginning a JAX self-play project against Pokémon Showdown's `[Gen 5] Random Battle` (`gen5randombattle`). The current foundation pins an upstream Showdown revision and provides a deterministic oracle fixture for its format rules, team generation, and first player requests.

## Development workflow

The single-deliverable ownership, review checkpoints, and GPU evidence rules
are recorded in [`docs/ORCHESTRATION.md`](docs/ORCHESTRATION.md) and
[`docs/ORCHESTRATION_STATE.json`](docs/ORCHESTRATION_STATE.json). The reusable
manual or scheduled task entry point is [`docs/SCHEDULED_PROMPT.md`](docs/SCHEDULED_PROMPT.md).
It authorizes direct-default-branch pushes for its task only; unrelated
requests remain local-only unless separately authorized.

## Set up the pinned reference

Use Node.js 22.18 or newer and npm. From the repository root, run:

```sh
node reference/setup.cjs
```

The command clones the revision recorded in [`reference-lock.json`](reference-lock.json) only when the ignored reference checkout is absent, verifies its origin, exact commit, and clean tracked/untracked status, then runs upstream `npm ci` and `npm run build`. It never resets, cleans, or updates an existing checkout. If the checkout has changed, inspect it and restore or remove it yourself before retrying.

## Inspect the reference outputs

```sh
node reference/oracle.cjs format
node reference/oracle.cjs smoke
node reference/oracle.cjs smoke --team-seed-p1=1,2,3,4 --team-seed-p2=5,6,7,8 --battle-seed=9,10,11,12
node reference/oracle.cjs provenance
node --test test/reference.test.cjs
node --test test/protocol.test.cjs
node --test test/*.test.cjs
```

The committed JSON files under `reference/fixtures/` are canonical semantic fixtures. The format export comes from Showdown's resolved Dex/rule table. The smoke artifact contains both complete generated teams, private initial requests, and raw omniscient update messages. Those updates can include `|split|` markers and private `SECRET` lines; this trusted oracle artifact is not a public player view or a policy training record. The harness validates Gen 5, singles, no Team Preview, team sizes, and that each player's request names match its submitted team. Wall-clock protocol lines are excluded. Provenance reports the upstream package lock hash and Node/npm versions separately from semantic fixture hashes.

## Protocol views

[`environment/protocol.cjs`](environment/protocol.cjs) exports `projectPackets(packets, audience)`. `packets` is an ordered array of complete Showdown object messages, each shaped `channel\nbody`; `audience` is exactly `p1`, `p2`, or `spectator`. It returns one object `{ schemaVersion, audience, records }` for that audience only. Records are ordered and typed: `{ type: "event", message }` for visible events, `{ type: "privateEvent", message }` for that player's private events, and `{ type: "request", request }` for that player's parsed request JSON. A hidden-only packet adds no placeholder record. For example:

```js
const { projectPackets } = require('./environment/protocol.cjs');
const p1View = projectPackets(completePackets, 'p1');
```

The projector routes `SECRET`/`PUBLIC` branches before any event becomes actor-visible; raw message bodies never enter a player view. It filters clock-only events, routes private requests/errors only to their owner, and drops trusted `end` metadata. Malformed, unsupported, and battle-error messages fail closed for the harness; their raw text is not an actor observation. It does not rebuild transport chunks or define model features. The hand-authored routing fixture is [`test/fixtures/protocol-transcript-v1.json`](test/fixtures/protocol-transcript-v1.json); tests also compare its split routing and live seeded battles against pinned Showdown.

## Random-policy baseline runner

[`environment/run-random.cjs`](environment/run-random.cjs) runs a complete seeded Gen 5 Random Battle against the pinned reference:

```sh
node environment/run-random.cjs --help
node environment/run-random.cjs \
  --team-seed-p1=1,2,3,4 --team-seed-p2=5,6,7,8 \
  --battle-seed=9,10,11,12 \
  --policy-seed-p1=13,14,15,16 --policy-seed-p2=17,18,19,20 \
  --output trusted-battle.json --result battle-result.json
node --test test/battle-runner.test.cjs
node --test test/*.test.cjs
```

All five seeds are separate four-limb Gen 5 seeds. The summary reports the pinned commit, seeds, result, turn count, packet/decision counts, and move/switch counts. `--output` writes a versioned trusted transcript containing complete private requests, split update branches, choice attempts, terminal outcome, and end metadata. Treat it as omniscient oracle data; the runner never passes that transcript to a policy.

The two random policies receive a frozen `{ history, request, random }` callback input. `history` contains only that side's `projectPackets` output and its own submitted-choice resolutions; `request` is the current request addressed to that side; `random` is that policy's independent seeded RNG. The runner submits each choice on its own request and relies on Showdown's choice commitment behavior to hold simultaneous choices. Candidate moves and switches come from the request payload. `trapped` is honored when disclosed; `maybeTrapped` is not resolved from simulator state. Wait requests create no action. Unavailable-choice errors and corrected requests stay inside a bounded decision episode.

`--result PATH` writes a separate version 1 per-battle result record on success or failure, including configuration and stream-initialization failures. The record identifies the pinned commit, format, five seeds, effective limits, `completed`/`failed` status, and validated outcome or `null`. Its counters separate actual policy invocations, successfully sent choices, accepted/rejected/pending submissions, retry requests and retry submissions, retry-limit exhaustion, selected versus sent moves/switches, public spectator events, the latest public `|turn|` marker, and terminal metadata turns. Partial counters remain available on failure; `runBattle` also attaches the record as `error.resultRecord`. The failure field contains a stable category and code, without private packet or raw error text.

`elapsedMs` is monotonic wall-clock milliseconds. `elapsedScope: "battle_runner"` covers the runner invocation through terminal validation and stream cleanup or failure. When a normal run generates its teams, the generation time is included. A run given prepared teams starts timing after their prior preparation, so shared preparation time is excluded. For a CLI error before `runBattle` starts, `elapsedScope: "cli_initialization"` covers command/reference initialization. Timing is instrumentation only; the runner does not report throughput. The new timing and counters stay outside the existing smoke summary and trusted transcript so their deterministic contents remain unchanged. A result record is trusted engineering metadata and must never be policy input. It carries only spectator-projected public event counts; it does not expose policy-private requests, split secret branches, timestamps, packet envelopes, or trusted terminal metadata beyond the validated outcome and terminal turn count.

## Request-aware singles action adapter

[`environment/action-space.cjs`](environment/action-space.cjs) maps one current Gen 5 singles request to a version 1 mask with eleven fixed slots: slots 0–3 represent the current request's move rows 1–4, slots 4–9 represent the player's roster positions 1–6 for switching, and slot 10 represents a sole Struggle request. Decoding returns Showdown commands such as `move 1` and `switch 2`. Reduced locked-move and recharge requests place their current row in slot 0; the adapter does not infer a move's original position. A sole Struggle row uses slot 10 and decodes to `move 1`. If a multi-row request contains a Struggle-named move, each row keeps its current request position and slot 10 remains disabled.

The mask follows only the player's own current request: disabled or exhausted moves are masked, `trapped` suppresses switches, `maybeTrapped` leaves them available for Showdown to resolve, and forced replacement masks include only eligible own roster positions. A `wait` request and `forceSwitch: [false]` return all-false masks; there is no WAIT action. Rejected command exclusions belong to the current request episode and are supplied by the caller. `decodeAction(request, slot, rejectedCommands)` recomputes the mask from the supplied request, so an old slot mapping is not a stale-action authorization token. The adapter accepts up to four current move rows and six own roster entries; it does not define observation features or freeze a future training API.

Pinned Showdown synthesizes an exact sole-Struggle row without PP: `{ move: "Struggle", id: "struggle", target: "randomNormal", disabled: false }`. The existing request parser accepts that one generated shape as `move 1`; malformed variants still fail. Task028 validates this row through the pinned Battle choice handler and runs complete seeded battles with adapter-decoded commands. This request adapter is a project step, not G1 or later milestone acceptance.

## Completed-battle decision trajectories

`environment/decision-trajectories.cjs` exports the version-1 `collectDecisionTrajectories(options)` library API. It composes one real `runBattle` call and accepts the runner's `ps`, `seeds`, `limits`, `preparedTeams`, and `streamFactory` options plus optional `policies.p1` and `policies.p2` callbacks. An omitted policy uses the existing `randomPolicy`; a supplied callback still receives exactly the runner's `{ history, request, random }` own-view input. Prepared-team handles are forwarded without cloning so their identity binding remains valid.

On completed pinned battles the API returns a JSON-serializable artifact with schema and action-space version 1, `gen5randombattle` and pinned upstream identity, the validated public terminal outcome, separate `players.p1.steps` and `players.p2.steps`, and fixed-seat `targets.terminalReward` values (`+1` winner, `-1` loser, `0` tie). Each step contains a zero-based `decisionIndex`, the runner's positive `requestEpisode` ID, `requestKind`, an `observation` with that player's current request and append-only `historyDelta`, the eleven-slot action `mask`, the selected `{ slot, command }`, and `behaviorProbability: null` because this collector does not know policy likelihoods. Concatenating one player's in-memory deltas reconstructs the exact own-view history prefix at each successive policy callback. Standard JSON serialization omits optional `undefined` properties in runner history records; parsed deltas reconstruct the JSON projection of that callback history. The delta captures that player's visible progress and resolution records, including non-actionable requests between decisions; retries use only rejection records from the current episode, while an identical request in a later episode is a new sample.

Only actionable policy invocations become steps: `wait`, `forceSwitch: [false]`, and duplicate unresolved requests add no decision. The request snapshot and history delta are detached before the policy is called. Observations contain no reward, outcome, other player's request or private history, seeds, packed teams, random function, raw packets, or trusted transcript. Terminal targets are attached only after the runner reports a validated completed battle and never enter the policy callback. The combined artifact is post-game private training data and must not be fed back as a policy observation. Schema/version fields describe its shape; serialization does not authenticate which process executed the battle. A selected command is not labelled as submitted, accepted, or executed. Failed runs reject with a fixed diagnostic and return no partial artifact. The schema omits events after the final policy decision, so its final delta is not a complete battle transcript. This is a trajectory schema step, not a learner, CLI, benchmark, performance result, or G1 closure.

## Move-first-v1 reference baseline

`environment/move-first-policy.cjs` exports `POLICY_ID` (`move-first-v1`) and the synchronous `moveFirstPolicy({ history, request, random })` callback. It composes the version 1 action mask and decoder: when any current move row is enabled, it chooses uniformly among those move rows, including the explicit Struggle slot, and otherwise chooses uniformly among enabled switches. A sole available action still consumes one draw from the supplied `random` function. Valid wait and `forceSwitch: [false]` requests return `null` without drawing; malformed or exhausted actionable requests and non-finite/out-of-range draws fail with a fixed diagnostic.

The callback is stateless and reads only its own current request, its own history, and the supplied RNG. Rejected commands are excluded only from the latest `requestEpisode` through the next visible episode marker. The callback follows the mask's request-only handling of `trapped`, `maybeTrapped`, fainted replacements, disabled moves, reduced locked/recharge rows, and Struggle; it does not inspect battle state. The runner's policy seed supplies the RNG stream, and this baseline does not change the existing default policy.

Use the same callback directly with `runBattle` or through the completed-battle collector:

```js
const { runBattle } = require('./environment/battle-runner.cjs');
const { collectDecisionTrajectories } = require('./environment/decision-trajectories.cjs');
const { moveFirstPolicy } = require('./environment/move-first-policy.cjs');
const { randomPolicy } = require('./environment/random-policy.cjs');

async function evaluate({ ps, seeds, preparedTeams }) {
  const policies = { p1: moveFirstPolicy, p2: randomPolicy };
  const battle = await runBattle({ ps, seeds, preparedTeams, policies });
  const trajectories = await collectDecisionTrajectories({ ps, seeds, preparedTeams, policies });
  return { battle, trajectories };
}
```

The generated paired-seed contract can keep the policy identity and its policy RNG seed together while exchanging seats. The two pinned paired battles in Task030 verify request masks, terminal reward mapping, and stream cleanup; they are integration checks, not strength estimates, a scored suite, or a tournament result.

Task022 adds [`environment/paired-seeds.cjs`](environment/paired-seeds.cjs), which prints a trusted manifest for a caller-selected positive pair count and explicit four-limb root seed:

```sh
node environment/paired-seeds.cjs --root-seed=1,2,3,4 --pairs=2 > paired-seeds.json
node --test test/paired-seeds.test.cjs
```

The versioned SHA-256 derivation uses separate named streams for team A, team B, the battle, and policies A/B. Each pair includes two expanded five-seed runs with opposite seat assignments, stable pair/run IDs, team and policy identities, and the pinned format revision. Output is byte-reproducible. Existing pair records and IDs remain identical when the requested count grows; the top-level count and additional records change. The command requires both options and accepts 1 through 100,000 pairs; that upper bound limits accidental allocations and does not select or recommend a scored suite size. Task022’s integration test verifies that deterministic regeneration gives the same packed team in the opposite seat. The manifest is engineering metadata and must never be policy input.

Task023 adds an explicit prepared-team API for reusing the same immutable packed team data in both runs of a pair. Each call to `runBattle` creates separate mutable battle and Pokémon state:

```js
async function main() {
	const ps = require('./reference/vendor/pokemon-showdown/dist/sim');
	const { generateManifest } = require('./environment/paired-seeds.cjs');
	const { runBattle } = require('./environment/battle-runner.cjs');
	const { prepareTeam } = require('./environment/prepared-teams.cjs');

	const [pair] = generateManifest([1, 2, 3, 4], 1).pairs;
	const teamA = prepareTeam(ps, pair.teams.A.seed);
	const teamB = prepareTeam(ps, pair.teams.B.seed);
	const first = await runBattle({ ps, seeds: pair.runs[0].seeds, preparedTeams: { p1: teamA, p2: teamB } });
	const second = await runBattle({ ps, seeds: pair.runs[1].seeds, preparedTeams: { p1: teamB, p2: teamA } });
}
main();
```

Each handle is immutable and tied to its generating Showdown object, pinned format revision, and team seed. Its packed data lives in a process-local `WeakMap`; the visible seed and digest metadata cannot restore a handle after serialization. The runner rejects copied, mismatched, or missing handles before opening a battle stream and records `teamSource: "prepared"` in that run’s result record. Preparing teams calls the generator and packer once per team; reusing handles does not repeat either call. Preparation metadata and handles are engineering data, never policy inputs. Task023 provides the reuse API and real paired-run coverage; Task024 adds explicit single-pair execution. Task025 now executes a caller-selected full manifest sequentially. Task026 adds recorded-outcome accounting, and Task027 adds observed additive-work cohorts. Rates, uncertainty, and scored-suite design remain proposed.

Task024 adds a single-pair library function and CLI. Both the root and zero-based pair index are required; the direct selector does not materialize earlier pairs:

```sh
node environment/run-pair.cjs --root-seed=1,2,3,4 --pair-index=0 --result pair-result.json
node --test test/pair-runner.test.cjs
```

The command prepares team A and B once, runs both seat assignments sequentially in separate battles, and writes their status and nested per-battle result records in a versioned pair record. On a battle failure it preserves any earlier completed run, records the attempted failed run, and leaves a remaining run marked `not_attempted`; preparation and configuration failures leave both runs unattempted. The pair `elapsedMs` starts before team preparation and covers both battles and their cleanup; nested prepared-battle times start when each `runBattle` begins, after the shared preparation. Pair results and per-battle results contain no raw transcript, packed team, policy input, or exception text. If parsing or pinned-reference initialization fails before the pair starts, an optional result file records `elapsedScope: "cli_initialization"` and the CLI exits nonzero. A result-file write error produces a concise nonzero error; no result file can be written when its own destination is unavailable. Task024 runs one explicitly selected pair; Task025 adds sequential execution across an explicit manifest.

Task026 adds a read-only offline report over Task025's `manifest.json` and `results.jsonl`:

```sh
node environment/summarize-results.cjs --manifest results/manifest.json --results results/results.jsonl
node --test test/summarize-results.test.cjs
```

The report validates the complete generated manifest and the journal's contiguous pair prefix, then counts recorded pair/run statuses and completed battle outcomes by player seat and policy identity. An empty journal is valid; each non-empty JSONL record must occupy a newline-terminated line, and blank lines are invalid. The command does not read or reconcile Task025's `summary.json`. A missing suffix is `unrecorded`: a write failure can leave a pair result absent after its battle was attempted. A full recorded prefix describes journal coverage only; it does not prove that the records came from an authentic runner process or make a scored benchmark claim. Offline checks validate serialized format, IDs, seeds, statuses, and terminal outcome shape. They cannot prove which process produced a result record; the report is accounting metadata without execution attestation. This report does not aggregate counters, timing, rates, or uncertainty intervals.

Task027 adds a separate offline additive-work report over the same manifest and journal:

```sh
node environment/summarize-work.cjs --manifest results/manifest.json --results results/results.jsonl
node --test test/summarize-work.test.cjs
```

The work report composes Task026's canonical-manifest and contiguous-prefix validation, then reports three run cohorts: completed runs, failed recorded attempts, and all recorded attempts. Each cohort contains its run count, counts of records with and without counters, and `observedTotals`. A failed attempted run with `resultRecord: null` is valid and increments `missingCounters`; it contributes no invented zero-work record. `not_attempted` runs and unrecorded suffix slots stay outside all work totals and remain visible in `coverage.runSlots`. A present nested result record must have every additive counter field. The report checks nonnegative safe integers, each `byPlayer` sum against its supplied total, overflow-safe arithmetic, and per-player choice-resolution, submission, and selection invariants. Failed-run counters remain included when present and may represent partial work.

`observedTotals.global` contains `objectPackets` and `publicEvents`. Each player-counter field has a `total`, `bySeat` counts for p1/p2, and `byPolicy` counts for policyA/policyB mapped through the validated run's policy seats. Selected actions stay separate from submitted actions; rejections and retries stay separate from policy invocations and choice submissions. `latestPublicTurn` and `terminalMetadataTurns` are metadata, not additive work, and are excluded. The report also excludes elapsed time, limits, run seeds, hashes, transcripts, and arbitrary record fields. It does not read or reconcile `summary.json`, authenticate which process emitted records, compute rates or uncertainty, select a scored suite, or make a performance claim. The pure API accepts finite JSON data with enumerable own data properties; accessors and custom prototypes are rejected with a fixed safe error. The CLI reads only the supplied manifest and result journal and does not bootstrap the pinned oracle.

Task025 provides a full-manifest library and pinned CLI. The caller chooses the root, pair count, and a path for a new output directory:

```sh
node environment/run-manifest.cjs --root-seed=1,2,3,4 --pairs=2 --output-dir=run-001
node --test test/manifest-runner.test.cjs
```

The output directory's parent must already exist, and the output directory itself must not exist. The runner creates it without recursion, then exclusively creates `manifest.json`, `results.jsonl`, and `summary.json`; an existing destination is left untouched and no pair starts. `manifest.json` serializes the exact versioned output of `generateManifest(rootSeed, pairCount)`. Each attempted pair's safe versioned pair result is appended as one compact JSON line in manifest index order before the next pair starts. A failure stops later pairs and preserves earlier journal lines; no resume or retry is provided. The version 1 summary records the root/count, pinned upstream and format, seed-derivation identity and manifest SHA-256, status, finite `manifest_runner` elapsed time, execution bookkeeping (`attemptedPairs`, `completedPairs`, `failedPairs`, and `unattemptedPairs`), `recordedPairs`, and safe failure stage/category/code with the pair index when applicable. A completed pair whose append failed remains counted as completed but is not described as journaled. CLI parse or pinned-reference bootstrap failures happen before the runner creates an output directory and may leave no artifacts. If summary writing fails, the command exits nonzero and reports that `summary.json` could not be confirmed complete.

Manifest `elapsedMs` is a monotonic duration over configuration snapshot/validation, manifest generation and output, attempted pairs and their cleanup, and JSONL appends. Its value is sampled immediately before final summary serialization and write; those final operations and CLI output are excluded. Pair records retain their existing `paired_runner` timing. These are timing boundaries for reproducibility and failure accounting, not performance or throughput claims. Manifests and pair records are trusted engineering data, never policy inputs; the journal contains no raw transcript, packed team, or exception text. The explicit `--pairs` count only chooses how many pairs this invocation attempts; it does not establish a scored suite size.

This is a deliberately small baseline. It does not provide a learned policy, observation features, a full game-memory contract, or JAX mechanics.

## Named baseline paired matchup

`environment/baseline-matchup.cjs` exports the version 1 library function `runBaselineMatchup(options)`. It requires an explicit pinned Showdown `ps` API, a four-limb `rootSeed`, a zero-based `pairIndex`, and `policyA` and `policyB` identifiers. It accepts only `legal-random-v1` (the unchanged `randomPolicy`) and `move-first-v1` (the existing stateless move-first callback); the same identifier may fill both roles.

```js
const oracle = require('./reference/oracle.cjs');
oracle.assertReady();
const ps = require('./reference/vendor/pokemon-showdown/dist/sim');
const { runBaselineMatchup } = require('./environment/baseline-matchup.cjs');

const artifact = await runBaselineMatchup({
  ps,
  rootSeed: [1, 2, 3, 4],
  pairIndex: 0,
  policyA: 'legal-random-v1',
  policyB: 'move-first-v1',
});
```

The `options` and `limits` containers must be plain records with enumerable own data properties; accessors and unknown option/limit keys are rejected. Optional `limits` may override positive finite `timeoutMs`, `maxPackets`, `maxDecisions`, and `maxRetriesPerRequest` values. Optional `streamFactory` is forwarded to each actual battle and must supply a fresh stream on each call. The API snapshots and validates its configuration before team generation. It selects the requested pair directly with `generatePair`, prepares and packs teams A and B once, then runs the generated A/p1 and B/p1 seat assignments sequentially with those same opaque handles.

Policy identifiers are attached to the generated `policyA` and `policyB` roles and mapped to each run's seats through `policySeats`. Each identity keeps its generated policy seed when its seat changes. The two roles have separate policy RNG streams even in same-ID self-play, and each battle starts fresh state. Baseline callbacks receive only the runner's existing recipient-scoped request, history, and random function.

The returned `schemaVersion: 1`, `kind: "gen5-named-baseline-paired-matchup"` artifact is distinct from Task024/025 pair and journal records. It includes the pinned format and upstream, root and pair identity, role policy IDs, effective battle limits, policy seeds, prepared-team seeds and packed-team digests, the exact generated run IDs/seats/policy-seat roles/seeds, policy IDs by seat, and two completed safe `battle-run-result` records. It does not include transcripts, packets, packed teams or handles, policy inputs, exception text, or caller-supplied arbitrary fields. The legacy `summarizeResults` API rejects this named-matchup envelope; it remains separate from the legacy pair accounting schema.

The function returns only when both battles have valid completed outcomes and the runner has completed cleanup. Configuration, preparation, battle, validation, or cleanup failures reject with the fixed diagnostic `Baseline paired matchup failed (baseline_matchup_failed).` No partial artifact is returned. A failed call may already have generated teams, opened a stream, or completed the first battle; the absence of an artifact says nothing about work attempted or completed. The per-battle `elapsedMs` values keep the runner's `battle_runner` timing scope and do not measure whole-matchup setup or claim performance. The serialized artifact is reproducibility metadata, not proof that these battles ran or that the metadata was not changed afterward. This API adds no CLI, persistence, journal, resume, aggregation, scored suite, rate, or strength claim.

### Run and save one named matchup

The CLI composes that library API with the pinned Showdown reference and saves its completed artifact:

```sh
node environment/run-baseline-matchup.cjs \
  --root-seed 1,2,3,4 \
  --pair-index 0 \
  --policy-a legal-random-v1 \
  --policy-b move-first-v1 \
  --output /tmp/gen5-pair-0.json
```

All five options are required; the seed, zero-based pair index, and both versioned policy IDs are explicit. The parser accepts either a separate value or `--name=value`, and `--help` must be used alone. Unknown and repeated flags, unsupported IDs, malformed seeds, and pair indices outside `0..99999` fail before reference setup or battle execution. Only `legal-random-v1` and `move-first-v1` are supported.

After the pinned reference check, the CLI exclusively creates the requested output file with owner-only permissions before running either battle. An existing file, symlink, or missing parent is left untouched and causes a safe error. While the pair runs, this reserved file is empty. The CLI writes the versioned artifact only after both battles complete and clean up, flushes the file, closes it, and then reports success. Ordinary execution and write failures return a fixed diagnostic, attempt to remove the reservation only when the output path still identifies the same file, and never emit a partial result or raw exception. Cleanup and filesystem failures can leave an empty or partial file; an abrupt process or machine failure can leave the reservation or interrupt a write. File flushing does not sync the containing directory entry. A failed invocation may have completed real battle work, and this CLI does not journal or resume it. Concurrent replacement or writing of the reserved output path is unsupported; identity checks avoid deleting a replacement that is observed at cleanup time.

The success line contains only the artifact kind, pair ID, completion status, and resolved output path. The saved artifact retains the library's per-battle timing scope, excludes earlier setup and output writing, and is reproducibility metadata rather than proof of execution or tamper resistance. No aggregate score, tournament result, strength estimate, or performance claim is produced.

## Trusted trace capture and replay

[`reference/trace.cjs`](reference/trace.cjs) captures and replays the full BattleStream input/output sequence:

```sh
node reference/trace.cjs capture --output /tmp/gen5-random-trace.json
node reference/trace.cjs capture --output /tmp/gen5-random-trace-seeded.json \
  --team-seed-p1=1,2,3,4 --team-seed-p2=5,6,7,8 \
  --battle-seed=9,10,11,12 \
  --policy-seed-p1=13,14,15,16 --policy-seed-p2=17,18,19,20
node reference/trace.cjs replay /tmp/gen5-random-trace.json
node --test test/trace.test.cjs
```

The version 1 trusted trace records the pinned SHA, format, all five seeds, complete ordered start/player/choice inputs, complete output packets, and verified outcome. Replay submits those recorded inputs directly to a fresh pinned simulator; it does not regenerate teams or call a policy. Only clock-only `|t:|` messages are normalized, with blank split branches retained in place. These files contain packed teams, private requests, split `SECRET` records, and trusted end metadata. Never expose a trace to an actor; use `projectPackets` to derive a single recipient's view.

## First JAX arithmetic slice

Task005 implements only the post-`ModifyBoost` non-HP stat-stage scaling substep from pinned Gen 5 `Pokemon.calculateStat`, before stat modifiers. The current pure int32 kernel accepts a nonnegative stored stat in `[0, 65535]` and a signed int32 stage; stages clamp to `[-6, 6]`. Task007 extends the original Task005 positive range because Gen 5 nature-product truncation can produce a valid stored zero. These are caller preconditions, not runtime checks. The scaler excludes base-stat derivation, natures, EVs, Wonder Room stat selection, event hooks, and later stat modifiers.

Create an isolated CPU environment without changing the existing GPU JAX installation:

```sh
python3 -m venv .venv-jax
. .venv-jax/bin/activate
python -m pip install -r requirements-jax-cpu.txt
node reference/stat-stage-oracle.cjs fixture
JAX_PLATFORMS=cpu python -m unittest discover -s test -p 'test_*.py' -v
node --test test/*.test.cjs
```

`requirements-jax-cpu.txt` pins JAX and JAXLIB to 0.12.0. The versioned oracle fixture is generated by the pinned Showdown runtime in a neutral Gen 5 custom game, calling `Pokemon.calculateStat` with an identity `ModifyBoost` event, no Wonder Room, and modifier 1. The differential test compares all 13 stages for stored stats 0 through 1024, plus the int32 stage extremes and stored-stat upper boundary. Tests cover eager/JIT/vmap/broadcast execution on CPU and GPU. The stat range `[0, 65535]` is a caller precondition, not a runtime check. This is a mechanics kernel parity slice, not a full stat calculator, battle simulator, or learner.

## Gen 5 stored-stat derivation

Task006 adds [`mechanics/stats.py`](mechanics/stats.py), which derives `hp, atk, def, spa, spd, spe` from base stats, level, IVs, EVs, and integer nature multipliers (`9` minus, `10` neutral, `11` plus). The final axis has six stats; level and the explicit `shedinja` flag broadcast over batch axes. Inputs must be int32 with base stats `[1,255]`, level `[1,100]`, IVs `[0,31]`, EVs `[0,255]`, and multipliers `[9,10,11]`; these value ranges are caller preconditions, not runtime checks. Shedinja's species override forces HP to 1. Base HP 1 by itself does not trigger that override.

The pinned `gen5randombattle` format does not enable `overflowstatmod`, so the kernel preserves Gen 5's unsigned low-16-bit truncation on nature products and does not apply the optional 595/728 caps. [`reference/stats-oracle.cjs`](reference/stats-oracle.cjs) constructs actual pinned Pokemon and records `maxhp`/`storedStats`; it also uses actual `Battle.statModify` calls for synthetic base-stat threshold cases. Regenerate the fixture and run the full Python suite with:

```sh
node reference/stats-oracle.cjs fixture
JAX_PLATFORMS=cpu python -m unittest discover -s test -p 'test_*.py' -v
node --test test/*.test.cjs
```

The fixture includes 125 real Pokemon across all 25 natures and levels `1, 17, 50, 99, 100`, plus nine synthetic cases across Adamant, Modest, and Hardy for pre-nature Attack 595, 596, and 609. At 595, `595*110=65450` stays below the 16-bit wrap; at 596, `596*110=65560` wraps to 24 and the stored Attack becomes 0. Tests compare eager, JIT, and JIT-vmap results on the pinned CPU environment and the available GPU environment. Task006 does not derive stat stages, model event hooks/forms, or validate competitive teams. Task007 extends stage scaling to accept zero, which is needed to compose the valid synthetic result.

## Attack stored-stat and stage composition

Task007's [`mechanics/attack.py`](mechanics/attack.py) composes the stored-stat and stage kernels for Attack only: derive the six Gen 5 stored stats, select Attack, and apply the requested stage. It matches `Pokemon.calculateStat('atk', stage, 1)` only with neutral `ModifyBoost`, no Wonder Room, and no stat modifier. It does not include abilities, items, status, or dynamic stat selection.

Generate the guarded reference fixture and run the same CPU/GPU Python test suites as Tasks005–006:

```sh
node reference/attack-oracle.cjs fixture
JAX_PLATFORMS=cpu python -m unittest discover -s test -p 'test_*.py' -v
node --test test/*.test.cjs
```

The compact fixture covers all 25 natures with initialized Pokemon and all 13 stages plus clamp inputs. Ordinary expected values come from the actual `Pokemon.calculateStat` call without overwriting `storedStats`. A separate synthetic case uses real `Battle.statModify` to derive zero at the Gen 5 nature-wrap boundary, then injects it only to check the real stage arithmetic. The fixture records the pin and source paths. On 2026-10-28, all 20 Python tests passed on both CPU and available GPU environments; all 34 Node tests passed. Task007 is accepted and complete.

## Damage randomizer arithmetic

Task008 implements the `Battle.randomizer(baseDamage)` arithmetic as an explicit-roll JAX kernel. It models the uint32-wrapped multiplication by `100-roll`, followed by floor division by 100. The input roll is supplied by the caller; this slice does not generate randomness or include the minimum-damage clamp, STAB, type effectiveness, or other modifiers. Its live oracle records battle seed `[101, 202, 303, 404]` and observes later `Battle.random(16)` results after Battle initialization; it does not label them as the PRNG's first draws.

The bounded oracle/test commands are:

```sh
node reference/damage-roll-oracle.cjs fixture
JAX_PLATFORMS=cpu python -m unittest discover -s test -p 'test_*.py' -v
node --test test/*.test.cjs
```

Task008 and Task009 are accepted and complete. Task008's 27-test Python suite passes on CPU and available GPU; Task009's 34-test Python suite passes on CPU and GPU. The existing Node suite passes 34/34. Both are arithmetic slices, not a complete damage calculator.

## Base-damage integer core

Task009 implements the nested integer core inside pinned `BattleActions.getDamage`, before `modifyDamage` and the later `+2`. Inputs are the effective level, base power, Attack, and Defense. It preserves uint32 product truncation and matches the zero-Defense result safely. The kernel does not derive stats or model events, critical hits, `+2`, the random roll, STAB, or type effectiveness.

```sh
node reference/base-damage-oracle.cjs fixture
JAX_PLATFORMS=cpu python -m unittest discover -s test -p 'test_*.py' -v
node --test test/*.test.cjs
```

Task009 is accepted and complete. Its full Python suite passes 34/34 on CPU and GPU, and the Node suite passes 34/34. Task010 composes the accepted core with Showdown's `+2`, the real neutral event's fixed-point identity behavior, and an explicit random-roll stage.

## Neutral damage after the random roll

Task010's pure JAX `neutral_damage_after_roll` returns the intermediate after `Battle.randomizer`, not final damage. It composes the base-damage core, the `+2` in `modifyDamage`, and `Battle.modify(value, 1)` as applied by the neutral `WeatherModifyDamage` event, then applies the supplied roll. That nominal identity event wraps to 20 bits because Showdown truncates `value * 4096` to uint32. This kernel omits the PRNG, full modifier engine, critical hits, STAB, type, burn, Gen 5 minimum-one check, and final 16-bit truncation.

```sh
node reference/neutral-damage-oracle.cjs fixture
JAX_PLATFORMS=cpu python -m unittest discover -s test -p 'test_*.py' -v
node --test test/*.test.cjs
```

Task010 is accepted and complete. Its guarded pinned oracle delegates to the real `modifyDamage`, neutral event, and `Battle.randomizer`, controlling only the explicit random draw. The 41-test Python suite passes on CPU and GPU; the Node suite passes 34/34. Task008 separately tests randomizer uint32-product wrapping; the neutral event bounds this composition's roll input to at most `2^20-1`.

## Fixed-point modifier arithmetic

Task011 adds `apply_fixedpoint_modifier(value, modifier)` for an already encoded Gen 5 modifier. Strict int32 inputs have caller-precondition ranges `value` `[0, 2147483647]` and `modifier` `[0, 65535]`. It follows `Battle.modify(value, modifier, 4096)`, including unsigned 32-bit product wrap and round-half-down behavior. It does not encode fractions or combine a chain of modifiers.

```sh
node reference/fixedpoint-oracle.cjs fixture
JAX_PLATFORMS=cpu python -m unittest discover -s test -p 'test_*.py' -v
node --test test/*.test.cjs
```

Task011 is accepted and complete. The 47-test Python suite passes on CPU and GPU; the Node suite passes 34/34. The same primitive implements Task010's neutral identity event, preserving its 20-bit wrap behavior. Task012 adds the bounded damage-finalization tail under explicit inputs.

## Gen 5 damage finalization

Task012 adds `finalize_damage(damage, modifier)` in `mechanics/finalize_damage.py`. It consumes damage after type effectiveness and burn, replaces zero with one, applies the already encoded final modifier through Task011, and truncates to 16 bits. The returned damage may be zero: Gen 5 does not repeat its minimum-one check after the final modifier or truncation. Both inputs are strict int32 and broadcast; caller-precondition ranges are damage `[0, 2147483647]` and modifier `[0, 65535]`.

The guarded oracle delegates to the actual pinned `getDamage`/`modifyDamage` tail. Eight ordinary cases use initialized Pokemon with no item or real Life Orb (observed modifier 5324). The 430 synthetic cases inject only an explicit randomizer output in a neutral path and register one real modifier event; they preserve real zero-check, event application, and final truncation. Tests also compose the primitive with Task010's existing neutral damage fixture. This slice does not implement STAB, effectiveness, burn, event dispatch, modifier chaining, bypassProtect, or a full damage calculator.

```sh
node reference/finalize-damage-oracle.cjs fixture > reference/fixtures/finalize-damage-v1.json
JAX_PLATFORMS=cpu .venv-jax/bin/python -m unittest discover -s test -p 'test_*.py' -v
node --test test/*.test.cjs
```

Task012 is accepted and complete. All 54 Python tests and 35 Node tests pass. GPU validation is unavailable in this resumed CPU-only environment; it has no CUDA backend. See `docs/TASKS.md` for current Node validation evidence and the Task013 proposal.

Task013 is accepted and complete after scheduled run 1 and architect review. `apply_type_effectiveness(damage, exponent)` takes strict broadcastable int32 inputs, clamps signed exponents to `[-6, 6]`, and uses the damage precondition `[0, 33554431]` so positive doubling stays within int32. Its live fixture covers ordinary real type lookup and a labeled synthetic grid across all clamped exponents. Four targeted CPU Python tests and the Node oracle restoration test pass; this environment has no GPU device. Type lookup, immunity, STAB, burn, modifiers, and the full damage engine remain outside this slice. Commit `5bd53aa` was pushed to `main`. See `docs/TASKS.md` for the fixture boundary and evidence.

Task014 is accepted and complete after scheduled run 2 and architect review. `apply_burn_reduction(damage, burned, physical, has_guts)` implements the pinned Gen 5 physical-burn arithmetic step. It applies the existing fixed-point `0.5` modifier only for `burned & physical & ~has_guts`; eligibility inputs are strict booleans and broadcast with strict int32 damage. Gen 5 does not exempt Facade. The guarded oracle covers real physical/special, Guts, Facade, and unburned moves, plus controlled randomizer boundary values observed before the minimum-one check. This kernel does not resolve runtime status, category, or ability, and does not dispatch events or include other damage steps. The focused Python tests pass 4/4 on CPU and the Node oracle/restoration tests pass 2/2; this environment exposes only a CPU device. See `docs/TASKS.md` for the exact fixture contract.

## Gen 5 STAB arithmetic

Task015 adds `apply_stab(damage, stab_eligible, adaptability, typeless)` in `mechanics/stab.py`. The strict int32 damage and strict boolean flags broadcast. Typed damage always uses the existing fixed-point kernel: 6144 for eligible STAB, 8192 for eligible Adaptability, and 4096 for typed non-STAB. `typeless` bypasses the modifying call and preserves the input, even if eligibility flags are true. The caller supplies already resolved eligibility and Adaptability state; this slice does not look up types or abilities, dispatch events, apply Tera, or include later damage steps.

The guarded oracle uses real pinned Gen 5 `getDamage`, `ModifySTAB`, and Adaptability behavior for ordinary moves. Synthetic cases replace only `Battle.randomizer` output and record the actual event result and delegated `Battle.modify`; they cover 0/1/2/3, product-wrap-adjacent values for all three typed modifiers, and int32 maximum. The typeless cases assert the STAB modify call is skipped.

```sh
node reference/stab-oracle.cjs fixture > reference/fixtures/stab-v1.json
JAX_PLATFORMS=cpu .venv-jax/bin/python -m unittest test.test_stab -v
node test/stab-oracle.test.cjs
```

Task015 is accepted and complete for scheduled run 3. The focused Python tests pass 4/4 on CPU and the Node fixture/restoration tests pass 2/2. This environment exposes only a CPU device; GPU validation is unavailable. See `docs/TASKS.md` for the exact fixture boundary and evidence.

## Bounded post-core damage composition

Task016 adds `damage_from_preroll(...)` in `mechanics/damage_composition.py`. It composes the existing random roll, STAB, type effectiveness, burn reduction, and finalization kernels in that order. Inputs remain strict int32/bool values with JAX broadcasting; the caller supplies already resolved flags, type exponent, and final modifier. The documented preconditions are preroll damage `[0, 33554431]`, roll `[0, 15]`, signed-int32 type exponent (clamped by the existing type kernel), and final modifier `[0, 65535]`. The domain is a caller contract and is not checked dynamically; out-of-domain preroll inputs do not claim to preserve the randomizer's broader uint32-wrap behavior. This slice starts after core and earlier modifiers and ends after final damage; it does not generate RNG or resolve runtime state/events, and does not include core, `+2`, weather, critical, spread, bypassProtect, or a full damage engine.

The pinned oracle delegates the actual `getDamage` return and original randomizer. Ordinary cases use real Gen 5 move/type/ability/status state, including physical burn, Guts bypass, Adaptability, and Life Orb; the `random(16)` damage draw is controlled for deterministic capture. Synthetic cases replace only the input passed to the original randomizer, then delegate to it, with explicit effectiveness and final modifier event control. Captured stages match each constituent kernel, including all 16 draws, an order-sensitive odd anchor, signed exponent clamps, the zero-before-final check, uint16 zero wrap, and maximum preroll. Eager, JIT, vmap, broadcast, and strict dtype checks pass on CPU. See `docs/TASKS.md` for exact evidence.

```sh
node reference/damage-composition-oracle.cjs fixture > reference/fixtures/damage-composition-v1.json
JAX_PLATFORMS=cpu .venv-jax/bin/python -m unittest test.test_damage_composition -v
node test/damage-composition-oracle.test.cjs
```

Task016 is accepted and complete for scheduled run 4. The final composition-focused rerun passed 4/4 Python tests on CPU and 2/2 direct Node oracle tests. A bounded Task017 proposal is to add the pre-roll prefix around this composition: compose `base_damage_core`, Gen 5 `+2`, and Task010's neutral `WeatherModifyDamage` fixed-point identity step, then pass that preroll into `damage_from_preroll` once. Keep effective Attack/Defense, flags, and roll explicit; end at final damage and exclude runtime event/type/ability lookup and RNG generation.

## Neutral full damage composition

Task017 adds `neutral_damage_before_roll(level, base_power, effective_attack, effective_defense)` and `damage_from_effective_stats_neutral(...)`. Their caller domains are level `[1, 100]`, base power `[1, 65535]`, effective Attack/Defense `[0, 65535]`, roll `[0, 15]`, signed-int32 type exponent, final modifier `[0, 65535]`, plus strict bool flags. The prefix composes the existing core, int32 `+2`, and neutral weather identity operation; `neutral_damage_after_roll` now delegates through that public prefix and applies one explicit roll. The full path then calls Task016's `damage_from_preroll` once with caller-supplied roll, STAB/type/burn flags, exponent, and final modifier. It derives no battle stats or runtime state. For the existing core domain, the prefix result is at most `1,048,575`, within Task016's pre-roll range.

The guarded fixture observes actual `BasePower` and category-appropriate effective attack/defense event outputs before the pinned core, captures the actual core and `WeatherModifyDamage` values, then delegates through the real randomizer and final tail. Ordinary cases include typed STAB/non-STAB, Adaptability, physical burn with and without Guts, and Life Orb. Six synthetic cases mutate level, base power, and stored Attack/Defense to exercise uint32 core wrap, zero Defense, neutral 20-bit wrap values 1,048,575 through 1,048,577, and maximum inputs; downstream type exponent and final modifier controls are explicit. Every case covers rolls 0–15. Critical is forced false and checked. Spread, Parental Bond, bypassProtect, event/runtime lookup, and RNG generation remain outside the arithmetic composition.

```sh
node reference/neutral-full-damage-oracle.cjs fixture > reference/fixtures/neutral-full-damage-v1.json
JAX_PLATFORMS=cpu .venv-jax/bin/python -m unittest test.test_neutral_damage test.test_neutral_damage_composition -v
node --test test/damage-composition-oracle.test.cjs test/neutral-full-damage-oracle.test.cjs
```

Task017 is accepted and complete for scheduled run 5. See `docs/TASKS.md` for validation evidence and the next bounded proposal.

## Mechanics coverage ledger and current validation boundary

[`docs/MECHANICS_COVERAGE.md`](docs/MECHANICS_COVERAGE.md) inventories every public function under `mechanics/`, linking source symbols to the pinned oracle, fixture, and Python differential test. It distinguishes atomic arithmetic from compositions, ordinary pinned battle paths from injected synthetic boundary probes, and both from reference-backed format/protocol/trace infrastructure. G0 specification closure was independently accepted on 2026-11-03 (see the [review report](docs/reviews/G0-specification.md)). The ledger remains a partial arithmetic inventory for the complete pinned target; it claims no game-coverage percentage or full-engine support.

GPU claims are historical and task-scoped: Tasks005–011 recorded GPU validation in their original environments. From Task012 onward, resumed task evidence reports CPU-only validation or GPU validation deferred; in particular Tasks013–017 record CPU-only results. The ledger is a documentation audit and did not rerun those kernels or establish current GPU availability.

[`docs/REUSE_AUDIT.md`](docs/REUSE_AUDIT.md) records Task019’s evidence-based reuse decision across Showdown, Jaxcalibur, the PokeJAX paper, the separate Gen 4 `JerJer2465/pokejax` repository, `poke-env`, Metamon, and Pgx. The pinned Showdown checkout remains the Gen 5 oracle. The paper’s throughput figures are reported with their RTX 5090 workload and are not treated as comparable project measurements. G0 specification closure was accepted on 2026-11-03 after review of the pinned target, reuse decision, charter, and mechanics inventory (see the [review report](docs/reviews/G0-specification.md)). Candidate inspections remain bounded to the evidence recorded here; the review did not rerun or extend them.

[`docs/BENCHMARK_CHARTER.md`](docs/BENCHMARK_CHARTER.md) records Task020’s evidence-backed source/version manifest and proposed future benchmark protocol. Task021 added version 1 per-battle result records to the pinned Showdown runner while preserving its smoke summary and trusted transcript. Task022 adds deterministic paired-seed manifests, Task023 adds seed-bound prepared-team handles, Task024 adds explicit single-pair execution, and Task025 adds sequential full-manifest execution with incremental result journaling. Task026 adds recorded-outcome accounting, and Task027 adds observed additive-work cohorts. Rates, uncertainty, scored-suite selection, and performance measurements remain future work; the full JAX battle and learning workloads remain unavailable. G0 specification closure was accepted on 2026-11-03 (see the [review report](docs/reviews/G0-specification.md)); G1 and later implementation gates remain open.

## Current task and next work

Tasks001–031 are accepted and complete. Task003's deterministic request-to-request random-policy baseline passed 15 targeted tests and contributed to the 29-test suite at that milestone; evidence includes the 102-turn default game, fresh-process transcript equality, and a real Shadow Tag unavailable-choice retry. Task004's full suite passed 34/34, and a fresh CLI process replayed the complete 553-event, 102-turn trace with matching outcome. GPU results for Tasks005–011 are historical; Tasks012–017 have CPU-only or deferred GPU evidence as recorded in `docs/TASKS.md`. Task018 audited the public mechanics inventory and G0 coverage boundary without rerunning mechanics tests. Task019 audited the named candidate reuse options without running candidate tests or benchmarks; it retains pinned Showdown as oracle and provisionally continues the custom incremental JAX engine path. Task020 documented the benchmark charter and source/version manifest, distinguishing current evidence, proposed protocol, and unresolved budgets. Task021 added separate per-battle timing, decision/submission/resolution/retry/event counters, and safe failure records; `elapsedMs` is instrumentation and makes no throughput claim. Its result record is trusted engineering metadata, excluded from policy input and the deterministic summary/transcript. Task022 adds explicit paired seed manifests. Task023 adds seed-bound prepared-team handles so packed teams can be reused across two separately initialized battles; each run has fresh mutable battle state, and prepared-run elapsed time excludes preparation. Task024 adds direct single-pair execution. Task025 adds accepted sequential full-manifest execution, incremental safe result journaling, and honest execution/recording counts. Task026 adds offline recorded-outcome summaries; Task027 adds observed additive-work cohorts. Scored suite size, rates and uncertainty, throughput, and concurrency/resume remain unresolved. No benchmark was run. G0 specification closure was accepted on 2026-11-03 for the complete pinned Showdown `gen5randombattle` target at `smogon/pokemon-showdown` commit `1242c1e2d9763ba5b85d33f13aecb89bae19ae3d` (see the [review report](docs/reviews/G0-specification.md)); the mechanics ledger remains partial, and this does not accept G1 or later implementation milestones in the larger [project plan](PROJECT_PLAN.md).

Task028 is accepted: it adds the request-aware eleven-slot singles adapter and the exact pinned sole-Struggle request-parser correction. The affected regression suite passed 97/97 before the final validation-boundary corrections, followed by the adapter suite passing 7/7. See [`docs/TASKS.md`](docs/TASKS.md) for scope and evidence. It does not close G1.

Task029 is accepted: it adds the separate completed-battle trajectory collector documented above. The affected suite passed 55/55, followed by focused tests passing 5/5 after test-only strengthening. See [`docs/TASKS.md`](docs/TASKS.md) for scope and evidence; G1 remains open.

See [`docs/TASKS.md`](docs/TASKS.md) for acceptance state and handoff details.

Task030 is accepted: `move-first-v1` passed its 71-test affected suite and the independent [T030 global checkpoint](docs/reviews/T030-global.md) accepted Tasks021–030. The review baseline is 30 and the next checkpoint is 40; G1 and later milestones remain open.

Task031 is accepted: the named baseline paired-matchup API passed 7/7 focused and 115/115 affected tests. The accepted count is 31; review baseline 30/next 40 and open G1 status are unchanged.
