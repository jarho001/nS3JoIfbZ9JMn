# Gen 5 Random Battle research foundation

This repository is beginning a JAX self-play project against Pokémon Showdown's `[Gen 5] Random Battle` (`gen5randombattle`). The current foundation pins an upstream Showdown revision and provides a deterministic oracle fixture for its format rules, team generation, and first player requests.

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
  --output trusted-battle.json
node --test test/battle-runner.test.cjs
node --test test/*.test.cjs
```

All five seeds are separate four-limb Gen 5 seeds. The summary reports the pinned commit, seeds, result, turn count, packet/decision counts, and move/switch counts. `--output` writes a versioned trusted transcript containing complete private requests, split update branches, choice attempts, terminal outcome, and end metadata. Treat it as omniscient oracle data; the runner never passes that transcript to a policy.

The two random policies receive a frozen `{ history, request, random }` callback input. `history` contains only that side's `projectPackets` output and its own submitted-choice resolutions; `request` is the current request addressed to that side; `random` is that policy's independent seeded RNG. The runner submits each choice on its own request and relies on Showdown's choice commitment behavior to hold simultaneous choices. Candidate moves and switches come from the request payload. `trapped` is honored when disclosed; `maybeTrapped` is not resolved from simulator state. Wait requests create no action. Unavailable-choice errors and corrected requests stay inside a bounded decision episode.

This is a deliberately small baseline. It does not provide a learned policy, observation features, a full game-memory contract, or JAX mechanics.

## Current task and next work

Task 003's deterministic request-to-request random-policy baseline is accepted and complete. Its targeted suite passed 15 tests; the full suite passed 29, including the 102-turn default game, fresh-process transcript equality, and a real Shadow Tag unavailable-choice retry. The Task002/003 foundation and request-episode learning are committed together. This work is a portion of G0/G1, not completion of either milestone in the larger [project plan](PROJECT_PLAN.md). The next bounded task is Task 004, a reference protocol trace and replay harness described in [`docs/TASKS.md`](docs/TASKS.md). Later work includes a first JAX mechanics slice.

See [`docs/TASKS.md`](docs/TASKS.md) for acceptance state and handoff details.
