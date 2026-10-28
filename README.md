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
```

The committed JSON files under `reference/fixtures/` are canonical semantic fixtures. The format export comes from Showdown's resolved Dex/rule table. The smoke artifact contains both complete generated teams, private initial requests, and raw omniscient update messages. Those updates can include `|split|` markers and private `SECRET` lines; this trusted oracle artifact is not a public player view or a policy training record. The harness validates Gen 5, singles, no Team Preview, team sizes, and that each player's request names match its submitted team. Wall-clock protocol lines are excluded. Provenance reports the upstream package lock hash and Node/npm versions separately from semantic fixture hashes.

## Current task and next work

Task 001 establishes a pinned oracle only. It does not complete G0 or G1 of the larger [project plan](PROJECT_PLAN.md). The next bounded tasks are:

1. Define the request/player-view contract and add visibility fixtures, projecting each player's view from the trusted raw split protocol.
2. Build a request-to-request baseline battle runner against Showdown.
3. Add a differential trace harness for a small JAX mechanics slice.
4. Implement and verify the first JAX mechanics slice.

See [`docs/TASKS.md`](docs/TASKS.md) for acceptance state and handoff details.
