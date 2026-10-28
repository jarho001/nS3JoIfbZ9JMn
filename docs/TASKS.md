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

**Scope:** Define how the pinned Showdown protocol becomes one policy-safe view per player. This task does not implement complete battles, a learner, or the JAX simulator.

**Acceptance criteria:**

- Convert `|split|PLAYER` sections into separate public events and the appropriate player's private events; never expose one player's `SECRET` lines or private requests to the other player's view.
- Add protocol fixtures covering public-only events, player-specific `SECRET`/`PUBLIC` pairs, private `sideupdate` requests, and clock-line filtering.
- Add paired-history tests showing that histories indistinguishable to one player produce identical projected views even when concealed state or the other player's private request differs.
- Preserve the raw transcript only as a trusted oracle fixture; name the projected view separately and document its information boundary.
- Keep generated artifacts deterministic and pass the focused Node test suite without adding full battle decision logic.

## Later bounded tasks — G0/G1 portions

1. **Request-to-request baseline battles:** implement legal random/simple policies and run complete battles by reading exact private requests and submitting choices through `BattleStream`.
2. **Differential trace harness:** serialize reference transitions and normalized event snapshots for deterministic paired testing.
3. **First JAX mechanics slice:** implement selected arithmetic and a normal move-resolution path, with parity fixtures through the trace harness.

Each task needs its own acceptance criteria and review. The current task establishes none of the player-view, full-battle, or JAX parity contracts.
