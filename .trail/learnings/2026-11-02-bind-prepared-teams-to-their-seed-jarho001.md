---
title: Bind prepared teams to their seed provenance
author: Jarho001
date: 2026-11-02
branch: main
id: l-e9a39748
scope: repo
status: active
src: agent
touches:
  - README.md
  - docs/BENCHMARK_CHARTER.md
  - docs/ORCHESTRATION_STATE.json
  - docs/TASKS.md
  - environment/battle-runner.cjs
---

## Decided
Represent a pre-generated team as an opaque, immutable handle bound to the Showdown API object, format revision, and four-limb team seed. Keep the packed team string and binding in a module-local WeakMap. Reuse only that immutable packed representation across separate BattleStreams; let each stream construct its own mutable battle and Pokémon state. Handles are process-local engineering inputs, not policy observations.

## Tried and ruled out
No alternatives were implemented or experimentally compared. Accepting an arbitrary packed-team string was ruled out by the seed-provenance contract: a string alone cannot establish which configured seed generated it. Regenerating a supplied team to verify its seed would defeat the generate-once behavior. Sharing mutable team or Pokémon objects across runs would couple those battles and is unnecessary when the packed string can be reused.

## Watch out
A handle's visible seed and digest metadata are descriptive only. Serialization drops its WeakMap binding, so another process cannot restore an authentic prepared team from those fields. Resolve both seat handles, API identity, and expected seeds before creating a stream. Never pass handles, packed strings, or hashes into policy callbacks.

In the code:
- `environment/prepared-teams.cjs` validates and copies each seed, calls `Teams.generate` and `Teams.pack` once, and stores authentic data in a module-private WeakMap.
- `environment/battle-runner.cjs` validates both seat handles before stream creation and reuses their packed strings while retaining the normal generated-team path when no handles are supplied.
- `test/prepared-teams.test.cjs` covers real paired battles, generation/packing call counts, seat-swapped team writes, and rejection of copied or mismatched handles.
