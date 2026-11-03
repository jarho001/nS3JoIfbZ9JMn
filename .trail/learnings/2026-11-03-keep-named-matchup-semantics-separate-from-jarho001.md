---
title: Keep named matchup semantics separate from legacy pair records
author: Jarho001
date: 2026-11-03
branch: main
id: l-ed326324
scope: repo
status: active
replaces: l-30da8511
src: agent
touches:
  - environment/run-pair.cjs
  - environment/summarize-results.cjs
  - environment/paired-seeds.cjs
---

## Decided
Keep the named baseline matchup in a separate wrapper over `generatePair`, opaque `prepareTeam` handles, and actual `runBattle` calls. The legacy pair executor hard-codes the default callback contract and preserves failed or unattempted pair slots; the named API instead needs allowlisted policy versions, explicit role-to-seat mapping, and a result only when both battles complete and clean up. A distinct artifact kind keeps those completion-only semantics out of the established pair and journal schemas.

## Tried and ruled out
No alternatives were experimentally compared. Adding the named policy choices to the legacy pair executor would change its documented default and failure-record boundary, both excluded from this task.

## Watch out
A policy role keeps its own generated RNG seed when its seat changes, including when both roles name the same policy. Pass only the existing recipient-scoped callback input to each policy. Serialized kind, schema, seeds, and result records are reproducibility metadata; they do not attest that execution occurred or remained unmodified.

In the code:
- `environment/run-pair.cjs:71-166` composes the generated pair with the existing default runner and preserves pair-level failure results.
- `environment/paired-seeds.cjs:48-61` defines the canonical role labels, exchanged seats, and identity-specific seed tuples.
- `environment/summarize-results.cjs:87-137` validates the legacy manifest-bound pair kind and rejects a named-matchup envelope.
