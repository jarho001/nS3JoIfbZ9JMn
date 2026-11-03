---
title: Offline outcome reports cannot attest who ran the battles
author: Jarho001
date: 2026-11-03
branch: main
id: l-86fd14c5
scope: repo
status: active
src: agent
touches:
  - README.md
  - docs/BENCHMARK_CHARTER.md
  - docs/ORCHESTRATION_STATE.json
  - docs/TASKS.md
  - environment/summarize-results.cjs
  - test/summarize-results.test.cjs
---

## Decided
Treat an offline report over serialized pair records as an accounting view only. Recompute expected manifest identity and validate pair/run IDs, seed assignments, seat exchange, terminal outcomes, and failure topology, then report outcomes by seat and policy identity. Do not claim the records came from the runner or that a complete recorded prefix proves the workload completed successfully.

## Tried and ruled out
Requiring the runner module's private error brand is impossible after a JSONL record crosses a process boundary: the private WeakSet is process-local and JSON serialization preserves no proof of origin. Treating valid kind/schema fields or matching deterministic IDs as execution attestation would let a fabricated record appear genuine.

## Watch out
A manifest or result journal can be edited or fabricated by its producer. Structural validation protects accounting against mismatched identities and malformed outcome topology; it does not cryptographically authenticate execution. Missing journal suffixes may hide pairs that ran but could not be written, so call them unrecorded rather than unattempted.

In the code:
- `environment/summarize-results.cjs` rebuilds the canonical manifest and only counts a contiguous prefix whose identities and run topology match it.
- `test/summarize-results.test.cjs` exercises forged identity rejection and real pinned runner success and failure records.
