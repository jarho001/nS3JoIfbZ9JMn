---
title: Capture trajectory observations from policy callbacks
author: Developer
date: 2026-11-03
branch: main
id: l-b9bd63dc
scope: repo
status: active
replaces: l-11d377e9
src: agent
touches:
  - environment/battle-runner.cjs
  - environment/protocol.cjs
---

## Decided
Use each recipient's live policy callback input as the authority for a trajectory observation. Snapshot the current request and new own-history records before policy selection; attach terminal targets only after completed-run validation. In memory, deltas reconstruct the exact callback history. JSON encoding omits optional `undefined` properties, so parsed deltas reconstruct that history's JSON projection. A selected command does not establish submission, acceptance, or execution.

## Tried and ruled out
Replaying the trusted transcript cannot establish an actor's exact callback history. It includes both players' private requests, split branches, trusted resolutions, and end metadata; the runner instead projects each packet for one recipient and interleaves that actor's own requests and choices.

## Watch out
- The combined artifact is post-game private data, not a policy observation or execution attestation.
- Preserve pre-action snapshots and per-player append-only deltas, including non-actionable events between decisions.
- Resolve rejected commands only from the current `requestEpisode`; a repeated request after progress is a new sample.

## In the code
- `environment/protocol.cjs` creates recipient-scoped packet records.
- `environment/battle-runner.cjs` supplies projected callback inputs and separately retains trusted records.
- `environment/decision-trajectories.cjs` captures deltas before selection and adds targets after completion.
- `test/decision-trajectories.test.cjs` checks in-memory reconstruction, JSON projection, privacy, targets, and unchanged choices.
