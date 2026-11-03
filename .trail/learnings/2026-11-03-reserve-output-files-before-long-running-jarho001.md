---
title: Reserve output files before long-running battles
author: Jarho001
date: 2026-11-03
branch: main
id: l-7f54bde1
scope: repo
status: active
src: agent
touches:
  - README.md
  - docs/ORCHESTRATION_STATE.json
  - docs/TASKS.md
---

## Decided
The `execute` function in `environment/run-baseline-matchup.cjs` reserves the explicit output with exclusive creation before starting expensive work and retains its descriptor. It writes only a validated completed artifact through that descriptor, syncs and closes it, then announces success. Do this because the two battles may run for a long time, and a later path-based write could overwrite a file created while they ran. The empty reservation is visible while the battles run.

## Tried and ruled out
No alternative storage protocol was benchmarked. Opening with exclusive creation directly reserves the requested final path; a later ordinary path write would not guarantee the path stays unclaimed during computation.

## Watch out
A crash can leave the empty reservation, and a write or cleanup error can leave an empty or partial file. File sync does not sync the containing directory entry. Compare the open file identity with the current path before writing or cleanup, and leave a replacement path untouched when a change is observed. Concurrent path replacement remains unsupported, including the race between an identity check and unlink. A failed run may already have performed real work, so a missing final artifact does not imply that no battle started.

In the code:
- `environment/run-baseline-matchup.cjs:83-107` exclusively reserves the destination before invoking the matchup and cleans up ordinary execution failures.
- `environment/run-baseline-matchup.cjs:26-42` compares file identity and only removes the owned reservation when the current path still matches.
- `environment/run-baseline-matchup.cjs:109-129` writes, syncs, closes, and only then reports success.
- `test/run-baseline-matchup.test.cjs:154-222` checks write/sync/close failures and preserves an output path replaced during execution.
