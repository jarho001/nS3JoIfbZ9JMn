---
title: Reserve output files before long-running battles
author: Jarho001
date: 2026-11-03
branch: main
id: l-e0423be3
scope: repo
status: active
replaces: l-7f54bde1
src: agent
touches:
  - README.md
  - docs/ORCHESTRATION_STATE.json
  - docs/TASKS.md
  - environment/run-baseline-matchup.cjs
  - test/run-baseline-matchup.test.cjs
---

## Decided
`execute` in `environment/run-baseline-matchup.cjs` reserves the requested file with exclusive creation before starting the pair and keeps its descriptor. It writes the completed artifact through that descriptor, syncs and closes it before success. Do this because an ordinary write after a long battle run could replace a file created while computation was in progress.

## Tried and ruled out
No alternative protocol was benchmarked. A write by path after the run was ruled out because it would not reserve the destination while battles are in progress.

## Watch out
Crashes and filesystem errors can leave empty or partial output; file sync does not sync the directory entry. Check file identity before writing or cleanup and leave a replacement path untouched when observed. Concurrent path replacement remains unsupported, including the race between identity check and unlink. Missing output does not mean that no battle work occurred.

In the code:
- `environment/run-baseline-matchup.cjs:99-126` reserves before execution and attempts identity-checked cleanup after run failure.
- `environment/run-baseline-matchup.cjs:52-78` checks the reserved inode before writing or removing it.
- `environment/run-baseline-matchup.cjs:128-145` writes, syncs and closes the completed artifact before reporting success.
- `test/run-baseline-matchup.test.cjs:171-221` checks write/close failures and preserves an observed path replacement.
