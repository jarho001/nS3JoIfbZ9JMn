---
title: Project each player view before exposing Showdown updates
author: Developer
date: 2026-10-28
id: l-9b34d84e
scope: repo
status: active
src: agent
touches:
  - reference/oracle.cjs
  - reference/fixtures/smoke-v1.json
  - docs/TASKS.md
---

## Decided
Keep raw Showdown update transcripts as trusted oracle data. Project each player's public and private information into separate views before any actor can consume it.

## Tried and ruled out
Treating the BattleStream `update` channel as inherently public is unsafe: split messages carry a player's `SECRET` lines alongside public lines. Retaining the complete update is useful for oracle evidence, but it is not a player observation.

## Watch out
Exact HP and other private details may occur in split `SECRET` records; choice requests are delivered through side-specific updates. A projection must respect the recipient for both channels and must not route raw transcripts into policy features.

In the code:
- `reference/oracle.cjs` stores raw messages under `rawUpdateEvents` and labels them as omniscient trusted data.
- `reference/fixtures/smoke-v1.json` contains those raw messages for oracle validation.
- Task 002 in `docs/TASKS.md` defines visibility projection and paired-history acceptance tests.
