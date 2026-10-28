---
title: Keep zero in the stat composition domain
author: Developer
date: 2026-10-28
id: l-7178b739
scope: repo
status: active
src: agent
touches:
  - mechanics/attack.py
  - mechanics/stat_stage.py
  - mechanics/stats.py
  - reference/attack-oracle.cjs
  - test/test_attack.py
---

## Decided
Keep the stage scaler's documented input domain at [0, 65535] and compose stored-stat derivation directly into Attack stage scaling. In pinned Gen 5 randombattle, real Battle.statModify can produce stored Attack 0 when the pre-nature value is 596 and the 110 product wraps to 24 before division. Excluding zero would make this valid arithmetic path impossible to compose.

## Tried and ruled out
A positive-only stage precondition was insufficient once the real Gen 5 nature truncation was included. Narrowing the composition to ordinary Pokemon would hide a valid simulator result instead of representing it.

## Watch out
The ordinary oracle cases must preserve initialized Pokemon storedStats. Keep the zero case separately labeled: derive zero with real Battle.statModify, then inject it only to test real calculateStat stage arithmetic. The Attack kernel covers neutral ModifyBoost, no Wonder Room, and modifier 1; it is not effective Attack after abilities, items, or status.

## In the code
mechanics/stats.py derive_stored_stats; mechanics/stat_stage.py scale_stat_stage; mechanics/attack.py derive_attack; reference/attack-oracle.cjs and test/test_attack.py. The committed attack-v1 fixture captures this boundary.
