---
title: Static duel capacity with per-lane budgets
author: "/root/task035"
date: 2026-11-03
branch: main
id: l-582fd216
scope: repo
status: active
src: agent
touches:
  - simulator/neutral_duel.py
  - reference/neutral-turn-oracle.cjs
  - test/test_neutral_duel.py
  - test/neutral-duel-oracle.test.cjs
---

## Decided
Keep the fixed trace capacity static and the per-lane execution budget dynamic. A common `max_turns` gives every vmapped result the same PyTree shape; only successful `STATUS_OK` transitions occupy the contiguous valid prefix. Budget exhaustion preserves the last successful state and its last transition diagnostic. A later call may resume from that state and observe an unsupported next request only if its budget includes the attempt. Terminal completion on the last allowed slot takes precedence over budget exhaustion.

## Tried and ruled out
Giving vmapped lanes different static capacities changes output shapes and cannot express heterogeneous budgets in one compiled batch. Running the turn kernel after a lane stops and masking only its count risks treating padding as experience or leaking a fabricated request/outcome. Treating budget exhaustion as a battle result also invents evidence that the native battle never supplied.

## Watch out
A valid zero budget returns an unchanged active state; an initially terminal state remains completed with zero turns. A representable runtime budget outside the static capacity returns a distinct invalid-budget diagnostic without advancing. Padding enum values may coincide with real statuses, so consume no trace field unless that row's validity mask is true. Both recipient views are trusted combined trace data: select one valid audience view before actor input.

## In the code:
- `simulator/neutral_duel.py:L122-L143` builds zeroed masked rows; `:146-L241` implements static scan capacity, dynamic budget, status handling, and result construction.
- `reference/neutral-turn-oracle.cjs:L309-L316` preserves default mode while selecting explicit `--duels`; `:352-L396` records bounded complete native battles from normal requests.
- `test/test_neutral_duel.py:L208-L250` compares every full-duel row; `:252-L299` checks resume and PP exhaustion; `:355-L413` checks mixed budgets and vmapped lane independence.
