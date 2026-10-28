---
title: Keep Gen 5 nature overflow and zero output explicit
author: Developer
date: 2026-10-28
id: l-d5aa2359
scope: repo
status: active
src: agent
touches:
  - mechanics/stats.py
  - reference/stats-oracle.cjs
  - test/test_stats.py
  - docs/TASKS.md
---

## Decided
For the pinned `gen5randombattle` path, honor `overflowstatmod` only when the effective format has it; the target rule table does not. Preserve Gen 5's unsigned low-16-bit truncation on nature products, and keep Shedinja's `maxHP: 1` as an explicit species override.

## Tried and ruled out
Assuming the optional 595/728 stat caps apply to every Gen 5 format was incorrect: the actual target format reports `overflowstatmod: false`. Assuming a positive base stat always produces a positive stored stat also fails within the documented synthetic input domain.

## Watch out
With base Attack 248, IV 31, EV 255, level 100, the pre-nature stat is 595: `595*110=65450` stays below 65536 and yields 654 after division. Base 249, IV 30 gives 596: `596*110=65560` wraps to 24, then yields stored stat 0. Task007 must reconcile this valid synthetic zero with Task005's positive stored-stat precondition before composing the kernels.

## In the code
`mechanics/stats.py:derive_stored_stats`, `reference/stats-oracle.cjs:generateSyntheticNatureCases`, `test/test_stats.py:test_real_stat_modify_nature_wrap_thresholds`, and `docs/TASKS.md` Task007.
