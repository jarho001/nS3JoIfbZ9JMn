---
title: Choose safe operands before integer division
author: Developer
date: 2026-10-28
id: l-581b4288
scope: repo
status: active
src: agent
touches:
  - mechanics/stat_stage.py
  - test/test_stat_stage.py
---

## Decided
Select the numerator and a strictly positive denominator before integer division in the JAX stat-stage kernel. This keeps both the eager and traced computation defined for every clamped stage.

## Tried and ruled out
Computing positive and negative branch quotients separately and selecting them with `jnp.where` leaves the unused branch evaluated; at stage +2 the negative denominator is zero. `where` over completed integer divisions is not a safe guard.

## Watch out
With JAX, select safe operands first, then divide once. Here clipping to [-6, 6] and choosing the denominator before division bounds all int32 intermediates; verify boundary stages under eager, JIT, and vmap.

## In the code
`mechanics/stat_stage.py:scale_stat_stage` and `test/test_stat_stage.py:test_all_stages_for_stats_through_1024_match_live_pinned_oracle`.
