---
title: Keep the random damage roll explicit and uint32-wrapped
author: Developer
date: 2026-10-28
id: l-5740fafd
scope: repo
status: active
src: agent
touches:
  - mechanics/damage_roll.py
  - reference/damage-roll-oracle.cjs
  - test/test_damage_roll.py
---

## Decided
Pass the roll into the JAX kernel and model the pinned Battle.randomizer arithmetic exactly: truncate the product to unsigned 32 bits, then truncate the division by 100. The useful precondition is base damage in int32 range and roll 0..15; the unsigned product can wrap even though JavaScript represents the intermediate exactly.

## Tried and ruled out
Widening or saturating the product changes high-damage results. Adding a minimum-one clamp here also changes Battle.randomizer, which only performs the random factor arithmetic.

## Watch out
Keep PRNG state outside this kernel. The controlled oracle replaces only Battle.random to exercise actual Battle.randomizer and restores it in finally. The live oracle records original seeded draws after Battle initialization; those are not necessarily the PRNG's first draws. Do not mistake the logged sequence for raw seed expansion.

## In the code
mechanics/damage_roll.py apply_damage_roll; reference/damage-roll-oracle.cjs controlledGrid and seededDraws; test/test_damage_roll.py. Pinned sources are sim/battle.ts:2407-2410 and sim/dex.ts:364-369.
