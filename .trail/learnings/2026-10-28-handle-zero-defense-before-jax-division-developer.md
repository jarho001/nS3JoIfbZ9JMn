---
title: Handle zero Defense before JAX division
author: Developer
date: 2026-10-28
id: l-6097610f
scope: repo
status: active
src: agent
touches:
  - mechanics/base_damage.py
  - reference/base-damage-oracle.cjs
  - test/test_base_damage.py
---

## Decided
For the pinned getDamage core, Defense zero yields zero: the division creates Infinity for a positive numerator or NaN for a zero numerator, and Battle.trunc/Dex.trunc converts either to unsigned 32-bit zero. In JAX, select divisor 1 when Defense is zero before dividing, then select the zero result afterward.

## Tried and ruled out
Putting a zero-Defense branch only after a division still asks JAX to evaluate division by zero. Saturating the denominator or returning a large damage value would not match the pinned unsigned conversion.

## Watch out
This is the arithmetic core before modifyDamage and the later +2, and its Attack/Defense inputs are effective values from the caller. Keep the zero-denominator rows separately labeled as controlled probes; ordinary Pokemon cases preserve initialized stored stats.

## In the code
mechanics/base_damage.py base_damage_core; reference/base-damage-oracle.cjs syntheticCases/invokeCore; test/test_base_damage.py test_zero_defense_matches_pinned_unsigned_conversion_without_invalid_division. Upstream paths: sim/battle-actions.ts:1772-1775 and sim/dex.ts:364-369.
