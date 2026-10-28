---
title: Neutral Showdown event identity can wrap large values
author: Developer
date: 2026-10-28
id: l-36960035
scope: repo
status: active
src: agent
touches:
  - mechanics/neutral_damage.py
  - reference/neutral-damage-oracle.cjs
  - test/test_neutral_damage.py
---

## Decided
Model the neutral `WeatherModifyDamage` event in the damage composition. Even with no event handlers, `runEvent` applies `Battle.modify(value, 1)` to integer relays, which is equivalent to keeping the low 20 bits before the randomizer.

## Tried and ruled out
Delegating the real `modifyDamage` and `Battle.randomizer` while assuming the core-plus-two value passed directly through the neutral event failed for large values: 85,822,016 became 887,360 before the randomizer. The pinned `Battle.modify` uses uint32 fixed-point intermediates, so its nominal 1x operation wraps. Bypassing the event or narrowing the accepted base-core domain would hide pinned behavior.

## Watch out
Keep this as the specific neutral event step, not a general modifier engine. The randomizer output is still an intermediate; Gen 5 minimum-one and final 16-bit truncation can change final `getDamage` output. Task008 separately checks randomizer product wrap outside the neutral event's bounded output range.

## In the code
`mechanics/neutral_damage.py` composes core, `+2`, the low-20-bit neutral event step, and an explicit roll. `reference/neutral-damage-oracle.cjs` transparently delegates the pinned event and randomizer, and `test/test_neutral_damage.py` checks the event boundaries and output distinction.
