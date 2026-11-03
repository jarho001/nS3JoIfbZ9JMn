---
title: Observe native Gen 5 turn draws without replacing RNG
author: Jarho001
date: 2026-11-03
branch: main
id: l-e0234c5a
scope: repo
status: active
src: agent
---

## Decided
For pinned Gen 5 turn parity, instrument `PRNG.prototype.random` only after the normal request boundary has been captured. Record each actual bound, runtime context, and four-limb seed before and after, then delegate and return the exact original result. In this slice each executed Tackle records accuracy `random(100)`, critical `random(16)`, then damage `random(16)`. See `reference/neutral-turn-oracle.cjs#withObservedBattleDraws` (L204-L231) and `#captureTurn` (L233-L267). Reproduce the LCG in JAX32 with byte-limb state arithmetic and split-half bounded multiplication; see `simulator/gen5_rng.py#next_state` (L36-L63) and `#sample_bounded` (L66-L76).

## Tried and ruled out
Wrapping only `Battle.random` misses calls made by native accuracy and critical helpers through the PRNG prototype. Replacing returned values or replaying a prerecorded tape would no longer test the pinned seeded turn. Assuming the constructor seed is the first turn's RNG state also ignores draws that may occur during initialization; begin comparison from the live normal-request state.

## Watch out
Keep trusted transition data separate from actor projections. The damaged owner can observe only HP actually lost, clamped to starting HP; an opponent's exact theoretical damage is hidden. Project each recipient's own HP exactly and opposing HP through the pinned percentage condition. `simulator/neutral_tackle.py#transition` (L361-L424) retains trusted event data, while `#project_view` (L427-L485) applies audience-specific visibility.

## Why
The native `PRNG.prototype.random` boundary sees all bounded calls while allowing the pinned implementation to advance normally. Capturing pre/post four-limb state proves both order and consumption, while keeping original values proves the oracle remains seeded Showdown. Recipient projections preserve observable critical, damage, faint, turn, and HP-condition semantics without exposing privileged internal state.
