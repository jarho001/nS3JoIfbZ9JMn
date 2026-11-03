---
title: Observe native Gen 5 turn draws without replacing RNG
author: Jarho001
date: 2026-11-03
branch: main
id: l-a0f44bbc
scope: repo
status: active
replaces: l-3c4d8fd1
src: agent
touches:
  - reference/neutral-turn-oracle.cjs
  - simulator/gen5_rng.py
  - simulator/neutral_tackle.py
---

## Decided
For exact seeded turn parity, observe PRNG.prototype.random after both normal move requests exist, record bounds, runtime contexts and four-limb pre/post states, and return the original result unchanged. Implement JAX state evolution with byte limbs and bounded sampling with split 16-bit products so x64 is unnecessary. Compare from the captured request-boundary seed, because initialization may already have consumed draws.

## Tried and ruled out
Observing only Battle.random missed the critical draw delegated through randomChance. An initial reading of raw Tackle data also missed the move constructor's default critical ratio of one; runtime active-move data and call stacks resolved that mistake without a move override. Replacing RNG values or replaying a tape would weaken the native seeded-turn evidence.

## Watch out
Keep theoretical damage and RNG traces trusted. Native owner-visible damage is HP actually lost, clamped to starting HP; opponents receive only the projected percentage condition and no exact numeric damage. Compare draw context as well as numeric values so an extra or reordered call cannot pass unnoticed.

In the code:
- reference/neutral-turn-oracle.cjs:204-231 observes the original PRNG method and restores its descriptor; :233-267 captures actual Battle.makeChoices transitions.
- simulator/gen5_rng.py:36-76 implements four-limb evolution and bounded draws.
- simulator/neutral_tackle.py:361-485 separates the trusted transition from recipient-specific projections.
- Pinned sim/dex-moves.ts:490 supplies ordinary critRatio=1; the tested Tackle draw order is accuracy100, critical16, damage16.
