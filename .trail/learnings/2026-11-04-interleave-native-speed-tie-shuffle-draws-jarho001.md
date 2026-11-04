---
title: Interleave native speed-tie shuffle draws
author: Jarho001
date: 2026-11-04
branch: main
id: l-0172a92e
scope: repo
status: active
src: agent
touches:
  - simulator/neutral_tackle_ties.py
  - reference/neutral-turn-oracle.cjs
---

## Decided
For the pinned restricted Tackle slice, execute each native speed-sort draw at its phase boundary. On ties, preserve the native random(0, 2) result and interleave scheduler draws with attack draws; unequal speeds do not consume a scheduler draw. Native phase names need semantic context from delegated `runAction` choice and `eachEvent` metadata, since invisible Update rows still consume RNG. This yields a 16-call surviving turn and terminal cutoffs at 7 or 13 calls.

## Tried and ruled out
Do not preconsume a tape of initiative draws: it moves RNG past intervening accuracy, critical, and damage calls. Do not infer every Update phase from draw position; inspect the native action choice and event metadata while delegating original methods unchanged.

## Watch out
Native `random(0, 2)` and JAX effective bound 2 are equivalent but represent different range conventions; record the raw native range and label the JAX bound as normalized. Keep `lax.cond` carries type-identical: cast the sampled scheduler value to the state machine's `int32` actor/context type before returning it alongside `uint16` RNG limbs.

## In the code:
`simulator/neutral_tackle_ties.py:_speed_sort_draw` conditionally calls `sample_bounded(..., 2)`; `_run_tie_turn` inserts those draws around the shared Tackle executor and cuts phases on terminal KO. `reference/neutral-turn-oracle.cjs:tieShufflePhase` uses the captured runAction choice, eachEvent event ID, and stack context; `withObservedBattleDraws` delegates random returns and restores wrapped methods.
