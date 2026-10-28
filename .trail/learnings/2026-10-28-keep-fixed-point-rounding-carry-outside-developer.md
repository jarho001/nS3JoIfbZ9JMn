---
title: Keep fixed-point rounding carry outside uint32 wrap
author: Developer
date: 2026-10-28
id: l-7a4699bd
scope: repo
status: active
src: agent
touches:
  - mechanics/fixedpoint.py
  - reference/fixedpoint-oracle.cjs
  - test/test_fixedpoint.py
---

## Decided
For one encoded Gen 5 modifier, wrap `value * modifier` to uint32 first, then apply Showdown's `+2047` as an ordinary JavaScript number before floor-dividing by 4096. The equivalent integer kernel adds one only when the wrapped product's remainder is greater than 2048.

## Tried and ruled out
Wrapping the rounding constant together with the product loses the max-product carry: `65537 * 65535` wraps to `2**32 - 1`, and the pinned result is 1,048,576. The addition after `tr(product)` is outside that uint32 conversion.

## Watch out
Exact remainder 2048 rounds down; 2049 rounds up. The result can be `2**20`, so do not apply a second mask. This is one modifier with an already encoded integer, not fraction encoding or chained event modifiers.

## In the code
`mechanics/fixedpoint.py` uses quotient/remainder arithmetic. `reference/fixedpoint-oracle.cjs` calls pinned `Battle.modify` directly, and `test/test_fixedpoint.py` includes the UINT32_MAX product carry and tie boundaries.
