---
title: Loop frequent ability updates and share the Start dispatcher to bound compilation
author: Jarho001
date: 2026-11-13
branch: wip/g4-trace-cpu-memory
id: l-21e4fb6d
scope: repo
status: active
src: agent
touches:
  - simulator/shared_engine.py
  - simulator/trace_ability.py
  - simulator/hook_effects.py
---

## Decided
Keep the two-holder Update visit in a JAX loop and factor ordinary ability Start after Trace acquisition at Start sites. Preserve the immediate copied Start effects and the original holder order, RNG and state; do not defer callbacks to a later scheduler boundary.

## Tried and ruled out
The preserved candidate called the ordinary Start dispatcher inside both Trace and ordinary branches and Python-unrolled each Update holder visit. That nested the weather/form Start subtree throughout frequent Update sites; its focused compilation was OOM-killed in an 8 GiB cgroup. Retrying unchanged source would not address that expansion.

## Watch out
A successful focused comparison does not accept full-format G4 or prove all runtime/resource bounds. The resumed host has no cgroup memory cap; retain that distinction when interpreting peak process RSS.

In the code: simulator/shared_engine.py _update uses lax.fori_loop; simulator/trace_ability.py copy separates acquisition from Start; simulator/hook_effects.py ability_start shares ordinary_ability_start. The exact focused test.test_trace_engine comparison passes all 21 native scenarios/26 boundaries, with original assertions, at 7,292,972 KiB peak process RSS. Historical failure is in test-logs/g4-cpu-conformance/schema32-wip.json.
