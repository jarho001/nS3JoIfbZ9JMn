---
title: Establish matched Showdown versus GPU performance before further benchmarking
author: Jarho001
date: 2026-11-06
branch: main
id: l-6cf79acb
scope: repo
status: active
src: agent
touches:
  - docs/G3_TESTING_PROFILING.md
  - docs/ORCHESTRATION_STATE.json
  - reference/shared-engine-cases.cjs
  - simulator/shared_engine.py
  - test/shared_engine_support.py
  - test/test_shared_engine_lifecycle.py
---

## Decided
The user stopped the extensive JAX CPU-versus-GPU profiling sweep on 2026-11-06. Do not run another benchmark until a matched uninstrumented legacy Showdown-versus-GPU benchmark is established. The performance objective is speed relative to legacy Showdown. Preserve intermediate evidence and correctness repairs.

## Tried and ruled out
The extensive sweep measured the new JAX simulator on CPU and GPU. Those measurements can describe JAX execution but do not establish the intended Showdown-relative speedup. Instrumented native correctness-oracle execution is unsuitable as the native performance baseline.

## Watch out
Do not resume the original sweep from a stale handoff or infer performance from correctness-campaign wall time. Agree equivalent workload, useful work, observable outputs, resources, transfer/retention costs, warmup/compilation and synchronized GPU completion before timing. G3 remains open.

In the code: docs/G3_TESTING_PROFILING.md records the overriding stop; docs/ORCHESTRATION_STATE.json records the preserved checkpoints and benchmark gate; docs/G3_TESTING_PROFILING_RESULTS.md links intermediate measurements and correctness evidence.
