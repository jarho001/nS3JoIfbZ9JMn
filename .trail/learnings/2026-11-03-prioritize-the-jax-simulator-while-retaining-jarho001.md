---
title: Prioritize the JAX simulator while retaining parity contracts
author: Jarho001
date: 2026-11-03
branch: main
id: l-c6cf4224
scope: repo
status: active
src: agent
touches:
  - PROJECT_PLAN.md
  - README.md
  - docs/ORCHESTRATION_STATE.json
  - docs/SCHEDULED_PROMPT.md
  - docs/TASKS.md
---

## Decided
Prioritize integrated, progressively playable JAX battle execution and Showdown parity validation. G3 simulator work may proceed while G1 reference-environment acceptance and G2 policy learning remain open. Preserve Showdown as semantic oracle, source of data and generated teams, and differential harness. Keep request/legal-action semantics, simultaneous choices, player-visible events/history, and outcomes in the minimal contract from the start. Defer PPO, tournament expansion, and additional baseline/reporting infrastructure unless they directly enable simulator semantics or parity validation. Compile and batch representative restricted slices early; measure JIT/vmap and throughput without requiring training. Use increasingly complete parity-tested playable battles as progress evidence rather than task count.

## Tried and ruled out
Continuing to prioritize learning and tournament/reporting work before integrated simulator execution does not match the immediate goal of replacing the battle engine and leaves the central mechanics and execution uncertainty unresolved.

## Watch out
G1 remains open and none of its acceptance criteria are waived; this decision is only about work sequencing. G2 learning remains open. A restricted development variant must declare supported mechanics and cannot imply full-format support. Validate on CPU first; require GPU evidence only when the task's acceptance criteria or a concrete device-specific property needs it, not for every slice.

In the code: Decision recorded 2026-11-03 in PROJECT_PLAN.md, docs/TASKS.md, docs/SCHEDULED_PROMPT.md, docs/ORCHESTRATION_STATE.json, and README.md. No source implementation or numbered task was started; accepted count remains 33, active task is null, review baseline is 30, and the next regular checkpoint remains 40. The next candidate is one bounded runnable JAX turn slice for a named restricted variant with ordinary move cases and exact RNG/state/request parity, subject to feasibility scoping at task kickoff.
