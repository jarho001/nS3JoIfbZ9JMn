---
title: Keep staged requests aligned with their episode
author: "/root/task037"
date: 2026-11-04
branch: main
id: l-75b4d172
scope: repo
status: active
replaces: l-8556df54
src: agent
touches:
  - "simulator/request_session.py#_idle_transition"
  - "simulator/request_session.py#project_submission_view"
---

## Decided
Current requests use `result.session.battle` and its local episode. Previous turn events use `result.before` and `result.transition` in a separate recipient projection.

## Tried and ruled out
Pairing the next token with the before-state gave stale HP/PP. Full transition-kernel calls in idle branches duplicated tracing: the initial focused suite took 402s; the corrected suite took 62s. This is a validation runtime observation, not a benchmark.

## Watch out
Empty transitions need the matching diagnostic status. Only the second valid commit resolves through the kernel. The local episode remains distinct from native Showdown rqid.

In the code:
`simulator/request_session.py`: `_idle_transition` and `project_submission_view`.
