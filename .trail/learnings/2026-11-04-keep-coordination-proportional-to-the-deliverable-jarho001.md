---
title: Keep coordination proportional to the deliverable
author: Jarho001
date: 2026-11-04
branch: main
id: l-b754dd51
scope: repo
status: active
src: agent
touches:
  - docs/ORCHESTRATION.md
---

## Decided
Use one concise delegation, checkpoints for unresolved architecture/evidence, and one complete review package plus required corrections. Batch affected checks and bookkeeping so coordination does not dominate implementation.

## Tried and ruled out
Repeated polling, partial reviews, and expanded final bookkeeping prolonged Task036. The user asked to capture a concise rule to prevent that overhead.

## Watch out
This changes coordination, not required validation, independent acceptance, correction limits, or review gates. Do not use it to skip checks.

In the code: docs/ORCHESTRATION.md, Ownership and escalation, records the rule; project-task count and review checkpoints are unchanged.
