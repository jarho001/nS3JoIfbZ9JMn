---
title: Keep routine validation short and history out of worker context
author: Jarho001
date: 2026-11-04
branch: main
id: l-d9ef2039
scope: repo
status: active
src: agent
touches:
  - docs/ORCHESTRATION.md
  - docs/SCHEDULED_PROMPT.md
  - ".codex/agents/implementer.toml"
---

## Decided
Use Luna/medium and a fresh scoped worker packet. Ordinary development runs only fast affected checks: 30 seconds per command and 90 seconds total. Defer long checks to explicit user-run sessions and report exact commands/source identity. Keep the current handoff/state compact, with immutable archives for completed detail.

## Tried and ruled out
Defaulting every bounded owner to max effort, full-history forks, replaying prior suites and duplicating acceptance narratives did not justify the resulting latency. Recent recorded legacy suites passed; new checks, native comparisons and review drove corrections. Lower effort is a new default to evaluate, not a measured speedup claim.

## Watch out
Preserve old tests and history; defer execution rather than deleting evidence. Missing genuinely required acceptance evidence stays pending. Do not claim deferred checks passed, and do not silently fall back to a max-effort runtime route.

In the code:
`docs/ORCHESTRATION.md` defines budgets and archived-history reads; `docs/SCHEDULED_PROMPT.md` applies them to routine entries; `.codex/agents/implementer.toml` sets medium. `docs/TASKS_ARCHIVE_001_037.md` and `docs/ORCHESTRATION_STATE_ARCHIVE_037.json` preserve pre-compaction bytes.
