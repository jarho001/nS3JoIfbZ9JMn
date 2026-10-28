---
title: Track choices by request episode
author: Developer
date: 2026-10-28
id: l-b1c64d9f
scope: repo
status: active
src: agent
touches:
  - environment/battle-runner.cjs
  - environment/random-policy.cjs
  - test/battle-runner.test.cjs
---

Decided
Deduplicate a choice only while that same request is still submitted and unresolved. Clear the pending submission when visible progress resolves it, then allow an identical request payload to represent a new decision. Keep unavailable-choice retry budgets and rejected-action exclusions within the current decision episode, and reset those exclusions only after protocol progress.

Tried
A guard that skipped any repeated request payload after any prior submission silently skipped a valid later choice when the same JSON request recurred. The runner kept waiting for a packet after it had sent no command. A larger timeout would mask this coordinator bug instead of resolving the missing submission. The corrected default seeded battle completes in about three seconds under the existing 60-second bound.

Watch out
Request JSON equality alone does not identify a pending decision. An unavailable-choice error and corrected request belong to the same retry episode; ordinary public progress resolves the pending choice and begins a new episode. Never reuse an old retry exclusion in a later episode. Preserve complete split branches until visibility projection, since removing clock lines before split parsing can shift the two branch positions.

In the code
`environment/battle-runner.cjs` tracks pending attempts and episode transitions; `environment/random-policy.cjs` filters only rejection evidence from the current episode; `test/battle-runner.test.cjs` covers identical repeated requests, corrected requests, retry bounds, and live seeded battles.
