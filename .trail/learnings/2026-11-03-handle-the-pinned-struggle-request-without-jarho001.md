---
title: Handle the pinned Struggle request without PP
author: Jarho001
date: 2026-11-03
branch: main
id: l-f186e3b6
scope: repo
status: active
src: agent
touches:
  - README.md
  - docs/ORCHESTRATION_STATE.json
  - docs/TASKS.md
  - environment/random-policy.cjs
  - test/battle-runner.test.cjs
---

## Decided
Accept the pinned Gen 5 Showdown sole-Struggle request row with `move: "Struggle"`, `id: "struggle"`, `disabled: false`, and no `pp` as the `move 1` command. Keep it on the explicit sole-Struggle action slot in the eleven-slot adapter. The runner validates requests before invoking a policy, so this request shape must be recognized in the shared request parser as well as mapped by the adapter.

## Tried and ruled out
Normalizing the row only in the policy adapter does not work: `runBattle.submit` calls `candidateActions` before invoking the policy, so the unmodified pinned request fails before the adapter can run. Broader runner special-casing would duplicate request parsing; the narrow parser exception accepts only the source-confirmed sole-Struggle shape and retains finite PP validation for ordinary move rows.

## Watch out
Pinned `Pokemon.getMoveRequestData()` synthesizes Struggle after no ordinary move has PP. The row carries `disabled: false` but omits `pp`; this differs from reduced locked and recharge rows, which omit both fields. Keep malformed variants rejected, submit the decoded Showdown command `move 1`, and scope retry exclusions to the current request episode. A serialized row is still only the current own request, not a reusable action authorization.

In the code:
- `reference/vendor/pokemon-showdown/sim/pokemon.ts` synthesizes the Struggle row in `getMoveRequestData()`.
- `environment/random-policy.cjs` accepts the exact generated sole-Struggle shape in `candidateActions`.
- `environment/action-space.cjs` maps that sole row to slot 10 and decodes it as `move 1`.
- `test/action-space.test.cjs` checks actual locked, recharge, and zero-PP request generation and pinned `Side.choose` acceptance.
- `test/battle-runner.test.cjs` checks adapter choices through the runner's real unavailable trapped-switch retry.
