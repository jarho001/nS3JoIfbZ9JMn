---
title: Native handler sorting can consume RNG before a skipped effect
author: Jarho001
date: 2026-11-06
branch: main
id: l-50b47586
scope: repo
status: active
src: agent
touches:
  - docs/ORCHESTRATION_STATE.json
  - reference/shared-engine-cases.cjs
  - simulator/shared_engine.py
  - test/shared_engine_support.py
  - test/test_shared_engine_lifecycle.py
---

## Decided
Preserve native callback collection and ordering before deciding whether to execute an effect. A callback that never heals or damages can still change the RNG stream. Refresh an incoming Pokemon's effective Speed at the native queue insertion boundary.

## Tried and ruled out
Filtering fainted Leftovers holders before residual sorting omitted a native shuffle draw. Extending only the item mask left the same bug for burn/Toxic callbacks; direct native family probes caught both. Refreshing Speed only at ordinary turns/residuals left a returning paralyzed reserve at cached base Speed after forced replacement. Minimized native regressions distinguish these paths from the older passing corpus.

## Watch out
Native fainting clears volatiles, but status and held item callbacks still enter residual sorting before fieldEvent skips their fainted holder. checkFainted changes the status to fnt only after the residual action when a replacement is needed; terminal statuses can remain. Trusted Pokemon.setStatus installation differs from an attempted status move: an immune fixture installation is silently rejected. Do not discard draws or change expected histories to accommodate these paths.

In the code: simulator/shared_engine.py (_switch, _residual); reference/shared-engine-cases.cjs cases paralyzed-forced-return, fainted-leftovers-handler-sort, fainted-brn/tox-handler-sort, trusted-toxic-immune-*; test/shared_engine_support.py initialize_case. Native source: battle.ts fieldEvent, resolveFaints and checkFainted. Evidence: test-logs/g3-testing-profiling/*-before.log and current campaign summary; timing evidence from before the status-family repair is superseded.
