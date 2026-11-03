---
title: Keep named matchup semantics out of legacy pair records
author: Jarho001
date: 2026-11-03
branch: main
id: l-30da8511
scope: repo
status: active
src: agent
touches:
  - environment/run-pair.cjs
  - environment/summarize-results.cjs
  - environment/paired-seeds.cjs
---

`environment/run-pair.cjs:71-166` is the legacy default pair executor: it selects the canonical `policyA`/`policyB` roles and seed tuples from `generatePair`, but calls `runBattle` without caller-selected policy callbacks and preserves failed/partially attempted pair records. `environment/summarize-results.cjs:87-137` validates that established pair-record kind and its manifest-bound run slots. A named baseline API has a different contract: allowlisted policy versions, exact role-to-seat callback mapping, and a result only when both battles complete and clean up.

Keep that contract in a separate wrapper over `generatePair`, `prepareTeam`, and `runBattle`. Extending the legacy executor would either alter its output/failure schema or force named-policy completion semantics into a schema whose consumers already count failures and unattempted runs. Give the wrapper a distinct allowlisted kind and verify the legacy summarizer rejects it. Continue passing each policy only the runner's existing recipient-scoped input; serialized matchup metadata is reproducibility data, not execution attestation.
