---
title: Preserve owned runner failure categories in summaries
author: jarho001
date: 2026-11-03
branch: main
id: l-ec89ac92
scope: repo
status: active
src: agent
---

Decided: When `runManifest` catches a privately owned `runPair` failure, preserve its safe `pairResult.failure.category` and code in the manifest summary. Validate categories against the categories emitted by `runBattle`; use a fixed fallback for unknown values. Keep execution ownership checks adjacent to the actual `runPair` call.

Ruled out: Do not map every non-runtime category to `initialization`, which erases packet, protocol, policy, timeout, decision, and retry-limit meaning. Do not read `category` or result-shaped metadata from an unowned thrown error.

Watch out: A `maxPackets: 1` reproducer showed the JSONL pair record retained `packet_limit` while the wrapper summary mislabeled it `initialization`. Trail inspection also found the supported `retry_limit` category in `battle-runner.cjs#runBattle`; category lists must include actual runner categories. `manifest-runner.test.cjs` checks that packet-limit and retry-limit category/code pairs reconcile between the owned journal record and summary.
