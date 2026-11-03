---
title: Capture trajectory observations from policy callbacks
author: Developer
date: 2026-11-03
branch: main
id: l-11d377e9
scope: repo
status: active
src: agent
touches:
  - environment/battle-runner.cjs
  - environment/protocol.cjs
---

Decided
Use each recipient's live policy callback input as the authority for a trajectory observation. Capture detached copies of that player's current request and new own-history records before invoking the policy. Attach terminal outcome and fixed-seat rewards only after the runner validates a completed battle. A selected command records a policy choice; it does not establish that the command was submitted, accepted, or executed.

Tried and ruled out
Reconstructing actor observations from the trusted transcript is unsafe. The transcript includes raw packets, both players' private requests, split branches, trusted choice resolutions, and end metadata. `runBattle` builds each callback history by projecting packets for one recipient and interleaving that actor's own requests, choices, and resolutions. Flattening or replaying the trusted record stream does not establish the exact recipient-visible history prefix at each callback.

Watch out
- Keep the completed artifact as post-game private data; never feed its outcomes, targets, or another player's trajectory into a policy observation.
- Snapshot before calling the policy so its current choice and later terminal target cannot appear in the pre-action observation.
- Append only new records per player; later deltas can include non-actionable requests and prior choice resolutions received since that player's last decision.
- Resolve rejection exclusions from the current `requestEpisode`. Equal request JSON after later progress is a separate sample.
- Serialized output is not execution attestation.

In the code
- `environment/protocol.cjs` projects each packet for an explicit recipient.
- `environment/battle-runner.cjs` supplies the projected callback input and separately retains trusted records and packets.
- `environment/decision-trajectories.cjs` snapshots callback observations before policy selection and finalizes targets after completion.
- `test/decision-trajectories.test.cjs` checks history reconstruction, paired privacy, fixed-seat targets, and same-seed policy choices.
