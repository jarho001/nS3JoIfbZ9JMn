# T050 global checkpoint review

**Disposition: ACCEPTED — 2026-11-05.** Independent reviewer: `/root/t050_global`, requested Astra/medium global-review route. Required corrections: **none**. Blockers: **none**.

This prospective checkpoint covers accepted Tasks041–049 and the unaccepted Task050 candidate. At review, durable accepted count was 49, review baseline 40, next checkpoint 50, pending corrections empty and GPU block null. Implementation owner `/root/task050_resume` had stopped editing and retains correction ownership. This reviewer changed only this report; no implementation, tests, configuration, state, staging or commits were changed.

## Reviewed identity and scope

Implementation base: `e5f4b21e52da78d7bcef37a1b4551808adec0b50`. Reviewed workspace HEAD: `cf11f44b6afd1ae671266c354a052c521d8ef669`; its commit changes only user-supplied long-test logs. Task050 is the frozen working-tree candidate, identified by independently verified Git blob hashes:

| File | Blob |
|---|---|
| `simulator/accuracy_choice.py` | `2480a85c574c604a06a705d39ebbf5ab38dbca19` |
| `simulator/neutral_mega_punch.py` | `44bac3dae2a4a6ef8fb7b7cc0519412ea5037098` |
| `simulator/__init__.py` | `fe5d497a96b36ef7fa3c078a916d461560c7f3f3` |
| `reference/accuracy-choice-oracle.cjs` | `901d14cc35c234e72cb46238452c7e44e9a70f8b` |
| `reference/neutral-mega-punch-oracle.cjs` | `698d668f67ba44e55186b0f5adc87760d6c2109e` |
| `test/test_accuracy_choice.py` | `f97b747dceae9572af7df65f8612048b027c957e` |

These match the active review record and Task050 handoff. No self-referential final commit identity is used. The complete new module, oracle and test, existing helper/export diffs, contract and API note were read. No existing test was changed or weakened; the v4 numerical/session sources are unchanged. Existing v5 changes are limited to optional power/accuracy, an action callback, and extraction of the identical miss mask. Defaults preserve the accepted v5 operation and public action constant; v6 exports use distinct prefixes.

## Candidate assessment

The v6 adapter satisfies its bounded direct-Mew simultaneous-choice contract. It reuses the existing two-slot layout and v5 numerical engine, supplies selected power/accuracy, and maps HP/RNG/turn/terminal/winner back while spending only the executed actor's selected public PP. An attempted miss spends PP and retains the attempted move ID. First-action knockout skips the other actor's PP, event and draws. Internal dummy PP and internal request phase do not leak into public continuation. Slot caps 56/32, IDs 33/5, variant 6, unequal positive speeds and at least one usable slot per side are enforced. Structural validation precedes dispatch; runtime precedence matches the established v4 dispatcher, including terminal absorption and unchanged rejected state/RNG.

The recipient view wraps the two-move view, preserves actor/target/order/attempted move on misses, and masks critical, damage, target HP and faint information for those rows. Its fields contain own request/stats/PP and public events, without full state, RNG, opponent PP or the unexecuted opponent choice. Static audience validation is inherited and exercised. Old v4 transition and legal masks fail closed on v6 input. The scope does not promise v6 sessions, reset, speed ties, Struggle or complete battles.

The oracle constructs actual Mew/Synchronize/Hardy/perfect-IV/itemless sets in unmodified pinned `gen5randombattle`, bypassing generation. It checks Gen 5 move facts and live two-slot PP, captures actual requests and split-selected events, and executes real native choices. Its eight scenarios cover all four choice pairs, both faster actors, the accuracy roll 93 producing Mega Punch miss versus Tackle hit, a critical, first-action knockout, both slot-depletion boundaries, and an actual next turn selecting remaining Tackle after Mega Punch reaches zero. The all-PP-zero native request is explicitly Struggle, and the implementation's typed unsupported boundary is not misrepresented as equivalent native request behavior.

I inspected the shared `assertReady` guard and `withObservedBattleDraws` implementation: checkout identity is checked, the observer calls the original PRNG and returns its value unchanged, while recording native bounds, contexts and before/after limbs. Task050 imports these from the unchanged neutral-turn oracle; its only old Mega Punch oracle edit exports the existing generic projector.

The focused test compares actual numeric results from one heterogeneous `jit(vmap(transition_accuracy_choice))` batch against independent native-derived expectations, including every state/event/draw leaf and unused trace slots. It includes successful native lanes, pending, negative/unknown choices, exhausted selected slot, all-PP-zero, excessive cap, speed tie, cross-variant and absorbing terminal lanes. Both recipients' current/next requests and native event fields are checked, including hit-only masking, exact/percentage HP visibility and attempted IDs. Separate compiled scalar continuation feeds the returned depleted state into the actual native follow-up comparison. Overkill arithmetic is independently calculated while native HP/events/RNG remain authoritative. Malformed shape/dtype, type hints, wrong move IDs and static audience rejection are exercised. This is adequate evidence for the declared bounded turn contract; it does not establish arbitrary-seed or full-format coverage.

## Checkpoint direction and milestones

Tasks041 and 047 add trusted terminal reset for v2 and v4 sessions. Sampled reset code and tests preserve the replacement battle/RNG exactly, advance a local episode once, clear commitments, reject stale pre-reset submissions and leave rejected sessions unchanged. The v4 test also resolves a fresh native turn with the new token. These are useful lifecycle semantics without claiming device-resident rollout or reset-pool throughput.

Task042 adds Pound through shared action and speed-tie machinery; sampled source and native assertions check distinct variant identity, draws and recipient views. Tasks043–044 add per-slot Tackle/Pound choices and complete native battle parity. Sampled adapter and full-battle assertions verify selected PP, move identity, bounded RNG traces, current/next requests and terminal behavior. Tasks045–046 add private staged submissions and full staged native battles. Sampled source resolves only the second commitment, advances the episode once and clears choices; tests compare native first-commit nonmutation and opponent-view invariance, then exact resolved state, events and draws.

Tasks048–049 introduce genuine misses and full Mega Punch battle evidence. I inspected the shared action/turn engine and the complete Task049 test: it threads returned state through native turns, compares complete transitions and both recipient projections, accounts independently for hits/misses/critical/KO coverage, and checks final snapshots, absent terminal requests and absorbing post-terminal submissions. Historical evidence records six native battles and 28 turns. These historical suites were sampled, not rerun.

This progression is consistent with the simulator-first decision in `PROJECT_PLAN.md`: increasingly complete parity-tested playable battles, explicit unsupported mechanics and early compiled/batched correctness. Task050 combines a choice with accuracy-dependent execution, rather than adding standalone tournament or reporting infrastructure. Coverage remains restricted to one Mew per side and a few ordinary moves. Switching/rosters, generated teams, status/items/weather and their interactions, full-format mechanics and broader lifecycle integration remain open.

G0 remains the only accepted milestone. G1 reference-environment milestone is open; G2 learning remains deferred. G3 has substantial declared-slice CPU differential evidence but still needs its own scope acceptance and profiler/device execution evidence. This task-count review closes no implementation milestone and certifies no GPU execution, performance, training or policy strength. Historical arithmetic GPU evidence cannot establish current battle/session GPU execution.

## Validation and evidence limits

Independent validation invocation, run once on the frozen hashes above:

```sh
time -p timeout --signal=TERM --kill-after=3s 15s env JAX_PLATFORMS=cpu /tmp/task040-jax-env/bin/python -m unittest test.test_accuracy_choice -v
```

**Exit 0; 4/4 tests passed in 5.516 seconds; outer real time 5.95 seconds.** This regenerated and exercised the guarded native fixture through the test setup. No other test/native execution was performed by this reviewer. Environment recorded by the owner: Python 3.12.14, JAX/JAXLIB 0.12.0.1 and Node v24.22.1; the independent command explicitly selected CPU. No additional environment/device provenance claim is made.

The owner's final focused run (4/4; 5.80 seconds outer) and required unchanged-default v5 regression (2/2; 2.49 seconds outer) are recorded in `docs/TASK050.md`, together with all earlier authoring attempts. Known owner execution is approximately 35 seconds, plus a conservative 30-second interrupted-owner allowance. Adding this review's 5.95 seconds gives approximately 70.95 seconds of accounted/reserved task execution, below the shared 90-second cap and this review's 20-second allowance. No fresh task budget, retry, broad historical suite or timing probe was introduced.

Read-only checks included `git status --short`, `git rev-parse HEAD`, `git hash-object` for all six files above, `git show --stat --oneline HEAD`, JSON parsing of compact state, and `git diff --check` (no whitespace errors). Governance read: `AGENTS.md`, global-review role contract, `docs/ORCHESTRATION.md`, compact state/TASKS, Task050 contract/API note, project-plan direction/gate table, and prior T040 report. Targeted source sampling is described above. The Trail index was read first; it describes an older inventory without the needed simulator cards, and `command -v trail` returned unavailable. No Trail graph pass is claimed.

User logs in HEAD report the same 21 legacy tests passing combined in 511.757 seconds and in separate module runs on the accepted base. They lack JAX/device placement provenance and are neither candidate evidence nor GPU evidence. The known-slow `test.test_neutral_two_move` is unchanged, already represented in those base logs, expressly excluded from Task050's routine checks and not an acceptance dependency. No repeat is requested. No required correctness evidence is deferred by this acceptance.

Repository configuration and the requested Astra/medium route do not independently attest the backend identity or automatic role loading. This limitation is recorded; no platform/backend attestation is claimed.

## Final synchronization conditions

The parent may accept Task050 after final focused coordination, recording accepted count **50**, review baseline **50**, next regular checkpoint **60**, and this **ACCEPTED** T050Global report. Preserve all earlier history/source identities, G0-only milestone approvals, existing deferred-validation history and null GPU block. Archive the complete Task050 contract/evidence as its accepted record, synchronize TASKS and both active-task locations, and close the active regular review before clearing candidate work. These are parent/owner bookkeeping actions, not implementation corrections or reasons to rerun passing checks.

No correction round or ownership transfer is required. Stop at this task boundary. The primary retains final coordination, commit and push ownership.
