# Pinned Showdown behavior catalog

Planning catalog prepared 2026-11-06 for the
[milestones and proposed first G3 goal](../PROJECT_PLAN.md#milestone-contracts).
Source base: `7068c36975fbe713d55c52c2c07659cf0fcefcbb`.
Every native case uses `gen5randombattle` from
[`reference-lock.json`](../reference-lock.json), Showdown commit
`1242c1e2d9763ba5b85d33f13aecb89bae19ae3d`, with its inherited Gen 5 rules.
Preparation observations alone establish no JAX support or milestone acceptance.
Shared execution evidence is linked below; full-format and G3 GPU closure remain open.

## Evidence and shared record

- **H — historical native fixture:** accepted tests and reproducible oracle
  entry points exist; not rerun during preparation. Preserve their precise scope.
- **N — observed native execution:** the appendix recipe ran successfully during
  preparation. Normal choices execute from directly supplied sets, with no
  mechanical callbacks or random outcomes replaced. These are reachable from
  those constructed starts; natural generator reachability/set legality is
  unproven and is not a full-format distribution claim.
- **S — synthetic trusted boundary:** native HP, PP or pre-existing status is
  explicitly installed before a live request, or a JAX-only invalid/capacity
  boundary is constructed. Label the setup; do not count it as an ordinary
  battle from natural initial conditions. The appendix verifies that its native
  setup mutations leave the battle RNG unchanged.
- **P — proposed, evidence pending:** expected behavior is provisional until
  observed in pinned Showdown, except local adapter/rejection policies explicitly
  defined by the goal. Source references guide the next probe, not a passed test.

`O(name, step)` below identifies the full native appendix record at
`cases[id=name].steps[step-1]`. Its input sets, choices and setup are in `setup`;
`initialEvents` retains each player's setup disclosures. `before`/`after` contain
each roster member's HP, max HP, PP, stats, status/counter,
item, active flag, turn, RNG, winner, requests and eleven-slot action spaces;
`draws` contain native arguments, values, context, stack and four-limb pre/post
states; `events.p1/p2` contain actual split-projected event lines. No JAX outputs
or hand-calculated damage values enter these expectations. Inline observations
and the replay recipe remain in this planning document so evidence does not
depend on an ephemeral `/tmp` artifact.

One proposed shared fixture schema extends those fields with stable mon IDs,
request-slot mappings, effect counters/boosts, pending choices and suspended
queue actions, initialization draws, schema/pin/source identities and visibility
history. These extra fields are **not yet captured by the appendix** and must be
added before the integrated-engine tests claim complete state parity. Normalize
only documented timing/diagnostic paths, never draws, hidden state or events.
The current low-level observer labels paralysis/stall draws `unsupported` and
some `random(from,to)` draws as `speed-sort`; retain raw arguments/stack until
the shared harness labels the actual phase correctly. New mechanic contexts
must not be discarded to fit a six- or sixteen-draw legacy trace.

Every detailed case below inherits this assertion contract:

1. Save native preconditions, accepted/rejected choices and request boundary;
   compare complete mechanically relevant state after each resolved boundary.
2. Assert ordered draw arguments/bounds, value, before/after limbs and final RNG,
   including scheduler draws and no draws on skipped actions. Initialization
   may consume RNG for gender/order: couple to the captured post-reset state,
   not blindly to the caller's original four-limb seed.
3. Compare request kind, acting sides, own move PP and legal command mapping;
   preserve native roster reordering and wait requests. Opposing requests and
   pending choices are trusted data, never actor inputs.
4. Compare p1 and p2 events/history in order, owner-exact versus opponent HP
   disclosure, status/item/ability reveals and real terminal outcome/rewards.
   Terminal has no legal actions; rejected/stopped work never invents a win/draw.
5. Run supported cases scalar/eager, JIT and heterogeneous `jit(vmap(...))` with
   scalar-lane equivalence, continuation and absorbing terminal/rejection lanes.
   Local validation policies are tested separately from native game behavior.

The proposed common runner is
`node reference/shared-engine-cases.cjs --batch all --output NEW_FILE`.
Its implementation and coverage map are above. The consumers are `test.test_shared_engine_cases`
(C), `test.test_shared_engine_battles` (F), and
`test.test_shared_engine_lifecycle` (L). Select case batches in one
runner and independent fixture reader, rather than creating an oracle/state
variant for each move. [V0–V11](../PROJECT_PLAN.md#validation-entry-points-and-remaining-decisions)
lists existing entry points and clearly marks proposed commands.

## Shared implementation and executable coverage

The user approved the first G3 brief on 2026-11-06. The runner and three Python
consumers proposed during preparation now exist, with one
[manifest and usage contract](shared-engine.md). Executed commands, source
identity and acceptance status live in [compact state](ORCHESTRATION_STATE.json).
The H/N/S/P labels and appendix below retain the preparation evidence; they are
not a substitute for that current execution record.

| Required family | Shared runner cases / assertions |
|---|---|
| B0 M01–M04 | `M01-duel-*`, `legacy-pound-tied`, `M02-duel-two-move`, `M03-*`, `M04-*`; old APIs retained and V3 controls run separately. |
| B1 A01/A06 | Legacy migration boundaries, `damage-level-1/50/100`, `last-pp`, `near-full-recover/switch`; native critical/miss/KO/full traces and PP exhaustion. |
| B1 A02/A03 | `normal-ghost`, `levitate`, `toxic-immunity/type-immunity`, `struggle-double-ko/protected`, `burned-struggle`, F04. |
| B1 A04/A05 | `protect-repeat`, `protect-chain-*`, `protect-success-2/4/8/16/32/64/128/256`, `protect-counter-cap`, `protect-paralyzed`; native successful/failing counter and full-width draws, expiry/switch interactions. Large counters are explicitly synthetic; capped success uses a real inverse-recurrence seed, not a replaced random outcome. |
| B2 S01–S09 | Original named appendix scenarios, `two-burn-reserves/swapped`, status/recovery/counter snapshots, both recipient streams. |
| B2 S10 | `toxic-counter-cap`, `already-statused/same-status`, `lum-inflicted-brn/par/tox`, `lum-brn/par/tox`, `toxic-pivot`, `paralysis-pivot-refresh`, F02/F03. |
| B3 W01–W06 | `voluntary-retarget`, `double-switch`, `forced-replacement`, `pivot-resume/protected/fast/slow/no-reserve/slower-resume/ko-resume/paralyzed/burn-last-pp`, `two-burn-reserves`, F01/F03; lifecycle tests split/reset these phases. |
| B4 P01/P02 | `commit-replace/cancel`, exact masks/requests/private pending choices, both submission orders; `test/shared-engine.test.cjs` compares actual BattleStream packets, including silence on pending/undo, to direct snapshots. |
| B4 P03 | Native `privacy-a/b` and `privacy-hp-a/b` paired histories plus lifecycle assertions; opposing reserve/item/PP/pending choices/exact HP do not enter p1's view. |
| B4 P04 | Typed malformed/unsupported/stale/terminal inputs; history/episode/queue bounds; synthetic smaller physical event/draw buffers exercise atomic rollback and empty traces. |
| B4 P05/P06 | NPZ round-trip at every ordinary/pending/pivot/forced boundary, terminal-only mixed reset; scalar/eager/JIT/full-battle `jit(vmap)` sizes 1/8/32; mixed active/pending/terminal/rejected/overflow lanes and lane-local RNG perturbation. All catalog boundaries also run eager/JIT and in 32-lane batches. CPU proof only. |
| B5 F01–F04 | `F01/02/03/04-p1/p2`: eight complete three-member native scripts, two seeds, both seats; F04 explicitly starts with depleted PP. Each boundary compares state, queue, RNG context/range/value/pre/post state, requests/masks, both views and real outcome/rewards. |

Initialization captures gender and speed-order draws, public reveals and private
requests. Six roster slots and four move slots are padded/masked, with stable
IDs distinct from command order. Draws have independent native phase labels;
the 2**32 range has an explicit tag and uint32 value. HP percentages retain the
native 99% cap for damaged HP. Only timing and diagnostic stack paths are omitted
from semantic comparisons. No G3 GPU/profile or full-format claim follows.

## B0 — migrate accepted slices without changing their evidence

| Case | Setup/choices and evidence | Required state/RNG/request/view/outcome assertions | Existing evidence → proposed consumer |
|---|---|---|---|
| M01 | H: itemless Mew Tackle duels, unequal/tied speeds; native seeded choices | Preserve every existing full transition and all tie/scheduler draws, initiative, KO-skipped PP/draws, both views and real end. Do not import the fixed trace capacity as an engine bound. | [Tackle oracle](../reference/neutral-turn-oracle.cjs), speed-tie mode `--ties`; `test.test_neutral_tackle`, `test.test_neutral_tackle_ties`, `test.test_neutral_duel` → C/F |
| M02 | H: `[Tackle,Pound]`, mixed choices, partial PP, staged submissions and terminal reset | Retain selected-slot PP/IDs, complete native traces, both commitment orders, privacy and reset invariants. Distinguish local duplicate/stale-token policy from native replaceable choices. | [Two-move oracle](../reference/neutral-two-move-oracle.cjs) modes `--duels`, `--session-duels`; `test.test_neutral_two_move_duel`, `test.test_two_move_session_duel`, `test.test_two_move_session_reset` → C/F/L |
| M03 | H: v6 eight one-turn cases and six full Tackle/Mega Punch duels | Preserve accuracy/crit/damage draw chains, both initiative directions, all four pairs, final-slot follow-up, both recipients and terminal absorption. Six duels contain 35 turns; this is v6 evidence only. | [Accuracy oracle](../reference/accuracy-choice-oracle.cjs) and `--duels`; [accepted boundary](accuracy-choice-variant.md), [T050 review](reviews/T050-global.md); `test.test_accuracy_choice`, `test.test_accuracy_choice_duel` → C/F |
| M04 | H/S: v7 no-burn control and five pre-installed single-burn boundaries | Burn reduction, exact clipped residual, direct-KO residual skip, residual KO and persistent native `brn` despite `0 fnt`; final requests and privacy. No inference of status-infliction support from this fixture. | [Burn oracle](../reference/burn-choice-oracle.cjs), [accepted boundary](burn-choice-variant.md), `test.test_burn_choice` → C |

Retain these APIs/tests until equivalent shared cases pass. Migrating new
expectations does not rewrite archived evidence or authorize cleanup now.

## B1 — moves, order, PP and termination

| Case | Setup/choices; classification | Expected state and terminal/boundary | RNG evidence | Requests and both visible streams | Evidence/consumer |
|---|---|---|---|---|---|
| A01 | H: M03 native Tackle/Mega Punch hit, miss, critical, overkill and last-PP fixtures | Spend PP on an attempted miss; spend none for an actor skipped after KO. Match native clipped HP and raw supported damage semantics independently. | Exact M03 traces; ordinary hits use accuracy `random(100)`, critical `random(16)`, damage `random(16)`; miss skips the last two. | Exhausted selected row disabled, remaining move still legal; miss reveals attempted actor/move without fabricated damage/crit/faint fields; terminal removes requests. | M03 → C/F |
| A02 | N: Snorlax Tackle or Earthquake versus Gengar; Gengar Toxic versus Snorlax | No damage on Normal/Ghost or Levitate immunity. Toxic fails against Immunity. Native attempts still spend selected PP. | `O(normal-ghost,1)`, `O(levitate,1)`: only Toxic accuracy draw; immune damage moves add no accuracy/crit/damage draw here. | Native ability/type immunity disclosure differs; do not expose every immunity before its event. Normal move requests continue. | O records → C |
| A03 | S: Mew/Gengar at HP 1, all ordinary PP zero; both submit native sole `move 1` | Faster Gengar Struggle KOs Mew, then recoil KOs Gengar; observed winner **p2**, not a draw. Preserve faint/outcome precedence. | `O(struggle-double-ko,1)`: critical/damage only, no accuracy draw; Mew never acts. | Sole Struggle maps to action slot 10; ordinary PP stays zero; two faint events then one winner, no next request. | O record → C/F |
| A04 | N: faster Mew Protect vs Mew Tackle, same choices for three turns | Protect succeeds, fails on repeat, then succeeds after failure; both attempted moves spend PP. Only the failed-Protect turn damages p1 (175→155). | `O(protect-repeat,1..3)`: scheduler draws on successful turns, `random(2)=1` on failed repeat then accuracy/crit/damage. | Preserve Protect start/fail/block events; no fabricated hit for a blocked move. | O records → C |
| A05 | P: consecutive successful Protect chain through counter cap; protect plus paralysis/switch | Gen 5 stall counter doubles to 256; at cap source uses chance `1/2**32`. Failure/expiry/switch reset needs native evidence. | Provisional: pinned `data/mods/gen5/conditions.ts` stall. Record the actual draws; a signed int32 bound cannot encode `2**32`, so use explicit limbs/tag in the new trace schema. | Native priority 4, one-turn Protect and valid follow-up masks; no hidden counter disclosed unless native request does. | Add native boundary observations → C |
| A06 | P: supplied supported sets across levels 1/50/100, type/STAB/burn/crit controls and HP/PP boundaries | Match resolved stats and damage rounding at low/high rolls, min damage, exact KO and overkill. Use native roll-producing seeds, never override RNG for ordinary cases. Synthetic arithmetic controls stay labeled. | Reuse [mechanics ledger](MECHANICS_COVERAGE.md) primitive fixtures, but obtain integrated native traces for all claimed branches. | Exact owner/opponent HP formatting including damaged HP that rounds near 100%; valid final-slot masks. | Existing arithmetic evidence + new integrated cases → C |

## B2 — status, ability, item, recovery and residual interactions

| Case | Setup/choices; classification | Expected state and terminal/boundary | RNG evidence | Requests and both visible streams | Evidence/consumer |
|---|---|---|---|---|---|
| S01 | N: Mew Will-O-Wisp vs Snorlax Tackle; caller seed `[0,2,3,4]` | Native roll 62 hits; Snorlax's same-turn physical attack is reduced. Final HP Mew 159/175, Snorlax 206/235 brn after 29 residual. | `O(wisp-hit,1)` includes Wisp accuracy then Tackle accuracy/crit/damage; residual draws none. | Status visible before Snorlax acts; owner sees exact HP and opponent sees native public HP. Normal next requests. | O record → C |
| S02 | N: same setup, caller seed `[1,2,3,4]` | Wisp roll 88 misses at Gen 5 accuracy 75, spends its PP, no status/residual; Mew ends 144/175, Snorlax 235/235. | `O(wisp-miss,1)`; seed is not assumed to survive initialization unchanged. | Miss and attempted move visible; no status or residual event. | O record → C |
| S03 | N: Mew Toxic vs Snorlax Tackle; separately Mew Toxic vs Gengar Toxic | Immunity prevents Snorlax poisoning. Gengar first poisons Mew; Synchronize reflection and Mew's Toxic both fail against Poison-type Gengar. Mew takes first toxic tick to 165/175, stage 1. | `O(toxic-immunity,1)`, `O(toxic-type-immunity,1)`; accuracy and reflection behavior recorded independently of JAX. | Preserve `[from] ability: Immunity` versus type immunity, Synchronize and `[from] psn` residual lines; own status conditions agree with requests. | O records → C |
| S04 | N: faster Mew Wisp vs slower Mew Tackle holding Lum Berry | Burn is reflected by Synchronize **before** Lum consumption/cure. P1 remains burned and ends 131/175; P2 cured, item consumed, 175/175. Reflection back attempts fail; preserve events. | `O(lum-synchronize,1)`; no invented reflection/cure RNG. | Visible status, ability activation, item consumption and cure ordering; item hidden until native disclosure; p2 own request reflects consumption. | O record → C |
| S05 | N: Mew Thunder Wave vs Snorlax Tackle; S: pre-paralyzed Mew vs Snorlax Mega Punch at seeds 0/1/21 | Same-turn paralysis check occurs. Pre-paralyzed Mew uses quarter-speed; Snorlax acts first. Seed 21 full paralysis skips Mew's PP/action/damage draws; seeds 0/1 permit it. | `O(thunder-wave,1)`, `O(paralysis-0/1/21,1)`; native `random(4)` follows the earlier actor's draws and precedes Mew accuracy when it acts. | Visible `cant ... par` on failure; no move event/PP decrement for that action. Normal requests retain status and actions. | O records → C |
| S06 | S: pre-toxic Mew/bench Gengar vs Snorlax Protect; Recover twice, switch out, switch back | Toxic stages 1,2; no bench residual; status persists. Native switch-in resets stage, next tick is 1: Mew HP 165,155,155,145 across boundaries. Recover is included in those numbers. | `O(toxic-switch-reset,1..4)` includes native stall/scheduler draws and zero draws on the two switch boundaries. | Native request roster swaps active to front; own exact and opposing public switch/HP lines differ. Hidden stage is trusted, not policy input. | O records → C/L |
| S07 | S: both burned Mew at HP 1, p1 faster; both Protect | Faster p1's residual KO ends this no-reserve battle immediately; p2 remains at HP 1 and wins. Do not process both residuals simultaneously or invent a tie. | `O(two-burn-ko-order,1)`; exact scheduler draw, no burn draw. | Burn/faint/winner order; no terminal request. Add P cases with reserves/both replacement requests, tied residual speed and swapped seats before generalizing. | O record + pending adjacent observations → C/F |
| S08 | S: Mew HP 15/175, brn and Leftovers, vs Snorlax; both Protect | Leftovers heals to 25 before burn reduces to 4. Protect does not block residual. Neither effect adds RNG. | `O(leftovers-before-burn,1)` includes a scheduler draw from turn execution. | Heal cause discloses item, then burn damage; exact/private and public HP follow native splits; game continues. | O record → C |
| S09 | S: Mew HP 10/175 Recover vs Snorlax Tackle; N/S: full-health Recover in S06 | Recover heals **88** to 98, then Tackle leaves HP 62; do not floor half of odd maximum HP. At full HP it fails while spending PP. | `O(recover-partial,1)`, `O(toxic-switch-reset,1)`; no recovery accuracy draw, but retain opposing action/scheduler draws. | Heal/fail events and final own condition/PP; no automatic win/draw. | O records → C |
| S10 | P: both statuses/items across pivot/replacement; Toxic stage cap 15; already-statused targets; Lum cures each included status | Capture application failure, correct counter saturation, no bench tick, consumption and which residuals run before/after interruptions. Native output must settle all order questions. | Provisional sources: `data/conditions.ts` tox; `data/items.ts` Leftovers/Lum; `data/abilities.ts` Synchronize/Immunity; inherited Gen 5 overrides. | Assert both per-player condition/history and hidden counters omitted. Generalizations beyond S01–S09 are pending. | Shared native batch extensions → C/F |

Sleep/freeze/confusion duration and ordinary poison are adjacent G4 families,
not extra first-goal moves. Protect duration/recovery and toxic counters already
exercise timed effects inside this package.

## B3 — switching and interrupted turns

| Case | Setup/choices; classification | Expected state and terminal/boundary | RNG evidence | Requests and both visible streams | Evidence/consumer |
|---|---|---|---|---|---|
| W01 | N: p1 Mew/bench Gengar switches to slot 2; p2 Snorlax Tackle | Incoming Gengar is targeted and immune; both remain at full HP. Snorlax spends one Tackle PP; bench Mew spends none. | `O(voluntary-retarget,1)` consumes no draws after setup. | Reveal Gengar on switch; own request roster now Gengar then Mew, so subsequent `switch 2` means Mew. No leak of unrevealed reserves. | O record → C/L |
| W02 | N: faster Mew U-turn with bench Snorlax; opponent Snorlax Mega Punch; then p1 switches | U-turn damages foe 235→196 and pauses with opposing action/PP unexecuted. Replacement Snorlax starts 235, receives pending Mega Punch, ends 154. Same native turn resumes rather than choosing again. | `O(pivot-resume,1..2)`: first boundary has 3 action draws; resumption has **10** including 7 equal-speed scheduler draws. Six-draw action-only tracing is insufficient. | First p1 forced-switch/p2 wait; next both move. Carry pending move privately, switch reveal `[from] U-turn`, exact/public recipient HP and correct history. | O records → C/F/L |
| W03 | N: level-100 Mew Mega Punch vs level-1 Snorlax with Gengar reserve; p2 chooses Mega Punch, then replaces | Snorlax faints without spending selected PP; p1 does not win while reserve remains. Only p2 replaces; next normal request advances to turn 2. | `O(forced-replacement,1..2)`: 3 first-action draws then none for replacement. | p1 wait/p2 forced-switch, no artificial p1 action; afterward p2 roster active Gengar first. Native transient faint-status fields must be captured rather than inferred. | O records → C/F/L |
| W04 | N: Mew U-turn with reserve vs Snorlax Protect | Protected U-turn spends PP but neither damages nor requests a pivot. Next request remains normal. | `O(pivot-protected,1)`: scheduler draw only; no action damage draws. | Protect/block events only, no phantom switch/reveal. | O record → C |
| W05 | P: U-turn slower/faster, KO with/without enemy reserves, no own reserve, last PP, burned/paralyzed actor | Observe exactly which continuation/replacement happens first and whether queued action survives. Rejected/failed/immune pivot must not invent replacement. | New native case batch required; do not infer all branches from W02. Trace all draw phases and pending action state. | Explicit acting-side/forced/wait masks at each interruption, followed by exact ordered views and true terminal outcome. | Shared native batch → C/F/L |
| W06 | P/S: two low-HP status targets with reserves; simultaneous voluntary switches; reset after replacement | Native replacement and residual precedence, status persistence, toxic/stall reset and entry update order. Synthetic low-HP setup must stay labeled. | New native observations needed for ties and two-side replacements; no fixed draw-count assumption. | Both forced requests only if native produces them; validate stable mon identity versus reordered command slots. | Shared native batch → C/F/L |

## B4 — protocol, privacy, lifecycle and execution bounds

| Case | Setup/choices; classification | State/RNG/result expectation | Requests/views and consuming assertion |
|---|---|---|---|
| P01 | N: Mew `[Tackle,Mega Punch]`; p1 selects Tackle, replaces it with Mega Punch, then p2 selects Tackle | `O(commit-replace,1..3)`: first two submissions leave numerical battle state/RNG unchanged with empty public logs; only final choice executes. P1 Tackle PP stays 56, Mega Punch becomes 31. | Before resolution opponent sees no pending choice; final events reveal only executed move. Proposed L must additionally capture private cancellation acknowledgements/undo and permitted native rejection paths; raw `Battle.log` alone does not establish full BattleStream parity. |
| P02 | H/N: both commitment orders, forced/wait/pivot, terminal and all-PP-zero requests | Use M02/M03 and all O raw requests plus `actionSpace`. WAIT is no policy action; terminal has none; Struggle slot differs from ordinary move slot. | Assert eleven-slot mapping, same-episode rejected exclusions, no all-masked acting softmax, own PP/status and both native event streams → C/L. Expanded unavailable/retry cases remain G1 work where reachable. |
| P03 | P: two valid histories indistinguishable to p1 but with different opposing reserve/item/PP/pending choice or exact HP inside the same visible interval | Identical p1 observations, masks, history-derived features and policy input; changing a secret that would already have caused a reveal is not a valid pair. Opponent's permitted private view can differ. | Native paired histories plus legacy actor-view tests guide L; historical views alone do not prove integrated non-leakage. |
| P04 | S/P local API: malformed arrays/IDs, >3 occupied slots, excluded mechanic, stale token, duplicate/replacement/cancel, terminal submission, queue/event overflow | Validate shape/dtype on host; typed runtime diagnostics preserve state/RNG for rejection. Overflow returns pre-advance state and no partial public events. Local episode overflow is explicit; a cap is not a draw. These are declared engine/adapter policies, not native game outcomes. | C/L must compare every state leaf, empty trace/mask and correct diagnostic; retain any old API's different duplicate policy only in that compatibility API. |
| P05 | H/P: split continuation at ordinary/pivot/forced boundaries and terminal reset to a supplied new battle | Resumed state/RNG/events equal uninterrupted execution. Reset installs exact supplied battle state/RNG/counters, clears old session choices/history/queue and advances local generation once. Live/stale reset rejection preserves all old state. | Reuse [v4 reset](../test/test_two_move_session_reset.py) and [batched sessions](../test/test_request_session_batch.py); proposed L extends to the shared engine and mixed active/pending/terminal/rejected lanes. |
| P06 | P: batch sizes 1, 8 and 32 with mixed supported phases, item/status branches, rejected inputs and terminal lanes | Every lane equals scalar/eager and JIT output, including unused event/draw masks and no cross-lane RNG. Pad capacities explicitly; synthetic overflow checks exercise the diagnostic. | C/F/L. First goal requires CPU; actual GPU placement/profile is required later for G3 closure via V6, never inferred from old arithmetic GPU runs. |

## B5 — full traces make the package playable

The F rows below were **P during preparation**; shared execution now uses the
eight named scripts in the coverage map above. Current proof is in compact state. Supply the exact team/seed/
choice scripts before implementation assertions, using the shared runner. Four
complete three-member scenarios must cover both acting seats and multiple seeds;
mirror/swap controls preserve team, policy and RNG identities correctly. A turn
limit is an explicit truncation failure, never terminal evidence. Boundary-only
companion cases may supply a rare branch, but cannot replace complete battles.

| Case | Scripted family and preconditions | Required finish and interaction evidence |
|---|---|---|
| F01 | Ordinary N candidate: Mew/Snorlax/Gengar, mixed physical attacks and voluntary switches; no trusted HP/PP edits | Multiple KOs and forced replacements through real last-member win; priority, type/ability immunity, critical/miss/PP branches via companion cases if absent. Thread actual returned state, not per-turn reinitialization. |
| F02 | Ordinary N candidate: statuses, Synchronize/Immunity, Lum/Leftovers, Recover/Protect, status-preserving switches | Full terminal history with status application/failure/cure, residual and item order; exact hidden counters plus both recipients. Must not terminate by skipping residual or calling a turn cap a draw. |
| F03 | Ordinary N candidate: faster/slower U-turn paths with reserves, pending opponent action and later KO | Native mid-turn requests, correct target after pivot, no double execution/PP spend, continuation to last-member win. Pause/resume at every request and compare uninterrupted run. |
| F04 | S candidate: explicitly depleted starting PP across a three-member roster, then normal native choices | Actual Struggle/recoil/last-member outcome, replacements and any double-KO precedence. Label trusted depleted initial state; add ordinary depletion follow-up from M03. It is not a natural generator-start claim. |

For each complete trace require C/F/L state and request comparisons at **every**
boundary, first-commit privacy, exact draw chains, both event histories, final
outcome and absorbing post-terminal submission. Replay the same scenarios as a
heterogeneous batch; paused/terminal lanes must not consume other lanes' draws.

## Later milestone families and missing proof

| Milestone | Relevant families | Evidence gap / required proof |
|---|---|---|
| G0 | Pin, resolved inherited rules/data and source identity | Existing accepted specification and V0; changed game identity needs versioned evidence, not a new mechanics implementation. |
| G1 | Complete BattleStream requests, cancel/undo/unavailable retry, trapping disclosure, forced/recharge/Struggle, trajectories and rewards | Broader native stream fixtures and end-to-end player-history/trajectory coverage; scored paired baseline protocol under D2. Appendix direct-Battle logs are insufficient alone. |
| G2 | Memory across waits/chunks, masked stochastic actions, policy RNG, recurrent burn-in, restart and observation/label separation | Learner/small-game controls, frozen baseline, independent training seeds and paired native held-out improvement; no current learning proof. |
| G3 closure | Shared B0–B5 CPU integration; GPU mixed-phase/device correctness, compilation, throughput and memory profile | Shared engine and most interactions remain unimplemented; actual device/profile proof required separately from first-goal CPU completion. |
| G4 | Sleep/freeze/confusion/durations, ordinary poison; hazards/entry abilities; weather/screens/boosts; trapping; recharge/locked/multiturn/multi-hit moves; substitute/recoil/contact; Illusion/Transform/copy/call/item exchange; Sleep Clause and all end/tie clauses | Audit complete reachable callback/data closure, generated six-member teams, minimized corpus and held-out fuzzing on CPU/GPU. Do not infer completeness from number of cases or sampled teams. |
| G5 | Reset pools, memory/label isolation, device rollout/learner, exact restart and same-policy transfer | Matched full-pipeline measurements, actual device/host-crossing counts, predeclared equivalence margin and useful learning gain; no current evidence. |
| G6 | Cross-play, tactical/rare cases, held-out opponents/teams, belief/league ablations | Multi-run native promotion, uncertainty/cycles/failure/latency evidence at fixed resources. |
| G7 | Matrix/partially observed games, information-set consistency, belief collapse/calibration, mixed-action search and best responses | Controlled method selection/ablations, exploiters and measured strength/cost benefit; failure to find an exploiter does not prove equilibrium. |
| G8 | Public-belief future states, distillation/adaptive compute, release replay/resume and audit | Locked native comparisons against best controls, deployment measurements, reproducible artifact identity/license/access; strongest claim limited to explored design space. |

## Observed native probe evidence

The following appendix recipe was run on the source base above with Node
`v24.22.1`, npm `11.9.0` and the guarded pinned Showdown build. Provenance reported
format SHA-256 `87b303476a12f0a14559c659c986fd13eb5112d26b2cac8371f393f84b498b71`
and upstream package-lock SHA-256
`10960416dfb6af611bcc7f6bd503e3b69d3ef7df0647fa9ce9e9ecf9ac857f82`.
The ordinary sandbox initially denied the provenance helper's child process;
the permitted execution route succeeded. No reference source was changed.
Only short native probes ran; no JAX suite, long integration, GPU or performance
run was performed. Probe runtimes are not performance evidence.

The excerpt below records each boundary's before/after RNG state and
ordered native draw arguments/values. Per-draw pre/post limbs, full state,
requests/actions and both views are reproduced by the recipe, which is the
persistent evidence location for each O reference. Diagnostic stack paths may
differ with invocation location and are excluded only from the semantic digest.

Observed **22 native scenarios / 31 boundaries**. Canonical JSON SHA-256 after removing only each
`draw.stack` and serializing with sorted keys, UTF-8, no whitespace:
`b4b5afc47eaaa7e21d889c4636bcbd0aaa658024e0191ea04fee3ace8502ea13`.

| Native record | Before → after RNG limbs | Ordered draw calls and returned values |
|---|---|---|
| `wisp-hit/1` | `61345,33910,19571,50263` → `37565,62519,52978,42619` | `(100)=62 (100)=55 (16)=3 (16)=9` |
| `wisp-miss/1` | `30982,33910,19571,50263` → `55250,62519,52978,42619` | `(100)=88 (100)=66 (16)=2 (16)=13` |
| `toxic-immunity/1` | `30982,33910,19571,50263` → `55250,62519,52978,42619` | `(100)=88 (100)=66 (16)=2 (16)=13` |
| `toxic-type-immunity/1` | `30982,33910,19571,50263` → `43514,9542,40559,8561` | `(100)=88 (100)=66` |
| `normal-ghost/1` | `57890,13032,12358,42006` → `43514,9542,40559,8561` | `(100)=66` |
| `levitate/1` | `57890,13032,12358,42006` → `43514,9542,40559,8561` | `(100)=66` |
| `lum-synchronize/1` | `1,2,3,4` → `8769,17248,37115,18776` | `(100)=47 (100)=88 (16)=10 (16)=2` |
| `two-burn-ko-order/1` | `1,2,3,4` → `30982,33910,19571,50263` | `(2,4)=2` |
| `leftovers-before-burn/1` | `30982,33910,19571,50263` → `57890,13032,12358,42006` | `(2,4)=3` |
| `thunder-wave/1` | `30982,33910,19571,50263` → `37673,46633,62039,8266` | `(100)=88 (4)=2 (100)=13 (16)=13 (16)=9` |
| `recover-partial/1` | `30982,33910,19571,50263` → `8769,17248,37115,18776` | `(100)=88 (16)=10 (16)=2` |
| `paralysis-0/1` | `61345,33910,19571,50263` → `52918,32001,50713,50540` | `(100)=62 (16)=8 (16)=3 (4)=2 (100)=8 (16)=1 (16)=12` |
| `paralysis-1/1` | `30982,33910,19571,50263` → `37673,46633,62039,8266` | `(100)=88 (4)=2 (100)=13 (16)=13 (16)=9` |
| `paralysis-21/1` | `13546,33910,19571,50263` → `15734,62519,52978,42619` | `(100)=2 (16)=13 (16)=6 (4)=0` |
| `toxic-switch-reset/1` | `57890,13032,12358,42006` → `43514,9542,40559,8561` | `(1,3)=2` |
| `toxic-switch-reset/2` | `43514,9542,40559,8561` → `55250,62519,52978,42619` | `(2)=0 (1,3)=2` |
| `toxic-switch-reset/3` | `55250,62519,52978,42619` → `55250,62519,52978,42619` | `none` |
| `toxic-switch-reset/4` | `55250,62519,52978,42619` → `55250,62519,52978,42619` | `none` |
| `voluntary-retarget/1` | `57890,13032,12358,42006` → `57890,13032,12358,42006` | `none` |
| `pivot-resume/1` | `57890,13032,12358,42006` → `55250,62519,52978,42619` | `(100)=66 (16)=2 (16)=13` |
| `pivot-resume/2` | `55250,62519,52978,42619` → `20898,18934,47363,16957` | `(0,2)=1 (0,2)=1 (0,2)=0 (100)=77 (16)=13 (16)=1 (0,2)=0 (0,2)=0 (0,2)=1 (0,2)=0` |
| `forced-replacement/1` | `1814,13032,12358,42006` → `15734,62519,52978,42619` | `(100)=82 (16)=6 (16)=3` |
| `forced-replacement/2` | `15734,62519,52978,42619` → `15734,62519,52978,42619` | `none` |
| `struggle-double-ko/1` | `30982,33910,19571,50263` → `43514,9542,40559,8561` | `(16)=14 (16)=10` |
| `protect-repeat/1` | `1,2,3,4` → `30982,33910,19571,50263` | `(0,2)=0` |
| `protect-repeat/2` | `30982,33910,19571,50263` → `55250,62519,52978,42619` | `(2)=1 (100)=66 (16)=2 (16)=13` |
| `protect-repeat/3` | `55250,62519,52978,42619` → `37673,46633,62039,8266` | `(0,2)=1` |
| `pivot-protected/1` | `57890,13032,12358,42006` → `43514,9542,40559,8561` | `(0,2)=1` |
| `commit-replace/1` | `1,2,3,4` → `1,2,3,4` | `none` |
| `commit-replace/2` | `1,2,3,4` → `1,2,3,4` | `none` |
| `commit-replace/3` | `1,2,3,4` → `37673,46633,62039,8266` | `(100)=47 (16)=14 (16)=10 (100)=13 (16)=13 (16)=9` |

### Replay recipe (short native probe, already observed)

Run from the repository root with the guarded reference already built. The
output contains trusted private state and is an engineering fixture, never an
actor observation. This is a native investigation recipe, not the proposed JAX
engine or a production shared-runner implementation.

```sh
node <<'JS' > /tmp/g3-native-probes.json
'use strict';
const path = require('node:path');
const root = process.cwd();
const {assertReady, withObservedBattleDraws} = require(path.join(root, 'reference/neutral-turn-oracle.cjs'));
const {projectPackets} = require(path.join(root, 'environment/protocol.cjs'));
const {actionSpace} = require(path.join(root, 'environment/action-space.cjs'));
assertReady();
const ps = require(path.join(root, 'reference/vendor/pokemon-showdown/dist/sim'));
const pin = require(path.join(root, 'reference-lock.json')).commit;
const ivs = {hp:31,atk:31,def:31,spa:31,spd:31,spe:31};
function mon(species, moves, options={}) {
  const ability = {Mew:'Synchronize', Snorlax:'Immunity', Gengar:'Levitate'}[species];
  return {species, ability, moves, item:'', level:50, nature:'Hardy', ivs,
    evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0}, ...options};
}
const mew = (moves, o) => mon('Mew', moves, o);
const lax = (moves, o) => mon('Snorlax', moves, o);
const ghost = (moves, o) => mon('Gengar', moves, o);
const attack = ['Tackle','Mega Punch'];
const cases = [
 {id:'wisp-hit',seed:0,p1:[mew(['Will-O-Wisp'])],p2:[lax(attack)],steps:[{p1:'move 1',p2:'move 1'}]},
 {id:'wisp-miss',seed:1,p1:[mew(['Will-O-Wisp'])],p2:[lax(attack)],steps:[{p1:'move 1',p2:'move 1'}]},
 {id:'toxic-immunity',seed:1,p1:[mew(['Toxic'])],p2:[lax(attack)],steps:[{p1:'move 1',p2:'move 1'}]},
 {id:'toxic-type-immunity',seed:1,p1:[mew(['Toxic'])],p2:[ghost(['Toxic'])],steps:[{p1:'move 1',p2:'move 1'}]},
 {id:'normal-ghost',seed:1,p1:[lax(['Tackle'])],p2:[ghost(['Toxic'])],steps:[{p1:'move 1',p2:'move 1'}]},
 {id:'levitate',seed:1,p1:[lax(['Earthquake'])],p2:[ghost(['Toxic'])],steps:[{p1:'move 1',p2:'move 1'}]},
 {id:'lum-synchronize',seed:1,p1:[mew(['Will-O-Wisp'],{evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:252}})],p2:[mew(['Tackle'],{item:'Lum Berry'})],steps:[{p1:'move 1',p2:'move 1'}]},
 {id:'two-burn-ko-order',seed:1,p1:[mew(['Protect'],{evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:252}})],p2:[mew(['Protect'])],setup:[{side:'p1',status:'brn',hp:1},{side:'p2',status:'brn',hp:1}],steps:[{p1:'move 1',p2:'move 1'}]},
 {id:'leftovers-before-burn',seed:1,p1:[mew(['Protect'],{item:'Leftovers'})],p2:[lax(['Protect'])],setup:[{side:'p1',status:'brn',hp:15}],steps:[{p1:'move 1',p2:'move 1'}]},
 {id:'thunder-wave',seed:1,p1:[mew(['Thunder Wave'])],p2:[lax(attack)],steps:[{p1:'move 1',p2:'move 1'}]},
 {id:'recover-partial',seed:1,p1:[mew(['Recover'])],p2:[lax(attack)],setup:[{side:'p1',hp:10}],steps:[{p1:'move 1',p2:'move 1'}]},
 ...[0,1,21].map(seed=>({id:`paralysis-${seed}`,seed,p1:[mew(attack)],p2:[lax(attack)],setup:[{side:'p1',status:'par'}],steps:[{p1:'move 2',p2:'move 2'}]})),
 {id:'toxic-switch-reset',seed:1,p1:[mew(['Recover']),ghost(['Protect'])],p2:[lax(['Protect'])],setup:[{side:'p1',status:'tox'}],steps:[{p1:'move 1',p2:'move 1'},{p1:'move 1',p2:'move 1'},{p1:'switch 2',p2:'move 1'},{p1:'switch 2',p2:'move 1'}]},
 {id:'voluntary-retarget',seed:1,p1:[mew(['Tackle']),ghost(['Protect'])],p2:[lax(['Tackle'])],steps:[{p1:'switch 2',p2:'move 1'}]},
 {id:'pivot-resume',seed:1,p1:[mew(['U-turn']),lax(attack)],p2:[lax(attack)],steps:[{p1:'move 1',p2:'move 2'},{p1:'switch 2'}]},
 {id:'forced-replacement',seed:21,p1:[mew(['Mega Punch'],{level:100})],p2:[lax(attack,{level:1}),ghost(['Protect'])],steps:[{p1:'move 1',p2:'move 2'},{p2:'switch 2'}]},
 {id:'struggle-double-ko',seed:1,p1:[mew(['Tackle'])],p2:[ghost(['Toxic'])],setup:[{side:'p1',hp:1,pp:0},{side:'p2',hp:1,pp:0}],steps:[{p1:'move 1',p2:'move 1'}]},
 {id:'protect-repeat',seed:1,p1:[mew(['Protect'],{evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:252}})],p2:[mew(['Tackle'])],steps:[{p1:'move 1',p2:'move 1'},{p1:'move 1',p2:'move 1'},{p1:'move 1',p2:'move 1'}]},
 {id:'pivot-protected',seed:1,p1:[mew(['U-turn']),lax(attack)],p2:[lax(['Protect'])],steps:[{p1:'move 1',p2:'move 1'}]},
 {id:'commit-replace',seed:1,p1:[mew(attack,{evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:252}})],p2:[mew(attack)],steps:[{p1:'move 1'},{p1:'move 2'},{p2:'move 1'}]},
];
function snapshot(battle) {
 return {rng:battle.prng.getSeed(),turn:battle.turn,ended:battle.ended,winner:battle.winner,
 sides:['p1','p2'].map(id=>({id,request:battle[id].activeRequest,
   actions:battle[id].activeRequest?actionSpace(battle[id].activeRequest):null,
   pokemon:battle[id].pokemon.map(p=>({name:p.species.name,active:p.isActive,hp:p.hp,maxhp:p.maxhp,
     status:p.status,toxicStage:p.statusState.stage??null,item:p.item,speed:p.speed,
     storedStats:p.storedStats,pp:p.moveSlots.map(m=>m.pp),
     stall:p.volatiles.stall?{counter:p.volatiles.stall.counter,duration:p.volatiles.stall.duration}:null}))}))};
}
const output={pin,source:'7068c36',kind:'planning native probes; constructed teams, not generator samples',cases:[]};
for(const scenario of cases) {
 const battle=new ps.Battle({formatid:'gen5randombattle',seed:[scenario.seed,2,3,4],
   p1:{name:'p1',team:scenario.p1},p2:{name:'p2',team:scenario.p2}});
 const setupRng=battle.prng.getSeed();
 for(const setup of scenario.setup??[]) {
   const pokemon=battle[setup.side].active[0];
   if(setup.status) pokemon.setStatus(setup.status,pokemon);
   if(setup.hp!==undefined) pokemon.sethp(setup.hp);
   if(setup.pp!==undefined) for(const move of pokemon.moveSlots)move.pp=setup.pp;
 }
 if(scenario.setup)battle.makeRequest('move');
 if(battle.prng.getSeed()!==setupRng)throw Error('setup changed RNG');
 const initialPackets=['update\n'+battle.log.join('\n')];
 const record={id:scenario.id,setup:scenario,
   initialEvents:Object.fromEntries(['p1','p2'].map(side=>[side,projectPackets(initialPackets,side).records])),steps:[]};
 for(const choices of scenario.steps) {
   const before=snapshot(battle);const start=battle.log.length;
   const {draws}=withObservedBattleDraws(ps,battle,()=>{
     for(const [side,choice] of Object.entries(choices)) {
       if(!battle.choose(side,choice))throw Error(`${scenario.id}: ${side} rejected ${choice}`);
     }
   });
   const packets=['update\n'+battle.log.slice(start).join('\n')];
   record.steps.push({choices,before,after:snapshot(battle),draws,
     events:Object.fromEntries(['p1','p2'].map(side=>[side,projectPackets(packets,side).records]))});
 }
 output.cases.push(record);battle.destroy();
}
process.stdout.write(JSON.stringify(output,null,2)+'\n');
JS
```

The digest is computed from the complete JSON output with Python:

```python
import hashlib, json
from pathlib import Path
data = json.loads(Path('/tmp/g3-native-probes.json').read_text())
for case in data['cases']:
    for step in case['steps']:
        for draw in step['draws']:
            draw.pop('stack', None)
canonical = json.dumps(data, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=False).encode()
print(hashlib.sha256(canonical).hexdigest())
```

## Approved bounded G4 roster/hazard package

`reference/roster-hazard-cases.cjs`, selected by
`node reference/shared-engine-cases.cjs --batch g4`, independently observes
six-member supplied rosters and Stealth Rock, Spikes and Gen 5 Rapid Spin.
`test-logs/g4-roster-hazards/native-summary.json` records the pinned source,
case counts, replay command and observed ordering before implementation.

| Family | Required behavior |
|---|---|
| Placement | foeSide request; Protect bypass; Rock repeat and fourth Spikes failure; all three layer amounts; exact scheduler Update draws |
| Entry | Slot six and stable command order; Levitate skips Spikes; Rock and Spikes insertion order; Leftovers after entry; simultaneous swaps |
| Entry KO | Pending attack loses target; forced replacement chain preserves turn; U-turn queue resumes; last-member hazard KO reaches native terminal |
| Removal | Gen 5 power/PP/STAB; clear own Spikes then Rock after damage, before faint/win; Ghost/Protect retain hazards |
| Integrated | Four complete six-member battles, two seeds and swapped seats; exact state/RNG/queue/request/events/history; eager/JIT/batched continuation/reset and lean rollout |

Existing species/abilities/items/status constraints remain. Toxic Spikes and
full-format G4 closure are excluded, as are GPU and performance runs.

CPU package acceptance: 22 scenarios / 408 new boundaries (four complete
six-member games), plus all retained shared-engine/lifecycle/rollout controls,
passed in the 26-check integration run. The two native BattleStream checks also
passed. See `test-logs/g4-roster-hazards/cpu-integration.log`, `native-stream.log`
and `source.json`. No GPU or performance evidence is implied.
