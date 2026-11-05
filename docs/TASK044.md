# Task044 exact complete-battle parity for two-move Mew

## Scope and status

Validate the accepted `gen5-neutral-two-move-mew-1v1-v4` API from actual
initial native requests through genuine Showdown terminal outcomes. Exercise
scripted per-turn Tackle/Pound choices, including continuing on Pound after a
side spends its last Tackle PP. This task adds a bounded native duel fixture
mode and a focused CPU differential test; it adds no simulator feature.

Owner: `/root/task044`. Source base: `1f46d378e2ea1352171164baf80b97a01a2e015b`.
Status: accepted by primary `/root` on 2026-11-05. Accepted task count is 44;
review baseline 40 and next checkpoint 50 remain unchanged. G0 is the only
accepted milestone. GPU evidence is not required.

Owned paths: `reference/neutral-two-move-oracle.cjs`, new
`test/test_neutral_two_move_duel.py`, `docs/neutral-two-move-variant.md`, this
handoff, `docs/TASKS.md`, and `docs/ORCHESTRATION_STATE.json`. Do not edit the
existing simulator or the Task043 one-turn test. If native duel evidence
reveals a concrete simulator parity failure, report the exact trace and source
evidence to `/root` before any simulator change.

## Acceptance criteria

- Add explicit `--duels` fixture mode using the pinned native Battle API and
  preserve the no-argument seven-case one-turn fixture behavior and semantic
  fields.
- Capture complete native battles for p1-faster, p2-faster, and tied-speed
  cases. Every turn uses actual `Battle.makeChoices`, current native requests,
  a deterministic alternating planned move slot per player, and fallback to
  the other currently legal native slot only when the planned slot is
  exhausted. The PP case sets p1 Tackle PP to one through the live Pokemon,
  regenerates requests with `battle.makeRequest('move')`, and proves request
  setup did not change the original PRNG limbs.
- Each case must actually execute Tackle then Pound on later turns; the PP
  case must spend Tackle's last PP and later execute Pound with the correct
  disabled mask. Require genuine `Battle.ended`, p1/p2 winner, and terminal
  outcome within 56 turns. Do not exhaust both move slots or synthesize a
  terminal/budget result.
- Record every pre/post native state, all PP, submitted choice slots, move
  identities and action order, exact draw trace/context/limbs, recipient event
  projections, current requests, next requests, and actual terminal winner.
- In the focused CPU test, use one shared jitted transition plus static
  audience projection helper across all cases and turns. Thread each JAX
  state from the prior JAX result. Compare every native state field, all draw
  data, events/visibility, requests/legal masks, and move/PP execution. At
  native terminal, verify terminal winner, empty legal masks, and absorbing
  JAX submission with unchanged state/RNG and no events or draws.
- Inspect the complete diff and report exact commands/results, all elapsed
  validator time including failures, source identities/hashes, changed paths,
  and unresolved concerns. Do not accept, commit, or push.

## Exclusions

No simulator/API edits without a concrete parity mismatch reported to the
parent first; no new moves, effects, Struggle execution, session/reset
integration, driver/library, policy/training/reporting/trajectory
infrastructure, GPU/performance claims, long suites, or milestone claims.

## Plan and progress

1. Inspect the current task state, variant contract, oracle seams, and focused
   test helpers. Trail CLI and relevant recent source cards are unavailable;
   the documented exact-file fallback is in use.
2. Persist this bounded plan and active ownership before implementation.
3. Add the opt-in native duel mode, focused differential test, and concise
   variant/task-index updates.
4. Run only the smallest relevant checks within 30 seconds per command and 90
   seconds total, inspect all owned-file changes, and return the evidence
   package to `/root` for primary acceptance review.

## Implementation and evidence

The oracle preserves its no-argument one-turn path and adds `--duels` with four
bounded cases. Each uses actual native move requests and `Battle.makeChoices`
until the real winner: p1 faster (8 turns), p2 faster (9), tied (8), and the
last-Tackle-PP case (8). Both players actually execute Tackle and later Pound
in every battle. The PP case starts p1 at `[1, 56]`, spends Tackle's final PP,
shows the next native request with Tackle disabled and Pound enabled, then
executes Pound on the following turn. On the third turn p1's alternating plan
returns to exhausted Tackle, so the fixture selects only the other legal
native request slot, Pound. `battle.makeRequest('move')` preserves all four
initial PRNG limbs. Every turn records its full pre/post snapshot,
both requests, planned and submitted slots, commands, exact original draw
context and limbs, action order, both recipient event projections, next
requests, and winner.

The focused CPU test shares one jitted transition/projection function across
all turns and cases. It seeds JAX only from each case's actual initial native
snapshot, then threads each transition result into the next turn. It compares
all state fields and PP, native current/next requests and legal masks, move
identity and deductions, action order, recipient events and HP visibility,
and every draw bound/value/context/pre/post limb. It confirms each case ends
with the actual native winner, both terminal legal masks are empty, and a
subsequent JAX submission is absorbing with unchanged state/RNG and no draws
or executed events. The final run uses Python 3.12.14 and JAX/JAXLIB 0.12.0.1
on CPU. No GPU run is required.

Validation commands and results:

| Command | Result and elapsed time |
|---|---|
| `time -p timeout --signal=TERM --kill-after=5s 30s node --check reference/neutral-two-move-oracle.cjs` | Passed; real 0.02s. |
| `time -p timeout --signal=TERM --kill-after=5s 30s node reference/neutral-two-move-oracle.cjs --duels > /tmp/task044-native-duels.json` | Initial attempt failed after generating the four battles because the fixture incorrectly assumed initial Pound PP was 35; the live Gen 5 request reports 56. Real 0.38s. The assertion was corrected to require Pound remain available. The same command then passed; real 0.29s. |
| `time -p timeout --signal=TERM --kill-after=5s 30s /tmp/task040-jax-env/bin/python -m py_compile test/test_neutral_two_move_duel.py` | Passed; real 0.02s. |
| `time -p timeout --signal=TERM --kill-after=5s 30s env JAX_PLATFORMS=cpu /tmp/task040-jax-env/bin/python -m unittest test.test_neutral_two_move_duel -v` | Passed twice, including the final winner assertion. Final: 1 test in 2.732s; shell real 3.14s. Earlier revision: 1 test in 2.782s; shell real 3.19s. |
| No-argument fixture comparison using the Task044 source base from `git show 1f46d378e2ea1352171164baf80b97a01a2e015b:reference/neutral-two-move-oracle.cjs` and the current oracle | Initial byte comparison failed because `draw.stack` embeds the temporary baseline filename; Python real 0.55s. Diagnostic comparison identified the path difference; real 0.57s. A parsed JSON comparison that strips only diagnostic `stack` strings passed for all seven scenarios and all other fields; real 0.56s. |
| `time -p timeout --signal=TERM --kill-after=5s 30s git diff --check` | Passed; real 0.00s. |
| `time -p timeout --signal=TERM --kill-after=5s 30s` Python state/link assertion script | Passed: JSON parsed, accepted-task history/count, review baseline/checkpoint, G0 approvals and active owner preserved, and all local links in touched Markdown resolve; real 0.02s. |

An attempted `/usr/bin/time -p ... node --check ...` command could not start
because `/usr/bin/time` is absent; the shell `time -p` form above worked. The
final parallel static checks also passed `node --check` (real 0.02s) and
`py_compile` (real 0.02s). The focused test command, native fixture attempts,
no-argument comparisons, and recorded static checks totaled about 8.8 seconds
of measured shell real time, well under the 90-second budget. No timeout
occurred.

The optional old full module command
`JAX_PLATFORMS=cpu /tmp/task040-jax-env/bin/python -m unittest test.test_neutral_two_move -v`
was deferred because it includes the known-slow boundary checks. It is not an
acceptance dependency. Its exact command and source blobs are recorded in
`deferred_validation` in the orchestration state. Trail is unavailable at the
documented CLI path, and its cached graph has no cards for these recent files;
the targeted source fallback was used.

Final code blob identities (Git blob IDs):

| File | Git blob | SHA-256 |
|---|---|---|
| `reference/neutral-two-move-oracle.cjs` | `31745d060126b9c8f9d259ab1169771f12be6cb4` | `0921411104ddadedf9ad4dc62cf4c9dabc03470dd26f0fe44c96dffa0ee903fd` |
| `test/test_neutral_two_move_duel.py` | `078331cee477207badb4edfdcd792ed26b472a70` | `6d1f5cc11b608c9352cae5c851b5c587a267fb49cd3efb5af572cd78e728d448` |
| unchanged `simulator/neutral_two_move.py` | `013dda42c232b47b2e5bd815b7c57a43d817199f` | — |
| unchanged `test/test_neutral_two_move.py` | `6ba18e596c1a18761f9d42a7787b9e351a13b6ab` | — |

Changed paths are `reference/neutral-two-move-oracle.cjs`,
`test/test_neutral_two_move_duel.py`, `docs/neutral-two-move-variant.md`, this
handoff, `docs/TASKS.md`, and `docs/ORCHESTRATION_STATE.json`. No simulator
file changed. Primary `/root` accepted the frozen oracle, focused test, exact
trace evidence, no-argument semantic compatibility, and complete runtime
record on 2026-11-05. No Astra review or milestone acceptance is claimed;
there is no unresolved simulator parity concern. No commit or push was made.
