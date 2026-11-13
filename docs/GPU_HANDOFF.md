# GPU handoff — G3 closure and G4 device conformance

**Deferred during travel, 2026-11-08.** The user will run GPU work after returning
home and can give the result bundle to an agent. This document is the living
record for all GPU work arising from G3 and G4. CPU implementation continues without a
GPU. Preparing this handoff does not run device tests, profiling or benchmarks.

**Validation order, 2026-11-13:** GPU correctness, broad CPU matrices,
complete-battle campaigns and profiling remain deferred until reachable mechanics
and inventory closure are complete. This supersedes earlier checkpoint timing;
all acceptance requirements and commands below remain pending.

## Current source and evidence

**Schema33 Mummy checkpoint passed focused CPU validation.** Native22 scenarios/
27 boundaries agree exactly in274.177s, including live changed ability/form
NPZ/journal16 persistence;2 native/inventory checks pass.
See [schema33-checkpoint.json](../test-logs/g4-cpu-conformance/schema33-checkpoint.json).
Current runner includes95 checks/39 modules,1055 scenarios/5134 boundaries,
`test.test_mummy_engine` and the `mummy` batch. Runner preparation and broad/device
checks remain deferred until reachable mechanics and inventory closure.

```sh
JAX_PLATFORMS=cuda "$JAX_PYTHON" -m unittest test.test_mummy_engine -v -f
```

Mummy preserves ordered Helmet/status-thaw/End/activation behavior, including
source/target KO, Substitute, suppression, Trace, Illusion and Zen Mode.
Schema32 state layout and journal16 remain unchanged. Retain the broad/migration,
full/lean and final campaign obligations below; no device run is claimed.

**Schema32 Trace checkpoint passed focused CPU validation.** Native21 scenarios/
26 boundaries agree exactly in305.122s, including saved seek NPZ/journal16.
Two native/inventory checks and one narrow both-holder Update status-cure order
regression pass. See [schema32-checkpoint.json](../test-logs/g4-cpu-conformance/schema32-checkpoint.json).
The compilation fix shares the holder-loop body and ordinary Start dispatch.
Peak focused-process RSS was7,292,972KiB; this host has no cgroup memory cap,
so this does not claim a replay under the original8GiB cap. Historical exit137
and OOM records remain in schema32-wip.json and its archive.

The schema32 checkpoint runner included94 checks/38 modules and1033 scenarios/5107 boundaries,
including `test.test_trace_engine`, `test.test_trace_update_regression` and the
`trace` batch. Preparation and all device checks remain deferred. Exact isolated
commands, pending until mechanics and inventory closure:

```sh
JAX_PLATFORMS=cuda "$JAX_PYTHON" -m unittest test.test_trace_engine -v -f
JAX_PLATFORMS=cuda "$JAX_PYTHON" -m unittest test.test_trace_update_regression -v -f
```

Retain the final migration/full-lean gates below. The original preliminary Mummy witnesses remain archived under `next-mummy/`
in schema32-wip-validation.tar.gz; the completed schema33 expansion is above.

**Schema31 Wonder Guard direct immunity checkpoint:** focused CPU parity passed18
scenarios/22 boundaries in303.667s; one native contract passed.
Wonder Guard covers direct immunity ordering, effective Hidden Power type,
Shedinja HP, Struggle/status/confusion/hazards/weather bypasses, suppression
and ability replacement restrictions; live NPZ/journal15 persistence.
See [schema31-checkpoint.json](../test-logs/g4-cpu-conformance/schema31-checkpoint.json).
The runner includes `test.test_wonder_guard_engine` and the `wonderguard` batch.
Isolated replay: `JAX_PLATFORMS=cuda "$JAX_PYTHON" -m unittest test.test_wonder_guard_engine -v -f`.
Broad execution/migration and current runner preparation remain pending.

**Schema30 Magic Coat condition reflection checkpoint:** focused CPU parity passed18
scenarios/23 boundaries in300.916s; one native contract passed.
Magic Coat covers one-turn expiry, precedence over Magic Bounce, reflected
statuses/hazards/abilities/phazing, Sleep Talk, and ordinary/Prankster pending
pivot NPZ/journal15 persistence.
See [schema30-checkpoint.json](../test-logs/g4-cpu-conformance/schema30-checkpoint.json).
The runner includes `test.test_magic_coat_engine` and the `magiccoat` batch.
Isolated replay: `JAX_PLATFORMS=cuda "$JAX_PYTHON" -m unittest test.test_magic_coat_engine -v -f`.
Broad execution/migration and current runner preparation remain pending.

**Schema29 Magic Bounce reflected frames checkpoint:** focused CPU parity passed22
scenarios/26 boundaries in275.568s; one native contract passed.
Magic Bounce reflection covers no-extra-PP accounting, Protect/immunity/air,
Gen5 side hazards in air, suppression, reflected phazing and a three-frame
Sleep Talk chain; live NPZ/journal14 persistence is included.
See [schema29-checkpoint.json](../test-logs/g4-cpu-conformance/schema29-checkpoint.json).
The runner includes `test.test_reflection_engine` and the `reflection` batch.
Isolated replay: `JAX_PLATFORMS=cuda "$JAX_PYTHON" -m unittest test.test_reflection_engine -v -f`.
Broad execution/migration and current runner preparation remain pending.

**Schema28 Copycat and Mirror Move frames checkpoint:** focused CPU parity passed24
scenarios/48 boundaries in280.722s; one native contract passed.
Copycat/Mirror Move cover Pressure, caller/global memory, priority, protection,
charges/rampage absent from own slots, pivot suspension, Torment and live
NPZ/journal14 persistence. Frame capacity proof remains pending.
See [schema28-checkpoint.json](../test-logs/g4-cpu-conformance/schema28-checkpoint.json).
The runner includes `test.test_copy_move_engine` and the `copymoves` batch.
Isolated replay: `JAX_PLATFORMS=cuda "$JAX_PYTHON" -m unittest test.test_copy_move_engine -v -f`.
Broad execution/migration and current runner preparation remain pending.

**Schema27 Nature Power shared frames checkpoint:** focused CPU parity passed14
scenarios/19 boundaries in271.105s; one native contract passed.
Nature Power uses the same child frame for Gen5 Earthquake; caller PP/action
identity remains distinct. Pressure, protection/immunity, priority, held items,
Sleep Talk filtering and live NPZ/journal13 persistence are covered.
See [schema27-checkpoint.json](../test-logs/g4-cpu-conformance/schema27-checkpoint.json).
The runner includes `test.test_nature_power_engine` and the `naturepower` batch.
Isolated replay: `JAX_PLATFORMS=cuda "$JAX_PYTHON" -m unittest test.test_nature_power_engine -v -f`.
Broad execution/migration and current runner preparation remain pending.

**Schema26 Sleep Talk active-move frames checkpoint:** focused CPU parity passed20
scenarios/51 boundaries in274.191s; one native contract passed.
The shared root/called useMove kernel uses explicit frames; caller PP/action
identity and called Pressure/lastMoveUsed are distinct. Pivot/phazing/recharge,
terminal calls and live NPZ/journal13 persistence are covered. Eight frames is
a guarded provisional capacity; full call/reflection closure proof remains open.
See [schema26-checkpoint.json](../test-logs/g4-cpu-conformance/schema26-checkpoint.json).
The runner includes `test.test_called_move_engine` and the `calledmoves` batch.
Isolated replay: `JAX_PLATFORMS=cuda "$JAX_PYTHON" -m unittest test.test_called_move_engine -v -f`.
Broad execution/migration and current runner preparation remain pending.

**Schema25 Own Tempo and Yawn checkpoint:** focused CPU parity passed18
scenarios/36 boundaries in214.526s; one native contract passed.
Delayed Yawn sleep retains the original caster through switching. Own Tempo
block/cure/fatigue, Sleep Clause, immunity/Lum and live NPZ/journal12 persistence
are covered.
See [schema25-checkpoint.json](../test-logs/g4-cpu-conformance/schema25-checkpoint.json).
The runner includes `test.test_sleep_control_engine` and the `sleepcontrol` batch.
Isolated replay: `JAX_PLATFORMS=cuda "$JAX_PYTHON" -m unittest test.test_sleep_control_engine -v -f`.
Broad execution/migration and current runner preparation remain pending.

**Schema24 Gastro Acid suppression checkpoint:** focused CPU parity passed23
scenarios/47 boundaries in195.494s; one native contract passed.
Raw/base ability is retained while modifiers/callbacks are suppressed. PP,
trapping, forms, copying, suppressed Start, restoration and live NPZ/journal11
persistence are covered.
See [schema24-checkpoint.json](../test-logs/g4-cpu-conformance/schema24-checkpoint.json).
The runner includes `test.test_suppression_engine` and the `suppression` batch.
Isolated replay: `JAX_PLATFORMS=cuda "$JAX_PYTHON" -m unittest test.test_suppression_engine -v -f`.
Broad execution/migration and current runner preparation remain pending.

**Schema23 ability creation and boost changes checkpoint:** focused CPU parity passed23
scenarios/48 boundaries in190.058s; one native contract passed.
Worry Seed/Simple Beam lifecycle, existing sleep cure, Zen/Illusion End, Simple/
Contrary boost paths/caps and live NPZ/journal10 persistence are covered.
See [schema23-checkpoint.json](../test-logs/g4-cpu-conformance/schema23-checkpoint.json).
The runner includes `test.test_ability_create_engine` and the `abilitycreate` batch.
Isolated replay: `JAX_PLATFORMS=cuda "$JAX_PYTHON" -m unittest test.test_ability_create_engine -v -f`.
Broad execution/migration and current runner preparation remain pending.

**Schema22 status immunity checkpoint:** focused CPU parity passed18
scenarios/39 boundaries in195.922s; one native contract passed.
Status prevention, acquisition cures before queued actions, simultaneous cure order,
Rest denial and live NPZ/journal10 persistence are covered.
See [schema22-checkpoint.json](../test-logs/g4-cpu-conformance/schema22-checkpoint.json).
The runner includes `test.test_status_immunity_engine` and the `statusimmunity` batch.
Isolated replay: `JAX_PLATFORMS=cuda "$JAX_PYTHON" -m unittest test.test_status_immunity_engine -v -f`.
Broad execution/migration and current runner preparation remain pending.

**Schema21 ability exchange checkpoint:** focused CPU parity passed29 scenarios/58
boundaries in176.860s; one native contract passed. Role Play, Skill Swap and
Entrainment cover callback order, original private ability, Illusion End,
protection/Substitute, Imposter Start semantics and switch/faint provenance.
Live acquired-ability NPZ/journal10 persistence is covered. See
[schema21-checkpoint.json](../test-logs/g4-cpu-conformance/schema21-checkpoint.json).
The runner includes `test.test_ability_change_engine` and the abilitychange batch.
Isolated replay: `JAX_PLATFORMS=cuda "$JAX_PYTHON" -m unittest test.test_ability_change_engine -v -f`.
Broad execution/migration and current runner preparation remain pending.

**Schema20 triggered forms checkpoint:** focused CPU parity passed17 scenarios/30
boundaries in161.755s; one native contract passed. Relic Song/Meloetta and
Zen Mode/Darmanitan triggers, recovery reversal, silent End on switching,
base-form starts, failure/Transform interactions and live Zen NPZ/journal9
persistence are covered. See
[schema20-checkpoint.json](../test-logs/g4-cpu-conformance/schema20-checkpoint.json).
The runner includes `test.test_trigger_form_engine` and the triggerforms batch.
Isolated replay: `JAX_PLATFORMS=cuda "$JAX_PYTHON" -m unittest test.test_trigger_form_engine -v -f`.
Broad execution/migration and current runner preparation remain pending.

**Schema19 weather forms checkpoint:** one focused CPU check passed16 scenarios/43
boundaries in155.306s; one native contract passed. Forecast/Flower Gift,
weather cycle/expiry/startup/entry, original requests, Transform exclusions,
Attack/Special Defense modifiers and live NPZ/journal8 persistence are covered.
See [schema19-checkpoint.json](../test-logs/g4-cpu-conformance/schema19-checkpoint.json).
The runner includes `test.test_form_engine` and the forms native batch.
Isolated replay: `JAX_PLATFORMS=cuda "$JAX_PYTHON" -m unittest test.test_form_engine -v -f`.
Broad execution/migration and current runner preparation remain pending.

**Schema18 Illusion checkpoint:** focused CPU parity passed12 scenarios/25
boundaries in138.869s; one native contract passed. Immutable event display names
and details preserve pre-reveal history, including mid-multihit reveals. Entry,
switch retention, hazards/status/Substitute, Transform blocking, lethal damage
and live disguised NPZ/journal8 persistence are covered. See
[schema18-checkpoint.json](../test-logs/g4-cpu-conformance/schema18-checkpoint.json).
The runner includes `test.test_illusion_engine` and the Illusion native batch.
Isolated replay: `JAX_PLATFORMS=cuda "$JAX_PYTHON" -m unittest test.test_illusion_engine -v -f`.
Broad execution/migration gates and current runner preparation remain pending.

**Schema17 Transform/Imposter checkpoint:** one focused CPU check passed17
scenarios/36 boundaries in111.411s; one native fixture contract passed. Base and
effective identity, virtual PP, copied boosts/abilities, original requests,
known/apparent type trap queries and live NPZ/journal7 roundtrip are covered.
See [schema17-checkpoint.json](../test-logs/g4-cpu-conformance/schema17-checkpoint.json).
The runner now includes `test.test_identity_engine` and the identity native batch.
Isolated GPU replay: `JAX_PLATFORMS=cuda "$JAX_PYTHON" -m unittest test.test_identity_engine -v -f`.
Broad execution/migration gates remain pending; no GPU or profiling ran.

**Schema16 item lifecycle checkpoint:** focused CPU parity passed28 scenarios/44
boundaries in94.580s; one native contract passed. It covers Trick/Switcheroo,
Gen5 Knock Off, Recycle, Sticky Hold, Choice Start/lock timing, immutable initial
item/pending provenance and post-exchange NPZ/journal persistence. See
[schema16-checkpoint.json](../test-logs/g4-cpu-conformance/schema16-checkpoint.json).
State schema16 appends item provenance; journal6 binds it. The return-home runner
includes `test.test_item_lifecycle_engine`; isolated replay is
`JAX_PLATFORMS=cuda "$JAX_PYTHON" -m unittest test.test_item_lifecycle_engine -v -f`.
Broad execution/legacy migrations and current preparation remain pending. Transform/Imposter is implemented in schema17 below; the earlier native-only
specification was preliminary.
No GPU work or profiling ran.

**Schema15 Pursuit checkpoint:** the focused CPU check passed23 scenarios/41
boundaries in89.858s; one native contract passed. It covers normal/intercepted
Pursuit, voluntary switches and both U-turn orders, skipped turns, PP/Pressure,
Substitute/Protect, Choice/Technician, KO/double-KO cancellation, mirrored seats,
and live pivot NPZ/journal persistence. See
[schema15-checkpoint.json](../test-logs/g4-cpu-conformance/schema15-checkpoint.json).
The source uses queued interception plus finish work through one move dispatcher;
being-called-back/activation/skip flags are persisted in schema15, journal5.
The return-home runner includes `test.test_pursuit_engine`; isolated correctness
replay is `JAX_PLATFORMS=cuda "$JAX_PYTHON" -m unittest test.test_pursuit_engine -v -f`.
Broader execution/lifecycle coverage and current preparation remain deferred.
No GPU work, profiling or budget sweeps ran.

**Schema14 phazing checkpoint:** focused CPU parity passed36 scenarios/46
boundaries in81.552s; the pinned native contract passed. Coverage includes four
phazing moves, Substitute/Protect/no-bench, Suction Cups/Soundproof, cancellation,
synchronous drag hazards, source KO and pivot interruption. See
[schema14-checkpoint.json](../test-logs/g4-cpu-conformance/schema14-checkpoint.json).
The return-home runner now includes `test.test_phazing_engine`; isolated device
replay is `JAX_PLATFORMS=cuda "$JAX_PYTHON" -m unittest test.test_phazing_engine -v -f`.
Broader execution-mode/lifecycle coverage and current-source preparation remain
pending; no GPU work or profiling ran. Trace adds transient force-switch flags;
persistent storage and journal schema4 are unchanged.

**Schema13 partial-trapping checkpoint:** the focused CPU check passed27
scenarios/105 boundaries in76.862s; one pinned native contract also passed.
Seven binding moves, Binding Band/Grip Claw, expiry/source cleanup, disclosed
switch rejection, Substitute/Spin/pivot interactions and current/legacy
persistence are covered. Exact source/input identities and remaining checks are
in [schema13-checkpoint.json](../test-logs/g4-cpu-conformance/schema13-checkpoint.json).
State schema13 adds binding duration/divisor/source state; journal schema4 binds
that shape. Broad eager/JIT/mixed/full-lean/reset testing was added for the
consolidated gate and has **not run**. Current-source preparation of all13 corpora
has **not been repeated**; the return-home runner regenerates them automatically.

The existing return-home block includes both new methods. For isolated device
replay after selecting the documented CUDA interpreter:

```bash
JAX_PLATFORMS=cuda "$JAX_PYTHON" -m unittest test.test_partial_trap_engine -v -f
```

The pending CPU matrix can be run at the agreed milestone gate with:

```bash
JAX_PLATFORMS=cpu "$JAX_PYTHON" -m unittest test.test_partial_trap_engine.PartialTrapEngineTest.test_deferred_eager_mixed_rollout_and_reset -v -f
```

These are correctness obligations; no extra profiling/ablation workload is added.

**Schema12 Substitute checkpoint:** one focused CPU differential check passed
all51 scenarios/184 boundaries in83.623s. Two completed same-source CPU checks
cover atomic metadata/persistence and complete-game core/journal/full/lean/reset.
Six native and six helper checks also passed. Exact sources, inputs and completed
versus pending evidence are in [schema12-checkpoint.json](../test-logs/g4-cpu-conformance/schema12-checkpoint.json).
The interrupted eager/JIT/mixed1/8/32 check and broader regressions remain pending;
they were not restarted under the user-approved reduced validation policy.

The runner includes the51 Substitute scenarios and two complete six-member games.
`test.test_substitute_engine` covers interception/RNG, native state/requests/both
histories, recoil/drain/secondary/contact/recharge/pivot/Spin interactions,
save/resume, full/lean/core/journal execution, reset and atomic rejection.
State schema12 and journal schema3 add the doll-HP slot with explicit migration.
Preparation of all twelve corpora passed with matching frozen source identities.
The return-home commands below are consolidated validation, not routine per-family
checks. No GPU execution or new profiling workload ran; full G4 remains open.

**Schema11 trapping/retry checkpoint:** eight affected CPU checks passed,
plus six native and six helper checks. The new31 scenarios/215 boundaries include
two complete games and mirrored private-state cases. Exact source identities,
fixture hashes and the descriptor-only version-label correction are recorded in
[schema11-checkpoint.json](../test-logs/g4-cpu-conformance/schema11-checkpoint.json).
The full68-check suite and GPU correctness remain pending. Preparation of all
eleven native corpora succeeded without JAX/GPU execution or profiling.

**Schema10 checkpoint:** three integrated CPU checks passed on `a9d0fbd`,
covering full/core native combat parity, mixed1/8/32,full/lean,core reset,
checksum-bound journal persistence and the exact1000-boundary turn-limit replay.
Five native and six helper checks also passed. Source/fixture hashes stayed
unchanged; see [schema10-checkpoint.json](../test-logs/g4-cpu-conformance/schema10-checkpoint.json).
The full64-check matching-source suite has not been repeated. GPU execution and
profiling remain deferred. The older schema9 evidence below is historical.

The preceding schema9 checkpoint, adding hidden causal state, rampages, eight charge moves,
semi-invulnerability, Power Herb, action restrictions and flinching. The four
phase execution/lifecycle checks and 13 prior scalar/migration checks passed on
engine commit `567f02c`, with unchanged source and fixture hashes. A separate
schema1/2 migration/rollout check also passed: **18 affected CPU checks total**.
Exact source and results are recorded in [schema9-checkpoint.json](../test-logs/g4-cpu-conformance/schema9-checkpoint.json).
The entire matching-source 61-check suite has not been repeated. Historical
schema8 results remain in
[schema8-checkpoint.json](../test-logs/g4-cpu-conformance/schema8-checkpoint.json).

The expanded runner expects **92 checks in 36 isolated modules**, using 31
independent native corpora: **1012 scenarios / 5081 boundaries**, including
32 complete constructed six-member combat games (plus the neutral switch-only
turn-limit game and eight historical three-member games). The 73 phase scenarios are in
`test.test_lock_phase_engine`. The runner recaptures causal state for the prior
corpora too. Use the existing commands below on matching committed source.
Full G4 remains open: remaining reachable mechanics, closure audit, history and
terminal repair and fixed full-format campaigns are still required. A genuine
reference-generated switch-only game proves the existing history bound
insufficient (6541 visible rows). That witness is not an implemented GPU test.
The 1024-row per-holder attack list is guarded atomically; its final reachable
capacity audit also remains pending. No device run or performance result is
claimed by this update.

The preceding **49-check schema 6 CPU checkpoint passed on `7631872`**, with
unchanged source/input hashes and actual CPU placement. Its
[record](../test-logs/g4-cpu-conformance/schema6-checkpoint.json) and regression
archive retain commands, fixture/source identities and every module log.
That result is historical evidence for schema 9. Earlier incomplete/OOM and
schema3/5 evidence remain in [active progress](ORCHESTRATION_STATE.json).
The runner isolates modules and integration suites release compiled caches;
cases and batch sizes are retained.

Use the **latest committed checkout** and retain its recorded identities. The
runner fingerprints engine, tables, IO, tests, oracle, configuration and benchmark
code; it separately hashes each freshly captured native fixture. Device results
cannot transfer across changed relevant source or fixtures.

## Deferred broad CPU and campaign checks

Run only after reachable mechanics and inventory closure. The current checkpoint
modules (including eager/JIT/mixed batches, lifecycle and full/lean checks) run in
isolated processes; this does not replace inventory or fixed-campaign acceptance.

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_COMPILATION_CACHE=false .venv-jax/bin/python - <<'PYCPU'
import os,runpy,subprocess,sys
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:3])
for module in runpy.run_path('scripts/g4_gpu_handoff.py')['MODULES']:
    subprocess.run([sys.executable,'-m','unittest',module,'-v','-f'],check=True)
PYCPU
```

Fixed 64-development / 128-held-out starts retain every acceptance condition in
PROJECT_PLAN.md. The exact intended CPU command is below; **pending harness
implementation**, since `evaluation.full_format_conformance` does not yet exist.
Keep source/fixture/config/policy identities and held-out reuse history. Never
substitute the checkpoint modules or truncated battles for this gate.

```bash
JAX_PLATFORMS=cpu "$JAX_PYTHON" -m evaluation.full_format_conformance \
  --config evaluation/g4-cpu-v1.json --suite acceptance --output NEW_CPU_DIRECTORY
```

## Return-home setup

Run from the repository root on Linux with one NVIDIA GPU compatible with the
chosen JAX CUDA wheel. Node 24 and the guarded pinned Showdown build are needed.
The reference pin remains `1242c1e2d9763ba5b85d33f13aecb89bae19ae3d`.

Use an existing CUDA-capable interpreter if available. Otherwise create a
separate environment; preserve the CPU environment. The project's recorded
JAX/JAXLIB version is `0.12.0.1`:

```bash
git pull --ff-only
python3 -m venv .venv-g4-gpu
.venv-g4-gpu/bin/python -m pip install 'jax[cuda13]==0.12.0.1'
export JAX_PYTHON="$PWD/.venv-g4-gpu/bin/python"

# Only if the pinned reference is missing/unbuilt:
node reference/setup.cjs

# Continue with the combined correctness + G3 closure block below.
```

CUDA 13 needs a compatible NVIDIA driver/device. `nvidia-smi` plus the strict
CUDA probe in the runner record what actually works. If this wheel is unavailable
or incompatible, retain the setup error for the reviewing agent; select a
compatible documented stack rather than silently changing versions or using CPU
fallback. The historical 8 GiB GPU is context, not a promise that future G4
workloads fit. No paid hardware, driver installation or infrastructure changes
are part of these commands. The runner targets one selected GPU; on a multi-GPU
machine set `CUDA_VISIBLE_DEVICES` to the intended device before invoking it.

The correctness runner (`"$JAX_PYTHON" scripts/g4_gpu_handoff.py`), used by
the combined block below:

1. Creates a fresh `work/g4-gpu-TIMESTAMP/` and records commit/tree, relevant file
   hashes, selected host affinity and independent native fixture hashes.
2. Freezes all sixteen legacy/current corpora, including move phases, the
   turn-limit witness, trapping/retries, Substitute and partial trapping. Runs the reference/native
   contracts. Unknown/unsupported families stay explicit.
3. Runs the same current 92-check scope on CPU, then under `JAX_PLATFORMS=cuda`,
   with each test module in a fresh process.
   A strict probe verifies actual eager/JIT GPU placement; shared case/lifecycle
   assertions also check device leaves. `cuda` initialization failure is a failure.
4. Saves package/device/driver information, exact command arrays, full logs and
   exit codes. Stops on failure/timeout, with no automatic retry or reduced batch.
5. Writes `summary.json` and a sibling `.tar.gz`, including on ordinary failures
   or interruption. The archive is the artifact to give to an agent.

The CPU worker budget is `min(8, available affinity CPUs)`, identical for both
backends; it is not a tuning sweep. GPU allocations grow on demand; persistent
compilation caching is disabled and recorded by the command environment. Each
isolated module has a 3600-second cap. The expanded suite is longer than the
historical 11-minute checkpoint; current GPU runtime/memory remain unmeasured. A timeout/OOM is incomplete evidence, not a
passing shard. Diagnose before changing the cap or workload.

Optional output location and a preparation-only check:

```bash
"$JAX_PYTHON" scripts/g4_gpu_handoff.py --output work/my-fresh-g4-gpu-run

# Usable while travelling; standard-library/Node preparation only, no JAX/GPU:
python3 scripts/g4_gpu_handoff.py --prepare-only --output work/my-fresh-g4-gpu-prep
```

Each output directory must be new. Relevant source/fixtures must be committed
for device validation. The runner never edits the engine, installs packages,
resets Git, commits, pushes, or launches training/profiling. Do not run both
correctness commands as a routine repeat; these are alternatives to the
checkpoint phase of the combined block below.

The added device checks are correctness checks, not extra profiling points:

- Status: sleep/freeze/confusion durations and RNG, Rest/Lum retained timers,
  Sleep Clause metadata, secondary effects and self-KO; exact saved continuation,
  full/lean complete games and mixed 1/8/32-lane isolation.
- Field: cached Speed after boosts, critical stage bypass and minimum screen
  damage, native weather/screen handler draws and expiry, weather double KO,
  rocks/Light Clay and hazard source/order. Replay complete games, restore/reset,
  invalid metadata rejection and mixed 1/8/32 lanes on the actual GPU.

- Hooks: startup ordering and cached Speed with permanent weather; Pressure PP
  saturation, Prankster priority, stat/power/critical/accuracy/damage modifier
  ordering and fixed-point rounding; Choice-lock requests/source metadata,
  Struggle and sleep; Life Orb double KO/pivot outcomes. Include heterogeneous
  batches, eager/JIT initialization, exact saved continuation/reset and both
  rollout outputs. These are the added `test.test_hook_engine` checks.

- Hit/thaw: exact 2/3/4/5-hit sampling and interruption; drain/recoil caps and
  rounding; Skill Link/Rock Head; immediate Rough Skin/Iron Barbs/Rocky Helmet
  and Liquid Ooze faints; single Life Orb application; self-destruction and
  distinct double-KO outcomes. Check frozen-user thaw versus frozen Scald
  targets, confusion blocking and Choice locks. Run the same eager/JIT,
  1/8/32-lane, save/restore/reset and full/lean checks on actual GPU leaves.
  The runner includes `test.test_hit_engine` and freezes `--batch hits` inputs.

- Recharge: the runner freezes `--batch recharge` and executes
  `test.test_recharge_engine` on CPU and CUDA. Check all six moves, minimal
  trapped requests, slot-2 moves versus the recharge pseudo-action, pending/undo
  and rejected switches/slots; PP exhaustion/Pressure/Choice; freeze, sleep,
  paralysis and confusion priority; duration-handler RNG, contact/Life Orb and
  residual faints, forced replacements and native terminal timing. Include
  eager/JIT and 1/8/32 mixed lanes, serialization of locked/private pending states,
  atomic invalid/stale/overflow behavior, lossless schema7 migration, and both
  complete six-member full/lean rollouts with continuation/reset.

## GPU obligations and readiness

| Obligation | Commands / current readiness | Evidence required |
|---|---|---|
| Current G4 data/hazard/generic/status/field/hook/hit/recharge device correctness | **Runnable checkpoint:** the runner above; check the CPU validation record first, then use the same CPU/CUDA modules and frozen fixtures | Both suites pass; real GPU leaf placement, exact native state/draw/request/view/outcome equality, batches 1/8/32, saved continuation/reset, full/lean equality, invalid/stale/unsupported/overflow isolation. Covers the current checkpoint only. |
| Subsequent G4 mechanics and schema changes | **Append before the next CPU checkpoint:** add implemented family tests and native batches to the runner and this handoff; record CPU source/evidence | GPU replay on matching final source, including new counters/volatiles, queues, visibility, per-lane RNG and serialization. Native-only cases cannot substitute for JAX checks. |
| Full G4 release conformance | **Blocked by implementation, not travel alone:** V7 executable and GPU config are absent; CPU conformance is still incomplete | Closed reachable ledger and all required directed cases; same fixed 64 development / 128 held-out starts and choices as CPU, true native terminals, zero divergence/truncation/capacity errors. Scalar JIT and heterogeneous 8/32 lanes; compare in chunks of at most 16 boundaries. Preserve held-out reuse history and source/fixture/config/policy identities. |
| G4 device-specific profiling | **No required G4 profiling experiment is currently defined.** Add a bounded command here if an actual device/capacity issue requires one | Exact workload/output contract, hardware/VRAM, CPU budget, cold compile versus synchronized warm execution, useful work counts, memory and actual GPU stream events. A profiler trace is not conformance or a speedup result. |

The full-format GPU entry point must be finalized by the implementing agent.
This is the planned command shape, **not runnable now**; do not create a fake
pass by running the checkpoint suite in its place:

```bash
JAX_PLATFORMS=cuda "$JAX_PYTHON" -m evaluation.full_format_conformance \
  --config evaluation/g4-gpu-v1.json --suite acceptance --output NEW_GPU_DIRECTORY
```

Before making that command runnable, implement the V7 module and GPU configuration
with the same frozen seeds, policies, teams, retry/turn caps and expected traces
as [the CPU configuration](../evaluation/g4-cpu-v1.json). The existing CPU config
explicitly has `gpu: false`; simply changing an environment variable is not a
completed full-format GPU harness. Refresh this section with the real exact CLI.

## G3 closure: remaining obligations and bounded commands

G3's CPU playable-engine goal is complete. Older GPU parity, matched
Showdown/CPU/GPU timings and a hardware capture exist in
[G3_TESTING_PROFILING_RESULTS.md](G3_TESTING_PROFILING_RESULTS.md), but describe
older source/output contracts. They must not be substituted for the current
shared engine, schema 3 tables or lean rollout. G3 does **not** require GPU speedup,
full-format G4, a large batch sweep or further optimization to close.

| G3 gate | Return-home evidence |
|---|---|
| Actual GPU correctness and placement | The current checkpoint runner already includes shared case/battle/lifecycle and lean-rollout tests on CPU and strict CUDA. Reuse that one run for G3 and the implemented G4 checkpoint; do not repeat the same suites under another name. All required phases, mixed 1/8/32 lanes, native equality, privacy, continuation/reset, error isolation and lean/full behavior must pass. |
| V6 homogeneous and heterogeneous measurements | Run the two-cohort command below: `ordinary` repeats the completed `M03-duel-accuracy`; `mixed` uses F01/F02/F03/F04 with phase offsets. One batch (32), one chunk (8), ten measured samples per timer with target 0.2s/sample. No ablation points or budget sweep. The existing harness additionally records host transfers/projection/submission/reset timings; those are ancillary, not device timer inputs. |
| Compile, synchronized warm execution and memory | Retain kernel/scan lowering and compilation, warmup, raw warmed samples, useful work/reset/error counts, compiler argument/output/temp/alias bytes, retained output bytes and `device_memory_stats`. Compiler/allocator statistics and payload bytes are **not measured physical peak VRAM**. Mark absent device counters explicitly; an agent must identify any remaining memory-evidence gap instead of fabricating peak usage. |
| Acceptance record | An agent audits the exact source/fixture/config/runtime identities and logs, confirms both cohorts completed without skips/errors, then reconciles V6 and the G3 plan/state. Successful commands alone do not automatically accept a milestone. Leave G4's remaining implementation and full-format GPU gates open. |

### One return-home block: correctness plus G3 V6 measurements

Run after setup above. This performs one CPU/CUDA checkpoint suite, then just
the two GPU measurement points; it does not launch a hardware capture, diagnostic
ablation, learning run or full G4 campaign. The measurement process has a
**300-second cap**; correctness has a 3600s cap per isolated test module on each backend.
These are caps, not expected runtimes. A timeout/OOM/skip/source change is failed
or incomplete evidence: preserve artifacts and stop, with no automatic retries.

The block runs in a child shell, limits host affinity to at most eight available
CPUs, creates a fresh directory, and always attempts to archive logs on exit.
`JAX_PYTHON` must be exported to the selected CUDA-capable interpreter. Review the
printed archive with an agent before running any optional follow-up.
The current V6 harness queries physical GPU 0 for its memory preflight. Use that
device for this combined block; on multi-GPU systems another selected device
needs that preflight mapping corrected first. The correctness-only runner can
still validate a different explicitly selected GPU.

```bash
bash <<'BASH'
set -euo pipefail
: "${JAX_PYTHON:?Set JAX_PYTHON to your CUDA-capable interpreter first}"
mkdir -p work
export GPU_RUN_ROOT
GPU_RUN_ROOT=$(mktemp -d "$PWD/work/gpu-handoff-XXXXXX")
finish() {
  run_code=$?
  trap - EXIT
  printf '%s\n' "$run_code" > "$GPU_RUN_ROOT/exit-code.txt"
  git status --porcelain > "$GPU_RUN_ROOT/git-status-after.txt"
  tar -czf "$GPU_RUN_ROOT.tar.gz" -C "$(dirname "$GPU_RUN_ROOT")" "$(basename "$GPU_RUN_ROOT")"
  printf 'Results: %s\nAgent bundle: %s.tar.gz\nExit code: %s\n' "$GPU_RUN_ROOT" "$GPU_RUN_ROOT" "$run_code"
  exit "$run_code"
}
trap finish EXIT
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_COMPILATION_CACHE=false
GPU_CPU_LIST=$(python3 -c 'import os; print(",".join(map(str,sorted(os.sched_getaffinity(0))[:8])))')
GPU_CPU_COUNT=$(python3 -c 'import os; print(min(8,len(os.sched_getaffinity(0))))')
export OMP_NUM_THREADS="$GPU_CPU_COUNT" OPENBLAS_NUM_THREADS="$GPU_CPU_COUNT"
"$JAX_PYTHON" scripts/g4_gpu_handoff.py --output "$GPU_RUN_ROOT/checkpoint" \
  > "$GPU_RUN_ROOT/checkpoint-driver.log" 2>&1

# Fresh bounded config; disable the profiler point that also triggers ablations.
python3 - <<'PY'
import json, os
from pathlib import Path
root=Path(os.environ['GPU_RUN_ROOT'])
config=json.loads(Path('benchmarks/g3-profile.json').read_text())
config.update(primary=['ordinary','mixed'], diagnostic=[], batches=[32], chunks=[8],
              repetitions=10, targetSeconds=0.2,
              profilerPoint={'cohort':'disabled','batch':0,'chunk':0})
(root/'g3-v6-config.json').write_text(json.dumps(config,indent=2)+'\n')
PY
printf '%s\n' g3-v6-measurements > "$GPU_RUN_ROOT/phase.txt"
timeout --signal=TERM --kill-after=5s 300s \
  taskset -c "$GPU_CPU_LIST" env JAX_PLATFORMS=cuda \
  "$JAX_PYTHON" -m benchmarks.shared_engine \
  --config "$GPU_RUN_ROOT/g3-v6-config.json" \
  --native "$GPU_RUN_ROOT/checkpoint/native-legacy.json" \
  --platform gpu --cohorts ordinary mixed --batches 32 --chunks 8 \
  --output "$GPU_RUN_ROOT/g3-v6" > "$GPU_RUN_ROOT/g3-v6.log" 2>&1

python3 - <<'PY'
import json, os
from pathlib import Path
from scripts.g4_gpu_handoff import source_identity
root=Path(os.environ['GPU_RUN_ROOT'])
checkpoint=json.loads((root/'checkpoint/summary.json').read_text())
assert checkpoint['status']=='passed_checkpoint_checks_not_full_G4', checkpoint
assert checkpoint['source_unchanged']
assert source_identity()==json.loads((root/'checkpoint/source.json').read_text())['sha256'], 'Source changed after correctness'
result=json.loads((root/'g3-v6/results.json').read_text())
assert result['platform']=='gpu' and not result['skipped'], result.get('skipped')
assert {(r['cohort'],r['batch'],r['chunk']) for r in result['rows']}=={('ordinary',32,8),('mixed',32,8)}
assert len(result['rows'])==2 and result['checkpoint']['phase']=='complete requested selection'
assert not any('experimental_diagnostics' in r or 'profiler_trace' in r for r in result['rows'])
assert all(r['cumulative_counts']['absorbing_or_rejected']==0 and r['useful_boundaries_per_call']>0 for r in result['rows'])
fixtures=json.loads((root/'checkpoint/fixtures.json').read_text())
assert result['native_fixture_sha256']==fixtures['legacy']['sha256']
assert result['native_pin']==json.loads(Path('reference-lock.json').read_text())['commit']
(root/'phase.txt').write_text('commands_passed_agent_acceptance_review_pending\n')
print('Commands passed. Agent must audit G3 closure; full G4 remains open.')
PY
BASH
```

The `benchmarks.shared_engine` rows measure the declared **full Result/trace**
contract. They are not timings of the optional lean wrapper, and are not a
matched-output speed comparison with Showdown. Lean GPU correctness is covered
by the checkpoint tests. A lean-performance experiment is optional follow-up,
not a new G3 completion threshold. Source fingerprints include the tables,
rules, reference, tests and benchmark code in the enclosing checkpoint bundle;
profile JSON also records its own config and fixture digest. Keep the complete
bundle, including failed/partial JSON, logs and exit codes.

### Optional: one current-source matched diagnostic and hardware capture

**Not part of the required block above.** Use only if an agent identifies a
concrete remaining device issue and the user authorizes that follow-up after
travel. Never use the old default baseline, which fails current source guards.
This sequence first creates a fresh current-source native/CPU/GPU-validated
96-battle lean baseline, then captures exactly that loop once, with one warmup.
Each device command has a 300-second cap. No full-versus-lean timing ablation,
extra batch sizes or automatic repeat is included.

From a fresh shell with `JAX_PYTHON` exported, run from the repository root:

```bash
bash <<'BASH'
set -euo pipefail
: "${JAX_PYTHON:?Set JAX_PYTHON first}"
mkdir -p work
export GPU_CAPTURE_ROOT
GPU_CAPTURE_ROOT=$(mktemp -d "$PWD/work/gpu-capture-XXXXXX")
finish() {
  capture_code=$?
  trap - EXIT
  printf '%s\n' "$capture_code" > "$GPU_CAPTURE_ROOT/exit-code.txt"
  tar -czf "$GPU_CAPTURE_ROOT.tar.gz" -C "$(dirname "$GPU_CAPTURE_ROOT")" "$(basename "$GPU_CAPTURE_ROOT")"
  printf 'Capture bundle: %s.tar.gz\nExit code: %s\n' "$GPU_CAPTURE_ROOT" "$capture_code"
  exit "$capture_code"
}
trap finish EXIT
python3 - <<'PY'
import json, os
from pathlib import Path
from scripts.g4_gpu_handoff import git, source_identity
root=Path(os.environ['GPU_CAPTURE_ROOT'])
(root/'source.json').write_text(json.dumps({'commit':git('rev-parse','HEAD'), 'sha256':source_identity()},indent=2)+'\n')
PY
GPU_CPU_LIST=$(python3 -c 'import os; print(",".join(map(str,sorted(os.sched_getaffinity(0))[:8])))')
GPU_CPU_COUNT=$(python3 -c 'import os; print(min(8,len(os.sched_getaffinity(0))))')
export OMP_NUM_THREADS="$GPU_CPU_COUNT" OPENBLAS_NUM_THREADS="$GPU_CPU_COUNT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_COMPILATION_CACHE=false
timeout --signal=TERM --kill-after=5s 300s taskset -c "$GPU_CPU_LIST" \
  env JAX_PLATFORMS=cpu,cuda "$JAX_PYTHON" -m benchmarks.g3_diagnostic \
  --copies 32 --retention lean --output "$GPU_CAPTURE_ROOT/diagnostic" \
  > "$GPU_CAPTURE_ROOT/diagnostic.log" 2>&1
timeout --signal=TERM --kill-after=5s 300s taskset -c "$GPU_CPU_LIST" \
  env JAX_PLATFORMS=cuda "$JAX_PYTHON" -m benchmarks.g3_profile_96 \
  --baseline "$GPU_CAPTURE_ROOT/diagnostic/results.json" \
  --output "$GPU_CAPTURE_ROOT/profile" > "$GPU_CAPTURE_ROOT/profile.log" 2>&1
"$JAX_PYTHON" -m benchmarks.g3_profile_96_analysis \
  --capture "$GPU_CAPTURE_ROOT/profile" > "$GPU_CAPTURE_ROOT/analysis.log" 2>&1
python3 - <<'PY'
import json, os
from pathlib import Path
from scripts.g4_gpu_handoff import source_identity
root=Path(os.environ['GPU_CAPTURE_ROOT'])
assert source_identity()==json.loads((root/'source.json').read_text())['sha256'], 'Source changed during diagnostic/capture'
PY
BASH
```

The diagnostic's `cpu,cuda` selector is intentional: that command initializes
both backends and explicitly demands GPU placement/digest equality. The capture
itself is strict CUDA. If the fixture/source/output-digest guard rejects any
step, retain the bundle and investigate; never bypass it. The offline analyzer
needs no GPU but requires the completed hardware stream export. Profiler overhead
changes timing; use the unprofiled diagnostic for backend comparisons. A trace
is neither conformance nor a causal claim that removing one output fixes speed.

## Give the results to an agent

Attach the printed `.tar.gz` (or make its directory available), then use:

> Review this GPU handoff bundle against `docs/GPU_HANDOFF.md` and
> `docs/ORCHESTRATION_STATE.json`. Check source/fixture/config identities,
> runtime/device placement, every exit code and test log, and any timeout/OOM or
> first divergence. Audit the G3 V6 checklist separately from full G4 release gates.
> Preserve failed evidence and minimize any mismatch. Record verified results
> against unchanged source, reconcile G3 closure only if all its CPU/GPU gates
> pass, update only the GPU obligations actually satisfied,
> and leave CPU/GPU/final-format gaps explicit. Do not infer completion from
> `summary.json` alone or launch profiling/training/new campaigns automatically.

The travel deferral does not block the active CPU goal or waive G4 GPU release
evidence. Update this same document and the compact deferred-GPU record as new
G3/G4 work adds obligations; do not create a handoff per mechanic.

## Architecture pass: future device obligations

The [2026-11-12 decisions](shared-engine.md#g4-architecture-decisions--2026-11-12)
add no implemented GPU tests yet. The runner now includes `test.test_core_journal` and prepares `TURN_LIMIT_NATIVE_CASE` automatically.
Once implemented, extend that same runner with core/journal full-versus-lean
parity across 16-boundary chunk flushes, save/restore including private disclosure
and journal pages, and mixed-lane called/reflected/transformed states. Verify
recipient histories and per-lane isolation as well as mechanics; never accept
truncated journals or a missing-page resume. Keep all JIT kernels device-resident
inside a chunk. These commands remain pending until those tests are implemented
and included in the runner; do not infer device correctness from native probes.

The design evidence itself can be checked now on CPU without JAX or profiling:

```sh
node --test test/g4-design-native.test.cjs test/gen5-reachability.test.cjs test/shared-engine.test.cjs
```

This command passed five native/source-contract checks. Its five constructed
native cases are design witnesses, not additions to the 526 implemented JAX cases.

The new module checks scalar core/full parity, mixed core batches1/8/32, lean/full
core rollout agreement, core reset, journal persistence/corruption and native
turn-limit warnings/tie across1000 boundaries. The lossless host journal is a CPU
adapter over device transitions. Device-side chunk collection and host journal
reset still require implementation; kernel reset alone does not validate them.
The earlier reference-generated6541-row witness still needs replay once all its
mechanics are supported. The neutral constructed witness does not prove generator
membership or final bounds. No profiling is part of this checkpoint.

The runner now includes `test.test_trapping_engine`, prepares `TRAP_NATIVE_CASES`
automatically, and preserves ordered retry fixtures. Its new checks cover private
feedback/request disclosure, opposing pending commitments, cannot-undo behavior,
Gen5 Ghost/grounded/Steel cases, Frisk RNG, eager/JIT/mixed1/8/32, retry-aware
core full/lean rollouts, saved private state and replay-required migration. Run
the same existing runner commands on matching committed source after travel.
No GPU or profiling command was executed for this candidate.
