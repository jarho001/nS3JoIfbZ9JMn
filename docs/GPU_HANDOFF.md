# GPU handoff — G3 closure and G4 device conformance

**Deferred during travel, 2026-11-08.** The user will run GPU work after returning
home and can give the result bundle to an agent. This document is the living
record for all GPU work arising from G3 and G4. CPU implementation continues without a
GPU. Preparing this handoff does not run device tests, profiling or benchmarks.

## Current source and evidence

The active engine is schema 6. The new ability/item package passes four CPU
integration checks: 70 independent native scenarios /324 boundaries, including
four complete six-member games; eager initialization and transition/JIT,
heterogeneous 1/8/32 batches, saved continuation/reset, full/lean equivalence,
Choice-lock metadata rejection and explicit legacy-state migration checks.
The matching-schema 45-check regression is still pending. Historical schema 5
[source/results](../test-logs/g4-cpu-conformance/schema5-source.json) and earlier
37-check evidence remain identified separately in [active progress](ORCHESTRATION_STATE.json).
The combined schema 4 run completed 39 checks before the 8 GiB limit killed its
status batch check; [that run is incomplete](../test-logs/g4-cpu-conformance/schema4-source.json).
The runner isolates test modules; the hook suite also clears executable caches
between groups to fit this host without dropping cases or batch sizes.

The current correctness scope is 49 checks and 322 independent native scenarios /
2102 boundaries, including 20 complete six-member games. Status 55, field 40 and
hook 70 cases are implemented JAX checks. This remains partial G4: remaining
ability/item hooks, complex phases, indirect effects, capacity proof and the
fixed full-format campaigns are pending. Passing this checkpoint cannot close G4.

Use the **latest committed checkout** and retain its recorded identities. The
runner fingerprints engine, tables, IO, tests, oracle, configuration and benchmark
code; it separately hashes each freshly captured native fixture. Device results
cannot transfer across changed relevant source or fixtures.

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
2. Freezes legacy, hazard, generic, status, field/boost and ability/item corpora. Runs the
   reference/native contracts. Unknown/unsupported families stay explicit.
3. Runs the same current 49-check scope on CPU, then under `JAX_PLATFORMS=cuda`,
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

## GPU obligations and readiness

| Obligation | Commands / current readiness | Evidence required |
|---|---|---|
| Current G4 data/hazard/generic/status/field/hook device correctness | **Runnable checkpoint:** the runner above; check the CPU validation record first, then use the same CPU/CUDA modules and frozen fixtures | Both suites pass; real GPU leaf placement, exact native state/draw/request/view/outcome equality, batches 1/8/32, saved continuation/reset, full/lean equality, invalid/stale/unsupported/overflow isolation. Covers the current checkpoint only. |
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
