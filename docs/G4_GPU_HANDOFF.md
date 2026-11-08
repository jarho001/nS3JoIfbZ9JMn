# G4 GPU handoff

**Deferred during travel, 2026-11-08.** The user will run GPU work after returning
home and can give the result bundle to an agent. This document is the living
record for all GPU work arising from G4. CPU implementation continues without a
GPU. Preparing this handoff does not run device tests, profiling or benchmarks.

## Current source and evidence

The starting checkpoint is `00ec1e7`. The current engine/table/IO/test bytes match
CPU-validated engine commit `4eab254`; native capture additions are at `578b352`.
See [source identities and results](../test-logs/g4-cpu-conformance/source.json)
and [active progress](ORCHESTRATION_STATE.json). If code changes during travel,
run the **latest committed checkout** and retain its own identities; do not reuse
GPU results for changed mechanics, tables, schemas, fixtures or test adapters.

Current CPU evidence: 37 checks, 157 native scenarios / 1059 boundaries, including
eight six-member games, eager/JIT/mixed batches 1/8/32, continuation/reset,
schema migration and full/lean rollout equivalence. This is partial G4 support.
The 48 status/duration cases are native expectations; their JAX implementation
is still pending. They must not be reported as GPU-validated mechanics merely
because their native tests pass.

## Return-home commands: current correctness checkpoint

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

"$JAX_PYTHON" scripts/g4_gpu_handoff.py
```

CUDA 13 needs a compatible NVIDIA driver/device. `nvidia-smi` plus the strict
CUDA probe in the runner record what actually works. If this wheel is unavailable
or incompatible, retain the setup error for the reviewing agent; select a
compatible documented stack rather than silently changing versions or using CPU
fallback. The historical 8 GiB GPU is context, not a promise that future G4
workloads fit. No paid hardware, driver installation or infrastructure changes
are part of these commands. The runner targets one selected GPU; on a multi-GPU
machine set `CUDA_VISIBLE_DEVICES` to the intended device before invoking it.

The single runner command:

1. Creates a fresh `work/g4-gpu-TIMESTAMP/` and records commit/tree, relevant file
   hashes, selected host affinity and independent native fixture hashes.
2. Freezes legacy, hazard, generic and native-only status corpora. Runs the
   reference/native contracts. Unknown/unsupported families stay explicit.
3. Runs the same current 37-check suite on CPU, then under `JAX_PLATFORMS=cuda`.
   A strict probe verifies actual eager/JIT GPU placement; shared case/lifecycle
   assertions also check device leaves. `cuda` initialization failure is a failure.
4. Saves package/device/driver information, exact command arrays, full logs and
   exit codes. Stops on failure/timeout, with no automatic retry or reduced batch.
5. Writes `summary.json` and a sibling `.tar.gz`, including on ordinary failures
   or interruption. The archive is the artifact to give to an agent.

The CPU worker budget is `min(8, available affinity CPUs)`, identical for both
backends; it is not a tuning sweep. GPU allocations grow on demand; persistent
compilation caching is disabled and recorded by the command environment. Each
suite has a 3600-second cap. Prior CPU runtime was about 11 minutes; current GPU
runtime/memory remain unmeasured. A timeout/OOM is incomplete evidence, not a
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
correctness commands above as a routine repeat; they are alternatives.

## GPU obligations and readiness

| Obligation | Commands / current readiness | Evidence required |
|---|---|---|
| Current G4 data/hazard/generic device correctness | **Ready:** the runner above; same CPU/CUDA modules and frozen fixtures | Both suites pass; real GPU leaf placement, exact native state/draw/request/view/outcome equality, batches 1/8/32, saved continuation/reset, full/lean equality, invalid/stale/unsupported/overflow isolation. Covers the current checkpoint only. |
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

## Profiling notes and commands to preserve

G4 requires device correctness for release. GPU compilation/memory/warm-execution
profiling is also a separate **G3** closure obligation in the plan; this handoff
does not reactivate the retired broad G3 sweep or introduce a G4 speed threshold.
The historical [G3 report](G3_TESTING_PROFILING_RESULTS.md) retains its commands,
hardware, output contract and results. Those results describe older source.

The existing profiler's CLI is:

```bash
JAX_PLATFORMS=cuda "$JAX_PYTHON" -m benchmarks.g3_profile_96 \
  --baseline PATH_TO_MATCHING_COMPLETE_DIAGNOSTIC_RESULTS_JSON \
  --output NEW_PROFILE_DIRECTORY
```

**Do not execute it with the old default baseline on current G4 source.** It
requires a complete matching GPU/native correctness diagnostic, measured-loop
hash, source hashes, frozen `native-pack.json` and output digest. The historical
baseline is incompatible with the expanded G4 engine; its guards should reject
that combination. A matching current baseline/profile adapter and resource scope
must be prepared before replacing this placeholder with a return-home command.
Keep this obligation pending rather than bypassing the guard. The broad
`benchmarks.shared_engine` sweep also includes diagnostic ablations and is not
part of the ready correctness command.

For any new GPU profiling need, the implementing agent must append the concrete
command, matching source/baseline/config, selected workload/batch/output retention,
memory headroom, warmup/capture count and timeout here. Record synchronized input
and output readiness, compilation separately, and device stream events separately
from host/nested annotations. Keep cold correctness testing and profiling results
separate. No automatic optimization sweep, learned-strength or speedup claim.

## Give the results to an agent

Attach the printed `.tar.gz` (or make its directory available), then use:

> Review this G4 GPU handoff bundle against `docs/G4_GPU_HANDOFF.md` and
> `docs/ORCHESTRATION_STATE.json`. Check source/fixture/config identities,
> runtime/device placement, every exit code and test log, and any timeout/OOM or
> first divergence. Distinguish checkpoint checks from full G4 release gates.
> Preserve failed evidence and minimize any mismatch. Record verified results
> against unchanged source, update only the GPU obligations actually satisfied,
> and leave CPU/GPU/final-format gaps explicit. Do not infer completion from
> `summary.json` alone or launch profiling/training/new campaigns automatically.

The travel deferral does not block the active CPU goal or waive G4 GPU release
evidence. Update this same document and the compact deferred-GPU record as new
G4 work adds obligations; do not create a handoff per mechanic.
