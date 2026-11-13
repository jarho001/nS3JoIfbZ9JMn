#!/usr/bin/env python3
"""Freeze inputs, run matching CPU/CUDA checks, and package results for review.

Preparation uses only the standard library and the pinned Node oracle. No JAX
import or GPU query occurs with --prepare-only. This is a correctness handoff,
not a benchmark or full-format acceptance harness.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    'test.test_gen5_tables', 'test.test_generic_engine', 'test.test_roster_hazards',
    'test.test_shared_engine_cases', 'test.test_shared_engine_battles',
    'test.test_shared_engine_lifecycle', 'test.test_rollout', 'test.test_g3_fixtures',
    'test.test_status_engine', 'test.test_field_engine',
    'test.test_hook_engine', 'test.test_hit_engine', 'test.test_recharge_engine', 'test.test_lock_phase_engine', 'test.test_core_journal', 'test.test_trapping_engine', 'test.test_substitute_engine', 'test.test_partial_trap_engine', 'test.test_phazing_engine', 'test.test_pursuit_engine', 'test.test_item_lifecycle_engine', 'test.test_identity_engine', 'test.test_illusion_engine', 'test.test_form_engine', 'test.test_trigger_form_engine', 'test.test_ability_change_engine', 'test.test_status_immunity_engine', 'test.test_ability_create_engine', 'test.test_suppression_engine', 'test.test_sleep_control_engine', 'test.test_called_move_engine', 'test.test_nature_power_engine', 'test.test_copy_move_engine', 'test.test_reflection_engine',
)
NATIVE_TESTS = (
    'test/reference.test.cjs', 'test/gen5-reachability.test.cjs',
    'test/shared-engine.test.cjs', 'test/status-duration-native.test.cjs',
    'test/field-boost-native.test.cjs',
    'test/ability-item-native.test.cjs', 'test/hit-effect-native.test.cjs', 'test/recharge-native.test.cjs', 'test/lock-phase-native.test.cjs', 'test/g4-design-native.test.cjs', 'test/trapping-native.test.cjs', 'test/substitute-native.test.cjs', 'test/partial-trap-native.test.cjs', 'test/phazing-native.test.cjs', 'test/pursuit-native.test.cjs', 'test/item-lifecycle-native.test.cjs', 'test/identity-native.test.cjs', 'test/illusion-native.test.cjs', 'test/form-native.test.cjs', 'test/trigger-form-native.test.cjs', 'test/ability-change-native.test.cjs', 'test/status-immunity-native.test.cjs', 'test/ability-create-native.test.cjs', 'test/suppression-native.test.cjs', 'test/sleep-control-native.test.cjs', 'test/called-move-native.test.cjs', 'test/nature-power-native.test.cjs', 'test/copy-move-native.test.cjs', 'test/reflection-native.test.cjs',
)
BATCHES = {'legacy': 'all', 'hazards': 'g4', 'generic': 'generic', 'status': 'status', 'field': 'field', 'hooks': 'hooks', 'hits': 'hits', 'recharge': 'recharge', 'locks': 'locks', 'turnlimit': 'turnlimit', 'trapping': 'trapping', 'substitute': 'substitute', 'partialtrap': 'partialtrap', 'phazing': 'phazing', 'pursuit': 'pursuit', 'items': 'items', 'identity': 'identity', 'illusion': 'illusion', 'forms': 'forms', 'triggerforms': 'triggerforms', 'abilitychange': 'abilitychange', 'statusimmunity': 'statusimmunity', 'abilitycreate': 'abilitycreate', 'suppression': 'suppression', 'sleepcontrol': 'sleepcontrol', 'calledmoves': 'calledmoves', 'naturepower': 'naturepower', 'copymoves': 'copymoves', 'reflection': 'reflection'}
SOURCE_PATHS = (
    'simulator', 'mechanics', 'rules', 'reference', 'test', 'environment',
    'evaluation', 'scripts', 'requirements-jax-cpu.txt', 'reference-lock.json',
    'benchmarks',
    'test-logs/g3-testing-profiling/diagnostic/native-pack.json',
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode().strip()


def source_identity():
    names = subprocess.check_output(['git', 'ls-files', '-z', '--', *SOURCE_PATHS], cwd=ROOT)
    paths = [p for p in names.decode().split('\0') if p]
    # Include this runner during a local pre-commit preparation check too.
    paths = sorted(set(paths) | {Path(__file__).resolve().relative_to(ROOT).as_posix()})
    return {name: sha256(ROOT / name) for name in paths}


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def run(command, log, env, timeout, records):
    """One attempt per phase; kill its process group on timeout/interruption."""
    keys = ('JAX_PLATFORMS', 'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'XLA_PYTHON_CLIENT_PREALLOCATE', 'JAX_ENABLE_COMPILATION_CACHE',
            'G3_NATIVE_CASES', 'G4_NATIVE_CASES', 'GEN5_NATIVE_CASES', 'STATUS_NATIVE_CASES',
            'FIELD_NATIVE_CASES', 'HOOK_NATIVE_CASES', 'HIT_NATIVE_CASES', 'RECHARGE_NATIVE_CASES', 'LOCK_NATIVE_CASES', 'TURN_LIMIT_NATIVE_CASE', 'TRAP_NATIVE_CASES', 'SUB_NATIVE_CASES', 'PARTIAL_NATIVE_CASES', 'PHAZE_NATIVE_CASES', 'PURSUIT_NATIVE_CASES', 'ITEM_NATIVE_CASES', 'IDENTITY_NATIVE_CASES', 'ILLUSION_NATIVE_CASES', 'FORM_NATIVE_CASES', 'TRIGGER_FORM_NATIVE_CASES', 'ABILITY_CHANGE_NATIVE_CASES', 'STATUS_IMMUNITY_NATIVE_CASES', 'ABILITY_CREATE_NATIVE_CASES', 'SUPPRESSION_NATIVE_CASES', 'SLEEP_CONTROL_NATIVE_CASES', 'CALLED_NATIVE_CASES', 'NATURE_POWER_NATIVE_CASES', 'G4_PROBE_OUTPUT')
    record = {'argv': list(map(str, command)), 'log': log.name, 'timeout_seconds': timeout,
              'environment':{key:env[key] for key in keys if key in env}}
    records.append(record)
    print(f'Running {log.name}', flush=True)
    with log.open('w') as output:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=output,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            record['exit_code'] = child.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            record['exit_code'] = child.returncode
            raise
    if record['exit_code']:
        raise RuntimeError(f'{log.name} exited {record["exit_code"]}; retain the bundle and stop')


PROBE = r'''
import importlib.metadata as md
import json, os, platform
from pathlib import Path
import jax, jaxlib, jax.numpy as jnp
expected = {'cpu':'cpu', 'cuda':'gpu'}[os.environ['JAX_PLATFORMS']]
assert jax.default_backend() == expected, (expected, jax.default_backend())
devices = jax.devices()
assert all(d.platform == expected for d in devices), devices
if expected == 'gpu':
    assert len(devices) == 1, 'Select one GPU with CUDA_VISIBLE_DEVICES before running this handoff'
x = jnp.arange(8, dtype=jnp.int32)
y = jax.jit(lambda z: z*z+1)(x)
jax.block_until_ready((x,y))
assert y.tolist() == [1,2,5,10,17,26,37,50]
assert all(d.platform == expected for v in (x,y) for d in v.devices())
result = {'python':platform.python_version(), 'jax':jax.__version__, 'jaxlib':jaxlib.__version__,
          'backend':jax.default_backend(), 'devices':list(map(str,devices)),
          'input_devices':list(map(str,x.devices())), 'jit_output_devices':list(map(str,y.devices())),
          'packages':{d.metadata['Name']:d.version for d in md.distributions()
                      if d.metadata['Name'].lower().startswith(('jax','nvidia','numpy'))}}
Path(os.environ['G4_PROBE_OUTPUT']).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Fresh directory; defaults to work/g4-gpu-TIMESTAMP')
    parser.add_argument('--prepare-only', action='store_true', help='Freeze native inputs and commands; no JAX/GPU execution')
    parser.add_argument('--suite-timeout', type=int, default=3600, help='Seconds per isolated CPU/CUDA test module; default 3600; no automatic retry')
    args = parser.parse_args(argv)
    if args.suite_timeout <= 0:
        parser.error('--suite-timeout must be positive')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = (args.output or ROOT / 'work' / f'g4-gpu-{stamp}').resolve()
    # A pre-existing output must never be overwritten or mistaken for this run.
    out.mkdir(parents=True, exist_ok=False)
    summary = {'schemaVersion':1, 'status':'running', 'kind':'G4-checkpoint-device-correctness',
               'started_utc':stamp, 'prepare_only':args.prepare_only, 'commands':[],
               'full_G4_accepted':False, 'profiling_run':False, 'training_run':False}
    failure = None
    identity = None
    original_affinity = None
    previous_term = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise InterruptedError(f'received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        available = sorted(os.sched_getaffinity(0))
        original_affinity = available
        selected = available[:min(8, len(available))]
        os.sched_setaffinity(0, selected)
        summary['host_cpu_affinity'] = selected
        env = dict(os.environ, OMP_NUM_THREADS=str(len(selected)), OPENBLAS_NUM_THREADS=str(len(selected)),
                   XLA_PYTHON_CLIENT_PREALLOCATE='false', JAX_ENABLE_COMPILATION_CACHE='false')
        # Ignore inherited test-fixture overrides; every test uses this bundle.
        for key in ('G3_NATIVE_CASES', 'G4_NATIVE_CASES', 'GEN5_NATIVE_CASES', 'STATUS_NATIVE_CASES', 'FIELD_NATIVE_CASES', 'HOOK_NATIVE_CASES', 'HIT_NATIVE_CASES', 'RECHARGE_NATIVE_CASES', 'LOCK_NATIVE_CASES', 'TURN_LIMIT_NATIVE_CASE', 'TRAP_NATIVE_CASES', 'SUB_NATIVE_CASES', 'PARTIAL_NATIVE_CASES', 'PHAZE_NATIVE_CASES', 'PURSUIT_NATIVE_CASES', 'ITEM_NATIVE_CASES', 'IDENTITY_NATIVE_CASES', 'ILLUSION_NATIVE_CASES', 'FORM_NATIVE_CASES', 'TRIGGER_FORM_NATIVE_CASES', 'ABILITY_CHANGE_NATIVE_CASES', 'STATUS_IMMUNITY_NATIVE_CASES', 'ABILITY_CREATE_NATIVE_CASES', 'SUPPRESSION_NATIVE_CASES', 'SLEEP_CONTROL_NATIVE_CASES', 'CALLED_NATIVE_CASES', 'NATURE_POWER_NATIVE_CASES'):
            env.pop(key, None)
        identity = source_identity()
        dirty = git('status','--porcelain','--untracked-files=all','--',*SOURCE_PATHS)
        save(out / 'source.json', {'commit':git('rev-parse','HEAD'), 'tree':git('rev-parse','HEAD^{tree}'),
                                  'working_tree_status':git('status','--porcelain'), 'sha256':identity})
        if dirty and not args.prepare_only:
            raise RuntimeError('Relevant code/fixtures are uncommitted; preserve and commit them before device validation')
        run(['node','reference/oracle.cjs','provenance'], out/'reference-provenance.log', env, 120, summary['commands'])
        run(['node','reference/oracle.cjs','format'], out/'reference-format.log', env, 120, summary['commands'])
        fixtures = {}
        for name,batch in BATCHES.items():
            path = out / f'native-{name}.json'
            command=(['node','reference/turn-limit-cases.cjs',str(path)] if batch=='turnlimit' else
                     ['node','reference/shared-engine-cases.cjs','--batch',batch,'--output',str(path)])
            run(command,
                out/f'prepare-{name}.log', env, 180, summary['commands'])
            data = json.loads(path.read_text())
            assert data['pin'] == json.loads((ROOT/'reference-lock.json').read_text())['commit']
            cases=[data] if batch=='turnlimit' else data['cases']
            fixtures[name] = {'file':path.name, 'sha256':sha256(path), 'cases':len(cases),
                              'boundaries':sum(len(c['steps']) for c in cases),
                              'jax_implemented':True}
        save(out/'fixtures.json', fixtures)
        env.update(G3_NATIVE_CASES=str(out/'native-legacy.json'),
                   G4_NATIVE_CASES=str(out/'native-hazards.json'),
                   GEN5_NATIVE_CASES=str(out/'native-generic.json'),
                   STATUS_NATIVE_CASES=str(out/'native-status.json'),
                   FIELD_NATIVE_CASES=str(out/'native-field.json'),
                   HOOK_NATIVE_CASES=str(out/'native-hooks.json'),
                   HIT_NATIVE_CASES=str(out/'native-hits.json'),
                   RECHARGE_NATIVE_CASES=str(out/'native-recharge.json'),
                   LOCK_NATIVE_CASES=str(out/'native-locks.json'),
                   TURN_LIMIT_NATIVE_CASE=str(out/'native-turnlimit.json'),
                   TRAP_NATIVE_CASES=str(out/'native-trapping.json'),
                   SUB_NATIVE_CASES=str(out/'native-substitute.json'),
                   PARTIAL_NATIVE_CASES=str(out/'native-partialtrap.json'),
                   PHAZE_NATIVE_CASES=str(out/'native-phazing.json'),
                   PURSUIT_NATIVE_CASES=str(out/'native-pursuit.json'),
                   ITEM_NATIVE_CASES=str(out/'native-items.json'),
                   IDENTITY_NATIVE_CASES=str(out/'native-identity.json'),
                   ILLUSION_NATIVE_CASES=str(out/'native-illusion.json'),
                   FORM_NATIVE_CASES=str(out/'native-forms.json'),
                   TRIGGER_FORM_NATIVE_CASES=str(out/'native-triggerforms.json'),
                   ABILITY_CHANGE_NATIVE_CASES=str(out/'native-abilitychange.json'),
                   STATUS_IMMUNITY_NATIVE_CASES=str(out/'native-statusimmunity.json'),
                   ABILITY_CREATE_NATIVE_CASES=str(out/'native-abilitycreate.json'),
                   SUPPRESSION_NATIVE_CASES=str(out/'native-suppression.json'),
                   SLEEP_CONTROL_NATIVE_CASES=str(out/'native-sleepcontrol.json'),
                   CALLED_NATIVE_CASES=str(out/'native-calledmoves.json'),
                   NATURE_POWER_NATIVE_CASES=str(out/'native-naturepower.json'),
                   COPY_MOVE_NATIVE_CASES=str(out/'native-copymoves.json'),
                   REFLECTION_NATIVE_CASES=str(out/'native-reflection.json'))
        summary['modules'] = list(MODULES)
        summary['fixtures'] = fixtures
        summary['future_full_format_gate'] = 'Pending implementation of V7 and remaining mechanics; see docs/GPU_HANDOFF.md'
        if args.prepare_only:
            summary['status'] = 'prepared_not_executed'
        else:
            run(['node','--test',*NATIVE_TESTS], out/'native-tests.log', env, 300, summary['commands'])
            for platform in ('cpu','cuda'):
                summary['phase'] = platform
                save(out/'summary.json', summary)
                phase_env = dict(env, JAX_PLATFORMS=platform, G4_PROBE_OUTPUT=str(out/f'{platform}-runtime.json'))
                if platform == 'cuda':
                    run(['nvidia-smi'], out/'nvidia-smi.log', env, 30, summary['commands'])
                run([sys.executable,'-c',PROBE], out/f'{platform}-probe.log', phase_env, 120, summary['commands'])
                # Release compiled executables between modules. A combined CPU
                # run exceeded the cloud's 8 GiB limit after 39 checks; process
                # isolation preserves every check without accumulating caches.
                counts = {}
                for module in MODULES:
                    log = out/f'{platform}-{module.rsplit(".",1)[-1]}.log'
                    run([sys.executable,'-m','unittest',module,'-v','-f'], log,
                        phase_env, args.suite_timeout, summary['commands'])
                    match = re.search(r'Ran (\d+) tests? in ([0-9.]+)s', log.read_text())
                    if match is None or int(match[1]) <= 0:
                        raise RuntimeError(f'{platform} {module} did not report a nonempty test run')
                    counts[module] = int(match[1])
                    summary[f'{platform}_tests'] = {'passed':False, 'modules':counts, 'count':sum(counts.values())}
                    save(out/'summary.json',summary)
                summary[f'{platform}_tests']['passed'] = True
            summary['status'] = 'passed_checkpoint_checks_not_full_G4'
    except BaseException as error:
        failure = error
        summary['status'] = 'failed'
        summary['error'] = repr(error)
    finally:
        try:
            summary['source_unchanged'] = identity is not None and source_identity() == identity
            if identity is not None and not summary['source_unchanged']:
                failure = failure or RuntimeError('Relevant source changed during the run; results are not acceptance evidence')
                summary['status'] = 'failed_source_changed'
            save(out/'summary.json', summary)
            archive = out.with_name(out.name + '.tar.gz')
            with archive.open('xb') as raw:
                with tarfile.open(fileobj=raw, mode='w:gz') as bundle:
                    bundle.add(out, arcname=out.name)
            print(f'Results: {out}\nAgent handoff bundle: {archive}', flush=True)
        except BaseException as error:
            failure = failure or error
            summary['status'] = 'failed_packaging_or_finalization'
            summary['finalization_error'] = repr(error)
            save(out/'summary.json', summary)
        signal.signal(signal.SIGTERM, previous_term)
        if original_affinity is not None:
            os.sched_setaffinity(0, original_affinity)
    if failure:
        print(f'NOT PASSED: {failure}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
