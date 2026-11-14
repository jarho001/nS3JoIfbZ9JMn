#!/usr/bin/env python3
"""Bounded caller-side diagnostic: trace/lower advance, without backend compilation.

Use timeout externally. Large StableHLO text stays under work/, never in git.
"""
import argparse
import collections
import gzip
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'work/g4-compilation')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:3])
    import jax
    from simulator import shared_engine as e
    from test.shared_engine_support import initialize_case, choices_array
    start = time.monotonic()
    case = json.loads(gzip.decompress(Path('test-logs/g4-cpu-conformance/switch-recovery.json.gz').read_bytes()))['cases'][0]
    state, _ = initialize_case(case)
    actions = choices_array(state, case['steps'][0]['choices'])
    initialization = time.monotonic()-start
    print(f'initialization: {initialization:.3f}s', flush=True)
    start = time.monotonic()
    closed = jax.make_jaxpr(e.advance)(state, actions)
    tracing = time.monotonic()-start
    print(f'tracing: {tracing:.3f}s', flush=True)
    counts = collections.Counter()
    sizes, seen = [], set()

    def walk(value):
        if hasattr(value, 'jaxpr') and not hasattr(value, 'eqns'):
            return walk(value.jaxpr)
        if hasattr(value, 'eqns'):
            if id(value) in seen:
                return
            seen.add(id(value))
            sizes.append(len(value.eqns))
            for equation in value.eqns:
                counts[equation.primitive.name] += 1
                walk(equation.params)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                walk(item)

    walk(closed)
    start = time.monotonic()
    lowered = jax.jit(e.advance).lower(state, actions)
    lowering = time.monotonic()-start
    start = time.monotonic()
    ir = lowered.as_text()
    (args.output/'advance.stablehlo.mlir').write_text(ir)
    result = dict(jax=jax.__version__, initialization_seconds=initialization,
                  tracing_seconds=tracing, lowering_seconds=lowering,
                  ir_export_seconds=time.monotonic()-start,
                  jaxpr_equations=sum(counts.values()), jaxpr_graphs=len(sizes),
                  largest_graph=max(sizes), primitives=counts.most_common(20),
                  stablehlo_bytes=len(ir.encode()), stablehlo_lines=len(ir.splitlines()),
                  compile_executed=False)
    (args.output/'inspection.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
