#!/usr/bin/env python3
"""Read bounded live context or explicitly selected documentation/history. No JAX."""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT/'docs/ORCHESTRATION_STATE.json'


def select(value, pointer):
    if not pointer.startswith('/'):
        raise ValueError('Use a JSON pointer, e.g. /active_task/next_work_package')
    for key in pointer[1:].split('/'):
        key = key.replace('~1', '/').replace('~0', '~')
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--field', help='JSON pointer into live state')
    modes.add_argument('--history-field', help='Explicit JSON pointer into pre-cleanup history')
    modes.add_argument('--headings', type=Path, help='List Markdown headings without reading body into context')
    modes.add_argument('--section', nargs=2, metavar=('PATH', 'HEADING'), help='Exact heading text; includes its subsections')
    parser.add_argument('--max-chars', type=int, default=12000, help='Output budget; fail rather than silently truncate')
    args = parser.parse_args()
    state = json.loads(STATE.read_text())
    if args.headings or args.section:
        path = args.headings or Path(args.section[0])
        lines = (ROOT/path).read_text().splitlines()
        headings = [(i, re.match(r'^(#{1,6}) (.+)$', line)) for i, line in enumerate(lines)]
        headings = [(i, m) for i, m in headings if m]
        if args.headings:
            output = '\n'.join(f'{i+1}: {m[0]}' for i, m in headings)
        else:
            found = [(i, m) for i, m in headings if m[2] == args.section[1]]
            if len(found) != 1:
                raise ValueError('Heading must match exactly once; use --headings first')
            start, match = found[0]
            end = next((i for i, m in headings if i > start and len(m[1]) <= len(match[1])), len(lines))
            output = '\n'.join(lines[start:end])
    else:
        if args.history_field:
            history = ROOT/state['historical_records']['archive']
            value = select(json.loads(history.read_text()), args.history_field)
        elif args.field:
            value = select(state, args.field)
        else:
            active = state['active_task']
            keys = ['id', 'status', 'approved_at', 'brief', 'scope', 'checkpoint', 'next',
                    'blockers', 'first_divergence', 'implementation_wip', 'cpu_block',
                    'acceptance', 'source_identity', 'next_work_package', 'stopping_reason',
                    'pending_acceptance']
            value = {'active_task': {key: active[key] for key in keys if key in active},
                     'gpu_block': state.get('gpu_block'),
                     'details': 'Use --field for live details; --history-field only for a relevant prior record. See docs/CURRENT.md.'}
        output = json.dumps(value, indent=2, ensure_ascii=False)
    if len(output) > args.max_chars:
        raise ValueError(f'Selection is {len(output)} characters, above budget {args.max_chars}; select a narrower field/subsection or explicitly raise --max-chars')
    print(output)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, IndexError, OSError) as error:
        print(f'Context selection failed: {error}', file=sys.stderr)
        sys.exit(1)
