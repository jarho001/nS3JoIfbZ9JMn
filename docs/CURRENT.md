# Current work entrypoint

Read AGENTS.md and verify Trail first. Then, from the repository root:

```sh
python scripts/read_context.py
```

This live view selects the approved goal, current checkpoint, next package,
blockers/WIP, source evidence and pending acceptance from ORCHESTRATION_STATE.json.
It has a 12,000-character output limit and fails rather than silently truncating.
There is no separately maintained status summary to go stale. Read the next
package's command status before running it; planned commands may need a module
implemented first. Preserve WIP when synchronizing with origin/main.

Load only the detail required for the action:

| Need | Selection |
|---|---|
| Live package or validation policy | `--field /active_task/next_work_package` or `--field /active_task/validation_policy` |
| Goal scope/acceptance | `--section PROJECT_PLAN.md 'Proposed G4 CPU implementation and conformance goal'`; narrow further using `--headings PROJECT_PLAN.md` if over budget |
| Implementation/checkpoint policy | `--section docs/ORCHESTRATION.md 'Compilation-aware G4 checkpoints'` |
| Reachability precheck | `--section docs/SHOWDOWN_BEHAVIOR_CASES.md 'Reachability audit and family precheck (2026-11-13)'` |
| Prior family evidence | Open the referenced checkpoint JSON; select historical state only if it answers a concrete question |
| Deferred GPU/broad gates | Search headings in GPU_HANDOFF.md, then read the relevant command section when preparing that work |

All flags above belong to `python scripts/read_context.py`. `--headings PATH`
lists headings with line numbers; `--section PATH 'Exact heading'` reads that
section including subsections. Use `rg -n` then bounded `sed -n` for source or
other documents. Increase the output budget explicitly only when needed. No
routine whole-file reads of the plan, case catalog, handoff, engine history or
state archives; authoritative scope and acceptance still apply in full.

Update live state at meaningful checkpoints. Keep current blockers, divergence,
WIP, next command and unfinished acceptance live. Put results in immutable test
evidence and link them; do not regrow a completed-family narrative in active
state. Historical records and their exact source/evidence remain available via
`--history-field /active_task/continuation/NAME` or a targeted archive read.

Pattern reference: AgenC's [bounded orientation map](https://github.com/tetsuo-ai/agenc-core/blob/97d8c1f5db9b78b73c122c7bdcff1bd1399fb756/runtime/src/context/orientation-map.ts)
and [on-request memory attachment](https://github.com/tetsuo-ai/agenc-core/blob/97d8c1f5db9b78b73c122c7bdcff1bd1399fb756/runtime/src/prompts/attachments/memory-on-request.ts).
This adapts bounded output and selective loading, with existing Trail/rg rather
than another retrieval service. It does not change host prompt injection,
model/effort, acceptance, approvals or GPU deferrals.
