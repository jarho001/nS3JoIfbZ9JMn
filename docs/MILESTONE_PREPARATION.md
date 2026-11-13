# Milestone preparation

Use this guidance when preparing a new goal or materially changing its scope.
The initial G0–G8 preparation is complete; the active goal and its next step are
in [ORCHESTRATION_STATE.json](ORCHESTRATION_STATE.json). This file does not
restart preparation, approve implementation or select Goal mode.

Follow [AGENTS.md](../AGENTS.md) for the configured model, ownership and validation
policy, and [ORCHESTRATION.md](ORCHESTRATION.md) for execution. Sharpen the relevant
[PROJECT_PLAN.md](../PROJECT_PLAN.md#milestone-contracts) sections around:

- Useful outcome, scope, dependencies, exclusions and concrete acceptance cases.
- Existing pinned native evidence, reusable handlers/harnesses and missing proof.
- Validation entry points, separating runnable commands from proposed harnesses,
  required acceptance from optional measurement, and pending from executed checks.
- Material user decisions and stopping conditions; routine choices stay with the
  primary. Preparation does not authorize implementation or device/performance runs.

Keep behavior cases in [SHOWDOWN_BEHAVIOR_CASES.md](SHOWDOWN_BEHAVIOR_CASES.md),
progress in the existing compact state and all device commands in
[GPU_HANDOFF.md](GPU_HANDOFF.md). Before implementing a family, use the existing
[reachability precheck](SHOWDOWN_BEHAVIOR_CASES.md#reachability-audit-and-family-precheck-2026-11-13).

The [completed 2026-11-06 assignment](archive/MILESTONE_PREPARATION_2026-11-06.md)
is preserved as historical planning context, including its original model/effort
and first-G3 instructions. It is not a current execution prompt.
