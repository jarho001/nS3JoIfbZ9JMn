"""Rollout contract regressions. Added without execution at the user's request."""
import json
from pathlib import Path
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from simulator import shared_engine as e
from simulator.rollout import SKIPPED, rollout, player_view
from simulator.shared_engine_io import player_view as state_view
from test.shared_engine_support import initialize_case, choices_array, assert_step


def equal(a, b):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_array_equal(x, y)


class RolloutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "test-logs/g3-testing-profiling/diagnostic/native-pack.json"
        cls.cases = json.loads(path.read_text())["cases"]

    def plan(self, case):
        state, _ = initialize_case(case)
        initial = state
        actions = []
        for step in case["steps"]:
            action = choices_array(state, step["choices"])
            actions.append(action)
            state = jax.jit(e.advance)(state, action).state
        return initial, jnp.stack(actions), state

    def test_native_full_outputs_and_final_only_match(self):
        for case in self.cases:
            with self.subTest(case=case["id"]):
                initial, actions, expected = self.plan(case)
                full = jax.jit(lambda s, a: rollout(s, a, retain=("states", "trace", "codes", "events", "draws", "rewards", "legal_actions")))(initial, actions)
                lean = jax.jit(rollout)(initial, actions)
                equal(lean.state, expected)
                equal(lean.state, full.state)
                self.assertEqual(lean.outputs, {})
                self.assertIsNone(lean.samples)
                self.assertEqual(int(lean.attempts), len(actions))
                self.assertEqual(int(lean.transitions), len(actions))
                self.assertTrue(bool(lean.terminal))
                self.assertFalse(bool(lean.failed) or bool(lean.truncated))
                self.assertEqual(int(lean.failure_step), -1)
                before = initial
                for k, step in enumerate(case["steps"]):
                    state = jax.tree.map(lambda x: x[k], full.outputs["states"])
                    trace = jax.tree.map(lambda x: x[k], full.outputs["trace"])
                    result = e.Result(state, full.outputs["codes"][k], trace)
                    assert_step(self, before, result, step, f"rollout/{case['id']}/{k}")
                    self.assertTrue(bool(full.outputs["valid"][k]))
                    np.testing.assert_array_equal(full.outputs["legal_actions"][k], e.legal_actions(state))
                    np.testing.assert_array_equal(full.outputs["events"].events[k], trace.events)
                    np.testing.assert_array_equal(full.outputs["draws"].args[k], trace.args)
                    before = state

    def test_first_error_freezes_successful_prefix(self):
        initial, actions, _ = self.plan(self.cases[0])
        prefix = jax.jit(e.advance)(initial, actions[0]).state
        plan = jnp.stack((actions[0], e.i32([11, 0]), actions[1]))
        result = jax.jit(lambda s, a: rollout(s, a, retain=("codes",)))(initial, plan)
        equal(result.state, prefix)
        self.assertEqual(int(result.code), e.INVALID)
        self.assertEqual(int(result.failure_step), 1)
        self.assertTrue(bool(result.failed))
        self.assertFalse(bool(result.truncated))
        self.assertEqual(int(result.attempts), 2)
        self.assertEqual(int(result.transitions), 1)
        self.assertEqual(result.outputs["codes"].tolist(), [e.OK, e.INVALID, SKIPPED])
        self.assertEqual(result.outputs["valid"].tolist(), [True, True, False])

    def test_pending_replace_undo_and_stale_token(self):
        initial, actions, _ = self.plan(self.cases[0])
        choice = actions[0]
        side = int(jnp.argmax(choice >= 0))
        pending = e.i32([-1, -1]).at[side].set(choice[side])
        cancel = e.i32([-1, -1]).at[side].set(-2)
        plan = jnp.stack((pending, pending, cancel, choice))
        result = jax.jit(rollout)(initial, plan)
        expected = jax.jit(e.advance)(initial, choice)
        equal(result.state, expected.state)
        self.assertEqual(int(result.attempts), 4)
        self.assertEqual(int(result.transitions), 1)
        self.assertTrue(bool(result.truncated))
        tokens = jnp.tile(jnp.stack((initial.generation, initial.episode)), (2, 1))
        tokens = tokens.at[1, 0].add(1)
        stale = jax.jit(lambda s, a, t: rollout(s, a, tokens=t))(initial, jnp.stack((pending, choice)), tokens)
        equal(stale.state, jax.jit(e.advance)(initial, pending).state)
        self.assertEqual(int(stale.code), e.STALE)
        self.assertEqual(int(stale.failure_step), 1)

    def test_capacity_unsupported_and_budget_are_not_terminal(self):
        initial, actions, _ = self.plan(self.cases[0])
        for state, code in ((initial._replace(hcount=e.i32(e.HISTORY)), e.CAPACITY),
                            (initial._replace(moves=initial.moves.at[0, 0, 0].set(99)), e.UNSUPPORTED)):
            result = jax.jit(rollout)(state, actions[:2])
            equal(result.state, state)
            self.assertEqual(int(result.code), code)
            self.assertEqual(int(result.failure_step), 0)
            self.assertEqual(int(result.attempts), 1)
            self.assertEqual(int(result.transitions), 0)
            self.assertFalse(bool(result.terminal))
        empty = jax.jit(lambda s, a: rollout(s, a, max_steps=0))(initial, actions)
        equal(empty.state, initial)
        self.assertEqual(int(empty.attempts), 0)
        self.assertTrue(bool(empty.truncated))
        partial = jax.jit(lambda s, a: rollout(s, a, max_steps=1))(initial, actions)
        resumed = jax.jit(rollout)(partial.state, actions[1:])
        whole = jax.jit(rollout)(initial, actions)
        equal(resumed.state, whole.state)
        no_rows = jax.jit(rollout)(initial, e.i32([]).reshape((0, 2)))
        equal(no_rows.state, initial)

    def test_terminal_padding_samples_and_safe_views(self):
        initial, actions, _ = self.plan(self.cases[2])
        padded = jnp.concatenate((actions, e.i32([[11, 0]])))
        samples = (0, len(actions) - 1, len(actions))
        result = jax.jit(lambda s, a: rollout(s, a, retain=("codes", "rewards"), sample_steps=samples))(initial, padded)
        self.assertEqual(int(result.attempts), len(actions))
        self.assertEqual(result.sample_valid.tolist(), [True, True, False])
        self.assertEqual(int(result.outputs["codes"][-1]), SKIPPED)
        self.assertEqual(result.outputs["rewards"][-1].tolist(), [0, 0])
        for side in range(2):
            view = player_view(result, side)
            self.assertEqual(view["final"], state_view(result.state, side))
            self.assertEqual([v["step"] for v in view["samples"]], list(samples[:2]))
        terminal = jax.jit(rollout)(result.state, actions[:1])
        equal(terminal.state, result.state)
        self.assertEqual(int(terminal.attempts), 0)
        self.assertEqual(int(terminal.code), e.TERMINAL)

    def test_mixed_vmap_lanes_and_disabled_rows(self):
        initial, actions, _ = self.plan(self.cases[0])
        bad = actions.at[1].set(e.i32([11, 0]))
        states = jax.tree.map(lambda x: jnp.stack((x, x)), initial)
        plans = jnp.stack((actions, bad))
        result = jax.jit(jax.vmap(rollout))(states, plans)
        for k, plan in enumerate((actions, bad)):
            equal(jax.tree.map(lambda x: x[k], result), jax.jit(rollout)(initial, plan))
        enabled = jnp.array([False, True])
        disabled = jax.jit(lambda s, a: rollout(s, a, enabled=enabled))(initial, jnp.stack((e.i32([11, 0]), actions[0])))
        equal(disabled.state, jax.jit(e.advance)(initial, actions[0]).state)
        self.assertEqual(int(disabled.attempts), 1)
        self.assertFalse(bool(disabled.failed))

    def test_host_shape_and_static_option_validation(self):
        initial, actions, _ = self.plan(self.cases[0])
        for kwargs in ({"max_steps": -1}, {"retain": ("unknown",)}, {"sample_steps": (1, 1)},
                       {"tokens": e.i32([0, 0])}, {"enabled": e.i32([1])}):
            with self.assertRaises(ValueError):
                rollout(initial, actions, **kwargs)


if __name__ == "__main__":
    unittest.main()
