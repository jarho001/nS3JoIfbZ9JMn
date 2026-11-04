"""Projection-only checks for initial per-player request-session views."""

from __future__ import annotations

import json
import pathlib
import subprocess
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from simulator import (
	ACTION_TACKLE,
	PHASE_MOVE_REQUEST,
	PHASE_TERMINAL,
	SessionRequestView,
	make_request_session,
	make_tie_state,
	project_request_view,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
ORACLE = ROOT / "reference" / "neutral-turn-oracle.cjs"


def _state(snapshot):
	return make_tie_state(
		hp=snapshot["hp"],
		max_hp=snapshot["maxHp"],
		pp=snapshot["pp"],
		level=snapshot["level"],
		attack=snapshot["attack"],
		defense=snapshot["defense"],
		speed=snapshot["speed"],
		rng=snapshot["rng"],
		turn=snapshot["turn"],
		request_phase=PHASE_TERMINAL if snapshot["terminal"] else PHASE_MOVE_REQUEST,
		terminal=snapshot["terminal"],
		winner=snapshot["winner"],
	)


def _request_projection(view):
	return {
		"phase": "move",
		"legalActions": ["move 1"] if bool(np.asarray(view.legal_move_mask)[0]) else [],
		"own": {
			"species": "Mew",
			"level": int(view.own_level),
			"hpCondition": f"{int(view.own_hp)}/{int(view.own_max_hp)}",
			"stats": {"atk": int(view.own_attack), "def": int(view.own_defense), "spe": int(view.own_speed)},
			"move": {"name": "Tackle", "id": "tackle", "pp": int(view.own_pp), "disabled": False},
		},
	}


def _leaf_equal(left, right):
	left_leaves = jax.tree_util.tree_leaves(left)
	right_leaves = jax.tree_util.tree_leaves(right)
	return len(left_leaves) == len(right_leaves) and all(
		np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(left_leaves, right_leaves)
	)


class RequestViewTests(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		completed = subprocess.run(
			["node", str(ORACLE), "--sessions"],
			cwd=ROOT,
			check=True,
			capture_output=True,
			text=True,
		)
		cls.fixture = json.loads(completed.stdout)

	def test_initial_request_projection_is_private_read_only_and_transformable(self):
		scenario = self.fixture["scenarios"][0]
		state = _state(scenario["initial"]["state"])
		session = make_request_session(state, episode=7)
		before = session

		for audience, side in ((0, "p1"), (1, "p2")):
			view = project_request_view(session, audience)
			self.assertIsInstance(view, SessionRequestView)
			self.assertEqual(_request_projection(view.player_view), scenario["initial"]["requests"][side])
			self.assertEqual(int(view.episode), 7)
			self.assertFalse(bool(view.own_choice_submitted))
			self.assertEqual(view._fields, ("player_view", "episode", "own_choice_submitted"))
			self.assertNotIn("rng", view.player_view._fields)
			self.assertNotIn("pending_choices", view.player_view._fields)

		first_commit = session._replace(pending_choices=jnp.asarray([ACTION_TACKLE, 0], dtype=jnp.int32))
		both_committed = session._replace(pending_choices=jnp.asarray([ACTION_TACKLE, ACTION_TACKLE], dtype=jnp.int32))
		p1_before_opponent = project_request_view(first_commit, 0)
		p1_after_opponent = project_request_view(both_committed, 0)
		self.assertTrue(_leaf_equal(p1_before_opponent, p1_after_opponent))
		self.assertTrue(bool(p1_before_opponent.own_choice_submitted))
		self.assertFalse(bool(project_request_view(first_commit, 1).own_choice_submitted))
		self.assertTrue(_leaf_equal(session, before))

		with self.assertRaisesRegex(TypeError, "session must be a RequestSession"):
			project_request_view(object(), 0)
		with self.assertRaisesRegex(ValueError, "pending_choices must have shape"):
			project_request_view(session._replace(pending_choices=jnp.zeros((1,), dtype=jnp.int32)), 0)
		with self.assertRaisesRegex(TypeError, "episode must have dtype"):
			project_request_view(session._replace(episode=jnp.asarray(7, dtype=jnp.float32)), 0)
		with self.assertRaisesRegex(TypeError, "audience must be an integer"):
			project_request_view(session, 0.0)
		with self.assertRaisesRegex(ValueError, "audience must be p1"):
			project_request_view(session, 2)

		compiled = jax.jit(project_request_view, static_argnums=1)(session, 0)
		self.assertTrue(_leaf_equal(project_request_view(session, 0), compiled))
		batched_sessions = jax.tree_util.tree_map(
			lambda *values: jnp.stack(values),
			session,
			first_commit,
			both_committed,
		)
		project_batch = jax.vmap(lambda item: project_request_view(item, 0))
		batched = project_batch(batched_sessions)
		compiled_batch = jax.jit(project_batch)(batched_sessions)
		self.assertTrue(_leaf_equal(batched, compiled_batch))
		for lane, item in enumerate((session, first_commit, both_committed)):
			self.assertTrue(_leaf_equal(jax.tree_util.tree_map(lambda value: value[lane], batched), project_request_view(item, 0)))
		self.assertTrue(_leaf_equal(session, before))


if __name__ == "__main__":
	unittest.main()
