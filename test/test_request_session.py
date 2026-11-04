"""Differential checks for staged private submissions around v2 turns."""

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
	DRAW_ACTION_UPDATE,
	DRAW_BEFORE_TURN,
	DRAW_MOVE_HIT_UPDATE,
	DRAW_QUEUE_SORT,
	DRAW_RESIDUAL_UPDATE,
	MAX_REQUEST_EPISODE,
	PHASE_MOVE_REQUEST,
	PHASE_TERMINAL,
	SESSION_DUPLICATE,
	SESSION_INVALID_ACTION,
	SESSION_INVALID_PLAYER,
	SESSION_OVERFLOW,
	SESSION_RESOLVED,
	SESSION_STALE_EPISODE,
	SESSION_TERMINAL,
	SESSION_UNSUPPORTED,
	SESSION_WAITING,
	STATUS_PENDING,
	STATUS_TERMINAL,
	STATUS_UNSUPPORTED,
	make_request_session,
	make_tie_state,
	project_submission_view,
	submit_choice,
	transition_with_speed_ties,
)
from simulator.neutral_tackle import (
	DRAW_ACCURACY,
	DRAW_CRITICAL,
	DRAW_DAMAGE,
	SPEED_TIE_VARIANT_ID,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
ORACLE = ROOT / "reference" / "neutral-turn-oracle.cjs"
DRAW_CONTEXTS = {
	"accuracy": DRAW_ACCURACY,
	"critical": DRAW_CRITICAL,
	"damage": DRAW_DAMAGE,
	"queue-sort": DRAW_QUEUE_SORT,
	"before-turn": DRAW_BEFORE_TURN,
	"turn-start-update": DRAW_ACTION_UPDATE,
	"move-hit-update": DRAW_MOVE_HIT_UPDATE,
	"action-update": DRAW_ACTION_UPDATE,
	"residual-update": DRAW_RESIDUAL_UPDATE,
}
_COMPILED_SUBMIT = jax.jit(submit_choice)
_COMPILED_VIEW = jax.jit(project_submission_view, static_argnums=1)


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


def _as_list(value):
	return np.asarray(value).tolist()


def _leaf_equal(left, right):
	left_leaves = jax.tree_util.tree_leaves(left)
	right_leaves = jax.tree_util.tree_leaves(right)
	return len(left_leaves) == len(right_leaves) and all(
		np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(left_leaves, right_leaves)
	)


def _view_projection(view):
	request = {
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
	events = []
	for index, actor in enumerate(_as_list(view.action_order)):
		if actor < 0:
			continue
		target = int(view.target_player[index])
		actor_id = "p1" if actor == 0 else "p2"
		target_id = "p1" if target == 0 else "p2"
		events.append({"type": "move", "actor": actor_id, "move": "tackle", "target": target_id})
		if bool(view.critical[index]):
			events.append({"type": "critical", "target": target_id})
		hp = int(view.target_hp_value[index])
		if hp == 0:
			condition = "0 fnt"
		elif bool(view.target_hp_is_percent[index]):
			condition = f"{hp}/100"
		else:
			condition = f"{hp}/{int(view.own_max_hp)}"
		events.append({"type": "damage", "target": target_id, "condition": condition})
		if bool(view.fainted[index]):
			events.append({"type": "faint", "target": target_id})
	if bool(view.terminal):
		winner = int(view.winner)
		events.append({"type": "winner", "side": "p1" if winner == 1 else "p2" if winner == 2 else "tie"})
	elif bool(view.turn_event):
		events.append({"type": "turn", "number": int(view.turn_number)})
	return {"request": request, "events": events}


class RequestSessionTests(unittest.TestCase):
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
		cls.scenarios = {scenario["id"]: scenario for scenario in cls.fixture["scenarios"]}

	def _assert_snapshot(self, state, snapshot, label):
		for name, field in (
			("hp", "hp"),
			("max_hp", "maxHp"),
			("pp", "pp"),
			("level", "level"),
			("attack", "attack"),
			("defense", "defense"),
			("speed", "speed"),
			("rng", "rng"),
		):
			self.assertEqual(_as_list(getattr(state, name)), snapshot[field], (*label, field))
		self.assertEqual(int(state.turn), snapshot["turn"], (*label, "turn"))
		self.assertEqual(int(state.request_phase), PHASE_TERMINAL if snapshot["requestPhase"] == "terminal" else PHASE_MOVE_REQUEST, (*label, "request phase"))
		self.assertEqual(bool(state.terminal), snapshot["terminal"], (*label, "terminal"))
		self.assertEqual(int(state.winner), snapshot["winner"], (*label, "winner"))

	def _assert_draws(self, transition, expected, label):
		self.assertEqual(int(transition.draws.count), len(expected), (*label, "draw count"))
		for index, draw in enumerate(expected):
			trace = transition.draws
			self.assertEqual(int(trace.bounds[index]), 2 if draw["context"] == "speed-sort" else draw["bound"], (*label, index, "bound"))
			self.assertEqual(int(trace.values[index]), draw["value"], (*label, index, "value"))
			self.assertEqual(int(trace.contexts[index]), DRAW_CONTEXTS[draw.get("phase", draw["context"])], (*label, index, "context"))
			self.assertEqual(_as_list(trace.before[index]), draw["before"], (*label, index, "before RNG"))
			self.assertEqual(_as_list(trace.after[index]), draw["after"], (*label, index, "after RNG"))
			if index:
				self.assertEqual(draw["before"], expected[index - 1]["after"], (*label, index, "draw chain"))

	def test_live_native_partial_and_second_commits_match_chained_tied_and_unequal_turns(self):
		self.assertEqual(self.fixture["kind"], "gen5-restricted-neutral-tackle-staged-session-oracle")
		self.assertRegex(self.fixture["captureContract"], r"each Battle\.choose separately")
		self.assertEqual(
			{scenario["id"] for scenario in self.fixture["scenarios"]},
			{"tied-first-p1", "tied-first-p2", "unequal-first-p1", "unequal-first-p2"},
		)
		for speed_control in ("tied", "unequal"):
			first_p1 = self.scenarios[f"{speed_control}-first-p1"]
			first_p2 = self.scenarios[f"{speed_control}-first-p2"]
			self.assertEqual(first_p1["initial"], first_p2["initial"], (speed_control, "independent identical initial battles"))
			self.assertEqual(first_p1["turns"][0]["firstPlayer"], "p1", (speed_control, "p1 first order"))
			self.assertEqual(first_p2["turns"][0]["firstPlayer"], "p2", (speed_control, "p2 first order"))

		for scenario in self.fixture["scenarios"]:
			state = _state(scenario["initial"]["state"])
			session = make_request_session(state)
			self.assertEqual(len(scenario["turns"]), 2, scenario["id"])
			for turn_index, turn in enumerate(scenario["turns"]):
				label = (scenario["id"], turn_index)
				self._assert_snapshot(session.battle, turn["before"], (*label, "before"))
				first_player = 0 if turn["firstPlayer"] == "p1" else 1
				second_player = 1 - first_player
				old_session = session
				first = submit_choice(session, first_player, ACTION_TACKLE, session.episode)
				compiled_first = _COMPILED_SUBMIT(session, jnp.asarray(first_player, jnp.int32), jnp.asarray(ACTION_TACKLE, jnp.int32), session.episode)
				self.assertTrue(_leaf_equal(first, compiled_first), (*label, "waiting eager/JIT"))
				self.assertEqual(int(first.status), SESSION_WAITING, (*label, "waiting status"))
				self.assertEqual(int(first.transition.status), STATUS_PENDING, (*label, "kernel pending status"))
				self.assertEqual(_as_list(first.session.pending_choices), [int(index == first_player) for index in range(2)], (*label, "private pending pair"))
				self.assertTrue(_leaf_equal(first.session.battle, old_session.battle), (*label, "battle unchanged"))
				self.assertTrue(_leaf_equal(first.transition.state, old_session.battle), (*label, "pending transition unchanged"))
				self.assertEqual(int(first.transition.draws.count), 0, (*label, "partial draws"))
				self.assertEqual(_as_list(first.transition.events.executed), [False, False], (*label, "partial events"))
				self._assert_snapshot(first.session.battle, turn["firstCommit"]["state"], (*label, "native partial"))
				self.assertEqual(turn["firstCommit"]["draws"], [], (*label, "native partial draws"))
				self.assertEqual(turn["totalDrawCount"], len(turn["firstCommit"]["draws"]) + len(turn["draws"]), (*label, "native total draws"))

				for audience, side in ((0, "p1"), (1, "p2")):
					view = project_submission_view(first, audience)
					compiled_view = _COMPILED_VIEW(first, audience)
					self.assertTrue(_leaf_equal(view, compiled_view), (*label, side, "view eager/JIT"))
					self.assertEqual(int(view.episode), int(old_session.episode), (*label, side, "local episode"))
					self.assertEqual(bool(view.own_choice_submitted), audience == first_player, (*label, side, "own commit bit"))
					self.assertEqual(_view_projection(view.player_view), {"request": turn["requests"][side], "events": []}, (*label, side, "partial current request"))
					self.assertEqual(_view_projection(view.resolved_events)["events"], [], (*label, side, "partial resolved events"))
					self.assertNotIn("pending_choices", view._fields)
					self.assertNotIn("rng", view.player_view._fields)
					self.assertNotIn("draws", view.player_view._fields)
					self.assertNotIn("pending_choices", view.resolved_events._fields)
					self.assertNotIn("opponent_choice_submitted", view._fields)

				# The local adapter rejects repeats and bad second submissions without
				# losing the other player's already accepted private commitment.
				for result, expected_status, suffix in (
					(submit_choice(first.session, first_player, ACTION_TACKLE, first.session.episode), SESSION_DUPLICATE, "duplicate"),
					(submit_choice(first.session, second_player, ACTION_TACKLE + 1, first.session.episode), SESSION_INVALID_ACTION, "invalid action"),
					(submit_choice(first.session, 2, ACTION_TACKLE, first.session.episode), SESSION_INVALID_PLAYER, "invalid player"),
					(submit_choice(first.session, second_player, ACTION_TACKLE, first.session.episode + 1), SESSION_STALE_EPISODE, "stale episode"),
				):
					self.assertEqual(int(result.status), expected_status, (*label, suffix))
					self.assertTrue(_leaf_equal(result.session, first.session), (*label, suffix, "session preserved"))
					self.assertTrue(_leaf_equal(result.before, first.session.battle), (*label, suffix, "battle preserved"))
					self.assertEqual(int(result.transition.draws.count), 0, (*label, suffix, "no draws"))

				resolved = submit_choice(first.session, second_player, ACTION_TACKLE, first.session.episode)
				compiled_resolved = _COMPILED_SUBMIT(first.session, jnp.asarray(second_player, jnp.int32), jnp.asarray(ACTION_TACKLE, jnp.int32), first.session.episode)
				self.assertTrue(_leaf_equal(resolved, compiled_resolved), (*label, "resolved eager/JIT"))
				self.assertEqual(int(resolved.status), SESSION_RESOLVED, (*label, "resolved status"))
				self.assertEqual(int(resolved.transition.status), 0, (*label, "kernel resolved status"))
				self.assertEqual(_as_list(resolved.session.pending_choices), [0, 0], (*label, "cleared commitments"))
				self.assertEqual(int(resolved.session.episode), int(old_session.episode) + 1, (*label, "next local episode"))
				self.assertTrue(_leaf_equal(resolved.session.battle, resolved.transition.state), (*label, "committed battle"))
				self._assert_snapshot(resolved.transition.state, turn["after"], (*label, "after"))
				self._assert_draws(resolved.transition, turn["draws"], label)
				actors = [int(actor) for actor in _as_list(resolved.transition.events.actors) if actor >= 0]
				self.assertEqual(["p1" if actor == 0 else "p2" for actor in actors], turn["actionOrder"], (*label, "native action order"))
				self.assertEqual(_as_list(resolved.transition.events.executed), [True, True], (*label, "native executed events"))
				for audience, side in ((0, "p1"), (1, "p2")):
					view = project_submission_view(resolved, audience)
					self.assertEqual(_view_projection(view.resolved_events), turn["views"][side], (*label, side, "resolved native events/view"))
					self.assertEqual(_view_projection(view.player_view), {"request": turn["nextRequests"][side], "events": []}, (*label, side, "next native request"))
					self.assertFalse(bool(view.own_choice_submitted), (*label, side, "cleared own commitment"))
				self.assertEqual(turn["before"]["speed"], [120, 120] if "tied" in scenario["id"] else [120, 152], (*label, "actual native speeds"))
				self.assertEqual(turn["totalDrawCount"], int(resolved.transition.draws.count), (*label, "native total draws"))
				session = resolved.session

	def test_actor_view_is_invariant_to_opponents_private_commit_bit(self):
		state = _state(self.scenarios["tied-first-p1"]["initial"]["state"])
		first = submit_choice(make_request_session(state), 0, ACTION_TACKLE, 0)
		changed_opponent = first._replace(
			session=first.session._replace(pending_choices=jnp.asarray([ACTION_TACKLE, ACTION_TACKLE], dtype=jnp.int32)),
		)
		self.assertTrue(_leaf_equal(project_submission_view(first, 0), project_submission_view(changed_opponent, 0)))
		self.assertTrue(bool(project_submission_view(first, 0).own_choice_submitted))
		self.assertTrue(bool(project_submission_view(changed_opponent, 1).own_choice_submitted))

	def test_status_precedence_terminal_unsupported_and_bounded_episode(self):
		state = _state(self.scenarios["tied-first-p1"]["initial"]["state"])
		base = make_request_session(state)
		stale_over_invalid = submit_choice(base, 9, -1, 1)
		self.assertEqual(int(stale_over_invalid.status), SESSION_STALE_EPISODE)
		self.assertTrue(_leaf_equal(stale_over_invalid.session, base))

		terminal_state = state._replace(terminal=jnp.asarray(True), request_phase=jnp.asarray(PHASE_TERMINAL, dtype=jnp.int32))
		terminal_session = make_request_session(terminal_state)
		terminal = submit_choice(terminal_session, 0, ACTION_TACKLE, 0)
		self.assertEqual(int(terminal.status), SESSION_TERMINAL)
		self.assertEqual(int(terminal.transition.status), STATUS_TERMINAL)
		self.assertTrue(_leaf_equal(terminal.session, terminal_session))
		terminal_view = project_submission_view(terminal, 0)
		self.assertFalse(bool(terminal_view.player_view.legal_move_mask[0]))
		self.assertTrue(bool(terminal_view.player_view.terminal))

		unsupported_state = state._replace(variant_id=jnp.asarray(SPEED_TIE_VARIANT_ID + 1, dtype=jnp.int32))
		unsupported_session = make_request_session(unsupported_state)
		unsupported = submit_choice(unsupported_session, 0, ACTION_TACKLE, 0)
		self.assertEqual(int(unsupported.status), SESSION_UNSUPPORTED)
		self.assertEqual(int(unsupported.transition.status), STATUS_UNSUPPORTED)
		self.assertTrue(_leaf_equal(unsupported.session, unsupported_session))

		out_of_pp_state = state._replace(pp=jnp.asarray([0, 56], dtype=jnp.int32))
		out_of_pp = submit_choice(make_request_session(out_of_pp_state), 1, ACTION_TACKLE, 0)
		out_of_pp_view = project_submission_view(out_of_pp, 0)
		self.assertFalse(bool(out_of_pp_view.player_view.legal_move_mask[0]))

		at_limit = make_request_session(state, episode=MAX_REQUEST_EPISODE)
		overflow = submit_choice(at_limit, 0, ACTION_TACKLE, MAX_REQUEST_EPISODE)
		self.assertEqual(int(overflow.status), SESSION_OVERFLOW)
		self.assertTrue(_leaf_equal(overflow.session, at_limit))

		last_usable = make_request_session(state, episode=MAX_REQUEST_EPISODE - 1)
		first = submit_choice(last_usable, 0, ACTION_TACKLE, MAX_REQUEST_EPISODE - 1)
		resolved = submit_choice(first.session, 1, ACTION_TACKLE, MAX_REQUEST_EPISODE - 1)
		self.assertEqual(int(resolved.status), SESSION_RESOLVED)
		self.assertEqual(int(resolved.session.episode), MAX_REQUEST_EPISODE)
		post_limit = submit_choice(resolved.session, 0, ACTION_TACKLE, MAX_REQUEST_EPISODE)
		self.assertEqual(int(post_limit.status), SESSION_OVERFLOW)
		self.assertTrue(_leaf_equal(post_limit.session, resolved.session))
		negative_epoch = base._replace(episode=jnp.asarray(-1, dtype=jnp.int32))
		self.assertEqual(int(submit_choice(negative_epoch, 0, ACTION_TACKLE, -1).status), SESSION_UNSUPPORTED)

		bad_pending = base._replace(pending_choices=jnp.asarray([2, 0], dtype=jnp.int32))
		self.assertEqual(int(submit_choice(bad_pending, 0, ACTION_TACKLE, 0).status), SESSION_UNSUPPORTED)
		with self.assertRaisesRegex(ValueError, "pending_choices"):
			submit_choice(base._replace(pending_choices=jnp.zeros((3,), dtype=jnp.int32)), 0, ACTION_TACKLE, 0)
		with self.assertRaises((TypeError, ValueError)):
			make_request_session(state, episode=True)
		with self.assertRaises(ValueError):
			make_request_session(state, episode=MAX_REQUEST_EPISODE + 1)
		with self.assertRaises(TypeError):
			submit_choice(base, 0.5, ACTION_TACKLE, 0)

	def test_heterogeneous_jit_vmap_matches_individual_lanes_and_permutation(self):
		state = _state(self.scenarios["tied-first-p1"]["initial"]["state"])
		base = make_request_session(state)
		sessions = [
			base,
			base._replace(pending_choices=jnp.asarray([0, ACTION_TACKLE], dtype=jnp.int32)),
			base._replace(pending_choices=jnp.asarray([ACTION_TACKLE, 0], dtype=jnp.int32)),
			base,
		]
		players = jnp.asarray([0, 0, 0, 1], dtype=jnp.int32)
		choices = jnp.asarray([ACTION_TACKLE, ACTION_TACKLE, ACTION_TACKLE, ACTION_TACKLE], dtype=jnp.int32)
		episodes = jnp.asarray([0, 0, 0, 1], dtype=jnp.int32)
		batched_sessions = jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *sessions)
		batched = jax.jit(jax.vmap(submit_choice))(batched_sessions, players, choices, episodes)
		self.assertEqual(_as_list(batched.status), [SESSION_WAITING, SESSION_RESOLVED, SESSION_DUPLICATE, SESSION_STALE_EPISODE])
		for index, session in enumerate(sessions):
			individual = submit_choice(session, players[index], choices[index], episodes[index])
			row = jax.tree_util.tree_map(lambda leaf: leaf[index], batched)
			self.assertTrue(_leaf_equal(row, individual), (index, "vmap versus individual"))
		self.assertTrue(_leaf_equal(jax.tree_util.tree_map(lambda leaf: leaf[0], batched).session.battle, sessions[0].battle), "waiting lane no-advance")
		permutation = jnp.asarray([3, 1, 0, 2], dtype=jnp.int32)
		reordered = jax.jit(jax.vmap(submit_choice))(
			jax.tree_util.tree_map(lambda value: value[permutation], batched_sessions),
			players[permutation],
			choices[permutation],
			episodes[permutation],
		)
		for new_index, old_index in enumerate(np.asarray(permutation).tolist()):
			self.assertTrue(
				_leaf_equal(
					jax.tree_util.tree_map(lambda leaf: leaf[new_index], reordered),
					jax.tree_util.tree_map(lambda leaf: leaf[old_index], batched),
				),
				(old_index, new_index, "permutation invariance"),
			)


if __name__ == "__main__":
	unittest.main()
