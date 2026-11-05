"""Heterogeneous batched parity for staged request-session submissions."""

from __future__ import annotations

import json
import pathlib
import subprocess
import unittest

import jax
import jax.numpy as jnp

from simulator import (
    ACTION_TACKLE,
    PHASE_TERMINAL,
    SESSION_DUPLICATE,
    SESSION_RESOLVED,
    SESSION_STALE_EPISODE,
    SESSION_TERMINAL,
    SESSION_WAITING,
    STATUS_OK,
    STATUS_PENDING,
    STATUS_TERMINAL,
    make_request_session,
    project_submission_view,
    submit_choice,
)
from test import test_request_session_duel as duel_test
from test.test_request_session import (
    _as_list,
    _leaf_equal,
    _state,
    _view_projection,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
ORACLE = ROOT / "reference" / "neutral-turn-oracle.cjs"
_COMPILED_BATCH_SUBMIT = jax.jit(jax.vmap(submit_choice))


def _stack_sessions(sessions):
    return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *sessions)


def _lane(tree, index):
    return jax.tree_util.tree_map(lambda value: value[index], tree)


class RequestSessionBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        completed = subprocess.run(
            ["node", str(ORACLE), "--session-duels"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=25,
        )
        cls.fixture = json.loads(completed.stdout)
        cls.scenarios = {scenario["id"]: scenario for scenario in cls.fixture["scenarios"]}

    def _assert_snapshot(self, state, snapshot, label):
        duel_test.RequestSessionDuelTests._assert_snapshot(self, state, snapshot, label)

    def _assert_draws(self, transition, expected, label):
        duel_test.RequestSessionDuelTests._assert_draws(self, transition, expected, label)

    def _assert_no_private_trace(self, view, label):
        duel_test.RequestSessionDuelTests._assert_no_private_trace(self, view, label)

    def _assert_quiet(self, result, source, expected_status, label, terminal=False):
        self.assertEqual(int(result.status), expected_status, (*label, "session status"))
        self.assertEqual(
            int(result.transition.status),
            STATUS_TERMINAL if terminal else STATUS_PENDING,
            (*label, "idle transition status"),
        )
        self.assertTrue(_leaf_equal(result.session, source), (*label, "session unchanged"))
        self.assertTrue(_leaf_equal(result.before, source.battle), (*label, "before state"))
        self.assertTrue(_leaf_equal(result.transition.state, source.battle), (*label, "battle and RNG unchanged"))
        self.assertEqual(int(result.transition.draws.count), 0, (*label, "no RNG draws"))
        self.assertEqual(_as_list(result.transition.events.executed), [False, False], (*label, "no events"))

    def _assert_waiting(self, result, source, scenario, turn_index, player, label):
        turn = scenario["turns"][turn_index]
        self.assertEqual(int(result.status), SESSION_WAITING, (*label, "waiting status"))
        self.assertEqual(int(result.transition.status), STATUS_PENDING, (*label, "pending transition"))
        self.assertTrue(_leaf_equal(result.before, source.battle), (*label, "before state"))
        self.assertTrue(_leaf_equal(result.session.battle, source.battle), (*label, "battle freeze"))
        self.assertTrue(_leaf_equal(result.transition.state, source.battle), (*label, "transition state freeze"))
        self.assertEqual(
            _as_list(result.session.pending_choices),
            [int(index == player) for index in range(2)],
            (*label, "lane-local commitment"),
        )
        self.assertEqual(int(result.session.episode), turn_index, (*label, "episode stays local"))
        self.assertEqual(int(result.transition.draws.count), 0, (*label, "waiting consumes no RNG"))
        self.assertEqual(_as_list(result.transition.events.executed), [False, False], (*label, "waiting emits no events"))
        self._assert_snapshot(result.session.battle, turn["firstCommit"]["state"], (*label, "native partial state"))
        self.assertTrue(turn["firstCommit"]["accepted"], (*label, "native partial accepted"))
        self.assertEqual(turn["firstCommit"]["draws"], [], (*label, "native partial draws"))
        self.assertEqual(turn["firstCommit"]["requests"], turn["requests"], (*label, "native requests frozen"))

        for audience, side in ((0, "p1"), (1, "p2")):
            view = project_submission_view(result, audience)
            self._assert_no_private_trace(view, (*label, side, "waiting view"))
            self.assertEqual(
                _view_projection(view.player_view),
                {"request": turn["requests"][side], "events": []},
                (*label, side, "audience request"),
            )
            self.assertEqual(_view_projection(view.resolved_events)["events"], [], (*label, side, "no partial events"))
            self.assertEqual(int(view.episode), turn_index, (*label, side, "projected episode"))
            self.assertEqual(bool(view.own_choice_submitted), audience == player, (*label, side, "own commit bit"))

    def _assert_resolved(self, result, source, scenario, turn_index, label):
        turn = scenario["turns"][turn_index]
        self.assertEqual(int(result.status), SESSION_RESOLVED, (*label, "resolved status"))
        self.assertEqual(int(result.transition.status), STATUS_OK, (*label, "resolved transition"))
        self._assert_snapshot(result.before, turn["before"], (*label, "native before state"))
        self._assert_snapshot(result.transition.state, turn["after"], (*label, "native after state"))
        self.assertTrue(_leaf_equal(result.session.battle, result.transition.state), (*label, "committed battle"))
        self.assertEqual(_as_list(result.session.pending_choices), [0, 0], (*label, "commitments cleared"))
        self.assertEqual(int(result.session.episode), turn_index + 1, (*label, "lane episode advanced once"))
        self.assertEqual(turn["totalDrawCount"], len(turn["draws"]), (*label, "native draw count"))
        self.assertEqual(int(result.transition.draws.count), turn["totalDrawCount"], (*label, "draw count"))
        self._assert_draws(result.transition, turn["draws"], label)
        actors = [int(actor) for actor in _as_list(result.transition.events.actors) if actor >= 0]
        self.assertEqual(
            ["p1" if actor == 0 else "p2" for actor in actors],
            turn["actionOrder"],
            (*label, "native action order"),
        )
        self.assertEqual(
            _as_list(result.transition.events.executed),
            [index < len(turn["actionOrder"]) for index in range(2)],
            (*label, "executed action mask"),
        )

        for audience, side in ((0, "p1"), (1, "p2")):
            view = project_submission_view(result, audience)
            self._assert_no_private_trace(view, (*label, side, "resolved view"))
            self.assertEqual(
                _view_projection(view.resolved_events),
                turn["views"][side],
                (*label, side, "recipient events"),
            )
            self.assertFalse(bool(view.own_choice_submitted), (*label, side, "commit cleared"))
            expected_request = turn["nextRequests"][side]
            if expected_request is None:
                self.assertTrue(bool(view.player_view.terminal), (*label, side, "terminal request"))
                self.assertEqual(int(view.player_view.request_phase), PHASE_TERMINAL, (*label, side, "terminal phase"))
                self.assertEqual(_as_list(view.player_view.legal_move_mask), [False], (*label, side, "terminal action mask"))
                self.assertEqual(int(view.player_view.winner), int(turn["after"]["winner"]), (*label, side, "terminal winner"))
            else:
                self.assertEqual(
                    _view_projection(view.player_view),
                    {"request": expected_request, "events": []},
                    (*label, side, "next native request"),
                )

    def _submit_batch(self, sessions, players, expected_episodes):
        batch = _COMPILED_BATCH_SUBMIT(
            _stack_sessions(sessions),
            jnp.asarray(players, dtype=jnp.int32),
            jnp.full((len(sessions),), ACTION_TACKLE, dtype=jnp.int32),
            jnp.asarray(expected_episodes, dtype=jnp.int32),
        )
        return [_lane(batch, index) for index in range(len(sessions))]

    def _current_first_player(self, scenario, turn_index):
        first = scenario["turns"][turn_index]["firstPlayer"]
        return 0 if first == "p1" else 1

    def _assert_terminal(self, session, scenario, label):
        outcome = scenario["outcome"]
        self.assertTrue(outcome["terminal"], (*label, "native terminal"))
        self.assertIn(outcome["winner"], ("p1", "p2"), (*label, "native winner"))
        self.assertEqual(outcome["pendingChoiceCount"], 0, (*label, "native choices cleared"))
        self._assert_snapshot(session.battle, outcome["state"], (*label, "native outcome"))
        self.assertTrue(bool(session.battle.terminal), (*label, "session terminal"))
        self.assertEqual(int(session.battle.winner), outcome["sideWinner"], (*label, "terminal winner"))
        self.assertEqual(_as_list(session.pending_choices), [0, 0], (*label, "terminal commitments"))
        self.assertEqual(int(session.episode), len(scenario["turns"]), (*label, "terminal episode"))

    def test_one_compiled_vmap_keeps_heterogeneous_full_duels_lane_local(self):
        self.assertEqual(
            self.fixture["kind"],
            "gen5-restricted-neutral-tackle-staged-session-full-duel-oracle",
        )
        self.assertEqual(
            set(self.scenarios),
            {"tied-first-p1", "tied-first-p2", "unequal-first-p1", "unequal-first-p2"},
        )
        labels = ("tied-first-p1", "tied-first-p2", "unequal-first-p1", "unequal-first-p2")
        scenarios = [self.scenarios[label] for label in labels]
        sessions = [make_request_session(_state(scenario["initial"]["state"])) for scenario in scenarios]
        turn_indices = [0] * len(sessions)
        terminal_checks = [0] * len(sessions)

        # Stage one private commit in all four independent fixture sessions.
        players = [self._current_first_player(scenario, 0) for scenario in scenarios]
        source = list(sessions)
        results = self._submit_batch(sessions, players, [0] * len(sessions))
        for lane_index, (result, scenario, player) in enumerate(zip(results, scenarios, players)):
            self._assert_waiting(result, source[lane_index], scenario, 0, player, (labels[lane_index], "initial batch"))
        sessions = [result.session for result in results]

        # Resolve one lane while duplicate and stale calls are rejected in two
        # others; every result still comes from the same batched executable.
        pending_by_lane = [_as_list(session.pending_choices).index(ACTION_TACKLE) for session in sessions]
        players = [
            1 - pending_by_lane[0],
            pending_by_lane[1],
            1 - pending_by_lane[2],
            pending_by_lane[3],
        ]
        episodes = [0, 0, -1, 0]
        source = list(sessions)
        results = self._submit_batch(sessions, players, episodes)
        self._assert_resolved(results[0], source[0], scenarios[0], 0, (labels[0], "mixed resolution"))
        self._assert_quiet(results[1], source[1], SESSION_DUPLICATE, (labels[1], "duplicate rejection"))
        self._assert_quiet(results[2], source[2], SESSION_STALE_EPISODE, (labels[2], "stale rejection"))
        self._assert_quiet(results[3], source[3], SESSION_DUPLICATE, (labels[3], "duplicate rejection"))
        turn_indices[0] += 1
        sessions = [result.session for result in results]

        # On this call lanes 0/1/2 return WAITING/RESOLVED/RESOLVED while lane
        # 3 rejects a duplicate. No lane's call consumes another lane's RNG.
        players = [
            self._current_first_player(scenarios[0], turn_indices[0]),
            1 - pending_by_lane[1],
            1 - pending_by_lane[2],
            pending_by_lane[3],
        ]
        episodes = [1, 0, 0, 0]
        source = list(sessions)
        results = self._submit_batch(sessions, players, episodes)
        self._assert_waiting(results[0], source[0], scenarios[0], 1, players[0], (labels[0], "asynchronous waiting"))
        self._assert_resolved(results[1], source[1], scenarios[1], 0, (labels[1], "asynchronous resolution"))
        self._assert_resolved(results[2], source[2], scenarios[2], 0, (labels[2], "asynchronous resolution"))
        self._assert_quiet(results[3], source[3], SESSION_DUPLICATE, (labels[3], "asynchronous duplicate"))
        turn_indices[1] += 1
        turn_indices[2] += 1
        sessions = [result.session for result in results]

        # Advance the remaining fixture turns with every lane present in each
        # call. Completed lanes receive terminal submissions while others wait
        # or resolve, proving terminal absorption under the same batch path.
        while not all(bool(session.battle.terminal) for session in sessions):
            source = list(sessions)
            players = []
            episodes = []
            modes = []
            for lane_index, (session, scenario) in enumerate(zip(sessions, scenarios)):
                pending = _as_list(session.pending_choices)
                if bool(session.battle.terminal):
                    modes.append(("terminal", None))
                    players.append(0)
                elif ACTION_TACKLE in pending:
                    held_player = pending.index(ACTION_TACKLE)
                    modes.append(("resolve", held_player))
                    players.append(1 - held_player)
                else:
                    self.assertLess(turn_indices[lane_index], len(scenario["turns"]))
                    first_player = self._current_first_player(scenario, turn_indices[lane_index])
                    modes.append(("wait", first_player))
                    players.append(first_player)
                episodes.append(int(session.episode))

            results = self._submit_batch(sessions, players, episodes)
            for lane_index, (result, session, scenario, mode) in enumerate(zip(results, source, scenarios, modes)):
                kind, player = mode
                label = (labels[lane_index], "scheduled", turn_indices[lane_index])
                if kind == "terminal":
                    self._assert_quiet(result, session, SESSION_TERMINAL, label, terminal=True)
                    terminal_checks[lane_index] += 1
                elif kind == "wait":
                    self._assert_waiting(result, session, scenario, turn_indices[lane_index], player, label)
                else:
                    self._assert_resolved(result, session, scenario, turn_indices[lane_index], label)
                    turn_indices[lane_index] += 1
            sessions = [result.session for result in results]

        for lane_index, (session, scenario) in enumerate(zip(sessions, scenarios)):
            self.assertEqual(turn_indices[lane_index], len(scenario["turns"]), (labels[lane_index], "all native turns resolved"))
            self._assert_terminal(session, scenario, (labels[lane_index], "terminal"))
        self.assertGreater(sum(terminal_checks), 0, "terminal submissions overlap active lanes")


if __name__ == "__main__":
    unittest.main()
