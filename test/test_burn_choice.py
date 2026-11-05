from __future__ import annotations

import json
import re
import subprocess
import unittest
from pathlib import Path
from typing import Sequence, get_type_hints

import jax
import jax.numpy as jnp
import numpy as np

import simulator
import test.test_accuracy_choice as accuracy_helpers
from simulator.accuracy_choice import AccuracyChoicePlayerView, transition_accuracy_choice
from simulator.burn_choice import (
	BurnChoicePlayerView,
	BurnChoiceState,
	BurnChoiceTransition,
	BurnResidualEvent,
	BurnResidualPlayerView,
	MAX_PP_BY_SLOT,
	VARIANT_ID,
	VARIANT_NAME,
	make_burn_choice_state,
	project_burn_choice_view,
	transition_burn_choice,
)
from simulator.neutral_mega_punch import (
	DRAW_ACCURACY,
	DRAW_CRITICAL,
	DRAW_DAMAGE,
	MegaPunchDrawTrace,
	MegaPunchEvents,
	transition_mega_punch,
)
from simulator.neutral_tackle import (
	PHASE_MOVE_REQUEST,
	PHASE_PENDING,
	PHASE_TERMINAL,
	PHASE_UNSUPPORTED,
	STATUS_INVALID_CHOICE,
	STATUS_OK,
	STATUS_PENDING,
	STATUS_TERMINAL,
	STATUS_UNSUPPORTED,
)
from simulator.neutral_two_move import TwoMoveState, _adapt_state


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / 'reference' / 'burn-choice-oracle.cjs'
MOVE_IDS = (33, 5)
CONTEXT_IDS = {'accuracy': DRAW_ACCURACY, 'critical': DRAW_CRITICAL, 'damage': DRAW_DAMAGE}
SIDE_INDEX = {'p1': 0, 'p2': 1}
BASE_POWER = {1: 50, 2: 80}


def _phase(native):
	if native['terminal']:
		return PHASE_TERMINAL
	if any(all(pp == 0 for pp in row) for row in native['pp']):
		return PHASE_UNSUPPORTED
	return PHASE_MOVE_REQUEST


def _state(native):
	return make_burn_choice_state(
		hp=native['hp'], max_hp=native['maxHp'], pp=native['pp'],
		level=native['level'], attack=native['attack'], defense=native['defense'],
		speed=native['speed'], rng=native['rng'], turn=native['turn'],
		request_phase=_phase(native), terminal=native['terminal'], winner=native['winner'],
		burned=native['burned'], species_id=(151, 151),
	)


def _hp_from_condition(condition):
	match = re.match(r'^(\d+)', condition)
	if not match:
		raise AssertionError(f'unrecognized native HP condition: {condition!r}')
	return int(match.group(1))


def _ceil_percent(hp, maximum):
	percent = (hp * 100 + maximum - 1) // maximum
	return 99 if percent == 100 and hp < maximum else percent


def _raw_overkill_damage(before, actor, action, critical, damage_roll, burned):
	"""Only used for the fixed direct-KO fixture where native HP clips overkill."""
	target = 1 - actor
	level_term = (2 * before['level'][actor]) // 5 + 2
	core = ((level_term * BASE_POWER[action] * before['attack'][actor]) // before['defense'][target]) // 50
	pre_roll = core + 2
	if critical:
		pre_roll *= 2
	rolled = ((pre_roll * (100 - damage_roll)) // 100) & 0xFFFFFFFF
	if burned:
		# The accepted 0.5 fixed-point modifier rounds exact halves down.
		rolled = rolled // 2
	return max(1, rolled)


def _native_attack_rows(case):
	turn = case['turn']
	before = turn['before']
	choice_by_actor = {SIDE_INDEX[side]: action for side, action in case['choices'].items()}
	moves_by_actor = {SIDE_INDEX[event['actor']]: event for event in turn['events']['p1'] if event['type'] == 'move'}
	misses_by_actor = {SIDE_INDEX[event['actor']]: event for event in turn['events']['p1'] if event['type'] == 'miss'}
	critical_targets = {SIDE_INDEX[event['target']] for event in turn['events']['p1'] if event['type'] == 'critical'}
	damage_events = {
		target: [event for event in turn['events'][target] if event['type'] == 'damage' and event['target'] == target and event.get('from') != 'brn']
		for target in ('p1', 'p2')
	}
	damage_cursors = {'p1': 0, 'p2': 0}
	current_hp = list(before['hp'])
	rows = []
	draw_cursor = 0
	for actor_side in turn['actionOrder']:
		actor = SIDE_INDEX[actor_side]
		move_event = moves_by_actor[actor]
		target_side = move_event['target']
		target = SIDE_INDEX[target_side]
		if target != 1 - actor:
			raise AssertionError(f"native move target disagrees with actor: {move_event}")
		target_side = 'p1' if target == 0 else 'p2'
		miss_event = misses_by_actor.get(actor)
		missed = miss_event is not None
		miss_target = SIDE_INDEX[miss_event['target']] if missed else -1
		if missed and miss_target != target:
			raise AssertionError(f"native miss target disagrees with move target: move={move_event}, miss={miss_event}")
		critical = (target in critical_targets) and not missed
		move_id = MOVE_IDS[choice_by_actor[actor] - 1]
		if missed:
			damage = 0
			target_hp = current_hp[target]
			draw_cursor += 1
		else:
			owner_events = damage_events[target_side]
			owner_event = owner_events[damage_cursors[target_side]]
			damage_cursors[target_side] += 1
			target_hp = _hp_from_condition(owner_event['condition'])
			actual_loss = current_hp[target] - target_hp
			if target_hp == 0 and actual_loss == current_hp[target]:
				damage_roll = turn['draws'][draw_cursor + 2]['value']
				damage = _raw_overkill_damage(
					before, actor, choice_by_actor[actor], critical, damage_roll,
					before['burned'][actor],
				)
				if damage <= current_hp[target]:
					damage = actual_loss
			else:
				damage = actual_loss
			draw_cursor += 3
		current_hp[target] = target_hp
		rows.append({
			'actor': actor, 'target': target, 'miss_target': miss_target,
			'missed': missed, 'critical': critical,
			'damage': damage, 'target_hp': target_hp,
			'target_max_hp': before['maxHp'][target], 'move_id': move_id,
		})
	assert draw_cursor == len(turn['draws'])
	return rows


def _native_expected(case):
	turn = case['turn']
	before = turn['before']
	after = turn['after']
	rows = _native_attack_rows(case)
	executed = [False, False]
	actors = [-1, -1]
	hit = [False, False]
	missed = [False, False]
	critical = [False, False]
	damage = [0, 0]
	target_hp = [0, 0]
	target_max_hp = [0, 0]
	move_ids = [-1, -1]
	for index, row in enumerate(rows):
		executed[index] = True
		actors[index] = row['actor']
		hit[index] = not row['missed']
		missed[index] = row['missed']
		critical[index] = row['critical']
		damage[index] = row['damage']
		target_hp[index] = row['target_hp']
		target_max_hp[index] = row['target_max_hp']
		move_ids[index] = row['move_id']
	events = MegaPunchEvents(
		jnp.asarray(executed, dtype=jnp.bool_), jnp.asarray(actors, dtype=jnp.int32),
		jnp.asarray(hit, dtype=jnp.bool_), jnp.asarray(missed, dtype=jnp.bool_),
		jnp.asarray(critical, dtype=jnp.bool_), jnp.asarray(damage, dtype=jnp.int32),
		jnp.asarray(target_hp, dtype=jnp.int32), jnp.asarray(target_max_hp, dtype=jnp.int32),
	)
	draw_rows = turn['draws']
	bounds = [0] * 6
	values = [-1] * 6
	contexts = [0] * 6
	seed_before = [[0] * 4 for _ in range(6)]
	seed_after = [[0] * 4 for _ in range(6)]
	for index, draw in enumerate(draw_rows):
		bounds[index] = draw['bound']
		values[index] = draw['value']
		contexts[index] = CONTEXT_IDS[draw['context']]
		seed_before[index] = draw['before']
		seed_after[index] = draw['after']
	draws = MegaPunchDrawTrace(
		jnp.asarray(len(draw_rows), dtype=jnp.int32), jnp.asarray(bounds, dtype=jnp.int32),
		jnp.asarray(values, dtype=jnp.int32), jnp.asarray(contexts, dtype=jnp.int32),
		jnp.asarray(seed_before, dtype=jnp.uint16), jnp.asarray(seed_after, dtype=jnp.uint16),
	)
	residual_rows = turn['residual']
	if residual_rows:
		row = residual_rows[0]
		actor = SIDE_INDEX[row['target']]
		owner_event = next(event for event in turn['events'][row['target']] if event['type'] == 'damage' and event.get('from') == 'brn')
		final_hp = _hp_from_condition(owner_event['condition'])
		attack_hp = list(before['hp'])
		for event_row in rows:
			attack_hp[event_row['target']] = event_row['target_hp']
		hp_loss = attack_hp[actor] - final_hp
		residual = BurnResidualEvent(
			jnp.asarray(True, dtype=jnp.bool_), jnp.asarray(actor, dtype=jnp.int32),
			jnp.asarray(hp_loss, dtype=jnp.int32), jnp.asarray(final_hp, dtype=jnp.int32),
			jnp.asarray(before['maxHp'][actor], dtype=jnp.int32),
		)
	else:
		residual = BurnResidualEvent(
			jnp.asarray(False, dtype=jnp.bool_), jnp.asarray(-1, dtype=jnp.int32),
			jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32),
			jnp.asarray(0, dtype=jnp.int32),
		)
	state = _state(after)
	return state, BurnChoiceTransition(
		state, jnp.asarray(STATUS_OK, dtype=jnp.int32), events, draws,
		jnp.asarray(move_ids, dtype=jnp.int32), residual,
	)


def _assert_native_event_order(test, case):
	turn = case['turn']
	for side in ('p1', 'p2'):
		events = turn['events'][side]
		residual_positions = [index for index, event in enumerate(events) if event['type'] == 'damage' and event.get('from') == 'brn']
		test.assertLessEqual(len(residual_positions), 1)
		if residual_positions:
			residual_index = residual_positions[0]
			for index, event in enumerate(events):
				if event['type'] in ('move', 'miss', 'critical') or (event['type'] == 'damage' and event.get('from') != 'brn'):
					test.assertLess(index, residual_index)
			for index, event in enumerate(events):
				if event['type'] in ('turn', 'winner', 'faint'):
					test.assertGreater(index, residual_index)
		for event in events:
			if event['type'] != 'damage':
				continue
			target = SIDE_INDEX[event['target']]
			hp = _hp_from_condition(event['condition'])
			if hp == 0:
				test.assertFalse(event['condition'].endswith(' brn'))
			else:
				test.assertEqual(event['condition'].endswith(' brn'), turn['before']['burned'][target])


def _empty(state, status):
	return BurnChoiceTransition(
		state, jnp.asarray(status, dtype=jnp.int32),
		MegaPunchEvents(
			jnp.zeros((2,), dtype=jnp.bool_), jnp.full((2,), -1, dtype=jnp.int32),
			jnp.zeros((2,), dtype=jnp.bool_), jnp.zeros((2,), dtype=jnp.bool_),
			jnp.zeros((2,), dtype=jnp.bool_), jnp.zeros((2,), dtype=jnp.int32),
			jnp.zeros((2,), dtype=jnp.int32), jnp.zeros((2,), dtype=jnp.int32),
		),
		MegaPunchDrawTrace(
			jnp.asarray(0, dtype=jnp.int32), jnp.zeros((6,), dtype=jnp.int32),
			jnp.full((6,), -1, dtype=jnp.int32), jnp.zeros((6,), dtype=jnp.int32),
			jnp.zeros((6, 4), dtype=jnp.uint16), jnp.zeros((6, 4), dtype=jnp.uint16),
		),
		jnp.full((2,), -1, dtype=jnp.int32),
		BurnResidualEvent(
			jnp.asarray(False, dtype=jnp.bool_), jnp.asarray(-1, dtype=jnp.int32),
			jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32),
		),
	)


def _stack_pytrees(trees):
	return jax.tree_util.tree_map(lambda *values: jnp.stack(values), *trees)


def _assert_request_and_view(test, player_view, request, native, side):
	index = SIDE_INDEX[side]
	game = player_view.attack.player_view
	legal = [not move['disabled'] for move in request['moves']]
	test.assertEqual(np.asarray(game.legal_move_mask).tolist(), legal)
	test.assertEqual(np.asarray(game.own_pp).tolist(), native['pp'][index])
	test.assertEqual(np.asarray(game.own_move_ids).tolist(), list(MOVE_IDS))
	test.assertEqual(int(game.own_hp), native['hp'][index])
	test.assertEqual(int(game.own_max_hp), native['maxHp'][index])
	for field in ('own_level', 'own_attack', 'own_defense', 'own_speed'):
		native_field = {'own_level': 'level', 'own_attack': 'attack', 'own_defense': 'defense', 'own_speed': 'speed'}[field]
		test.assertEqual(int(getattr(game, field)), native[native_field][index])
	test.assertEqual(bool(player_view.burned[index]), native['burned'][index])
	condition = request['hpCondition']
	test.assertEqual(_hp_from_condition(condition), native['hp'][index])
	test.assertEqual(condition.endswith(' brn'), native['burned'][index])


def _assert_projection(test, before_state, result, case, rows):
	turn = case['turn']
	for audience, side in enumerate(('p1', 'p2')):
		view = project_burn_choice_view(before_state, result, audience)
		base = view.attack.player_view
		request = turn['requests'][side]
		_assert_request_and_view(test, view, request, turn['before'], side)
		test.assertEqual(bool(base.terminal), turn['after']['terminal'])
		test.assertEqual(int(base.winner), turn['after']['winner'])
		expected_turn = next((event['number'] for event in turn['events'][side] if event['type'] == 'turn'), 0)
		test.assertEqual(bool(base.turn_event), expected_turn > 0)
		test.assertEqual(int(base.turn_number), expected_turn)
		test.assertEqual(np.asarray(base.action_order).tolist(), [row['actor'] for row in rows] + [-1] * (2 - len(rows)))
		test.assertEqual(np.asarray(base.target_player).tolist(), [row['target'] for row in rows] + [-1] * (2 - len(rows)))
		test.assertEqual(np.asarray(base.event_move_ids).tolist(), [row['move_id'] for row in rows] + [-1] * (2 - len(rows)))
		test.assertEqual(np.asarray(view.attack.missed).tolist(), [row['missed'] for row in rows] + [False] * (2 - len(rows)))
		for row_index, row in enumerate(rows):
			target = row['target']
			is_owner = audience == target
			test.assertEqual(bool(base.critical[row_index]), row['critical'] and not row['missed'])
			if row['missed']:
				test.assertEqual(row['miss_target'], row['target'])
				test.assertEqual(int(base.target_player[row_index]), row['miss_target'])
				test.assertEqual(int(base.damage_value[row_index]), -1)
				test.assertFalse(bool(base.damage_visible[row_index]))
				test.assertEqual(int(base.target_hp_value[row_index]), -1)
				test.assertFalse(bool(base.target_hp_is_percent[row_index]))
				test.assertFalse(bool(base.fainted[row_index]))
			else:
				actual_loss = min(turn['before']['hp'][target], row['damage'])
				test.assertEqual(bool(base.damage_visible[row_index]), is_owner)
				test.assertEqual(int(base.damage_value[row_index]), actual_loss if is_owner else -1)
				test.assertEqual(bool(base.target_hp_is_percent[row_index]), not is_owner)
				visible_hp = row['target_hp'] if is_owner else _ceil_percent(row['target_hp'], row['target_max_hp'])
				test.assertEqual(int(base.target_hp_value[row_index]), visible_hp)
				test.assertEqual(bool(base.fainted[row_index]), row['target_hp'] == 0)
		residual_events = [event for event in turn['events'][side] if event['type'] == 'damage' and event.get('from') == 'brn']
		rview = view.residual
		test.assertEqual(bool(rview.executed), bool(residual_events))
		if residual_events:
			actor = SIDE_INDEX[residual_events[0]['target']]
			owner_event = next(event for event in turn['events'][residual_events[0]['target']] if event['type'] == 'damage' and event.get('from') == 'brn')
			final_hp = _hp_from_condition(owner_event['condition'])
			attack_hp = list(turn['before']['hp'])
			for row in rows:
				attack_hp[row['target']] = row['target_hp']
			loss = attack_hp[actor] - final_hp
			owner = audience == actor
			test.assertEqual(int(rview.actor), actor)
			test.assertEqual(int(rview.target_player), actor)
			test.assertEqual(bool(rview.damage_visible), owner)
			test.assertEqual(int(rview.damage_value), loss if owner else -1)
			test.assertEqual(bool(rview.target_hp_is_percent), not owner)
			test.assertEqual(int(rview.target_hp_value), final_hp if owner else _ceil_percent(final_hp, turn['before']['maxHp'][actor]))
			test.assertEqual(bool(rview.fainted), final_hp == 0)
		test.assertNotIn('state', base._fields)
		test.assertNotIn('rng', base._fields)
		test.assertNotIn('opponent_pp', base._fields)
		test.assertNotIn('rng', rview._fields)
		next_request = turn['nextRequests'][side]
		next_state = result.state
		next_status = STATUS_TERMINAL if turn['after']['terminal'] else STATUS_PENDING
		next_view = project_burn_choice_view(next_state, _empty(next_state, next_status), audience)
		if next_request is None:
			test.assertEqual(np.asarray(next_view.attack.player_view.legal_move_mask).tolist(), [False, False])
			test.assertTrue(bool(next_view.attack.player_view.terminal))
			test.assertEqual(int(next_view.attack.player_view.winner), turn['after']['winner'])
		else:
			_assert_request_and_view(test, next_view, next_request, turn['after'], side)


class BurnChoiceTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		completed = subprocess.run(
			['node', str(ORACLE)], cwd=ROOT, check=True, capture_output=True, text=True,
		)
		cls.fixture = json.loads(completed.stdout)

	def test_native_parity_batch_privacy_and_rejections(self):
		self.assertEqual(self.fixture['upstream']['commit'], '1242c1e2d9763ba5b85d33f13aecb89bae19ae3d')
		self.assertEqual(VARIANT_ID, 7)
		self.assertEqual(VARIANT_NAME, 'gen5-single-burn-choice-mew-1v1-v7')
		self.assertEqual(simulator.BURN_CHOICE_VARIANT_ID, 7)
		self.assertEqual(simulator.ACTION_MEGA_PUNCH, 1)
		self.assertEqual(simulator.ACCURACY_CHOICE_ACTION_MEGA_PUNCH, 2)
		self.assertEqual(simulator.BURN_CHOICE_ACTION_MEGA_PUNCH, 2)
		cases = self.fixture['cases']
		self.assertEqual(len(cases), 6)
		self.assertEqual(MAX_PP_BY_SLOT, (56, 32))
		states = []
		choices = []
		expected = []
		native_rows = []
		for case in cases:
			setup = case['setup']
			self.assertEqual(setup['setupRngBefore'], setup['setupRngAfter'])
			self.assertEqual(setup['before']['rng'], setup['setupRngBefore'])
			for side in ('p1', 'p2'):
				self.assertEqual(setup['requests'][side]['hpCondition'].endswith(' brn'), side == case['burnedSetup'])
				self.assertEqual(case['turn']['requests'][side]['hpCondition'].endswith(' brn'), case['turn']['before']['burned'][SIDE_INDEX[side]])
			for side in ('p1', 'p2'):
				status_lines = [event for event in setup['statusEvents'][side] if event['type'] == 'status']
				if case['burnedSetup']:
					self.assertEqual(status_lines, [{'type': 'status', 'target': case['burnedSetup'], 'status': 'brn'}])
				else:
					self.assertEqual(status_lines, [])
			state, native_transition = _native_expected(case)
			rows = _native_attack_rows(case)
			self.assertEqual(case['turn']['after']['burned'], case['turn']['after']['statuses'].__class__(status == 'brn' for status in case['turn']['after']['statuses']))
			self.assertEqual(len(case['turn']['residual']), sum(event['type'] == 'damage' and event.get('from') == 'brn' for event in case['turn']['events']['p1']))
			self.assertEqual(len(case['turn']['draws']), len(rows) + 2 * sum(not row['missed'] for row in rows))
			if case['turn']['residual']:
				residual_row = case['turn']['residual'][0]
				actor = SIDE_INDEX[residual_row['target']]
				attack_hp = list(case['turn']['before']['hp'])
				for row in rows:
					attack_hp[row['target']] = row['target_hp']
				owner_event = next(event for event in case['turn']['events'][residual_row['target']] if event['type'] == 'damage' and event.get('from') == 'brn')
				residual_hp = _hp_from_condition(owner_event['condition'])
				self.assertEqual(attack_hp[actor] - residual_hp, min(attack_hp[actor], max(1, case['turn']['before']['maxHp'][actor] // 8)))
			_assert_native_event_order(self, case)
			self.assertEqual(len([e for e in case['turn']['events']['p1'] if e['type'] == 'winner']), int(case['turn']['after']['terminal']))
			if case['turn']['after']['terminal']:
				self.assertEqual(case['turn']['after']['winner'], 2 if case['turn']['after']['hp'][0] == 0 else 1)
				self.assertEqual(len([e for e in case['turn']['events']['p1'] if e['type'] == 'turn']), 0)
			else:
				self.assertEqual(len([e for e in case['turn']['events']['p1'] if e['type'] == 'turn']), 1)
			states.append(_state(case['turn']['before']))
			choices.append([case['choices']['p1'], case['choices']['p2']])
			expected.append(native_transition)
			native_rows.append(rows)

		# Append fail-closed semantic lanes to this one heterogeneous compiled batch.
		base = states[0]
		pending = base._replace(battle=base.battle._replace(request_phase=jnp.asarray(PHASE_PENDING, dtype=jnp.int32)))
		negative = base
		unknown = base
		dual = base._replace(burned=jnp.asarray([True, True], dtype=jnp.bool_))
		cap = base._replace(battle=base.battle._replace(pp=base.battle.pp.at[0, 0].set(57)))
		wrong_variant = base._replace(battle=base.battle._replace(variant_id=jnp.asarray(6, dtype=jnp.int32)))
		tied = base._replace(battle=base.battle._replace(speed=jnp.asarray([120, 120], dtype=jnp.int32)))
		exhausted = base._replace(battle=base.battle._replace(pp=base.battle.pp.at[0, 0].set(0)))
		all_pp_zero = base._replace(battle=base.battle._replace(pp=base.battle.pp.at[0, :].set(jnp.asarray([0, 0], dtype=jnp.int32))))
		terminal_case = next(case for case in cases if case['id'] == 'burned-direct-attack-ko-skips-residual-seed-21')
		terminal = _state(terminal_case['turn']['after'])
		lanes = [pending, negative, unknown, dual, cap, wrong_variant, tied, exhausted, all_pp_zero, terminal]
		lane_choices = [[1, 0], [1, -1], [3, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [99, -1]]
		lane_status = [STATUS_PENDING, STATUS_INVALID_CHOICE, STATUS_UNSUPPORTED, STATUS_UNSUPPORTED, STATUS_UNSUPPORTED, STATUS_UNSUPPORTED, STATUS_UNSUPPORTED, STATUS_INVALID_CHOICE, STATUS_UNSUPPORTED, STATUS_TERMINAL]
		states.extend(lanes)
		choices.extend(lane_choices)
		expected.extend(_empty(state, status) for state, status in zip(lanes, lane_status))

		batch_states = _stack_pytrees(states)
		batch_choices = jnp.asarray(choices, dtype=jnp.int32)
		batched_transition = jax.jit(jax.vmap(transition_burn_choice))
		actual_batch = batched_transition(batch_states, batch_choices)
		expected_batch = _stack_pytrees(expected)
		accuracy_helpers._assert_tree_equal(self, actual_batch, expected_batch, 'complete native and rejection batch')

		for index, case in enumerate(cases):
			_assert_projection(self, states[index], jax.tree_util.tree_map(lambda leaf: leaf[index], actual_batch), case, native_rows[index])

		# Structural boundaries fail before choice dispatch; v6/v5 cannot consume the nested v7 wrapper.
		self.assertEqual(int(transition_burn_choice(base, jnp.asarray([0, 0], dtype=jnp.int32)).status), STATUS_PENDING)
		with self.assertRaises(ValueError):
			transition_burn_choice(BurnChoiceState(base.battle, jnp.asarray([1], dtype=jnp.bool_)), [1, 1])
		with self.assertRaises(ValueError):
			make_burn_choice_state(
				hp=[10, 10], max_hp=[10, 10], pp=[[1, 1], [1, 1]],
				level=[50, 50], attack=[100, 100], defense=[100, 100], speed=[120, 100],
				rng=[1, 2, 3, 4], burned=[True],
			)
		with self.assertRaises(TypeError):
			transition_burn_choice(BurnChoiceState(base.battle, jnp.asarray([1, 0], dtype=jnp.int32)), [1, 1])
		with self.assertRaises(TypeError):
			transition_accuracy_choice(base, [1, 1])
		with self.assertRaises(TypeError):
			transition_mega_punch(base, [1, 1])
		old_v6 = transition_accuracy_choice(base.battle, jnp.asarray([1, 1], dtype=jnp.int32))
		self.assertEqual(int(old_v6.status), STATUS_UNSUPPORTED)
		accuracy_helpers._assert_tree_equal(self, old_v6.state, base.battle, 'v6 rejects inner v7 state unchanged')
		wrong_v5 = _adapt_state(base.battle)
		old_v5 = transition_mega_punch(wrong_v5, jnp.asarray([1, 1], dtype=jnp.int32))
		self.assertEqual(int(old_v5.status), STATUS_UNSUPPORTED)
		accuracy_helpers._assert_tree_equal(self, old_v5.state, wrong_v5, 'v5 rejects wrong variant unchanged')
		with self.assertRaises(TypeError):
			project_burn_choice_view(base, expected[0], 0.0)
		with self.assertRaises(ValueError):
			project_burn_choice_view(base, expected[0], 2)
		with self.assertRaises(TypeError):
			jax.jit(project_burn_choice_view)(base, expected[0], jnp.asarray(0, dtype=jnp.int32))
		with self.assertRaises(TypeError):
			transition_burn_choice(base, [1.0, 1.0])
		with self.assertRaises(ValueError):
			transition_burn_choice(base, [1])
		wrong_view = project_burn_choice_view(wrong_variant, _empty(wrong_variant, STATUS_UNSUPPORTED), 0)
		self.assertEqual(np.asarray(wrong_view.attack.player_view.legal_move_mask).tolist(), [False, False])
		self.assertEqual(np.asarray(actual_batch.state.burned).shape, (len(states), 2))

		int_vector = Sequence[int] | np.ndarray | jax.Array
		bool_vector = Sequence[bool] | np.ndarray | jax.Array
		make_hints = get_type_hints(make_burn_choice_state)
		self.assertIs(make_hints['return'], BurnChoiceState)
		self.assertEqual(make_hints['hp'], int_vector)
		self.assertEqual(make_hints['burned'], bool_vector)
		transition_hints = get_type_hints(transition_burn_choice)
		self.assertIs(transition_hints['state'], BurnChoiceState)
		self.assertEqual(transition_hints['choices'], int_vector)
		self.assertIs(transition_hints['return'], BurnChoiceTransition)
		projection_hints = get_type_hints(project_burn_choice_view)
		self.assertIs(projection_hints['before'], BurnChoiceState)
		self.assertIs(projection_hints['result'], BurnChoiceTransition)
		self.assertIs(projection_hints['audience'], int)
		self.assertIs(projection_hints['return'], BurnChoicePlayerView)
		namedtuple_hints = {
			BurnChoiceState: {'battle': TwoMoveState, 'burned': jax.Array},
			BurnChoiceTransition: {
				'state': BurnChoiceState, 'status': jax.Array, 'events': MegaPunchEvents,
				'draws': MegaPunchDrawTrace, 'event_move_ids': jax.Array, 'residual': BurnResidualEvent,
			},
			BurnResidualEvent: {
				'executed': jax.Array, 'actor': jax.Array, 'hp_loss': jax.Array,
				'target_hp': jax.Array, 'target_max_hp': jax.Array,
			},
			BurnChoicePlayerView: {
				'attack': AccuracyChoicePlayerView, 'burned': jax.Array, 'residual': BurnResidualPlayerView,
			},
			BurnResidualPlayerView: {
				'executed': jax.Array, 'actor': jax.Array, 'target_player': jax.Array,
				'damage_value': jax.Array, 'damage_visible': jax.Array,
				'target_hp_value': jax.Array, 'target_hp_is_percent': jax.Array, 'fainted': jax.Array,
			},
		}
		for record_type, expected_hints in namedtuple_hints.items():
			self.assertEqual(get_type_hints(record_type), expected_hints)


if __name__ == '__main__':
	unittest.main()
