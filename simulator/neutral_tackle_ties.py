"""Seeded speed-tie extension of the restricted neutral Tackle simulator.

This is a separate v2 variant because each tied native turn can consume up to
ten additional phase-order draws. The accepted v1 transition and its six-slot
draw PyTree remain unchanged.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .gen5_rng import sample_bounded
from .neutral_tackle import (
	ACTION_TACKLE,
	CHOICE_PENDING,
	DRAW_ACCURACY,
	DRAW_CRITICAL,
	DRAW_DAMAGE,
	PHASE_MOVE_REQUEST,
	PHASE_PENDING,
	PHASE_TERMINAL,
	PHASE_UNSUPPORTED,
	STATUS_INVALID_CHOICE,
	STATUS_OK,
	STATUS_PENDING,
	STATUS_TERMINAL,
	STATUS_UNSUPPORTED,
	SPEED_TIE_VARIANT_ID,
	BattleState,
	PlayerView,
	Transition,
	TurnEvents,
	_TackleActionResult,
	_dispatch_transition,
	_empty_events,
	_execute_tackle_action,
	_project_view,
	_record_action_event,
	_supported_tie_state,
	_choice_array,
)


VARIANT_NAME = "gen5-neutral-tackle-speed-ties-mew-1v1-v2"
MAX_TIE_DRAWS_PER_TURN = 16

DRAW_QUEUE_SORT = 4
DRAW_BEFORE_TURN = 5
DRAW_MOVE_HIT_UPDATE = 6
DRAW_ACTION_UPDATE = 7
DRAW_RESIDUAL_UPDATE = 8


class TieDrawTrace(NamedTuple):
	"""Native-order draw prefix; tie shuffles store effective bound two."""

	count: jax.Array
	bounds: jax.Array
	values: jax.Array
	contexts: jax.Array
	before: jax.Array
	after: jax.Array


class TieTransition(NamedTuple):
	state: BattleState
	status: jax.Array
	events: TurnEvents
	draws: TieDrawTrace


class TieDuelTrace(NamedTuple):
	"""Fixed successful-prefix trace with trusted views for both recipients."""

	valid: jax.Array
	before: BattleState
	transition: TieTransition
	p1_view: PlayerView
	p2_view: PlayerView


class TieDuelResult(NamedTuple):
	state: BattleState
	disposition: jax.Array
	turn_count: jax.Array
	diagnostic_status: jax.Array
	trace: TieDuelTrace


def make_tie_state(
	*,
	hp,
	max_hp,
	pp,
	level,
	attack,
	defense,
	speed,
	rng,
	turn=1,
	request_phase=PHASE_MOVE_REQUEST,
	terminal=False,
	winner=-1,
	species_id=(151, 151),
	move_id=33,
) -> BattleState:
	"""Build variant v2 at the pinned normal-request boundary."""
	from .neutral_tackle import make_state

	return make_state(
		hp=hp,
		max_hp=max_hp,
		pp=pp,
		level=level,
		attack=attack,
		defense=defense,
		speed=speed,
		rng=rng,
		turn=turn,
		request_phase=request_phase,
		terminal=terminal,
		winner=winner,
		variant_id=SPEED_TIE_VARIANT_ID,
		species_id=species_id,
		move_id=move_id,
	)


def _empty_tie_draws() -> TieDrawTrace:
	return TieDrawTrace(
		jnp.asarray(0, dtype=jnp.int32),
		jnp.full((MAX_TIE_DRAWS_PER_TURN,), -1, dtype=jnp.int32),
		jnp.zeros((MAX_TIE_DRAWS_PER_TURN,), dtype=jnp.int32),
		jnp.full((MAX_TIE_DRAWS_PER_TURN,), -1, dtype=jnp.int32),
		jnp.zeros((MAX_TIE_DRAWS_PER_TURN, 4), dtype=jnp.uint16),
		jnp.zeros((MAX_TIE_DRAWS_PER_TURN, 4), dtype=jnp.uint16),
	)


def _record_tie_draw(draws, bound, value, context, before, after):
	index = draws.count
	return TieDrawTrace(
		draws.count + jnp.asarray(1, dtype=jnp.int32),
		draws.bounds.at[index].set(jnp.asarray(bound, dtype=jnp.int32)),
		draws.values.at[index].set(jnp.asarray(value, dtype=jnp.int32)),
		draws.contexts.at[index].set(jnp.asarray(context, dtype=jnp.int32)),
		draws.before.at[index].set(before),
		draws.after.at[index].set(after),
	)


def _speed_sort_draw(state, draws, context):
	"""Match a tied native ``random(0, 2)``; unequal speeds consume nothing."""

	def draw(_):
		before = state.rng
		rng, value = sample_bounded(before, 2)
		value = value.astype(jnp.int32)
		updated = state._replace(rng=rng)
		trace = _record_tie_draw(draws, 2, value, context, before, rng)
		return updated, trace, value

	def skip(_):
		return state, draws, jnp.asarray(0, dtype=jnp.int32)

	return jax.lax.cond(state.speed[0] == state.speed[1], draw, skip, operand=None)


def _execute_action(state, events, draws, actor, event_slot):
	target = 1 - actor
	can_act = (state.hp[actor] > 0) & (state.hp[target] > 0)

	def execute(_):
		action = _execute_tackle_action(state, actor)
		new_events = _record_action_event(events, event_slot, actor, action)
		new_draws = draws
		for offset, bound, context in (
			(0, 100, DRAW_ACCURACY),
			(1, 16, DRAW_CRITICAL),
			(2, 16, DRAW_DAMAGE),
		):
			new_draws = _record_tie_draw(
				new_draws,
				bound,
				action.values[offset],
				context,
				action.before[offset],
				action.after[offset],
			)
		return action.state, new_events, new_draws

	return jax.lax.cond(can_act, execute, lambda _: (state, events, draws), operand=None)


def _finish_turn(state, events, draws):
	terminal = jnp.any(state.hp == 0)
	winner = jnp.where(
		(state.hp[0] == 0) & (state.hp[1] == 0),
		0,
		jnp.where(state.hp[0] == 0, 2, jnp.where(state.hp[1] == 0, 1, -1)),
	).astype(jnp.int32)
	phase = jnp.where(
		terminal,
		PHASE_TERMINAL,
		jnp.where(jnp.any(state.pp == 0), PHASE_UNSUPPORTED, PHASE_MOVE_REQUEST),
	).astype(jnp.int32)
	state = state._replace(
		turn=state.turn + jnp.where(terminal, 0, 1).astype(jnp.int32),
		request_phase=phase,
		terminal=terminal,
		winner=winner,
	)
	return TieTransition(state, jnp.asarray(STATUS_OK, dtype=jnp.int32), events, draws)


def _run_tie_turn(state):
	events = _empty_events()
	draws = _empty_tie_draws()
	state, draws, queue_swap = _speed_sort_draw(state, draws, DRAW_QUEUE_SORT)
	state, draws, _ = _speed_sort_draw(state, draws, DRAW_BEFORE_TURN)
	state, draws, _ = _speed_sort_draw(state, draws, DRAW_ACTION_UPDATE)
	first_actor = jnp.where(
		state.speed[0] == state.speed[1],
		jnp.where(queue_swap == 0, 0, 1),
		jnp.where(state.speed[0] > state.speed[1], 0, 1),
	).astype(jnp.int32)
	state, events, draws = _execute_action(state, events, draws, first_actor, 0)
	first_action_ended = jnp.any(state.hp == 0)
	first_carry = (state, events, draws, first_actor)

	def ended_after_first(carry):
		state, events, draws, _ = carry
		# Native hitStepMoveHitLoop reaches one Update before faint resolution ends it.
		state, draws, _ = _speed_sort_draw(state, draws, DRAW_MOVE_HIT_UPDATE)
		return _finish_turn(state, events, draws)

	def continue_after_first(carry):
		state, events, draws, first_actor = carry
		state, draws, _ = _speed_sort_draw(state, draws, DRAW_MOVE_HIT_UPDATE)
		state, draws, _ = _speed_sort_draw(state, draws, DRAW_MOVE_HIT_UPDATE)
		state, draws, _ = _speed_sort_draw(state, draws, DRAW_ACTION_UPDATE)
		second_actor = 1 - first_actor
		state, events, draws = _execute_action(state, events, draws, second_actor, 1)
		second_action_ended = jnp.any(state.hp == 0)
		second_carry = (state, events, draws)

		def ended_after_second(second_state):
			state, events, draws = second_state
			state, draws, _ = _speed_sort_draw(state, draws, DRAW_MOVE_HIT_UPDATE)
			return _finish_turn(state, events, draws)

		def continue_after_second(second_state):
			state, events, draws = second_state
			state, draws, _ = _speed_sort_draw(state, draws, DRAW_MOVE_HIT_UPDATE)
			state, draws, _ = _speed_sort_draw(state, draws, DRAW_MOVE_HIT_UPDATE)
			state, draws, _ = _speed_sort_draw(state, draws, DRAW_ACTION_UPDATE)
			state, draws, _ = _speed_sort_draw(state, draws, DRAW_RESIDUAL_UPDATE)
			return _finish_turn(state, events, draws)

		return jax.lax.cond(second_action_ended, ended_after_second, continue_after_second, second_carry)

	return jax.lax.cond(first_action_ended, ended_after_first, continue_after_first, first_carry)


def _empty_tie_transition(state, status):
	return TieTransition(state, jnp.asarray(status, dtype=jnp.int32), _empty_events(), _empty_tie_draws())


def transition_with_speed_ties(state: BattleState, choices) -> TieTransition:
	"""Resolve simultaneous Tackle choices with v2 initiative ties and traces."""
	return _dispatch_transition(
		state,
		_choice_array(choices),
		_supported_tie_state,
		_run_tie_turn,
		_empty_tie_transition,
	)


def project_tie_view(before: BattleState, result: TieTransition, audience) -> PlayerView:
	"""Project v2 output through the same visibility rules as the v1 API."""
	return _project_view(before, result, audience, _supported_tie_state)


def _padding_tie_row(state):
	zero_state = jax.tree_util.tree_map(jnp.zeros_like, state)
	zero_events = _empty_events()
	zero_result = TieTransition(
		zero_state,
		jnp.asarray(STATUS_TERMINAL, dtype=jnp.int32),
		zero_events,
		_empty_tie_draws(),
	)
	zero_p1 = jax.tree_util.tree_map(jnp.zeros_like, project_tie_view(zero_state, zero_result, 0))
	zero_p2 = zero_p1._replace(audience=jnp.asarray(1, dtype=jnp.int32))
	return TieDuelTrace(jnp.asarray(False), zero_state, zero_result, zero_p1, zero_p2)


def run_tie_duel(initial_state: BattleState, *, max_turns: int, turn_budget=None) -> TieDuelResult:
	"""Run a bounded v2 duel using the shared static-capacity executor."""
	from .neutral_duel import _run_duel_core

	return _run_duel_core(
		initial_state,
		max_turns=max_turns,
		turn_budget=turn_budget,
		transition_fn=transition_with_speed_ties,
		view_fn=project_tie_view,
		padding_fn=_padding_tie_row,
		trace_type=TieDuelTrace,
		result_type=TieDuelResult,
	)
