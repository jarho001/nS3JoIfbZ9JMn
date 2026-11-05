"""Fixed-state Gen 5 Mew turn API for simultaneous Tackle/Pound choices."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .neutral_tackle import (
	ACTION_TACKLE,
	PHASE_MOVE_REQUEST,
	PHASE_TERMINAL,
	PHASE_UNSUPPORTED,
	SPECIES_MEW,
	STATUS_INVALID_CHOICE,
	STATUS_OK,
	TACKLE_BASE_POWER,
	BattleState,
	TurnEvents,
	_execute_neutral_action,
	_int_scalar,
	_int_vector,
	_dispatch_transition,
	_project_view,
	_supported_common,
	_choice_array,
)
from .neutral_tackle_ties import (
	MAX_TIE_DRAWS_PER_TURN,
	TieDrawTrace,
	TieTransition,
	_empty_tie_transition,
	_run_tie_turn,
)


VARIANT_ID = 4
VARIANT_NAME = "gen5-neutral-two-move-mew-1v1-v4"
MOVE_TACKLE = 33
MOVE_POUND = 1
MOVE_IDS = (MOVE_TACKLE, MOVE_POUND)
ACTION_POUND = 2
POUND_BASE_POWER = 40
MAX_MOVE_PP = 56
INTERNAL_MOVE_ID = 0
INVALID_MOVE_ID = -1


class TwoMoveState(NamedTuple):
	"""Trusted v4 state; PP and move IDs are ordered by player and slot."""

	hp: jax.Array
	max_hp: jax.Array
	pp: jax.Array
	level: jax.Array
	attack: jax.Array
	defense: jax.Array
	speed: jax.Array
	rng: jax.Array
	turn: jax.Array
	request_phase: jax.Array
	terminal: jax.Array
	winner: jax.Array
	variant_id: jax.Array
	species_id: jax.Array
	move_ids: jax.Array


class TwoMoveTransition(NamedTuple):
	state: TwoMoveState
	status: jax.Array
	events: TurnEvents
	draws: TieDrawTrace
	event_move_ids: jax.Array


class TwoMovePlayerView(NamedTuple):
	"""One recipient's request, own PP/moves, and observed executed events."""

	audience: jax.Array
	request_phase: jax.Array
	legal_move_mask: jax.Array
	own_hp: jax.Array
	own_max_hp: jax.Array
	own_pp: jax.Array
	own_move_ids: jax.Array
	own_level: jax.Array
	own_attack: jax.Array
	own_defense: jax.Array
	own_speed: jax.Array
	action_order: jax.Array
	event_move_ids: jax.Array
	critical: jax.Array
	damage_value: jax.Array
	damage_visible: jax.Array
	target_player: jax.Array
	target_hp_value: jax.Array
	target_hp_is_percent: jax.Array
	fainted: jax.Array
	turn_event: jax.Array
	turn_number: jax.Array
	terminal: jax.Array
	winner: jax.Array


def _int_matrix(name: str, value, shape: tuple[int, int]) -> jax.Array:
	array = np.asarray(value)
	if array.shape != shape or array.dtype.kind not in "iu":
		raise ValueError(f"{name} must be an integer matrix with shape {shape}.")
	if np.any(array < np.iinfo(np.int32).min) or np.any(array > np.iinfo(np.int32).max):
		raise ValueError(f"{name} values must fit signed int32.")
	return jnp.asarray(array, dtype=jnp.int32)


_STATE_LAYOUT = {
	"hp": ((2,), jnp.dtype(jnp.int32)),
	"max_hp": ((2,), jnp.dtype(jnp.int32)),
	"pp": ((2, 2), jnp.dtype(jnp.int32)),
	"level": ((2,), jnp.dtype(jnp.int32)),
	"attack": ((2,), jnp.dtype(jnp.int32)),
	"defense": ((2,), jnp.dtype(jnp.int32)),
	"speed": ((2,), jnp.dtype(jnp.int32)),
	"rng": ((4,), jnp.dtype(jnp.uint16)),
	"turn": ((), jnp.dtype(jnp.int32)),
	"request_phase": ((), jnp.dtype(jnp.int32)),
	"terminal": ((), jnp.dtype(jnp.bool_)),
	"winner": ((), jnp.dtype(jnp.int32)),
	"variant_id": ((), jnp.dtype(jnp.int32)),
	"species_id": ((2,), jnp.dtype(jnp.int32)),
	"move_ids": ((2,), jnp.dtype(jnp.int32)),
}


def _validate_two_move_state_structure(state: TwoMoveState) -> None:
	if not isinstance(state, TwoMoveState):
		raise TypeError("state must be a simulator.neutral_two_move.TwoMoveState.")
	for name, (shape, dtype) in _STATE_LAYOUT.items():
		value = getattr(state, name)
		if not hasattr(value, "shape") or tuple(value.shape) != shape:
			raise ValueError(f"state.{name} must have shape {shape}.")
		if jnp.dtype(value.dtype) != dtype:
			raise TypeError(f"state.{name} must have dtype {dtype}.")


def make_two_move_state(
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
	variant_id=VARIANT_ID,
	species_id=(SPECIES_MEW, SPECIES_MEW),
	move_ids=MOVE_IDS,
) -> TwoMoveState:
	"""Build the independent v4 PyTree from initialized Showdown values."""
	seed = np.asarray(rng)
	if seed.shape != (4,) or seed.dtype.kind not in "iu" or np.any(seed < 0) or np.any(seed > 0xFFFF):
		raise ValueError("rng must contain four unsigned 16-bit Gen 5 limbs.")
	if not isinstance(terminal, (bool, np.bool_)):
		raise ValueError("terminal must be boolean.")
	state = TwoMoveState(
		_int_vector("hp", hp, 2),
		_int_vector("max_hp", max_hp, 2),
		_int_matrix("pp", pp, (2, 2)),
		_int_vector("level", level, 2),
		_int_vector("attack", attack, 2),
		_int_vector("defense", defense, 2),
		_int_vector("speed", speed, 2),
		jnp.asarray(seed, dtype=jnp.uint16),
		_int_scalar("turn", turn),
		_int_scalar("request_phase", request_phase),
		jnp.asarray(terminal, dtype=jnp.bool_),
		_int_scalar("winner", winner),
		_int_scalar("variant_id", variant_id),
		_int_vector("species_id", species_id, 2),
		_int_vector("move_ids", move_ids, 2),
	)
	_validate_two_move_state_structure(state)
	return state


def _adapt_state(state: TwoMoveState) -> BattleState:
	"""Supply a transient internal neutral boundary for the shared turn engine."""
	return BattleState(
		state.hp,
		state.max_hp,
		jnp.ones((2,), dtype=jnp.int32),
		state.level,
		state.attack,
		state.defense,
		state.speed,
		state.rng,
		state.turn,
		state.request_phase,
		state.terminal,
		state.winner,
		state.variant_id,
		state.species_id,
		jnp.asarray(INTERNAL_MOVE_ID, dtype=jnp.int32),
	)


def _supported_two_move_state(state: TwoMoveState, internal: BattleState) -> jax.Array:
	return (
		_supported_common(internal, move_id=INTERNAL_MOVE_ID, max_pp=MAX_MOVE_PP)
		& (state.variant_id == VARIANT_ID)
		& jnp.all(state.species_id == SPECIES_MEW)
		& jnp.all(state.move_ids == jnp.asarray(MOVE_IDS, dtype=jnp.int32))
		& jnp.all(state.pp >= 0)
		& jnp.all(state.pp <= MAX_MOVE_PP)
		& jnp.all(jnp.any(state.pp > 0, axis=1))
	)


def _execute_selected_action(state: BattleState, actor, choices):
	choice = choices[actor]
	base_power = jnp.where(choice == ACTION_TACKLE, TACKLE_BASE_POWER, POUND_BASE_POWER)
	return _execute_neutral_action(state, actor, base_power)


def _public_transition(state: TwoMoveState, internal: TieTransition, choices) -> TwoMoveTransition:
	pp = state.pp
	choice_slots = jnp.clip(choices, ACTION_TACKLE, ACTION_POUND) - 1
	for row in range(2):
		actor = jnp.clip(internal.events.actors[row], 0, 1)
		choice_slot = choice_slots[actor]
		decrement = internal.events.executed[row].astype(jnp.int32)
		pp = pp.at[actor, choice_slot].add(-decrement)
	terminal = internal.state.terminal
	request_phase = jnp.where(
		terminal,
		PHASE_TERMINAL,
		jnp.where(jnp.any(jnp.all(pp == 0, axis=1)), PHASE_UNSUPPORTED, PHASE_MOVE_REQUEST),
	).astype(jnp.int32)
	public_state = state._replace(
		hp=internal.state.hp,
		pp=pp,
		rng=internal.state.rng,
		turn=internal.state.turn,
		request_phase=request_phase,
		terminal=terminal,
		winner=internal.state.winner,
	)
	actors = jnp.clip(internal.events.actors, 0, 1)
	move_ids = state.move_ids[choice_slots[actors]]
	event_move_ids = jnp.where(internal.events.executed, move_ids, INVALID_MOVE_ID).astype(jnp.int32)
	return TwoMoveTransition(public_state, internal.status, internal.events, internal.draws, event_move_ids)


def transition_two_move(state: TwoMoveState, choices) -> TwoMoveTransition:
	"""Resolve simultaneous Tackle/Pound choices through the shared v2 engine.

	Choices are ordered p1, p2: 0 is pending, 1 selects Tackle, 2 selects
	Pound, negatives are invalid, and values above 2 are unsupported. Selecting
	a known slot with zero PP is invalid. No rejected or pending result advances
	state or RNG.
	"""
	_validate_two_move_state_structure(state)
	choice_array = _choice_array(choices)
	internal = _adapt_state(state)

	def supported(candidate):
		return _supported_two_move_state(state, candidate)

	def run_turn(candidate):
		selected_pp = state.pp[jnp.arange(2), jnp.clip(choice_array, 1, 2) - 1]
		selected_available = jnp.all(selected_pp > 0)

		def execute(_):
			return _run_tie_turn(candidate, lambda current, actor: _execute_selected_action(current, actor, choice_array))

		return jax.lax.cond(
			selected_available,
			execute,
			lambda _: _empty_tie_transition(candidate, STATUS_INVALID_CHOICE),
			operand=None,
		)

	internal_result = _dispatch_transition(
		internal,
		choice_array,
		supported,
		run_turn,
		_empty_tie_transition,
		sole_action_id=ACTION_POUND,
	)
	return jax.lax.cond(
		internal_result.status == STATUS_OK,
		lambda result: _public_transition(state, result, choice_array),
		lambda result: TwoMoveTransition(
			state,
			result.status,
			result.events,
			result.draws,
			jnp.full((2,), INVALID_MOVE_ID, dtype=jnp.int32),
		),
		internal_result,
	)


def project_two_move_view(before: TwoMoveState, result: TwoMoveTransition, audience) -> TwoMovePlayerView:
	"""Project one v4 result to p1 (0) or p2 (1), without trusted internals."""
	_validate_two_move_state_structure(before)
	_validate_two_move_state_structure(result.state)
	if not isinstance(result, TwoMoveTransition):
		raise TypeError("result must be a simulator.neutral_two_move.TwoMoveTransition.")
	before_internal = _adapt_state(before)
	after_internal = _adapt_state(result.state)
	internal_result = TieTransition(after_internal, result.status, result.events, result.draws)
	common = _project_view(
		before_internal,
		internal_result,
		audience,
		lambda candidate: _supported_two_move_state(before, candidate),
	)
	viewer = common.audience
	own_pp = before.pp[viewer]
	legal = (
		_supported_two_move_state(before, before_internal)
		& (before.hp[viewer] > 0)
		& (before.request_phase == PHASE_MOVE_REQUEST)
		& (own_pp > 0)
	)
	return TwoMovePlayerView(
		common.audience,
		common.request_phase,
		legal.astype(jnp.bool_),
		common.own_hp,
		common.own_max_hp,
		own_pp,
		before.move_ids,
		common.own_level,
		common.own_attack,
		common.own_defense,
		common.own_speed,
		common.action_order,
		jnp.where(result.events.executed, result.event_move_ids, INVALID_MOVE_ID).astype(jnp.int32),
		common.critical,
		common.damage_value,
		common.damage_visible,
		common.target_player,
		common.target_hp_value,
		common.target_hp_is_percent,
		common.fainted,
		common.turn_event,
		common.turn_number,
		common.terminal,
		common.winner,
	)


__all__ = [
	"ACTION_POUND",
	"ACTION_TACKLE",
	"INTERNAL_MOVE_ID",
	"INVALID_MOVE_ID",
	"MAX_MOVE_PP",
	"MAX_TIE_DRAWS_PER_TURN",
	"MOVE_IDS",
	"MOVE_POUND",
	"MOVE_TACKLE",
	"POUND_BASE_POWER",
	"TieDrawTrace",
	"TwoMovePlayerView",
	"TwoMoveState",
	"TwoMoveTransition",
	"VARIANT_ID",
	"VARIANT_NAME",
	"make_two_move_state",
	"project_two_move_view",
	"transition_two_move",
]
