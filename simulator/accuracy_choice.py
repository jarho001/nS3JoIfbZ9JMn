"""Restricted v6 simultaneous Tackle/Mega Punch accuracy-choice turn."""

from __future__ import annotations

from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .neutral_mega_punch import (
	MEGA_PUNCH_ACCURACY,
	MEGA_PUNCH_BASE_POWER,
	MegaPunchDrawTrace,
	MegaPunchEvents,
	MegaPunchTransition,
	_mask_missed_projection,
	_empty_draws,
	_empty_events,
	_execute_action,
	_run_turn,
)
from .neutral_tackle import (
	CHOICE_PENDING,
	PHASE_MOVE_REQUEST,
	PHASE_PENDING,
	PHASE_TERMINAL,
	PHASE_UNSUPPORTED,
	SPECIES_MEW,
	STATUS_INVALID_CHOICE,
	STATUS_OK,
	STATUS_PENDING,
	STATUS_TERMINAL,
	STATUS_UNSUPPORTED,
	BattleState,
	PlayerView,
	_supported_common,
	_project_view,
)
from .neutral_two_move import (
	INVALID_MOVE_ID,
	MAX_MOVE_PP,
	MOVE_TACKLE,
	TwoMovePlayerView,
	TwoMoveState,
	TwoMoveTransition,
	_adapt_state,
	_choice_array,
	_int_scalar,
	_int_vector,
	_validate_two_move_state_structure,
	make_two_move_state,
	project_two_move_view,
)


VARIANT_ID = 6
VARIANT_NAME = "gen5-accuracy-choice-mew-1v1-v6"
MOVE_MEGA_PUNCH = 5
MOVE_IDS = (MOVE_TACKLE, MOVE_MEGA_PUNCH)
ACTION_TACKLE = 1
ACTION_MEGA_PUNCH = 2
TACKLE_BASE_POWER = 50
TACKLE_ACCURACY = 100
TACKLE_MAX_PP = 56
MEGA_PUNCH_MAX_PP = 32
MAX_PP_BY_SLOT = (TACKLE_MAX_PP, MEGA_PUNCH_MAX_PP)

_IntVectorInput = Sequence[int] | np.ndarray | jax.Array
_IntScalarInput = int | np.ndarray | jax.Array


class AccuracyChoiceTransition(NamedTuple):
	state: TwoMoveState
	status: jax.Array
	events: MegaPunchEvents
	draws: MegaPunchDrawTrace
	event_move_ids: jax.Array


class AccuracyChoicePlayerView(NamedTuple):
	player_view: TwoMovePlayerView
	missed: jax.Array


def make_accuracy_choice_state(
	*,
	hp: _IntVectorInput,
	max_hp: _IntVectorInput,
	pp: _IntVectorInput,
	level: _IntVectorInput,
	attack: _IntVectorInput,
	defense: _IntVectorInput,
	speed: _IntVectorInput,
	rng: _IntVectorInput,
	turn: _IntScalarInput = 1,
	request_phase: _IntScalarInput = PHASE_MOVE_REQUEST,
	terminal: bool | np.bool_ | np.ndarray | jax.Array = False,
	winner: _IntScalarInput = -1,
	species_id: _IntVectorInput = (SPECIES_MEW, SPECIES_MEW),
) -> TwoMoveState:
	"""Build the v6 state using the shared fixed two-slot layout."""
	state = make_two_move_state(
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
		variant_id=VARIANT_ID,
		species_id=species_id,
		move_ids=MOVE_IDS,
	)
	_validate_two_move_state_structure(state)
	return state


def _supported_accuracy_choice_state(state: TwoMoveState, internal: BattleState) -> jax.Array:
	return (
		_supported_common(internal, move_id=0, max_pp=MAX_MOVE_PP)
		& (state.variant_id == VARIANT_ID)
		& jnp.all(state.species_id == SPECIES_MEW)
		& jnp.all(state.move_ids == jnp.asarray(MOVE_IDS, dtype=jnp.int32))
		& jnp.all(state.pp >= 0)
		& jnp.all(state.pp <= jnp.asarray(MAX_PP_BY_SLOT, dtype=jnp.int32)[None, :])
		& jnp.all(jnp.any(state.pp > 0, axis=1))
		& (state.speed[0] != state.speed[1])
	)


def _empty_result(state: TwoMoveState, status: int) -> AccuracyChoiceTransition:
	return AccuracyChoiceTransition(
		state,
		jnp.asarray(status, dtype=jnp.int32),
		_empty_events(),
		_empty_draws(),
		jnp.full((2,), INVALID_MOVE_ID, dtype=jnp.int32),
	)


def _run_accuracy_choice_turn(
	state: TwoMoveState,
	choices: jax.Array,
	burned: jax.Array | None = None,
) -> AccuracyChoiceTransition:
	internal = _adapt_state(state)

	def execute(current, actor, slot, events, draws):
		action = choices[actor]
		base_power = jnp.where(action == ACTION_TACKLE, TACKLE_BASE_POWER, MEGA_PUNCH_BASE_POWER)
		accuracy = jnp.where(action == ACTION_TACKLE, TACKLE_ACCURACY, MEGA_PUNCH_ACCURACY)
		actor_burned = jnp.asarray(False, dtype=jnp.bool_) if burned is None else burned[actor]
		return _execute_action(
			current,
			actor,
			slot,
			events,
			draws,
			base_power=base_power,
			accuracy=accuracy,
			burned=actor_burned,
		)

	internal_result = _run_turn(internal, action_fn=execute)
	pp = state.pp
	choice_slots = jnp.clip(choices, ACTION_TACKLE, ACTION_MEGA_PUNCH) - 1
	for row in range(2):
		actor = jnp.clip(internal_result.events.actors[row], 0, 1)
		choice_slot = choice_slots[actor]
		decrement = internal_result.events.executed[row].astype(jnp.int32)
		pp = pp.at[actor, choice_slot].add(-decrement)
	terminal = internal_result.state.terminal
	request_phase = jnp.where(
		terminal,
		PHASE_TERMINAL,
		jnp.where(jnp.any(jnp.all(pp == 0, axis=1)), PHASE_UNSUPPORTED, PHASE_MOVE_REQUEST),
	).astype(jnp.int32)
	public_state = state._replace(
		hp=internal_result.state.hp,
		pp=pp,
		rng=internal_result.state.rng,
		turn=internal_result.state.turn,
		request_phase=request_phase,
		terminal=terminal,
		winner=internal_result.state.winner,
	)
	actors = jnp.clip(internal_result.events.actors, 0, 1)
	move_slots = choice_slots[actors]
	event_move_ids = jnp.where(
		internal_result.events.executed,
		state.move_ids[move_slots],
		INVALID_MOVE_ID,
	).astype(jnp.int32)
	return AccuracyChoiceTransition(
		public_state,
		internal_result.status,
		internal_result.events,
		internal_result.draws,
		event_move_ids,
	)


def transition_accuracy_choice(
	state: TwoMoveState,
	choices: _IntVectorInput,
) -> AccuracyChoiceTransition:
	"""Resolve simultaneous choices: 1=Tackle, 2=Mega Punch, 0=pending."""
	_validate_two_move_state_structure(state)
	choice_array = _choice_array(choices)
	internal = _adapt_state(state)

	def terminal_result(_):
		return _empty_result(state, STATUS_TERMINAL)

	def active_result(_):
		negative = jnp.any(choice_array < 0)
		unknown = jnp.any(choice_array > ACTION_MEGA_PUNCH)
		pending = jnp.any(choice_array == CHOICE_PENDING) | (state.request_phase == PHASE_PENDING)

		def active_after_negative(__):
			return jax.lax.cond(
				unknown,
				lambda ___: _empty_result(state, STATUS_UNSUPPORTED),
				lambda ___: jax.lax.cond(
					pending,
					lambda ____: _empty_result(state, STATUS_PENDING),
					lambda ____: execute_or_reject(None),
					None,
				),
				None,
			)

		def execute_or_reject(___):
			supported = _supported_accuracy_choice_state(state, internal)

			def selected_pp_or_execute(____):
				selected_pp = state.pp[jnp.arange(2), jnp.clip(choice_array, ACTION_TACKLE, ACTION_MEGA_PUNCH) - 1]
				return jax.lax.cond(
					jnp.all(selected_pp > 0),
					lambda _____: _run_accuracy_choice_turn(state, choice_array),
					lambda _____: _empty_result(state, STATUS_INVALID_CHOICE),
					None,
				)

			return jax.lax.cond(
				supported,
				selected_pp_or_execute,
				lambda ____: _empty_result(state, STATUS_UNSUPPORTED),
				None,
			)

		return jax.lax.cond(
			negative,
			lambda __: _empty_result(state, STATUS_INVALID_CHOICE),
			active_after_negative,
			None,
		)

	return jax.lax.cond(state.terminal, terminal_result, active_result, None)


def _supported_for_view(state: TwoMoveState) -> jax.Array:
	return _supported_accuracy_choice_state(state, _adapt_state(state))


def project_accuracy_choice_view(
	before: TwoMoveState,
	result: AccuracyChoiceTransition,
	audience: int,
) -> AccuracyChoicePlayerView:
	"""Project a v6 result to p1 (0) or p2 (1), preserving miss visibility."""
	_validate_two_move_state_structure(before)
	if not isinstance(result, AccuracyChoiceTransition):
		raise TypeError("result must be a simulator.accuracy_choice.AccuracyChoiceTransition.")
	_validate_two_move_state_structure(result.state)
	common = TwoMoveTransition(
		result.state,
		result.status,
		result.events,
		result.draws,
		result.event_move_ids,
	)
	base = project_two_move_view(before, common, audience)
	viewer = base.audience
	legal = (
		_supported_for_view(before)
		& (before.hp[viewer] > 0)
		& (before.request_phase == PHASE_MOVE_REQUEST)
		& (before.pp[viewer] > 0)
	)
	base = base._replace(
		own_pp=before.pp[viewer],
		own_move_ids=before.move_ids,
		legal_move_mask=legal.astype(jnp.bool_),
	)
	missed = result.events.missed
	internal_before = _adapt_state(before)
	internal_after = _adapt_state(result.state)
	internal_result = result.events
	projected = _project_view(
		internal_before,
		MegaPunchTransition(internal_after, result.status, internal_result, result.draws),
		audience,
		lambda candidate: _supported_accuracy_choice_state(before, candidate),
	)
	projected = _mask_missed_projection(projected, missed)
	# Keep the established two-move recipient contract while replacing its
	# common projection with the v6 supported-state and accuracy-aware result.
	base = base._replace(
		action_order=projected.action_order,
		event_move_ids=jnp.where(result.events.executed, result.event_move_ids, INVALID_MOVE_ID).astype(jnp.int32),
		critical=projected.critical,
		damage_value=projected.damage_value,
		damage_visible=projected.damage_visible,
		target_player=projected.target_player,
		target_hp_value=projected.target_hp_value,
		target_hp_is_percent=projected.target_hp_is_percent,
		fainted=projected.fainted,
		turn_event=projected.turn_event,
		turn_number=projected.turn_number,
		terminal=projected.terminal,
		winner=projected.winner,
	)
	return AccuracyChoicePlayerView(base, missed)


__all__ = [
	"ACTION_MEGA_PUNCH",
	"ACTION_TACKLE",
	"AccuracyChoicePlayerView",
	"AccuracyChoiceTransition",
	"MAX_PP_BY_SLOT",
	"MEGA_PUNCH_MAX_PP",
	"MOVE_IDS",
	"MOVE_MEGA_PUNCH",
	"TACKLE_ACCURACY",
	"TACKLE_BASE_POWER",
	"TACKLE_MAX_PP",
	"VARIANT_ID",
	"VARIANT_NAME",
	"make_accuracy_choice_state",
	"project_accuracy_choice_view",
	"transition_accuracy_choice",
]
