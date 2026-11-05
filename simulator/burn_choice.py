"""Restricted Gen 5 single-pre-existing-burn choice turn (variant 7)."""

from __future__ import annotations

from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .accuracy_choice import (
	ACTION_MEGA_PUNCH,
	ACTION_TACKLE,
	AccuracyChoicePlayerView,
	AccuracyChoiceTransition,
	MOVE_IDS,
	VARIANT_ID as ACCURACY_CHOICE_VARIANT_ID,
	make_accuracy_choice_state,
	_run_accuracy_choice_turn,
	_supported_accuracy_choice_state,
	project_accuracy_choice_view,
)
from .neutral_mega_punch import MegaPunchDrawTrace, MegaPunchEvents, _empty_draws, _empty_events
from .neutral_tackle import (
	CHOICE_PENDING,
	PHASE_MOVE_REQUEST,
	PHASE_PENDING,
	PHASE_TERMINAL,
	PHASE_UNSUPPORTED,
	SPECIES_MEW,
	STATUS_INVALID_CHOICE,
	STATUS_PENDING,
	STATUS_TERMINAL,
	STATUS_UNSUPPORTED,
)
from .neutral_two_move import TwoMoveState, _adapt_state, _choice_array, _validate_two_move_state_structure


VARIANT_ID = 7
VARIANT_NAME = "gen5-single-burn-choice-mew-1v1-v7"
MAX_PP_BY_SLOT = (56, 32)
INVALID_MOVE_ID = -1

_IntVectorInput = Sequence[int] | np.ndarray | jax.Array
_IntScalarInput = int | np.ndarray | jax.Array
_BoolVectorInput = Sequence[bool] | np.ndarray | jax.Array


class BurnChoiceState(NamedTuple):
	"""v7 wrapper around the fixed v6 battle layout and public native burn flags."""

	battle: TwoMoveState
	burned: jax.Array


class BurnResidualEvent(NamedTuple):
	"""One fixed row for the once-per-turn residual, if it occurs."""

	executed: jax.Array
	actor: jax.Array
	hp_loss: jax.Array
	target_hp: jax.Array
	target_max_hp: jax.Array


class BurnChoiceTransition(NamedTuple):
	state: BurnChoiceState
	status: jax.Array
	events: MegaPunchEvents
	draws: MegaPunchDrawTrace
	event_move_ids: jax.Array
	residual: BurnResidualEvent


class BurnResidualPlayerView(NamedTuple):
	"""Recipient-safe projection of the separate burn residual event."""

	executed: jax.Array
	actor: jax.Array
	target_player: jax.Array
	damage_value: jax.Array
	damage_visible: jax.Array
	target_hp_value: jax.Array
	target_hp_is_percent: jax.Array
	fainted: jax.Array


class BurnChoicePlayerView(NamedTuple):
	"""The accepted v6 attack view, known public burn flags, and residual view."""

	attack: AccuracyChoicePlayerView
	burned: jax.Array
	residual: BurnResidualPlayerView


def _burned_array(value) -> jax.Array:
	if isinstance(value, jax.core.Tracer):
		if tuple(value.shape) != (2,):
			raise ValueError("burned must have shape (2,).")
		if jnp.dtype(value.dtype) != jnp.dtype(jnp.bool_):
			raise TypeError("burned must have boolean dtype.")
		return value
	try:
		raw = np.asarray(value)
	except (TypeError, ValueError) as error:
		raise TypeError("burned must be a boolean vector of length two.") from error
	if raw.shape != (2,):
		raise ValueError("burned must have shape (2,).")
	if raw.dtype.kind != "b":
		raise TypeError("burned must have boolean dtype.")
	return jnp.asarray(raw, dtype=jnp.bool_)


def _validate_burn_choice_state_structure(state: BurnChoiceState) -> None:
	if not isinstance(state, BurnChoiceState):
		raise TypeError("state must be a simulator.burn_choice.BurnChoiceState.")
	_validate_two_move_state_structure(state.battle)
	if not hasattr(state.burned, "shape") or tuple(state.burned.shape) != (2,):
		raise ValueError("state.burned must have shape (2,).")
	if jnp.dtype(state.burned.dtype) != jnp.dtype(jnp.bool_):
		raise TypeError("state.burned must have boolean dtype.")


def make_burn_choice_state(
	*,
	hp: _IntVectorInput,
	max_hp: _IntVectorInput,
	pp: _IntVectorInput,
	level: _IntVectorInput,
	attack: _IntVectorInput,
	defense: _IntVectorInput,
	speed: _IntVectorInput,
	rng: _IntVectorInput,
	burned: _BoolVectorInput = (False, False),
	turn: _IntScalarInput = 1,
	request_phase: _IntScalarInput = PHASE_MOVE_REQUEST,
	terminal: bool | np.bool_ | np.ndarray | jax.Array = False,
	winner: _IntScalarInput = -1,
	species_id: _IntVectorInput = (SPECIES_MEW, SPECIES_MEW),
) -> BurnChoiceState:
	"""Build the v7 nested state, retaining the v6 two-slot physical layout."""
	battle6 = make_accuracy_choice_state(
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
		species_id=species_id,
	)
	battle = battle6._replace(variant_id=jnp.asarray(VARIANT_ID, dtype=jnp.int32))
	state = BurnChoiceState(battle, _burned_array(burned))
	_validate_burn_choice_state_structure(state)
	return state


def _as_accuracy_choice_state(battle: TwoMoveState) -> TwoMoveState:
	return battle._replace(variant_id=jnp.asarray(ACCURACY_CHOICE_VARIANT_ID, dtype=jnp.int32))


def _supported_burn_choice_state(state: BurnChoiceState) -> jax.Array:
	battle6 = _as_accuracy_choice_state(state.battle)
	return (
		(state.battle.variant_id == VARIANT_ID)
		& jnp.all(state.battle.species_id == SPECIES_MEW)
		& jnp.all(state.battle.move_ids == jnp.asarray(MOVE_IDS, dtype=jnp.int32))
		& (jnp.sum(state.burned.astype(jnp.int32)) <= 1)
		& _supported_accuracy_choice_state(battle6, _adapt_state(battle6))
	)


def _empty_residual() -> BurnResidualEvent:
	return BurnResidualEvent(
		jnp.asarray(False, dtype=jnp.bool_),
		jnp.asarray(-1, dtype=jnp.int32),
		jnp.asarray(0, dtype=jnp.int32),
		jnp.asarray(0, dtype=jnp.int32),
		jnp.asarray(0, dtype=jnp.int32),
	)


def _empty_result(state: BurnChoiceState, status: int) -> BurnChoiceTransition:
	return BurnChoiceTransition(
		state,
		jnp.asarray(status, dtype=jnp.int32),
		_empty_events(),
		_empty_draws(),
		jnp.full((2,), INVALID_MOVE_ID, dtype=jnp.int32),
		_empty_residual(),
	)


def _run_burn_choice_turn(state: BurnChoiceState, choices: jax.Array) -> BurnChoiceTransition:
	battle = state.battle
	attack = _run_accuracy_choice_turn(battle, choices, burned=state.burned)
	attack_after = attack.state
	residual_actor = jnp.argmax(state.burned.astype(jnp.int32)).astype(jnp.int32)
	residual_runs = (~attack_after.terminal) & jnp.any(state.burned)
	base_hp_loss = jnp.maximum(1, battle.max_hp[residual_actor] // 8)
	actual_hp_loss = jnp.where(
		residual_runs,
		jnp.minimum(attack_after.hp[residual_actor], base_hp_loss),
		jnp.asarray(0, dtype=jnp.int32),
	).astype(jnp.int32)
	final_hp = attack_after.hp.at[residual_actor].add(-actual_hp_loss)
	terminal = jnp.any(final_hp == 0)
	winner = jnp.where(
		(final_hp[0] == 0) & (final_hp[1] == 0),
		0,
		jnp.where(final_hp[0] == 0, 2, jnp.where(final_hp[1] == 0, 1, -1)),
	).astype(jnp.int32)
	request_phase = jnp.where(
		terminal,
		PHASE_TERMINAL,
		jnp.where(jnp.any(jnp.all(attack_after.pp == 0, axis=1)), PHASE_UNSUPPORTED, PHASE_MOVE_REQUEST),
	).astype(jnp.int32)
	final_battle = attack_after._replace(
		hp=final_hp,
		turn=battle.turn + (~terminal).astype(jnp.int32),
		request_phase=request_phase,
		terminal=terminal,
		winner=winner,
	)
	final_state = BurnChoiceState(final_battle, state.burned)
	residual = BurnResidualEvent(
		residual_runs,
		jnp.where(residual_runs, residual_actor, -1).astype(jnp.int32),
		actual_hp_loss,
		jnp.where(residual_runs, final_hp[residual_actor], 0).astype(jnp.int32),
		jnp.where(residual_runs, battle.max_hp[residual_actor], 0).astype(jnp.int32),
	)
	return BurnChoiceTransition(
		final_state,
		attack.status,
		attack.events,
		attack.draws,
		attack.event_move_ids,
		residual,
	)


def transition_burn_choice(
	state: BurnChoiceState,
	choices: _IntVectorInput,
) -> BurnChoiceTransition:
	"""Resolve simultaneous choices: 1=Tackle, 2=Mega Punch, 0=pending."""
	_validate_burn_choice_state_structure(state)
	choice_array = _choice_array(choices)

	def terminal_result(_):
		return _empty_result(state, STATUS_TERMINAL)

	def active_result(_):
		negative = jnp.any(choice_array < 0)
		unknown = jnp.any(choice_array > ACTION_MEGA_PUNCH)
		pending = jnp.any(choice_array == CHOICE_PENDING) | (state.battle.request_phase == PHASE_PENDING)

		def execute_or_reject(__):
			def selected_pp_or_execute(____):
				selected_pp = state.battle.pp[
					jnp.arange(2), jnp.clip(choice_array, ACTION_TACKLE, ACTION_MEGA_PUNCH) - 1
				]
				return jax.lax.cond(
					jnp.all(selected_pp > 0),
					lambda _____: _run_burn_choice_turn(state, choice_array),
					lambda _____: _empty_result(state, STATUS_INVALID_CHOICE),
					None,
				)

			return jax.lax.cond(
				_supported_burn_choice_state(state),
				selected_pp_or_execute,
				lambda ____: _empty_result(state, STATUS_UNSUPPORTED),
				None,
			)

		return jax.lax.cond(
			negative,
			lambda ___: _empty_result(state, STATUS_INVALID_CHOICE),
			lambda ___: jax.lax.cond(
				unknown,
				lambda ____: _empty_result(state, STATUS_UNSUPPORTED),
				lambda ____: jax.lax.cond(
					pending,
					lambda _____: _empty_result(state, STATUS_PENDING),
					execute_or_reject,
					None,
				),
				None,
			),
			None,
		)

	return jax.lax.cond(state.battle.terminal, terminal_result, active_result, None)


def _residual_player_view(result: BurnChoiceTransition, audience: int) -> BurnResidualPlayerView:
	viewer = jnp.asarray(audience, dtype=jnp.int32)
	event = result.residual
	actor = jnp.maximum(event.actor, 0)
	is_owner = viewer == actor
	percent = (event.target_hp * 100 + jnp.maximum(event.target_max_hp, 1) - 1) // jnp.maximum(event.target_max_hp, 1)
	percent = jnp.where((percent == 100) & (event.target_hp < event.target_max_hp), 99, percent)
	visible_hp = jnp.where(is_owner, event.target_hp, percent).astype(jnp.int32)
	visible_hp = jnp.where(event.executed, visible_hp, 0)
	visible_loss = jnp.where(is_owner, event.hp_loss, -1).astype(jnp.int32)
	visible_loss = jnp.where(event.executed, visible_loss, 0)
	return BurnResidualPlayerView(
		event.executed,
		event.actor,
		jnp.where(event.executed, actor, -1).astype(jnp.int32),
		visible_loss,
		event.executed & is_owner,
		visible_hp,
		event.executed & (~is_owner),
		event.executed & (event.target_hp == 0),
	)


def project_burn_choice_view(
	before: BurnChoiceState,
	result: BurnChoiceTransition,
	audience: int,
) -> BurnChoicePlayerView:
	"""Project attack and residual rows without exposing trusted battle state."""
	_validate_burn_choice_state_structure(before)
	if not isinstance(result, BurnChoiceTransition):
		raise TypeError("result must be a simulator.burn_choice.BurnChoiceTransition.")
	_validate_burn_choice_state_structure(result.state)
	if isinstance(audience, jax.core.Tracer):
		raise TypeError("audience must be a static p1/p2 integer.")
	try:
		raw_audience = np.asarray(audience)
	except (TypeError, ValueError) as error:
		raise TypeError("audience must be an integer p1/p2 identifier.") from error
	if raw_audience.shape != () or raw_audience.dtype.kind not in "iu":
		raise TypeError("audience must be an integer p1/p2 identifier.")
	if int(raw_audience) not in (0, 1):
		raise ValueError("audience must be p1 (0) or p2 (1).")
	before6 = _as_accuracy_choice_state(before.battle)
	after6 = _as_accuracy_choice_state(result.state.battle)
	accuracy_result = AccuracyChoiceTransition(
		after6,
		result.status,
		result.events,
		result.draws,
		result.event_move_ids,
	)
	attack_view = project_accuracy_choice_view(before6, accuracy_result, int(raw_audience))
	viewer = attack_view.player_view.audience
	legal = (
		_supported_burn_choice_state(before)
		& (before.battle.hp[viewer] > 0)
		& (before.battle.request_phase == PHASE_MOVE_REQUEST)
		& (before.battle.pp[viewer] > 0)
	)
	attack_view = attack_view._replace(
		player_view=attack_view.player_view._replace(legal_move_mask=legal.astype(jnp.bool_)),
	)
	return BurnChoicePlayerView(
		attack_view,
		result.state.burned,
		_residual_player_view(result, int(raw_audience)),
	)


__all__ = [
	"ACTION_MEGA_PUNCH",
	"ACTION_TACKLE",
	"BurnChoicePlayerView",
	"BurnChoiceState",
	"BurnChoiceTransition",
	"BurnResidualEvent",
	"BurnResidualPlayerView",
	"MAX_PP_BY_SLOT",
	"VARIANT_ID",
	"VARIANT_NAME",
	"make_burn_choice_state",
	"project_burn_choice_view",
	"transition_burn_choice",
]
