"""Fixed-state JAX turn for the Gen 5 neutral Pound Mew-versus-Mew variant.

Both players have Pound as their sole move. This separately versioned slice
reuses the accepted accuracy, critical-hit, damage-roll, speed-tie, event, and
recipient-projection machinery while enforcing its own move and variant IDs.
"""

from __future__ import annotations

import jax

from .neutral_duel import _validate_state_structure
from .neutral_tackle import (
	PHASE_MOVE_REQUEST,
	SPECIES_MEW,
	BattleState,
	PlayerView,
	_execute_neutral_action,
	_project_view,
	_supported_common,
	make_state,
	_choice_array,
	_dispatch_transition,
)
from .neutral_tackle_ties import (
	MAX_TIE_DRAWS_PER_TURN,
	TieDrawTrace,
	TieTransition,
	_empty_tie_transition,
	_run_tie_turn,
)


VARIANT_ID = 3  # gen5-neutral-pound-mew-1v1-v3
VARIANT_NAME = "gen5-neutral-pound-mew-1v1-v3"
MOVE_POUND = 1
POUND_BASE_POWER = 40
POUND_BASE_PP = 35
POUND_MAX_PP = 56  # 35 base PP with all three PP Ups.
POUND_ACCURACY = 100
POUND_CRITICAL_RATIO = 1  # Gen 5 default: one critical result in sixteen.

ACTION_POUND = 1  # The sole move is submitted as request move 1.


def make_pound_state(
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
	species_id=(SPECIES_MEW, SPECIES_MEW),
) -> BattleState:
	"""Build v3 from pinned Showdown initialized stats and RNG state."""
	state = make_state(
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
		move_id=MOVE_POUND,
	)
	_validate_state_structure(state)
	return state


def _supported_pound_state(state: BattleState) -> jax.Array:
	return _supported_common(
		state,
		move_id=MOVE_POUND,
		max_pp=POUND_MAX_PP,
	) & (state.variant_id == VARIANT_ID)


def _execute_pound_action(state: BattleState, actor):
	return _execute_neutral_action(state, actor, POUND_BASE_POWER)


def _run_pound_turn(state: BattleState) -> TieTransition:
	return _run_tie_turn(state, _execute_pound_action)


def transition_pound(state: BattleState, choices) -> TieTransition:
	"""Resolve simultaneous sole-Pound actions with the v3 tie trace contract.

	``choices`` is an int32 vector ordered p1, p2. ``1`` means Pound / request
	``move 1`` and ``0`` means not yet submitted. Invalid, pending, terminal,
	unsupported, or cross-variant inputs return an unchanged state and no draws.
	The result retains the fixed sixteen-slot native-order tie trace and supports
	``jax.jit`` and ``jax.vmap``.
	"""
	_validate_state_structure(state)
	return _dispatch_transition(
		state,
		_choice_array(choices),
		_supported_pound_state,
		_run_pound_turn,
		_empty_tie_transition,
		sole_action_id=ACTION_POUND,
	)


def project_pound_view(before: BattleState, result: TieTransition, audience) -> PlayerView:
	"""Project one v3 result to p1 (0) or p2 (1).

	The fixed ``PlayerView`` has one legal-action bit and no move-name field; its
	sole action denotes Pound because this API validates the v3 Pound boundary.
	Compile with ``jax.jit(project_pound_view, static_argnums=2)``.
	"""
	_validate_state_structure(before)
	_validate_state_structure(result.state)
	return _project_view(before, result, audience, _supported_pound_state)


__all__ = [
	"ACTION_POUND",
	"MAX_TIE_DRAWS_PER_TURN",
	"MOVE_POUND",
	"POUND_ACCURACY",
	"POUND_BASE_PP",
	"POUND_BASE_POWER",
	"POUND_CRITICAL_RATIO",
	"POUND_MAX_PP",
	"TieDrawTrace",
	"TieTransition",
	"VARIANT_ID",
	"VARIANT_NAME",
	"make_pound_state",
	"project_pound_view",
	"transition_pound",
]
