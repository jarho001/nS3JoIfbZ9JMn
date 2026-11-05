"""Fixed-state JAX turn for the restricted Gen 5 Mega Punch Mew variant.

This v5 slice supports one active Mew per side, Mega Punch as the sole move,
and distinct positive speeds. It models the native 85% accuracy check,
including misses. It is a development boundary, not full-format support.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from mechanics.damage_composition import damage_from_preroll
from mechanics.neutral_damage import neutral_damage_before_roll

from .neutral_duel import _validate_state_structure
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
	make_state,
)
from .gen5_rng import sample_bounded


VARIANT_ID = 5  # gen5-neutral-mega-punch-mew-1v1-v5
VARIANT_NAME = "gen5-neutral-mega-punch-mew-1v1-v5"
MOVE_MEGA_PUNCH = 5
MEGA_PUNCH_BASE_POWER = 80
MEGA_PUNCH_ACCURACY = 85
MEGA_PUNCH_BASE_PP = 20
MEGA_PUNCH_MAX_PP = 32
MEGA_PUNCH_CRITICAL_RATIO = 1
ACTION_MEGA_PUNCH = 1

DRAW_ACCURACY = 1
DRAW_CRITICAL = 2
DRAW_DAMAGE = 3

_IntVectorInput = Sequence[int] | np.ndarray | jax.Array
_IntScalarInput = int | np.ndarray | jax.Array


class MegaPunchEvents(NamedTuple):
	"""At most two attempted move rows in actual action order."""

	executed: jax.Array
	actors: jax.Array
	hit: jax.Array
	missed: jax.Array
	critical: jax.Array
	damage: jax.Array
	target_hp: jax.Array
	target_max_hp: jax.Array


class MegaPunchDrawTrace(NamedTuple):
	"""Six native-order draw slots with an explicit packed prefix count."""

	count: jax.Array
	bounds: jax.Array
	values: jax.Array
	contexts: jax.Array
	before: jax.Array
	after: jax.Array


class MegaPunchTransition(NamedTuple):
	state: BattleState
	status: jax.Array
	events: MegaPunchEvents
	draws: MegaPunchDrawTrace


class MegaPunchPlayerView(NamedTuple):
	"""Existing recipient projection plus explicit per-action miss rows."""

	player_view: PlayerView
	missed: jax.Array


def make_mega_punch_state(
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
) -> BattleState:
	"""Build the v5 fixed layout from pinned Showdown initialized values."""
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
		move_id=MOVE_MEGA_PUNCH,
	)
	_validate_state_structure(state)
	return state


def _empty_events() -> MegaPunchEvents:
	return MegaPunchEvents(
		jnp.zeros((2,), dtype=jnp.bool_),
		jnp.full((2,), -1, dtype=jnp.int32),
		jnp.zeros((2,), dtype=jnp.bool_),
		jnp.zeros((2,), dtype=jnp.bool_),
		jnp.zeros((2,), dtype=jnp.bool_),
		jnp.zeros((2,), dtype=jnp.int32),
		jnp.zeros((2,), dtype=jnp.int32),
		jnp.zeros((2,), dtype=jnp.int32),
	)


def _empty_draws() -> MegaPunchDrawTrace:
	return MegaPunchDrawTrace(
		jnp.asarray(0, dtype=jnp.int32),
		jnp.zeros((6,), dtype=jnp.int32),
		jnp.full((6,), -1, dtype=jnp.int32),
		jnp.zeros((6,), dtype=jnp.int32),
		jnp.zeros((6, 4), dtype=jnp.uint16),
		jnp.zeros((6, 4), dtype=jnp.uint16),
	)


def _empty_transition(state: BattleState, status: int) -> MegaPunchTransition:
	return MegaPunchTransition(
		state,
		jnp.asarray(status, dtype=jnp.int32),
		_empty_events(),
		_empty_draws(),
	)


def _supported_mega_punch_state(state: BattleState) -> jax.Array:
	return (
		_supported_common(state, move_id=MOVE_MEGA_PUNCH, max_pp=MEGA_PUNCH_MAX_PP)
		& (state.variant_id == VARIANT_ID)
		& (state.speed[0] != state.speed[1])
	)


def _record_draw(
	draws: MegaPunchDrawTrace,
	bound: int,
	value,
	context: int,
	before,
	after,
) -> MegaPunchDrawTrace:
	index = draws.count
	return MegaPunchDrawTrace(
		draws.count + jnp.asarray(1, dtype=jnp.int32),
		draws.bounds.at[index].set(jnp.asarray(bound, dtype=jnp.int32)),
		draws.values.at[index].set(jnp.asarray(value, dtype=jnp.int32)),
		draws.contexts.at[index].set(jnp.asarray(context, dtype=jnp.int32)),
		draws.before.at[index].set(before),
		draws.after.at[index].set(after),
	)


def _execute_action(
	state: BattleState,
	actor,
	slot,
	events,
	draws,
	*,
	base_power=MEGA_PUNCH_BASE_POWER,
	accuracy=MEGA_PUNCH_ACCURACY,
):
	target = 1 - actor
	seed_before_accuracy = state.rng
	seed_after_accuracy, accuracy_word = sample_bounded(seed_before_accuracy, 100)
	accuracy_roll = accuracy_word.astype(jnp.int32)
	after_accuracy = state._replace(
		pp=state.pp.at[actor].add(jnp.asarray(-1, dtype=jnp.int32)),
		rng=seed_after_accuracy,
	)
	draws = _record_draw(
		draws,
		100,
		accuracy_roll,
		DRAW_ACCURACY,
		seed_before_accuracy,
		seed_after_accuracy,
	)

	def set_event(inner_events, *, hit, missed, critical, damage, target_hp):
		return MegaPunchEvents(
			inner_events.executed.at[slot].set(jnp.asarray(True)),
			inner_events.actors.at[slot].set(jnp.asarray(actor, dtype=jnp.int32)),
			inner_events.hit.at[slot].set(hit),
			inner_events.missed.at[slot].set(missed),
			inner_events.critical.at[slot].set(critical),
			inner_events.damage.at[slot].set(damage),
			inner_events.target_hp.at[slot].set(target_hp),
			inner_events.target_max_hp.at[slot].set(state.max_hp[target]),
		)

	def miss_branch(_):
		return (
			after_accuracy,
			set_event(
				events,
				hit=jnp.asarray(False),
				missed=jnp.asarray(True),
				critical=jnp.asarray(False),
				damage=jnp.asarray(0, dtype=jnp.int32),
				target_hp=state.hp[target],
			),
			draws,
		)

	def hit_branch(_):
		seed_after_critical, critical_word = sample_bounded(after_accuracy.rng, 16)
		seed_after_damage, damage_word = sample_bounded(seed_after_critical, 16)
		critical_roll = critical_word.astype(jnp.int32)
		damage_roll = damage_word.astype(jnp.int32)
		critical = critical_roll == 0
		pre_critical_damage = neutral_damage_before_roll(
			after_accuracy.level[actor],
			jnp.asarray(base_power, dtype=jnp.int32),
			after_accuracy.attack[actor],
			after_accuracy.defense[target],
		)
		critical_damage = jnp.where(
			critical,
			(pre_critical_damage.astype(jnp.uint32) * jnp.asarray(2, dtype=jnp.uint32)).astype(jnp.int32),
			pre_critical_damage,
		)
		damage = damage_from_preroll(
			critical_damage,
			damage_roll,
			False,
			False,
			False,
			jnp.asarray(0, dtype=jnp.int32),
			False,
			False,
			False,
			jnp.asarray(4096, dtype=jnp.int32),
		)
		target_hp = jnp.maximum(jnp.asarray(0, dtype=jnp.int32), state.hp[target] - damage)
		new_state = after_accuracy._replace(
			hp=after_accuracy.hp.at[target].set(target_hp),
			rng=seed_after_damage,
		)
		new_draws = _record_draw(
			_record_draw(
				draws,
				16,
				critical_roll,
				DRAW_CRITICAL,
				after_accuracy.rng,
				seed_after_critical,
			),
			16,
			damage_roll,
			DRAW_DAMAGE,
			seed_after_critical,
			seed_after_damage,
		)
		return (
			new_state,
			set_event(
				events,
				hit=jnp.asarray(True),
				missed=jnp.asarray(False),
				critical=critical,
				damage=damage,
				target_hp=target_hp,
			),
			new_draws,
		)

	return jax.lax.cond(
		accuracy_roll < accuracy,
		hit_branch,
		miss_branch,
		operand=None,
	)


def _run_turn(state: BattleState, action_fn=_execute_action) -> MegaPunchTransition:
	first_actor = jnp.where(state.speed[0] > state.speed[1], 0, 1).astype(jnp.int32)
	events = _empty_events()
	draws = _empty_draws()

	def process_action(step, carry):
		current_state, current_events, current_draws = carry
		actor = jnp.where(step == 0, first_actor, 1 - first_actor).astype(jnp.int32)
		can_act = (current_state.hp[actor] > 0) & (current_state.hp[1 - actor] > 0)
		return jax.lax.cond(
			can_act,
			lambda value: action_fn(value[0], actor, step, value[1], value[2]),
			lambda value: value,
			(current_state, current_events, current_draws),
		)

	state, events, draws = jax.lax.fori_loop(0, 2, process_action, (state, events, draws))
	terminal = jnp.any(state.hp == 0)
	winner = jnp.where(
		(state.hp[0] == 0) & (state.hp[1] == 0),
		0,
		jnp.where(state.hp[0] == 0, 2, jnp.where(state.hp[1] == 0, 1, -1)),
	).astype(jnp.int32)
	next_phase = jnp.where(
		terminal,
		PHASE_TERMINAL,
		jnp.where(jnp.any(state.pp == 0), PHASE_UNSUPPORTED, PHASE_MOVE_REQUEST),
	).astype(jnp.int32)
	state = state._replace(
		turn=state.turn + jnp.where(terminal, 0, 1).astype(jnp.int32),
		request_phase=next_phase,
		terminal=terminal,
		winner=winner,
	)
	return MegaPunchTransition(state, jnp.asarray(STATUS_OK, dtype=jnp.int32), events, draws)


def _choice_array(choices: _IntVectorInput) -> jax.Array:
	if isinstance(choices, jax.core.Tracer):
		result = choices
		if result.dtype != jnp.int32:
			raise TypeError("compiled choices must have dtype int32.")
	else:
		try:
			raw = np.asarray(choices)
		except (TypeError, ValueError) as error:
			raise TypeError("choices must be integral values that fit signed int32.") from error
		if raw.dtype.kind not in "iu" or np.any(raw < np.iinfo(np.int32).min) or np.any(raw > np.iinfo(np.int32).max):
			raise TypeError("choices must be integral values that fit signed int32.")
		result = jnp.asarray(raw, dtype=jnp.int32)
	if result.shape != (2,):
		raise ValueError("choices must be a length-two p1/p2 vector.")
	return result


def transition_mega_punch(
	state: BattleState,
	choices: _IntVectorInput,
) -> MegaPunchTransition:
	"""Resolve simultaneous sole-Mega-Punch choices for a supported v5 state.

	Choice ``1`` is request ``move 1`` and ``0`` is pending. Negative values are
	invalid; unknown positive values and unsupported states fail closed. Runtime
	non-success statuses preserve state and have empty events/draws.
	"""
	_validate_state_structure(state)
	choice_array = _choice_array(choices)

	def terminal_result(_):
		return _empty_transition(state, STATUS_TERMINAL)

	def active_result(_):
		negative = jnp.any(choice_array < 0)
		unknown = jnp.any(choice_array > ACTION_MEGA_PUNCH)
		pending = jnp.any(choice_array == CHOICE_PENDING) | (state.request_phase == PHASE_PENDING)
		return jax.lax.cond(
			negative,
			lambda _: _empty_transition(state, STATUS_INVALID_CHOICE),
			lambda _: jax.lax.cond(
				unknown,
				lambda __: _empty_transition(state, STATUS_UNSUPPORTED),
				lambda __: jax.lax.cond(
					pending,
					lambda ___: _empty_transition(state, STATUS_PENDING),
					lambda ___: jax.lax.cond(
						_supported_mega_punch_state(state),
						lambda ____: _run_turn(state),
						lambda ____: _empty_transition(state, STATUS_UNSUPPORTED),
						None,
					),
					None,
				),
				None,
			),
			None,
		)

	return jax.lax.cond(state.terminal, terminal_result, active_result, None)


def project_mega_punch_view(
	before: BattleState,
	result: MegaPunchTransition,
	audience: int,
) -> MegaPunchPlayerView:
	"""Project a v5 result to p1 (0) or p2 (1), keeping miss rows explicit."""
	if not isinstance(result, MegaPunchTransition):
		raise TypeError("result must be a MegaPunchTransition.")
	_validate_state_structure(before)
	_validate_state_structure(result.state)
	base = _project_view(before, result, audience, _supported_mega_punch_state)
	missed = result.events.missed
	base = _mask_missed_projection(base, missed)
	return MegaPunchPlayerView(base, missed)


def _mask_missed_projection(base: PlayerView, missed: jax.Array) -> PlayerView:
	"""Hide hit-only projection fields on rows whose selected move missed."""
	return base._replace(
		critical=base.critical & (~missed),
		damage_value=jnp.where(missed, -1, base.damage_value).astype(jnp.int32),
		damage_visible=base.damage_visible & (~missed),
		target_hp_value=jnp.where(missed, -1, base.target_hp_value).astype(jnp.int32),
		target_hp_is_percent=base.target_hp_is_percent & (~missed),
		fainted=base.fainted & (~missed),
	)


__all__ = [
	"ACTION_MEGA_PUNCH",
	"DRAW_ACCURACY",
	"DRAW_CRITICAL",
	"DRAW_DAMAGE",
	"MEGA_PUNCH_ACCURACY",
	"MEGA_PUNCH_BASE_POWER",
	"MEGA_PUNCH_BASE_PP",
	"MEGA_PUNCH_CRITICAL_RATIO",
	"MEGA_PUNCH_MAX_PP",
	"MOVE_MEGA_PUNCH",
	"MegaPunchDrawTrace",
	"MegaPunchEvents",
	"MegaPunchPlayerView",
	"MegaPunchTransition",
	"VARIANT_ID",
	"VARIANT_NAME",
	"make_mega_punch_state",
	"project_mega_punch_view",
	"transition_mega_punch",
]
