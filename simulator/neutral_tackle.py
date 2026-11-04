"""A fixed-state JAX turn for the Gen 5 neutral Mew-versus-Mew Tackle variant.

The supported boundary is intentionally narrow: one active Mew per side,
Tackle as the sole move, no battle effects, and distinct positive speeds. The
reference side initializes those values from pinned Showdown sets. This module
receives the resulting ordinary level, HP, PP, Attack, Defense, and Speed values;
it does not implement team or stat construction.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from mechanics.damage_composition import damage_from_preroll
from mechanics.neutral_damage import neutral_damage_before_roll

from .gen5_rng import sample_bounded


VARIANT_ID = 1  # gen5-neutral-tackle-mew-1v1-v1
SPEED_TIE_VARIANT_ID = 2  # gen5-neutral-tackle-speed-ties-mew-1v1-v2
SPECIES_MEW = 151
MOVE_TACKLE = 33
TACKLE_BASE_POWER = 50  # Pinned Gen 5 move data.
TACKLE_MAX_PP = 56  # 35 base PP with all three PP Ups.
MEW_MAX_HP = 404  # Level 100, 31 IV, 252 EV, no HP-changing effect.
MEW_MAX_STAT = 299  # Level 100, neutral nature, 31 IV, 252 EV.

ACTION_TACKLE = 1
CHOICE_PENDING = 0

PHASE_MOVE_REQUEST = 0
PHASE_PENDING = 1
PHASE_TERMINAL = 2
PHASE_UNSUPPORTED = 3

STATUS_OK = 0
STATUS_PENDING = 1
STATUS_INVALID_CHOICE = 2
STATUS_UNSUPPORTED = 3
STATUS_TERMINAL = 4

DRAW_ACCURACY = 1
DRAW_CRITICAL = 2
DRAW_DAMAGE = 3


class BattleState(NamedTuple):
	"""Fixed-shape trusted battle state; arrays are ordered p1, p2."""

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
	move_id: jax.Array


class TurnEvents(NamedTuple):
	"""At most two action rows, stored in actual move order."""

	executed: jax.Array
	actors: jax.Array
	critical: jax.Array
	damage: jax.Array
	target_hp: jax.Array
	target_max_hp: jax.Array


class DrawTrace(NamedTuple):
	"""Six fixed draw slots (accuracy, critical, damage for each action)."""

	count: jax.Array
	bounds: jax.Array
	values: jax.Array
	contexts: jax.Array
	before: jax.Array
	after: jax.Array


class Transition(NamedTuple):
	state: BattleState
	status: jax.Array
	events: TurnEvents
	draws: DrawTrace


class PlayerView(NamedTuple):
	"""Minimal per-recipient request and semantic event projection.

	This value contains only the recipient's exact own HP/PP/stats and the HP
	value visible to that recipient for each damage event. It never exposes the
	opponent's trusted HP, complete state, or unprojected protocol.
	"""

	audience: jax.Array
	request_phase: jax.Array
	legal_move_mask: jax.Array
	own_hp: jax.Array
	own_max_hp: jax.Array
	own_pp: jax.Array
	own_level: jax.Array
	own_attack: jax.Array
	own_defense: jax.Array
	own_speed: jax.Array
	action_order: jax.Array
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


class _TackleActionResult(NamedTuple):
	state: BattleState
	critical: jax.Array
	damage: jax.Array
	target_hp: jax.Array
	target_max_hp: jax.Array
	values: jax.Array
	before: jax.Array
	after: jax.Array


def _int_vector(name: str, value, length: int) -> jax.Array:
	array = np.asarray(value)
	if array.shape != (length,) or array.dtype.kind not in "iu":
		raise ValueError(f"{name} must be an integer vector of length {length}.")
	if np.any(array < np.iinfo(np.int32).min) or np.any(array > np.iinfo(np.int32).max):
		raise ValueError(f"{name} values must fit signed int32.")
	return jnp.asarray(array, dtype=jnp.int32)


def _int_scalar(name: str, value) -> jax.Array:
	array = np.asarray(value)
	if array.shape != () or array.dtype.kind not in "iu":
		raise ValueError(f"{name} must be an integer scalar.")
	if int(array) < np.iinfo(np.int32).min or int(array) > np.iinfo(np.int32).max:
		raise ValueError(f"{name} must fit signed int32.")
	return jnp.asarray(array, dtype=jnp.int32)


def make_state(
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
	move_id=MOVE_TACKLE,
) -> BattleState:
	"""Build the fixed PyTree from pinned Showdown's initialized values.

	Structural errors raise ``ValueError`` before simulation. In-range and
	variant-boundary errors are returned as ``STATUS_UNSUPPORTED`` by
	:func:`transition` without changing the state or RNG.
	"""
	seed = np.asarray(rng)
	if seed.shape != (4,) or seed.dtype.kind not in "iu" or np.any(seed < 0) or np.any(seed > 0xFFFF):
		raise ValueError("rng must contain four unsigned 16-bit Gen 5 limbs.")
	if not isinstance(terminal, (bool, np.bool_)):
		raise ValueError("terminal must be boolean.")

	return BattleState(
		_int_vector("hp", hp, 2),
		_int_vector("max_hp", max_hp, 2),
		_int_vector("pp", pp, 2),
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
		_int_scalar("move_id", move_id),
	)


def _empty_events() -> TurnEvents:
	return TurnEvents(
		jnp.zeros((2,), dtype=jnp.bool_),
		jnp.full((2,), -1, dtype=jnp.int32),
		jnp.zeros((2,), dtype=jnp.bool_),
		jnp.zeros((2,), dtype=jnp.int32),
		jnp.zeros((2,), dtype=jnp.int32),
		jnp.zeros((2,), dtype=jnp.int32),
	)


def _empty_draws() -> DrawTrace:
	return DrawTrace(
		jnp.asarray(0, dtype=jnp.int32),
		jnp.zeros((6,), dtype=jnp.int32),
		jnp.full((6,), -1, dtype=jnp.int32),
		jnp.zeros((6,), dtype=jnp.int32),
		jnp.zeros((6, 4), dtype=jnp.uint16),
		jnp.zeros((6, 4), dtype=jnp.uint16),
	)


def _empty_transition(state: BattleState, status: int) -> Transition:
	return Transition(state, jnp.asarray(status, dtype=jnp.int32), _empty_events(), _empty_draws())


def _supported_common(state: BattleState) -> jax.Array:
	return (
		(state.request_phase == PHASE_MOVE_REQUEST)
		& (~state.terminal)
		& (state.winner == -1)
		& jnp.all(state.species_id == SPECIES_MEW)
		& (state.move_id == MOVE_TACKLE)
		& jnp.all(state.max_hp > 0)
		& jnp.all(state.max_hp <= MEW_MAX_HP)
		& jnp.all(state.hp > 0)
		& jnp.all(state.hp <= state.max_hp)
		& jnp.all(state.pp > 0)
		& jnp.all(state.pp <= TACKLE_MAX_PP)
		& jnp.all(state.level >= 1)
		& jnp.all(state.level <= 100)
		& jnp.all(state.attack > 0)
		& jnp.all(state.attack <= MEW_MAX_STAT)
		& jnp.all(state.defense > 0)
		& jnp.all(state.defense <= MEW_MAX_STAT)
		& jnp.all(state.speed > 0)
		& jnp.all(state.speed <= MEW_MAX_STAT)
		& (state.turn >= 1)
		& (state.turn < 0x7FFFFFFF)
	)


def _supported_state(state: BattleState) -> jax.Array:
	return _supported_common(state) & (state.variant_id == VARIANT_ID) & (state.speed[0] != state.speed[1])


def _supported_tie_state(state: BattleState) -> jax.Array:
	return _supported_common(state) & (state.variant_id == SPEED_TIE_VARIANT_ID)


def _record_draw(draws: DrawTrace, index, bound, value, context, before, after) -> DrawTrace:
	return DrawTrace(
		draws.count + jnp.asarray(1, dtype=jnp.int32),
		draws.bounds.at[index].set(jnp.asarray(bound, dtype=jnp.int32)),
		draws.values.at[index].set(jnp.asarray(value, dtype=jnp.int32)),
		draws.contexts.at[index].set(jnp.asarray(context, dtype=jnp.int32)),
		draws.before.at[index].set(before),
		draws.after.at[index].set(after),
	)


def _execute_tackle_action(state: BattleState, actor) -> _TackleActionResult:
	target = 1 - actor
	seed_before_accuracy = state.rng
	seed_after_accuracy, accuracy_word = sample_bounded(seed_before_accuracy, 100)
	seed_after_critical, critical_word = sample_bounded(seed_after_accuracy, 16)
	seed_after_damage, damage_word = sample_bounded(seed_after_critical, 16)
	accuracy_roll = accuracy_word.astype(jnp.int32)
	critical_roll = critical_word.astype(jnp.int32)
	damage_roll = damage_word.astype(jnp.int32)
	critical = critical_roll == 0

	pre_critical_damage = neutral_damage_before_roll(
		state.level[actor],
		jnp.asarray(TACKLE_BASE_POWER, dtype=jnp.int32),
		state.attack[actor],
		state.defense[target],
	)
	critical_product = pre_critical_damage.astype(jnp.uint32) * jnp.asarray(2, dtype=jnp.uint32)
	critical_damage = jnp.where(critical, critical_product.astype(jnp.int32), pre_critical_damage)
	damage = damage_from_preroll(
		critical_damage,
		damage_roll,
		False,  # Mew is Psychic; Normal Tackle is never STAB.
		False,
		False,
		jnp.asarray(0, dtype=jnp.int32),  # Normal-on-Psychic is neutral.
		False,
		False,
		False,
		jnp.asarray(4096, dtype=jnp.int32),  # Neutral final modifier.
	)
	hp_after = jnp.maximum(jnp.asarray(0, dtype=jnp.int32), state.hp[target] - damage)
	new_state = state._replace(
		hp=state.hp.at[target].set(hp_after),
		pp=state.pp.at[actor].add(jnp.asarray(-1, dtype=jnp.int32)),
		rng=seed_after_damage,
	)
	return _TackleActionResult(
		new_state,
		critical,
		damage,
		hp_after,
		state.max_hp[target],
		jnp.stack((accuracy_roll, critical_roll, damage_roll)),
		jnp.stack((seed_before_accuracy, seed_after_accuracy, seed_after_critical)),
		jnp.stack((seed_after_accuracy, seed_after_critical, seed_after_damage)),
	)


def _record_action_event(events: TurnEvents, slot, actor, action: _TackleActionResult) -> TurnEvents:
	return TurnEvents(
		events.executed.at[slot].set(jnp.asarray(True)),
		events.actors.at[slot].set(jnp.asarray(actor, dtype=jnp.int32)),
		events.critical.at[slot].set(action.critical),
		events.damage.at[slot].set(action.damage),
		events.target_hp.at[slot].set(action.target_hp),
		events.target_max_hp.at[slot].set(action.target_max_hp),
	)


def _run_turn(state: BattleState) -> Transition:
	first_actor = jnp.where(state.speed[0] > state.speed[1], 0, 1).astype(jnp.int32)
	events = _empty_events()
	draws = _empty_draws()

	def process_action(step, carry):
		current_state, current_events, current_draws = carry
		actor = jnp.where(step == 0, first_actor, 1 - first_actor).astype(jnp.int32)
		target = 1 - actor
		can_act = (current_state.hp[actor] > 0) & (current_state.hp[target] > 0)

		def execute(inner_carry):
			inner_state, inner_events, inner_draws = inner_carry
			action = _execute_tackle_action(inner_state, actor)
			new_events = _record_action_event(inner_events, step, actor, action)
			first_draw = step * jnp.asarray(3, dtype=jnp.int32)
			new_draws = inner_draws
			for offset, bound, context in ((0, 100, DRAW_ACCURACY), (1, 16, DRAW_CRITICAL), (2, 16, DRAW_DAMAGE)):
				new_draws = _record_draw(
					new_draws,
					first_draw + offset,
					bound,
					action.values[offset],
					context,
					action.before[offset],
					action.after[offset],
				)
			# Accuracy is exactly 100 for the pinned Tackle data, but its bounded
			# draw is still consumed and retained in the trace.
			_ = action.values[0] < 100
			return action.state, new_events, new_draws

		return jax.lax.cond(can_act, execute, lambda value: value, carry)

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
	return Transition(state, jnp.asarray(STATUS_OK, dtype=jnp.int32), events, draws)


def _choice_array(choices):
	if isinstance(choices, jax.core.Tracer):
		choice_array = choices
		if choice_array.dtype != jnp.int32:
			raise TypeError("compiled choices must have dtype int32.")
	else:
		raw_choices = np.asarray(choices)
		if raw_choices.dtype.kind not in "iu" or np.any(raw_choices < np.iinfo(np.int32).min) or np.any(raw_choices > np.iinfo(np.int32).max):
			raise TypeError("choices must be integral values that fit signed int32.")
		choice_array = jnp.asarray(raw_choices, dtype=jnp.int32)
	if choice_array.shape != (2,):
		raise ValueError("choices must be a length-two p1/p2 vector.")
	return choice_array


def _dispatch_transition(state, choice_array, supported_state, run_turn, empty_transition):
	def empty(status):
		return empty_transition(state, status)

	def terminal_result(_):
		return empty(STATUS_TERMINAL)

	def active_result(_):
		negative = jnp.any(choice_array < 0)
		unknown = jnp.any(choice_array > ACTION_TACKLE)
		pending_choice = jnp.any(choice_array == CHOICE_PENDING)
		def invalid_result(__):
			return empty(STATUS_INVALID_CHOICE)

		def unsupported_result(__):
			return empty(STATUS_UNSUPPORTED)

		def pending_result(__):
			return empty(STATUS_PENDING)

		def success_result(__):
			return run_turn(state)

		def success_or_unsupported(__):
			return jax.lax.cond(supported_state(state), success_result, unsupported_result, None)

		return jax.lax.cond(
			negative,
			invalid_result,
			lambda __: jax.lax.cond(
				unknown,
				unsupported_result,
				lambda ___: jax.lax.cond(
					pending_choice | (state.request_phase == PHASE_PENDING),
					pending_result,
					success_or_unsupported,
					None,
				),
				None,
			),
			None,
		)

	return jax.lax.cond(state.terminal, terminal_result, active_result, None)


def transition(state: BattleState, choices) -> Transition:
	"""Resolve both simultaneous Tackle choices with exact Gen 5 RNG.

	``choices`` is a two-element int32 vector ordered p1, p2. ``1`` means the
	sole legal ``move 1`` action, ``0`` means that side has not submitted yet,
	and negative values are malformed choices. Positive unknown action IDs are
	unsupported. No non-success result changes state or RNG. The result has
	fixed-size event and draw traces and is directly usable with ``jax.jit`` and
	``jax.vmap``.
	"""
	return _dispatch_transition(state, _choice_array(choices), _supported_state, _run_turn, _empty_transition)


def _project_view(before: BattleState, result, audience, supported_state) -> PlayerView:
	"""Common recipient projection; each public variant supplies its own boundary."""

	if isinstance(audience, jax.core.Tracer):
		raise TypeError("audience must be a static p1/p2 integer.")
	raw_audience = np.asarray(audience)
	if raw_audience.shape != () or raw_audience.dtype.kind not in "iu":
		raise TypeError("audience must be an integer p1/p2 identifier.")
	if int(raw_audience) not in (0, 1):
		raise ValueError("audience must be p1 (0) or p2 (1).")
	viewer = jnp.asarray(int(raw_audience), dtype=jnp.int32)
	own_hp = before.hp[viewer]
	own_max_hp = before.max_hp[viewer]
	legal = supported_state(before) & (before.hp[viewer] > 0) & (before.pp[viewer] > 0)
	actors = result.events.actors
	valid = result.events.executed
	targets = 1 - jnp.maximum(actors, 0)
	target_hp = result.events.target_hp
	target_max_hp = jnp.maximum(result.events.target_max_hp, 1)
	target_hp_before = before.hp[targets]
	is_owner = viewer == targets
	percent = (target_hp * 100 + target_max_hp - 1) // target_max_hp
	percent = jnp.where((percent == 100) & (target_hp < target_max_hp), 99, percent)
	visible_hp = jnp.where(is_owner, target_hp, percent).astype(jnp.int32)
	visible_hp = jnp.where(valid, visible_hp, 0)
	target_is_percent = valid & (~is_owner)
	damage_is_visible = valid & is_owner
	observed_hp_loss = jnp.minimum(target_hp_before, jnp.maximum(0, target_hp_before - target_hp))
	visible_damage = jnp.where(damage_is_visible, observed_hp_loss, -1).astype(jnp.int32)
	fainted = valid & (target_hp == 0)
	turn_event = (result.status == STATUS_OK) & (~result.state.terminal)
	return PlayerView(
		viewer,
		before.request_phase,
		jnp.asarray([legal], dtype=jnp.bool_),
		own_hp,
		own_max_hp,
		before.pp[viewer],
		before.level[viewer],
		before.attack[viewer],
		before.defense[viewer],
		before.speed[viewer],
		actors,
		result.events.critical & valid,
		visible_damage,
		damage_is_visible,
		jnp.where(valid, targets, -1).astype(jnp.int32),
		visible_hp,
		target_is_percent,
		fainted,
		turn_event,
		jnp.where(turn_event, result.state.turn, 0).astype(jnp.int32),
		result.state.terminal,
		result.state.winner,
	)


def project_view(before: BattleState, result: Transition, audience) -> PlayerView:
	"""Project a v1 turn to static audience p1 (0) or p2 (1).

	For compiled projection, pass ``audience`` as a static argument, for
	example ``jax.jit(project_view, static_argnums=2)``. Dynamic or unsupported
	audience values fail closed instead of indexing another side's private HP.
	"""
	return _project_view(before, result, audience, _supported_state)
