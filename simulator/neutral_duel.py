"""Bounded compiled duels for the restricted neutral Gen 5 Tackle variant.

``max_turns`` is a static trace capacity in ``[1, MAX_DUEL_TURNS]``. A
``turn_budget`` may be smaller and dynamic, which lets vmapped lanes stop at
different points without changing the trace shape. Only rows selected by the
successful-prefix ``trace.valid`` mask are executed turns.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .neutral_tackle import (
	BattleState,
	DrawTrace,
	PlayerView,
	Transition,
	TurnEvents,
	STATUS_OK,
	STATUS_TERMINAL,
	TACKLE_MAX_PP,
	project_view,
	transition,
)


MAX_DUEL_TURNS = TACKLE_MAX_PP

DISPOSITION_COMPLETED = 0
DISPOSITION_STOPPED = 1
DISPOSITION_BUDGET_EXHAUSTED = 2
DISPOSITION_INVALID_BUDGET = 3

DIAGNOSTIC_NONE = -1
DIAGNOSTIC_INVALID_BUDGET = -2

_DISPOSITION_RUNNING = 4
_STATE_LAYOUT = {
	"hp": ((2,), jnp.dtype(jnp.int32)),
	"max_hp": ((2,), jnp.dtype(jnp.int32)),
	"pp": ((2,), jnp.dtype(jnp.int32)),
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
	"move_id": ((), jnp.dtype(jnp.int32)),
}


class DuelTrace(NamedTuple):
	"""Fixed-capacity transition rows and both trusted recipient projections.

	The two views form one trusted combined trace. Actor input must select exactly
	one audience from a row whose ``valid`` mask is true.
	"""

	valid: jax.Array
	before: BattleState
	transition: Transition
	p1_view: PlayerView
	p2_view: PlayerView


class DuelResult(NamedTuple):
	"""Final trusted state and honest bounded-execution diagnostics."""

	state: BattleState
	disposition: jax.Array
	turn_count: jax.Array
	diagnostic_status: jax.Array
	trace: DuelTrace


def _validate_state_structure(state: BattleState) -> None:
	if not isinstance(state, BattleState):
		raise TypeError("initial_state must be a simulator.neutral_tackle.BattleState.")
	for name, (shape, dtype) in _STATE_LAYOUT.items():
		value = getattr(state, name)
		if not hasattr(value, "shape") or tuple(value.shape) != shape:
			raise ValueError(f"initial_state.{name} must have shape {shape}.")
		if jnp.dtype(value.dtype) != dtype:
			raise TypeError(f"initial_state.{name} must have dtype {dtype}.")


def _capacity(max_turns: int) -> int:
	if type(max_turns) is not int:
		raise TypeError("max_turns must be a static Python int.")
	if not 1 <= max_turns <= MAX_DUEL_TURNS:
		raise ValueError(f"max_turns must be in [1, {MAX_DUEL_TURNS}].")
	return max_turns


def _budget_array(turn_budget, capacity: int) -> jax.Array:
	if isinstance(turn_budget, jax.core.Tracer):
		if tuple(turn_budget.shape) != () or turn_budget.dtype != jnp.int32:
			raise TypeError("compiled turn_budget must be a scalar int32.")
		return turn_budget

	try:
		budget = np.asarray(turn_budget)
	except (TypeError, ValueError) as error:
		raise TypeError("turn_budget must be an integer scalar.") from error
	if budget.shape != () or budget.dtype.kind not in "iu":
		raise TypeError("turn_budget must be an integer scalar.")
	value = int(budget)
	if not np.iinfo(np.int32).min <= value <= np.iinfo(np.int32).max:
		raise ValueError("turn_budget must fit signed int32.")
	return jnp.asarray(value, dtype=jnp.int32)


def _padding_row(state: BattleState) -> DuelTrace:
	zero_state = jax.tree_util.tree_map(jnp.zeros_like, state)
	zero_events = TurnEvents(
		jnp.zeros((2,), dtype=jnp.bool_),
		jnp.zeros((2,), dtype=jnp.int32),
		jnp.zeros((2,), dtype=jnp.bool_),
		jnp.zeros((2,), dtype=jnp.int32),
		jnp.zeros((2,), dtype=jnp.int32),
		jnp.zeros((2,), dtype=jnp.int32),
	)
	zero_draws = DrawTrace(
		jnp.asarray(0, dtype=jnp.int32),
		jnp.zeros((6,), dtype=jnp.int32),
		jnp.zeros((6,), dtype=jnp.int32),
		jnp.zeros((6,), dtype=jnp.int32),
		jnp.zeros((6, 4), dtype=jnp.uint16),
		jnp.zeros((6, 4), dtype=jnp.uint16),
	)
	result = Transition(zero_state, jnp.asarray(STATUS_TERMINAL, dtype=jnp.int32), zero_events, zero_draws)
	zero_p1 = jax.tree_util.tree_map(jnp.zeros_like, project_view(zero_state, result, 0))
	zero_p2 = zero_p1._replace(audience=jnp.asarray(1, dtype=jnp.int32))
	return DuelTrace(jnp.asarray(False), zero_state, result, zero_p1, zero_p2)


def _run_duel_core(
	initial_state,
	*,
	max_turns,
	turn_budget,
	transition_fn,
	view_fn,
	padding_fn,
	trace_type,
	result_type,
):
	capacity = _capacity(max_turns)
	_validate_state_structure(initial_state)
	if turn_budget is None:
		turn_budget = capacity
	budget = _budget_array(turn_budget, capacity)
	padding = padding_fn(initial_state)

	budget_valid = (budget >= 0) & (budget <= capacity)
	initially_terminal = initial_state.terminal
	initial_disposition = jnp.where(
		budget_valid,
		jnp.where(initially_terminal, DISPOSITION_COMPLETED, _DISPOSITION_RUNNING),
		DISPOSITION_INVALID_BUDGET,
	).astype(jnp.int32)
	initial_diagnostic = jnp.where(
		budget_valid,
		jnp.where(initially_terminal, STATUS_TERMINAL, DIAGNOSTIC_NONE),
		DIAGNOSTIC_INVALID_BUDGET,
	).astype(jnp.int32)
	initial_active = budget_valid & (~initially_terminal)
	initial_carry = (
		initial_state,
		initial_disposition,
		jnp.asarray(0, dtype=jnp.int32),
		initial_diagnostic,
		initial_active,
	)

	def execute(carry):
		state, _, count, _, _ = carry
		result = transition_fn(state, jnp.asarray([1, 1], dtype=jnp.int32))
		succeeded = result.status == STATUS_OK
		finished = succeeded & result.state.terminal
		disposition = jnp.where(
			succeeded,
			jnp.where(finished, DISPOSITION_COMPLETED, _DISPOSITION_RUNNING),
			DISPOSITION_STOPPED,
		).astype(jnp.int32)
		diagnostic = jnp.where(succeeded, STATUS_OK, result.status).astype(jnp.int32)
		next_carry = (
			result.state,
			disposition,
			count + succeeded.astype(jnp.int32),
			diagnostic,
			succeeded & (~result.state.terminal),
		)

		def successful_row(_):
			return trace_type(
				jnp.asarray(True),
				state,
				result,
				view_fn(state, result, 0),
				view_fn(state, result, 1),
			)

		row = jax.lax.cond(succeeded, successful_row, lambda _: padding, operand=None)
		return next_carry, row

	def scan_step(carry, slot):
		should_execute = carry[4] & (slot < budget)
		return jax.lax.cond(should_execute, execute, lambda current: (current, padding), carry)

	(final_state, disposition, turn_count, diagnostic_status, _), trace = jax.lax.scan(
		scan_step,
		initial_carry,
		jnp.arange(capacity, dtype=jnp.int32),
	)
	disposition = jnp.where(
		disposition == _DISPOSITION_RUNNING,
		DISPOSITION_BUDGET_EXHAUSTED,
		disposition,
	).astype(jnp.int32)
	return result_type(final_state, disposition, turn_count, diagnostic_status, trace)


def run_duel(initial_state: BattleState, *, max_turns: int, turn_budget=None) -> DuelResult:
	"""Execute up to a static capacity of simultaneous sole-move Tackle turns.

	The capacity is limited to the accepted move's maximum PP (56). The optional
	dynamic budget defaults to that capacity and can be vmapped across lanes; zero
	returns an immediate budget-exhausted result. A successful prefix has shape
	``(max_turns,)``. Each valid row contains the trusted before/after transition,
	fixed draw/event trace, and both recipient-specific projections. The returned
	views are combined trusted output, not a joint actor observation: select only
	p1 or p2 for a valid row.

	An initially terminal state completes with no rows and diagnostic
	``STATUS_TERMINAL``. A non-success transition stops at its unchanged state and
	preserves that transition status. If the capacity or dynamic budget is spent
	while nonterminal, the result is ``DISPOSITION_BUDGET_EXHAUSTED``; it never
	invents a winner, loss, draw, or padded experience. Resume by passing the
	returned state to another call and concatenating only valid trace rows.

	Compile with ``jax.jit(run_duel, static_argnames=("max_turns",))``. ``max_turns``
	must be a Python integer from 1 through 56. ``turn_budget`` is an integer scalar
	from zero through the capacity; batched values must be int32 scalars per lane.
	"""
	return _run_duel_core(
		initial_state,
		max_turns=max_turns,
		turn_budget=turn_budget,
		transition_fn=transition,
		view_fn=project_view,
		padding_fn=_padding_row,
		trace_type=DuelTrace,
		result_type=DuelResult,
	)
