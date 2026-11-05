"""Private staged requests for the restricted v4 Tackle/Pound turn API."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .neutral_tackle import STATUS_OK, STATUS_PENDING, STATUS_TERMINAL, STATUS_UNSUPPORTED
from .neutral_tackle_ties import _empty_tie_transition
from .neutral_two_move import (
	ACTION_POUND,
	ACTION_TACKLE,
	INVALID_MOVE_ID,
	TwoMovePlayerView,
	TwoMoveState,
	TwoMoveTransition,
	_adapt_state,
	_supported_two_move_state,
	_validate_two_move_state_structure,
	project_two_move_view,
	transition_two_move,
)
from .request_session import (
	MAX_REQUEST_EPISODE,
	SESSION_DUPLICATE,
	SESSION_INVALID_ACTION,
	SESSION_INVALID_PLAYER,
	SESSION_NOT_TERMINAL,
	SESSION_OVERFLOW,
	SESSION_RESOLVED,
	SESSION_RESET,
	SESSION_STALE_EPISODE,
	SESSION_STOPPED,
	SESSION_TERMINAL,
	SESSION_UNSUPPORTED,
	SESSION_WAITING,
	_integral_scalar,
)


class TwoMoveRequestSession(NamedTuple):
	"""Trusted fixed-shape session; choices are ordered p1, p2."""

	battle: TwoMoveState
	pending_choices: jax.Array  # int32[2], zero until that player's commit.
	episode: jax.Array  # int32 scalar; local session version, not native rqid.


class TwoMoveSessionSubmission(NamedTuple):
	"""Coordinator result with trusted state and the resolved prior turn."""

	session: TwoMoveRequestSession
	status: jax.Array
	before: TwoMoveState
	transition: TwoMoveTransition


class TwoMoveSessionReset(NamedTuple):
	"""Trusted coordinator result for a terminal v4-session reset."""

	session: TwoMoveRequestSession
	status: jax.Array


class TwoMoveRequestView(NamedTuple):
	"""One actor's current v4 request and only their own commitment bit."""

	player_view: TwoMovePlayerView
	episode: jax.Array
	own_choice_submitted: jax.Array


class TwoMoveSubmissionView(NamedTuple):
	"""Current request plus a separate projection of the prior resolved turn."""

	player_view: TwoMovePlayerView
	resolved_events: TwoMovePlayerView
	episode: jax.Array
	own_choice_submitted: jax.Array


def _validate_two_move_session_structure(session: TwoMoveRequestSession) -> None:
	if not isinstance(session, TwoMoveRequestSession):
		raise TypeError("session must be a TwoMoveRequestSession.")
	_validate_two_move_state_structure(session.battle)
	if not hasattr(session.pending_choices, "shape") or tuple(session.pending_choices.shape) != (2,):
		raise ValueError("session.pending_choices must have shape (2,).")
	if jnp.dtype(session.pending_choices.dtype) != jnp.int32:
		raise TypeError("session.pending_choices must have dtype int32.")
	if not hasattr(session.episode, "shape") or tuple(session.episode.shape) != ():
		raise ValueError("session.episode must be a scalar.")
	if jnp.dtype(session.episode.dtype) != jnp.int32:
		raise TypeError("session.episode must have dtype int32.")


def make_two_move_request_session(battle: TwoMoveState, *, episode: int = 0) -> TwoMoveRequestSession:
	"""Create an empty v4 request session with a bounded local episode."""
	_validate_two_move_state_structure(battle)
	raw = np.asarray(episode)
	if raw.shape != () or raw.dtype.kind not in "iu" or isinstance(episode, (bool, np.bool_)):
		raise TypeError("episode must be an integral scalar.")
	value = int(raw)
	if not 0 <= value <= MAX_REQUEST_EPISODE:
		raise ValueError(f"episode must be in [0, {MAX_REQUEST_EPISODE}].")
	return TwoMoveRequestSession(
		battle,
		jnp.zeros((2,), dtype=jnp.int32),
		jnp.asarray(value, dtype=jnp.int32),
	)


def reset_two_move_request_session(
	session: TwoMoveRequestSession,
	battle: TwoMoveState,
	expected_episode: int | jax.Array,
) -> TwoMoveSessionReset:
	"""Reset a terminal v4 session onto a trusted supplied request state.

	The replacement must be a supported nonterminal v4 state. The old session
	must be structurally valid, terminal, have no pending choices, and carry the
	matching nonnegative local episode. A held choice must still have PP in its
	selected slot. Rejections preserve the entire old session; an accepted reset
	copies ``battle`` exactly, clears commitments, and increments the old local
	episode once. No turn transition or RNG draw occurs.

	Runtime precedence is stale episode, unsupported session metadata or
	replacement, old session not terminal, episode overflow, then reset.
	Compile with ``jax.jit(reset_two_move_request_session)`` and
	``jax.jit(jax.vmap(reset_two_move_request_session))``.
	"""
	_validate_two_move_session_structure(session)
	_validate_two_move_state_structure(battle)
	episode_id = _integral_scalar("expected_episode", expected_episode)
	old_battle = session.battle
	choices = session.pending_choices

	def rejected(status):
		return TwoMoveSessionReset(session, jnp.asarray(status, dtype=jnp.int32))

	def check_metadata(_):
		valid_pending = jnp.all((choices == 0) | (choices == ACTION_TACKLE) | (choices == ACTION_POUND))
		pending_slots = jnp.clip(choices - 1, 0, 1)
		pending_pp = old_battle.pp[jnp.arange(2), pending_slots]
		valid_held_pp = jnp.all((choices == 0) | (pending_pp > 0))
		valid_episode = session.episode >= 0
		terminal_commitments_clear = (~old_battle.terminal) | jnp.all(choices == 0)
		replacement_supported = _supported_two_move_state(battle, _adapt_state(battle))

		def check_terminal(__):
			def overflow(___):
				return rejected(SESSION_OVERFLOW)

			def accept(___):
				return TwoMoveSessionReset(
					session._replace(
						battle=battle,
						pending_choices=jnp.zeros((2,), dtype=jnp.int32),
						episode=session.episode + jnp.asarray(1, dtype=jnp.int32),
					),
					jnp.asarray(SESSION_RESET, dtype=jnp.int32),
				)

			return jax.lax.cond(
				session.episode == MAX_REQUEST_EPISODE,
				overflow,
				accept,
				operand=None,
			)

		def unsupported(__):
			return rejected(SESSION_UNSUPPORTED)

		def supported(__):
			return jax.lax.cond(
				old_battle.terminal,
				check_terminal,
				lambda ___: rejected(SESSION_NOT_TERMINAL),
				operand=None,
			)

		return jax.lax.cond(
			valid_pending & valid_held_pp & valid_episode & terminal_commitments_clear & replacement_supported,
			supported,
			unsupported,
			operand=None,
		)

	return jax.lax.cond(
		episode_id == session.episode,
		check_metadata,
		lambda _: rejected(SESSION_STALE_EPISODE),
		operand=None,
	)


def _idle_two_move_transition(battle: TwoMoveState) -> TwoMoveTransition:
	"""Synthesize a zero-trace v4 result for a request with no new turn."""
	internal = _adapt_state(battle)
	status = jnp.where(
		battle.terminal,
		jnp.asarray(STATUS_TERMINAL, dtype=jnp.int32),
		jnp.where(
			_supported_two_move_state(battle, internal),
			jnp.asarray(STATUS_PENDING, dtype=jnp.int32),
			jnp.asarray(STATUS_UNSUPPORTED, dtype=jnp.int32),
		),
	).astype(jnp.int32)
	result = _empty_tie_transition(internal, status)
	return TwoMoveTransition(
		battle,
		result.status,
		result.events,
		result.draws,
		jnp.full((2,), INVALID_MOVE_ID, dtype=jnp.int32),
	)


def project_two_move_request_view(session: TwoMoveRequestSession, audience: int) -> TwoMoveRequestView:
	"""Project a fresh or current recipient request without running a turn.

	The view contains no full session or opponent commitment. Compile with
	``jax.jit(project_two_move_request_view, static_argnums=1)``.
	"""
	_validate_two_move_session_structure(session)
	player_view = project_two_move_view(session.battle, _idle_two_move_transition(session.battle), audience)
	return TwoMoveRequestView(
		player_view,
		session.episode,
		session.pending_choices[player_view.audience] != 0,
	)


def submit_two_move_choice(
	session: TwoMoveRequestSession,
	player: int | jax.Array,
	choice: int | jax.Array,
	expected_episode: int | jax.Array,
) -> TwoMoveSessionSubmission:
	"""Privately stage a move or resolve the accepted simultaneous pair once.

	Runtime precedence is stale episode, invalid player, invalid action,
	terminal battle, unsupported battle/session metadata, episode overflow,
	duplicate commitment, unavailable selected PP, then new commitment.
	Only an accepted second commitment invokes ``transition_two_move``. Stale
	and duplicate rejection is this local session contract; native
	``Side.choose`` may replace a prior choice.
	"""
	_validate_two_move_session_structure(session)
	player_id = _integral_scalar("player", player)
	action_id = _integral_scalar("choice", choice)
	episode_id = _integral_scalar("expected_episode", expected_episode)
	battle = session.battle
	idle_transition = _idle_two_move_transition(battle)

	def rejected(status, transition=None):
		if transition is None:
			transition = idle_transition
		return TwoMoveSessionSubmission(session, jnp.asarray(status, dtype=jnp.int32), battle, transition)

	def process_commit(_):
		choices = session.pending_choices
		valid_pending = jnp.all((choices == 0) | (choices == ACTION_TACKLE) | (choices == ACTION_POUND))
		pending_slots = jnp.clip(choices - 1, 0, 1)
		pending_pp = battle.pp[jnp.arange(2), pending_slots]
		valid_held_pp = jnp.all((choices == 0) | (pending_pp > 0))
		valid_metadata = valid_pending & valid_held_pp & (session.episode >= 0)
		internal = _adapt_state(battle)
		valid_battle = _supported_two_move_state(battle, internal)

		def unsupported(_):
			return rejected(SESSION_UNSUPPORTED)

		def supported(_):
			index = jnp.clip(player_id, 0, 1)
			already = choices[index] != 0

			def duplicate(__):
				return rejected(SESSION_DUPLICATE)

			def check_selected_pp(__):
				selected_pp = battle.pp[index, jnp.clip(action_id - 1, 0, 1)]

				def unavailable(___):
					return rejected(SESSION_INVALID_ACTION)

				def accept(___):
					new_choices = choices.at[index].set(action_id)

					def wait(____):
						return TwoMoveSessionSubmission(
							session._replace(pending_choices=new_choices),
							jnp.asarray(SESSION_WAITING, dtype=jnp.int32),
							battle,
							TwoMoveTransition(
								idle_transition.state,
								idle_transition.status,
								idle_transition.events,
								idle_transition.draws,
								idle_transition.event_move_ids,
							),
						)

					def resolve(____):
						turn = transition_two_move(battle, new_choices)

						def completed(_____):
							next_session = session._replace(
								battle=turn.state,
								pending_choices=jnp.zeros((2,), dtype=jnp.int32),
								episode=session.episode + jnp.asarray(1, dtype=jnp.int32),
							)
							return TwoMoveSessionSubmission(
								next_session,
								jnp.asarray(SESSION_RESOLVED, dtype=jnp.int32),
								battle,
								turn,
							)

						def stopped(_____):
							return rejected(SESSION_STOPPED, turn)

						return jax.lax.cond(turn.status == STATUS_OK, completed, stopped, operand=None)

					return jax.lax.cond(choices[1 - index] != 0, resolve, wait, operand=None)

				return jax.lax.cond(selected_pp > 0, accept, unavailable, operand=None)

			def available_episode(__):
				return jax.lax.cond(already, duplicate, check_selected_pp, operand=None)

			return jax.lax.cond(
				session.episode == MAX_REQUEST_EPISODE,
				lambda __: rejected(SESSION_OVERFLOW),
				available_episode,
				operand=None,
			)

		return jax.lax.cond(valid_metadata & valid_battle, supported, unsupported, operand=None)

	def check_supported(_):
		return jax.lax.cond(
			battle.terminal,
			lambda __: rejected(SESSION_TERMINAL),
			process_commit,
			operand=None,
		)

	def check_action(_):
		return jax.lax.cond(
			(action_id == ACTION_TACKLE) | (action_id == ACTION_POUND),
			check_supported,
			lambda __: rejected(SESSION_INVALID_ACTION),
			operand=None,
		)

	def check_player(_):
		return jax.lax.cond(
			(player_id == 0) | (player_id == 1),
			check_action,
			lambda __: rejected(SESSION_INVALID_PLAYER),
			operand=None,
		)

	return jax.lax.cond(
		episode_id == session.episode,
		check_player,
		lambda _: rejected(SESSION_STALE_EPISODE),
		operand=None,
	)


def project_two_move_submission_view(
	result: TwoMoveSessionSubmission,
	audience: int,
) -> TwoMoveSubmissionView:
	"""Project current request separately from recipient-filtered prior events.

	Compile with ``jax.jit(project_two_move_submission_view, static_argnums=1)``.
	"""
	if not isinstance(result, TwoMoveSessionSubmission):
		raise TypeError("result must be a TwoMoveSessionSubmission.")
	request_view = project_two_move_request_view(result.session, audience)
	resolved_events = project_two_move_view(result.before, result.transition, audience)
	return TwoMoveSubmissionView(
		request_view.player_view,
		resolved_events,
		request_view.episode,
		request_view.own_choice_submitted,
	)


__all__ = [
	"TwoMoveRequestSession",
	"TwoMoveSessionReset",
	"TwoMoveRequestView",
	"TwoMoveSessionSubmission",
	"TwoMoveSubmissionView",
	"make_two_move_request_session",
	"reset_two_move_request_session",
	"project_two_move_request_view",
	"project_two_move_submission_view",
	"submit_two_move_choice",
]
