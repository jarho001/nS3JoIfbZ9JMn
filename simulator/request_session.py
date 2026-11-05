"""Staged private choices for one restricted v2 JAX turn."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .neutral_duel import _validate_state_structure
from .neutral_tackle import ACTION_TACKLE, BattleState, PlayerView
from .neutral_tackle_ties import (
	STATUS_PENDING as TURN_PENDING,
	STATUS_OK as TURN_OK,
	STATUS_TERMINAL as TURN_TERMINAL,
	STATUS_UNSUPPORTED as TURN_UNSUPPORTED,
	_empty_tie_transition,
	TieTransition,
	_supported_tie_state,
	project_tie_view,
	transition_with_speed_ties,
)

MAX_REQUEST_EPISODE = 0x7FFFFFFF
SESSION_WAITING = 0
SESSION_RESOLVED = 1
SESSION_STALE_EPISODE = 2
SESSION_DUPLICATE = 3
SESSION_INVALID_PLAYER = 4
SESSION_INVALID_ACTION = 5
SESSION_TERMINAL = 6
SESSION_UNSUPPORTED = 7
SESSION_OVERFLOW = 8
SESSION_STOPPED = 9
SESSION_RESET = 10
SESSION_NOT_TERMINAL = 11


class RequestSession(NamedTuple):
	"""Trusted fixed-shape session; pending choices are ordered p1, p2."""

	battle: BattleState
	pending_choices: jax.Array  # int32[2], zero until that player's commit.
	episode: jax.Array  # int32 scalar; local session version, not Showdown rqid.


class SessionSubmission(NamedTuple):
	"""Trusted coordinator result. Project a player view before actor input."""

	session: RequestSession
	status: jax.Array
	before: BattleState
	transition: TieTransition


class SessionReset(NamedTuple):
	"""Trusted coordinator result for a terminal-session reset."""

	session: RequestSession
	status: jax.Array


class SessionView(NamedTuple):
	"""Current actor request plus a separate recipient projection of prior events."""

	player_view: PlayerView
	resolved_events: PlayerView
	episode: jax.Array
	own_choice_submitted: jax.Array


class SessionRequestView(NamedTuple):
	"""Read-only view of one player's current request in a request session."""

	player_view: PlayerView
	episode: jax.Array
	own_choice_submitted: jax.Array


def _integral_scalar(name: str, value) -> jax.Array:
	if isinstance(value, jax.core.Tracer):
		if value.shape != ():
			raise ValueError(f"{name} must be a scalar.")
		if jnp.dtype(value.dtype) != jnp.int32:
			raise TypeError(f"compiled {name} must have dtype int32.")
		return value
	raw = np.asarray(value)
	if raw.shape != ():
		raise ValueError(f"{name} must be a scalar.")
	if raw.dtype.kind not in "iu" or raw.item() < np.iinfo(np.int32).min or raw.item() > np.iinfo(np.int32).max:
		raise TypeError(f"{name} must be an integer fitting signed int32.")
	return jnp.asarray(raw, dtype=jnp.int32)


def _validate_session_structure(session: RequestSession) -> None:
	if not isinstance(session, RequestSession):
		raise TypeError("session must be a RequestSession.")
	_validate_state_structure(session.battle)
	if not hasattr(session.pending_choices, "shape") or tuple(session.pending_choices.shape) != (2,):
		raise ValueError("session.pending_choices must have shape (2,).")
	if jnp.dtype(session.pending_choices.dtype) != jnp.int32:
		raise TypeError("session.pending_choices must have dtype int32.")
	if not hasattr(session.episode, "shape") or tuple(session.episode.shape) != ():
		raise ValueError("session.episode must be a scalar.")
	if jnp.dtype(session.episode.dtype) != jnp.int32:
		raise TypeError("session.episode must have dtype int32.")


def make_request_session(battle: BattleState, *, episode: int = 0) -> RequestSession:
	"""Start an empty session; episode is local int32, bounded and not an rqid."""
	_validate_state_structure(battle)
	raw = np.asarray(episode)
	if raw.shape != () or raw.dtype.kind not in "iu" or isinstance(episode, (bool, np.bool_)):
		raise TypeError("episode must be an integral scalar.")
	value = int(raw)
	if not 0 <= value <= MAX_REQUEST_EPISODE:
		raise ValueError(f"episode must be in [0, {MAX_REQUEST_EPISODE}].")
	return RequestSession(
		battle,
		jnp.zeros((2,), dtype=jnp.int32),
		jnp.asarray(value, dtype=jnp.int32),
	)


def _idle_transition(battle: BattleState) -> TieTransition:
	status = jnp.where(
		battle.terminal,
		TURN_TERMINAL,
		jnp.where(_supported_tie_state(battle), TURN_PENDING, TURN_UNSUPPORTED),
	).astype(jnp.int32)
	return _empty_tie_transition(battle, status)


def project_request_view(session: RequestSession, audience: int) -> SessionRequestView:
	"""Project one player's current request without advancing the session.

	The view is available immediately after ``make_request_session``. It contains
	only the recipient's projected request, local episode and own commit bit.
	Compile with ``jax.jit(project_request_view, static_argnums=1)``.
	"""
	_validate_session_structure(session)
	player_view = project_tie_view(session.battle, _idle_transition(session.battle), audience)
	return SessionRequestView(
		player_view,
		session.episode,
		session.pending_choices[player_view.audience] == ACTION_TACKLE,
	)


def submit_choice(session: RequestSession, player, choice, expected_episode) -> SessionSubmission:
	"""Stage one Tackle commit or resolve the simultaneous pair exactly once.

	Malformed structures and non-int32 compiled scalar inputs fail before
	simulation. Runtime precedence is stale token, player, action, terminal,
	unsupported state, episode overflow, then duplicate commit. A first accepted
	choice updates trusted pending metadata only; no battle field, turn or RNG
	changes. The second accepted choice calls the existing v2 transition once,
	clears both choices and advances the local episode. Duplicate/stale rejection
	is this adapter's policy; native Side.choose can replace a prior choice, but
	choice replacement is outside this single-commit interface.
	"""
	_validate_session_structure(session)
	player_id = _integral_scalar("player", player)
	action_id = _integral_scalar("choice", choice)
	episode_id = _integral_scalar("expected_episode", expected_episode)
	battle = session.battle

	def rejected(status, turn=None):
		if turn is None:
			turn = _idle_transition(battle)
		return SessionSubmission(session, jnp.asarray(status, jnp.int32), battle, turn)

	def process_commit(_):
		valid_pending = jnp.all((session.pending_choices == 0) | (session.pending_choices == ACTION_TACKLE))
		valid_episode = session.episode >= 0

		def malformed_pending(_):
			return rejected(SESSION_UNSUPPORTED)

		def well_formed_pending(_):
			index = jnp.clip(player_id, 0, 1)
			other = 1 - index
			already = session.pending_choices[index] != 0
			other_committed = session.pending_choices[other] == ACTION_TACKLE

			def duplicate(_):
				return rejected(SESSION_DUPLICATE)

			def new_commit(_):
				new_pending = session.pending_choices.at[index].set(action_id)

				def wait(__):
					pending_session = session._replace(pending_choices=new_pending)
					turn = _empty_tie_transition(battle, TURN_PENDING)
					return SessionSubmission(
						pending_session,
						jnp.asarray(SESSION_WAITING, jnp.int32),
						battle,
						turn,
					)

				def resolve(__):
					turn = transition_with_speed_ties(battle, new_pending)

					def completed(___):
						next_session = session._replace(
							battle=turn.state,
							pending_choices=jnp.zeros((2,), jnp.int32),
							episode=session.episode + jnp.asarray(1, jnp.int32),
						)
						return SessionSubmission(
							next_session,
							jnp.asarray(SESSION_RESOLVED, jnp.int32),
							battle,
							turn,
						)

					def stopped(___):
						return SessionSubmission(
							session,
							jnp.asarray(SESSION_STOPPED, jnp.int32),
							battle,
							turn,
						)

					return jax.lax.cond(turn.status == TURN_OK, completed, stopped, operand=None)

				return jax.lax.cond(other_committed, resolve, wait, operand=None)

			def available_episode(__):
				return jax.lax.cond(already, duplicate, new_commit, operand=None)

			return jax.lax.cond(
				session.episode == MAX_REQUEST_EPISODE,
				lambda __: rejected(SESSION_OVERFLOW),
				available_episode,
				operand=None,
			)

		return jax.lax.cond(
			valid_pending & valid_episode,
			well_formed_pending,
			malformed_pending,
			operand=None,
		)

	def check_supported(_):
		return jax.lax.cond(
			battle.terminal,
			lambda __: rejected(SESSION_TERMINAL, _empty_tie_transition(battle, TURN_TERMINAL)),
			lambda __: jax.lax.cond(
				_supported_tie_state(battle),
				process_commit,
				lambda ___: rejected(SESSION_UNSUPPORTED, _empty_tie_transition(battle, TURN_UNSUPPORTED)),
				operand=None,
			),
			operand=None,
		)

	def check_action(_):
		return jax.lax.cond(
			action_id == ACTION_TACKLE,
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


def reset_request_session(session: RequestSession, battle: BattleState, expected_episode) -> SessionReset:
	"""Start a new local request episode from a trusted prepared battle.

	The replacement must be a supported nonterminal restricted-v2 request
	state. The old session must be structurally valid, terminal, have no pending
	choices, and carry the matching local episode. A terminal session with a
	held choice is unsupported; a valid live session returns
	SESSION_NOT_TERMINAL. The replacement battle is copied verbatim, pending
	choices are cleared, and the local episode advances once. Rejections retain
	the full input session and do not transition or draw RNG.
	"""
	_validate_session_structure(session)
	_validate_state_structure(battle)
	episode_id = _integral_scalar("expected_episode", expected_episode)

	def rejected(status):
		return SessionReset(session, jnp.asarray(status, jnp.int32))

	def check_reset_metadata(_):
		valid_pending = jnp.all((session.pending_choices == 0) | (session.pending_choices == ACTION_TACKLE))
		valid_episode = session.episode >= 0
		terminal_commitments_clear = (~session.battle.terminal) | jnp.all(session.pending_choices == 0)
		supported_replacement = _supported_tie_state(battle)

		def check_terminal(__):
			return jax.lax.cond(
				session.battle.terminal,
				lambda ___: jax.lax.cond(
					session.episode == MAX_REQUEST_EPISODE,
					lambda ____: rejected(SESSION_OVERFLOW),
					lambda ____: SessionReset(
						session._replace(
							battle=battle,
							pending_choices=jnp.zeros((2,), dtype=jnp.int32),
							episode=session.episode + jnp.asarray(1, jnp.int32),
						),
						jnp.asarray(SESSION_RESET, jnp.int32),
					),
					operand=None,
				),
				lambda ___: rejected(SESSION_NOT_TERMINAL),
				operand=None,
			)

		return jax.lax.cond(
			valid_pending & valid_episode & terminal_commitments_clear & supported_replacement,
			check_terminal,
			lambda __: rejected(SESSION_UNSUPPORTED),
			operand=None,
		)

	return jax.lax.cond(
		episode_id == session.episode,
		check_reset_metadata,
		lambda _: rejected(SESSION_STALE_EPISODE),
		operand=None,
	)


def project_submission_view(result: SessionSubmission, audience: int) -> SessionView:
	"""Project the next request and this result's recipient-filtered events.

	Compile with ``jax.jit(project_submission_view, static_argnums=1)``.
	"""
	request_view = project_request_view(result.session, audience)
	resolved_events = project_tie_view(result.before, result.transition, audience)
	return SessionView(
		request_view.player_view,
		resolved_events,
		request_view.episode,
		request_view.own_choice_submitted,
	)
