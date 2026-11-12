"""Bounded exact rollouts with opt-in trajectory retention.

This is a trusted environment API, not a policy observation. Battle mechanics,
current history and capacity checks remain in shared_engine.advance. By default
scan returns no per-step arrays; only the current state and accounting survive.
"""
from __future__ import annotations

import operator
from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import shared_engine as e

SKIPPED = -1
OUTPUTS = frozenset(("states", "events", "draws", "rewards", "legal_actions", "codes", "trace"))


class EventOutput(NamedTuple):
    events: jax.Array
    count: jax.Array


class DrawOutput(NamedTuple):
    args: jax.Array
    values: jax.Array
    context: jax.Array
    before: jax.Array
    after: jax.Array
    count: jax.Array


class RolloutResult(NamedTuple):
    state: e.State
    code: jax.Array
    attempts: jax.Array
    transitions: jax.Array
    failed: jax.Array
    failure_step: jax.Array
    terminal: jax.Array
    truncated: jax.Array
    outputs: dict
    samples: e.State | None
    sample_steps: jax.Array
    sample_valid: jax.Array


def rollout(state, choices, *, max_steps=None, tokens=None, enabled=None,
            retain=(), sample_steps=()):
    """Apply one battle's int32[T,2] choice plan, stopping per lane on failure/end.

    Use ``jax.jit`` and ``jax.vmap`` for compilation and independent lanes.
    ``max_steps``, ``retain`` and ``sample_steps`` are static Python options.
    Tokens are optional int32[T,2] generation/episode pairs; enabled is bool[T].
    Each enabled row is one submission, potentially only a private commitment.
    Attempts count submitted rows; transitions count only engine OK results.

    Rejection preserves the successful prefix, latches its first error and
    zero-based plan index, and prevents later choices from hiding the failure.
    A live state at the plan/budget boundary is truncated, never a fabricated
    terminal. Call again with the returned state to continue a truncated plan.

    Retained events/draws/states are trusted data. The outputs' ``valid`` mask
    distinguishes attempted rows from disabled/stopped rows (codes SKIPPED).
    State samples are post-submission snapshots, stored only at requested plan
    indices; sample_valid must be checked before consuming a sample.
    """
    e._validate_structure(state)
    choices = jnp.asarray(choices)
    if choices.ndim != 2 or choices.shape[1] != 2 or choices.dtype != jnp.int32:
        raise ValueError("choices must be int32[T,2]")
    length = choices.shape[0]
    if max_steps is None:
        max_steps = length
    if type(max_steps) is not int or not 0 <= max_steps <= length:
        raise ValueError("max_steps must be a static integer in [0,T]")
    if not isinstance(retain, tuple) or any(name not in OUTPUTS for name in retain):
        raise ValueError("retain must be a static tuple of supported output names")
    if len(set(retain)) != len(retain):
        raise ValueError("retain entries must be distinct")
    if (not isinstance(sample_steps, tuple)
            or any(type(k) is not int or not 0 <= k < max_steps for k in sample_steps)
            or len(set(sample_steps)) != len(sample_steps)):
        raise ValueError("sample_steps must be distinct static plan indices below max_steps")
    if tokens is not None:
        tokens = jnp.asarray(tokens)
        if tokens.shape != (length, 2) or tokens.dtype != jnp.int32:
            raise ValueError("tokens must be int32[T,2]")
    if enabled is None:
        enabled = jnp.ones((length,), jnp.bool_)
    else:
        enabled = jnp.asarray(enabled)
        if enabled.shape != (length,) or enabled.dtype != jnp.bool_:
            raise ValueError("enabled must be bool[T]")

    indices = e.i32(sample_steps)
    samples = (jax.tree.map(lambda x: jnp.zeros((len(sample_steps),) + x.shape, x.dtype), state)
               if sample_steps else None)
    sample_valid = jnp.zeros((len(sample_steps),), jnp.bool_)
    initial = (state, jnp.where(state.winner >= 0, e.TERMINAL, e.PENDING).astype(jnp.int32),
               e.i32(0), e.i32(0), jnp.array(False), e.i32(-1), samples, sample_valid)

    def step(carry, row):
        current, code, attempts, transitions, failed, failure_step, saved, valid = carry
        index, choice, allowed, token = row
        active = allowed & (index < max_steps) & (~failed) & (current.winner < 0)

        def advance(_):
            return e.advance(current, choice, token if tokens is not None else None)

        result = jax.lax.cond(active, advance,
                              lambda _: e.Result(current, e.i32(SKIPPED), e.empty_trace()), None)
        bad = active & (result.code != e.OK) & (result.code != e.PENDING) & (result.code != e.TERMINAL) & (result.code != e.RETRY)
        code = jnp.where(active, result.code, code)
        failure_step = jnp.where(bad & (~failed), index, failure_step)
        failed = failed | bad
        attempts = attempts + active.astype(jnp.int32)
        transitions = transitions + (active & (result.code == e.OK)).astype(jnp.int32)

        if sample_steps:
            matches = indices == index
            slot = jnp.argmax(matches)

            def save(pair):
                saved, valid = pair
                saved = jax.tree.map(lambda buf, leaf: buf.at[slot].set(leaf), saved, result.state)
                return saved, valid.at[slot].set(True)

            saved, valid = jax.lax.cond(active & jnp.any(matches), save,
                                       lambda pair: pair, (saved, valid))

        out = {}
        if retain:
            out["valid"] = active
        for name in retain:
            if name == "states":
                out[name] = result.state
            elif name == "events":
                out[name] = EventOutput(result.trace.events, result.trace.ecount)
            elif name == "draws":
                t = result.trace
                out[name] = DrawOutput(t.args, t.values, t.context, t.before, t.after, t.dcount)
            elif name == "rewards":
                out[name] = jnp.where(active, e.rewards(result.state), jnp.zeros(2, jnp.int32))
            elif name == "legal_actions":
                out[name] = jnp.where(active, e.legal_actions(result.state), jnp.zeros((2, 11), jnp.bool_))
            elif name == "codes":
                out[name] = result.code
            elif name == "trace":
                out[name] = result.trace
        return (result.state, code, attempts, transitions, failed, failure_step, saved, valid), out

    # A zero-length plan still has a well-defined result and output shapes.
    token_rows = tokens if tokens is not None else jnp.zeros((length, 2), jnp.int32)
    carry, outputs = jax.lax.scan(step, initial,
                                   (jnp.arange(max_steps, dtype=jnp.int32), choices[:max_steps],
                                    enabled[:max_steps], token_rows[:max_steps]))
    final, code, attempts, transitions, failed, failure_step, samples, sample_valid = carry
    terminal = final.winner >= 0
    return RolloutResult(final, code, attempts, transitions, failed, failure_step,
                         terminal, (~terminal) & (~failed), outputs, samples, indices, sample_valid)


def player_view(result, audience):
    """Host-only recipient-safe view of the final state and valid state samples.

    Raw rollout outputs remain trusted engineering data, never actor inputs.
    This adapter uses the existing visibility projection, including private
    requests and public HP formatting; it exposes no RNG or pending choices.
    """
    from .shared_engine_io import player_view as project

    if isinstance(audience, bool):
        raise ValueError("audience must be player index 0 or 1")
    try:
        audience = operator.index(audience)
    except TypeError as error:
        raise ValueError("audience must be player index 0 or 1") from error
    if audience not in (0, 1):
        raise ValueError("audience must be player index 0 or 1")
    view = {"final": project(result.state, audience), "samples": []}
    if result.samples is not None:
        for k, valid in enumerate(jax.device_get(result.sample_valid)):
            if valid:
                state = jax.tree.map(lambda x: x[k], result.samples)
                view["samples"].append({"step": int(result.sample_steps[k]), "view": project(state, audience)})
    return view
