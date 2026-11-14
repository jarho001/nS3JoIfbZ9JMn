"""Pinned raw stat selection; source-side modifiers and category stay separate."""
import jax.numpy as jnp
from . import gen5_engine_tables as t
TARGET_ATTACK=jnp.asarray([key=='foulplay' for key in t.MOVE_KEYS],jnp.bool_)
TARGET_DEF=jnp.asarray([key in ('psyshock','psystrike','secretsword') for key in t.MOVE_KEYS],jnp.bool_)
