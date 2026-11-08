"""Gen 3–5 Hidden Power type index/power from stored-order IVs, in JAX."""
import jax.numpy as jnp
from .stat_stage import _as_int32


def hidden_power(ivs):
    """Return type index 0..15 and power 30..70; IV final axis hp/atk/def/spa/spd/spe.

    Type ordering is Fighting/Flying/Poison/Ground/Rock/Bug/Ghost/Steel/Fire/
    Water/Grass/Electric/Psychic/Ice/Dragon/Dark. Native bit order puts Speed
    before Special Attack and Special Defense. Input IVs are int32 in 0..31.
    """
    values=_as_int32(ivs,'ivs')
    if values.ndim<1 or values.shape[-1]!=6: raise ValueError('IVs must end in six stats')
    weights=jnp.asarray([1,2,4,16,32,8],jnp.int32)
    kind=jnp.sum((values & 1)*weights,axis=-1,dtype=jnp.int32)*15//63
    power=jnp.sum(((values >> 1) & 1)*weights,axis=-1,dtype=jnp.int32)*40//63+30
    return kind,power
