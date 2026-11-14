"""Original-species stat modifiers for Light Ball, Thick Club and Gen5 Soul Dew."""
import jax.numpy as jnp
from . import gen5_engine_tables as t
I=t.ITEM_IDS
ITEM_IMPLEMENTED=jnp.asarray([key in ('lightball','thickclub','souldew')for key in t.ITEMS],jnp.bool_)
BASE_NAMES=('',)+tuple(t.SP_ROWS[t.g.to_id(name)]['baseSpecies']for name in t.SPECIES[1:])
BALL=jnp.asarray([name=='Pikachu'for name in BASE_NAMES],jnp.bool_)
CLUB=jnp.asarray([name in ('Cubone','Marowak')for name in BASE_NAMES],jnp.bool_)
DEW=jnp.asarray([False]+[t.SP_ROWS[t.g.to_id(name)]['num']in (380,381)for name in t.SPECIES[1:]],jnp.bool_)


def offense(s,side,physical):
    mon=s.active[side];species=s.base_species[side,mon];item=s.item[side,mon]
    double=((item==I['lightball'])&BALL[species])|((item==I['thickclub'])&CLUB[species]&physical)
    dew=(item==I['souldew'])&DEW[species]&(~physical)
    return jnp.where(double,8192,jnp.where(dew,6144,4096)).astype(jnp.int32)


def defense(s,side,physical):
    mon=s.active[side];active=(s.item[side,mon]==I['souldew'])&DEW[s.base_species[side,mon]]&(~physical)
    return jnp.where(active,6144,4096).astype(jnp.int32)
