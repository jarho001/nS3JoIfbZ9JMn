"""Explicit capability IDs for the integrated ability/item hook family."""
import jax.numpy as jnp
from . import gen5_engine_tables as t
from . import field_tables as f

A=t.ABILITY_IDS
I=t.ITEM_IDS
ABILITIES=('pressure','intimidate','prankster','hugepower','purepower','adaptability',
    'technician','hustle','compoundeyes','superluck','sniper','guts','quickfeet',
    'marvelscale','battlearmor','shellarmor','thickfat','filter','solidrock','tintedlens',
    'drizzle','drought','sandstream','snowwarning','swiftswim','chlorophyll','sandrush','icebody')
ITEMS=('choiceband','choicespecs','choicescarf','lifeorb','expertbelt','muscleband',
       'wiseglasses','silkscarf','mysticwater','scopelens','eviolite')
ABILITY_IMPLEMENTED=jnp.asarray([n<=3 or key in ABILITIES for n,key in enumerate(t.ABILITIES)],jnp.bool_)
ITEM_IMPLEMENTED=f.ITEM_IMPLEMENTED|jnp.asarray([key in ITEMS for key in t.ITEMS],jnp.bool_)
ABILITY_WEATHER=jnp.asarray([{'drizzle':1,'drought':2,'sandstream':3,'snowwarning':4}.get(key,0) for key in t.ABILITIES],jnp.int32)
CHOICE=jnp.asarray([key in ('choiceband','choicespecs','choicescarf') for key in t.ITEMS],jnp.bool_)
NFE=jnp.asarray([False]+[t.SP_ROWS[t.g.to_id(name)].get('nfe',False) for name in t.SPECIES[1:]],jnp.bool_)
CRIT_RATIO=t.mv(lambda r:r['critRatio'])
CRIT_DENOMINATOR=jnp.asarray([0,16,8,4,3,2],jnp.int32)
ABILITY_NAMES=('',)+tuple(next(r['name'] for r in t.g.DATA['abilities'] if r['id']==key) for key in t.ABILITIES[1:])
ITEM_NAMES=('',)+tuple(next(r['name'] for r in t.g.DATA['items'] if r['id']==key) for key in t.ITEMS[1:])

PRESSURE_TARGET=t.mv(lambda r:bool(r.get('flags',{}).get('mustpressure')) or r['target'] in ('normal','any','all','allAdjacent','allAdjacentFoes','randomNormal','adjacentFoe'))
