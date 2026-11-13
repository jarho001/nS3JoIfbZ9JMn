"""Weather-driven Gen5 Castform/Cherrim effective forms."""
import jax.numpy as jnp
from . import suppression as suppression_ops
from . import gen5_engine_tables as t
A=t.ABILITY_IDS;S=t.SPECIES_IDS
FORECAST=A['forecast'];FLOWER=A['flowergift']
CAST=S['castform'];CHERRIM=S['cherrim']
CAST_FORMS=jnp.asarray([CAST,S['castformrainy'],S['castformsunny'],CAST,S['castformsnowy']],jnp.int32)
CAST_GROUP=jnp.asarray([k.startswith('Castform')for k in t.SPECIES],jnp.bool_)
FLOWER_GROUP=jnp.asarray([k.startswith('Cherrim')for k in t.SPECIES],jnp.bool_)
SPECIES_IMPLEMENTED=CAST_GROUP|FLOWER_GROUP
ABILITY_IMPLEMENTED=jnp.asarray([k in ('forecast','flowergift')for k in t.ABILITIES],jnp.bool_)

def valid_pair(s):
    return (CAST_GROUP[s.base_species]&CAST_GROUP[s.species])|(FLOWER_GROUP[s.base_species]&FLOWER_GROUP[s.species])

def flower(s,side):
    mon=s.active[side]
    return (suppression_ops.abilities(s)[side,mon]==FLOWER)&FLOWER_GROUP[s.base_species[side,mon]]&(s.weather_state[0]==2)

def change(c,side,e):
    s,_=c;mon=s.active[side];ability=suppression_ops.abilities(s)[side,mon];weather=s.weather_state[0]
    cast=(ability==FORECAST)&CAST_GROUP[s.base_species[side,mon]]
    flower=(ability==FLOWER)&FLOWER_GROUP[s.base_species[side,mon]]
    species=jnp.where(cast,CAST_FORMS[weather],jnp.where(weather==2,S['cherrimsunshine'],CHERRIM))
    allowed=(cast|flower)&(~s.transformed[side,mon])&(s.hp[side,mon]>0)&(species!=s.species[side,mon])
    return e._when(allowed,lambda c:set_form(c,side,species,e,ability=ability),c)


def set_form(c,side,species,e,ability=0,message=0):
    s,tr=c;mon=s.active[side]
    stats=e.derive_stored_stats(t.BASE_STATS[species],s.level[side,mon],s.ivs[side,mon],s.evs[side,mon],e.generic.NATURE_MULTIPLIERS[s.nature[side,mon]],t.SHEDINJA[species])
    stats=stats.at[0].set(s.stats[side,mon,0])
    s=s._replace(species=s.species.at[side,mon].set(species),stats=s.stats.at[side,mon].set(stats),base_stats=s.base_stats.at[side,mon].set(stats),speed=s.speed.at[side,mon].set(stats[5]),types=s.types.at[side,mon].set(t.SPECIES_TYPES[species]),apparent_types=s.apparent_types.at[side,mon].set(t.SPECIES_TYPES[species]),known_type=s.known_type.at[side,mon].set(True))
    return e._event((s,tr),e.FORM_CHANGE,side,mon,value=species,aux=ability,status=message)


def weather_change(c,e,context):
    s,_=c;speeds=s.speed[jnp.arange(2),s.active];tie=(speeds[0]==speeds[1])&jnp.all(~s.fainted[jnp.arange(2),s.active])
    c,word=e.jax.lax.cond(tie,lambda c:e._draw(c,0,2,context,2),lambda c:(c,jnp.uint32(0)),c)
    first=jnp.where(tie,word.astype(jnp.int32),jnp.argmax(speeds))
    for k in range(2):
        side=jnp.where(k==0,first,1-first)
        c=e._when(~c[0].fainted[side,c[0].active[side]],lambda c:change(c,side,e),c)
    return c
