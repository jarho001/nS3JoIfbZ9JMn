"""Pinned declarative field/stat capabilities, separate from data availability."""
import jax.numpy as jnp
from . import gen5_engine_tables as t

WEATHER_KEYS=('', 'raindance','sunnyday','sandstorm','hail')
WEATHER_NAMES=('none','RainDance','SunnyDay','Sandstorm','Hail')
SCREEN_KEYS=('reflect','lightscreen')
SCREEN_MOVES=tuple(t.MOVE_IDS[k] for k in SCREEN_KEYS)
STATS=('atk','def','spa','spd','spe','accuracy','evasion')
# Native resolved property order differs from stat order for these two moves.
# Observed with Dex.mod('gen5').moves at the pinned source; never use sorted JSON
# keys as the boost-event order.
BOOST_ORDER_OVERRIDES={'shellsmash':('def','spd','atk','spa','spe'),'shiftgear':('spe','atk')}
WEATHER=t.mv(lambda r:WEATHER_KEYS.index(t.g.to_id(r['weather'])) if t.g.to_id(r.get('weather','')) in WEATHER_KEYS and r.get('weather') else 0)
SCREEN=t.mv(lambda r:SCREEN_KEYS.index(r['sideCondition'])+1 if r.get('sideCondition') in SCREEN_KEYS else 0)
BOOSTS=jnp.asarray([[0]*7]+[[r.get('boosts',{}).get(k,0) for k in STATS] for r in (t.MV_ROWS[key] for key in t.MOVE_KEYS[1:])],jnp.int32)
BOOST_ORDER=jnp.asarray([list(range(7))]+[[STATS.index(k) for k in BOOST_ORDER_OVERRIDES.get(key,STATS)]+[i for i,k in enumerate(STATS) if k not in BOOST_ORDER_OVERRIDES.get(key,STATS)] for key in t.MOVE_KEYS[1:]],jnp.int32)
ACTOR_TARGET=jnp.asarray([target in ('self','all','allySide','allyTeam','allies') for target in t.TARGETS],jnp.bool_)
WEATHER_ROCKS=jnp.asarray([0]+[t.ITEM_IDS[k] for k in ('damprock','heatrock','smoothrock','icyrock')],jnp.int32)
LIGHT_CLAY=t.ITEM_IDS['lightclay']
ITEM_IMPLEMENTED=jnp.asarray([i<=2 or key in ('damprock','heatrock','smoothrock','icyrock','lightclay') for i,key in enumerate(t.ITEMS)],jnp.bool_)


def simple_boost(row):
    if not row.get('boosts') or row['category']!='Status' or t.has_callback(row):return False
    if row['target'] not in ('self','normal','allAdjacentFoes'):return False
    if any(row.get(k) for k in ('secondary','secondaries','self','selfSwitch','forceSwitch','status','volatileStatus',
                               'heal','selfdestruct','weather','sideCondition','pseudoWeather','slotCondition','damage')):return False
    if set(row['boosts'])-set(STATS):return False
    return True


IS_BOOST=jnp.asarray([False]+[simple_boost(t.MV_ROWS[key]) for key in t.MOVE_KEYS[1:]],jnp.bool_)
MOVE_IMPLEMENTED=IS_BOOST|(WEATHER>0)|(SCREEN>0)
