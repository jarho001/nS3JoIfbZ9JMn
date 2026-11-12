"""Shared-engine numeric tables, preserving schema 1/2 entity IDs.

The complete data universe is decoupled from the current handler capability
mask. Unsupported handlers are rejected before RNG/event mutation. Adding data
does not make an unimplemented callback a no-op.
"""
import jax.numpy as jnp
from . import gen5_tables as g

LEGACY_SPECIES=('', 'Mew', 'Snorlax', 'Gengar')
LEGACY_MOVES=('', 'Tackle', 'Pound', 'Mega Punch', 'Earthquake', 'U-turn', 'Will-O-Wisp',
              'Toxic', 'Thunder Wave', 'Recover', 'Protect', 'Struggle', 'Stealth Rock', 'Spikes', 'Rapid Spin')
LEGACY_ITEMS=('', 'leftovers', 'lumberry')
SPECIES=LEGACY_SPECIES+tuple(r['name'] for r in g.DATA['species'] if r['name'] not in LEGACY_SPECIES)
MOVE_NAMES=LEGACY_MOVES+tuple(r['name'] for r in g.DATA['moves'] if r['name'] not in LEGACY_MOVES)
MOVE_KEYS=tuple(g.to_id(name) for name in MOVE_NAMES)
ITEMS=LEGACY_ITEMS+tuple(r['id'] for r in g.DATA['items'] if r['id'] not in LEGACY_ITEMS)
ABILITIES=('', 'synchronize', 'immunity', 'levitate')+tuple(
    r['id'] for r in g.DATA['abilities'] if r['id'] not in ('synchronize','immunity','levitate'))
SPECIES_IDS={g.to_id(name):i for i,name in enumerate(SPECIES)}
MOVE_IDS={name:i for i,name in enumerate(MOVE_KEYS)}
ABILITY_IDS={name:i for i,name in enumerate(ABILITIES)}
ITEM_IDS={name:i for i,name in enumerate(ITEMS)}


def remap(kind,ids):
    return jnp.asarray([0]+[ids[r['id']] for r in g.DATA[kind]],jnp.int32)


G_SPECIES=remap('species',SPECIES_IDS)
G_MOVES=remap('moves',MOVE_IDS)
G_ABILITIES=remap('abilities',ABILITY_IDS)
G_ITEMS=remap('items',ITEM_IDS)
SP_ROWS={r['id']:r for r in g.DATA['species']}
MV_ROWS={r['id']:r for r in g.DATA['moves']}
BASE_STATS=jnp.asarray([[1]*6]+[[SP_ROWS[g.to_id(name)]['baseStats'][s] for s in g.STATS] for name in SPECIES[1:]],jnp.int32)
SPECIES_TYPES=jnp.asarray([[0,0]]+[[g.TYPES[g.to_id(t)] for t in SP_ROWS[g.to_id(name)]['types']]+
    [0]*(2-len(SP_ROWS[g.to_id(name)]['types'])) for name in SPECIES[1:]],jnp.int32)
SHEDINJA=jnp.asarray([False]+[SP_ROWS[g.to_id(name)].get('maxHP')==1 for name in SPECIES[1:]])
SPECIES_GENDER=jnp.asarray([0]+[{'M':1,'F':2,'N':0}.get(SP_ROWS[g.to_id(name)].get('gender'),-1) for name in SPECIES[1:]],jnp.int32)
SPECIES_ABILITY=jnp.asarray([0]+[ABILITY_IDS[g.to_id(SP_ROWS[g.to_id(name)]['abilities']['0'])] for name in SPECIES[1:]],jnp.int32)
SPECIES_NAME=jnp.asarray([0]+[SPECIES_IDS[g.to_id(SP_ROWS[g.to_id(name)]['baseSpecies'])] for name in SPECIES[1:]],jnp.int32)
SPECIES_IMPLEMENTED=jnp.asarray([False]+[not bool(SP_ROWS[g.to_id(name)].get('battleOnly') or
    SP_ROWS[g.to_id(name)].get('changesFrom')) for name in SPECIES[1:]])


def mv(extract,empty=0):
    return jnp.asarray([empty]+[extract(MV_ROWS[key]) for key in MOVE_KEYS[1:]],jnp.int32)


PP=mv(lambda r:r['pp'] if r.get('noPPBoosts') or r['id']=='trumpcard' else r['pp']*8//5)
POWER=mv(lambda r:r['basePower'])
ACCURACY=mv(lambda r:0 if r['accuracy'] is True else r['accuracy'])
CATEGORY=mv(lambda r:{'Status':0,'Physical':1,'Special':2}[r['category']])
MOVE_TYPE=mv(lambda r:g.TYPES.get(g.to_id(r['type']),0))
PRIORITY=mv(lambda r:r['priority'])
TARGETS=('',)+tuple(MV_ROWS[key]['target'] for key in MOVE_KEYS[1:])
SELF_TARGET=jnp.asarray([target=='self' for target in TARGETS],jnp.bool_)


def has_callback(value):
    if isinstance(value,dict): return '$callback' in value or any(has_callback(v) for v in value.values())
    if isinstance(value,list): return any(has_callback(v) for v in value)
    return False


def plain(row):
    if row['category']=='Status' or row['basePower']<=0 or row['target'] not in ('normal','any','allAdjacent','allAdjacentFoes'):
        return False
    if has_callback(row) or row['critRatio']!=1: return False
    # Unknown declarative fields must fail closed just like unknown callbacks
    # (e.g. False Swipe's noFaint cannot become an ordinary damaging move).
    known={'name','id','fullname','effectType','exists','num','gen','isNonstandard','duration',
        'noCopy','affectsFainted','sourceEffect','target','type','basePower','accuracy','critRatio','baseMoveType',
        'secondary','secondaries','hasSheerForceBoost','priority','category','overrideOffensiveStat',
        'overrideOffensivePokemon','overrideDefensiveStat','overrideDefensivePokemon','ignoreNegativeOffensive',
        'ignorePositiveDefensive','ignoreOffensive','ignoreDefensive','ignoreImmunity','pp','noPPBoosts',
        'status','weather','isZ','isMax','flags','selfSwitch','ignoreAbility','damage','spreadHit',
        'forceSTAB','volatileStatus','maxMove','zMove','contestType'}
    if set(row)-known:return False
    effects=('secondary','secondaries','self','selfdestruct','recoil','drain','multihit','status',
        'volatileStatus','weather','pseudoWeather','sideCondition','slotCondition','boosts',
        'selfSwitch','forceSwitch','ohko','damage','willCrit','breaksProtect','ignoreAbility',
        'ignoreDefensive','ignoreOffensive','ignoreNegativeOffensive','ignorePositiveDefensive',
        'overrideOffensiveStat','overrideDefensiveStat','overrideOffensivePokemon','overrideDefensivePokemon',
        'forceSTAB','ignoreImmunity','spreadHit','isZ','isMax')
    if any(row.get(k) for k in effects): return False
    # Wind consumers are Gen9 Wind Power/Wind Rider and gen9ssb conditions;
    # neither ability/condition can enter the pinned Gen5 data/format.
    if set(row.get('flags',{}))-{'protect','mirror','contact','sound','punch','pulse','bite','authentic',
                                'slicing','bullet','distance','metronome','nonsky','wind'}:return False
    return True


STATUS_CODES={'brn':1,'par':2,'tox':3,'psn':5,'slp':6,'frz':7}
SECONDARY_CHANCE=mv(lambda r:(r.get('secondary') or {}).get('chance',0) if
    set(r.get('secondary') or {})<= {'chance','status'} and (r.get('secondary') or {}).get('status') in STATUS_CODES else 0)
SECONDARY_STATUS=mv(lambda r:STATUS_CODES.get((r.get('secondary') or {}).get('status'),0))


def simple_secondary(row):
    if not row.get('secondary') or set(row['secondary'])-{'chance','status'}:return False
    if row['secondary'].get('status') not in STATUS_CODES:return False
    if row.get('secondaries') not in (None,[row['secondary']]):return False
    clean=dict(row);clean.pop('secondary',None);clean.pop('secondaries',None)
    return plain(clean)


MOVE_IMPLEMENTED=jnp.asarray([False]+[name in LEGACY_MOVES or key in ('hiddenpower','return','frustration','spore','confuseray','poisonpowder','rest') or
    plain(MV_ROWS[key]) or simple_secondary(MV_ROWS[key]) for name,key in zip(MOVE_NAMES[1:],MOVE_KEYS[1:])])
