"""Immutable pinned Gen 5 tables and pre-entry set decoding.

This table namespace is versioned independently of restricted schema 1/2 IDs.
Callback tokens are audit data. There is deliberately no callback interpreter
or native/Python simulation fallback. Decoding a set does not authorize battle
execution; the shared engine still preflights its implemented domain.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from mechanics.stats import derive_stored_stats
from mechanics.hidden_power import hidden_power
from .gen5_rng import sample_bounded

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / 'rules/gen5-data-v1.json').read_text())
PIN = json.loads((ROOT / 'reference-lock.json').read_text())['commit']
if DATA['pin'] != PIN or DATA['schemaVersion'] != 1:
    raise ValueError('incompatible pinned Gen 5 data')
DATA_SHA256 = hashlib.sha256(json.dumps(DATA, sort_keys=True, separators=(',', ':'),
                                      ensure_ascii=False).encode()).hexdigest()
STATS = ('hp', 'atk', 'def', 'spa', 'spd', 'spe')


def to_id(value):
    return re.sub('[^a-z0-9]', '', value.lower())


def index(kind):
    return {row['id']: n + 1 for n, row in enumerate(DATA[kind])}


SPECIES = index('species')
MOVES = index('moves')
ABILITIES = index('abilities')
ITEMS = index('items')
TYPES = index('types')
NATURES = index('natures')


def _table(kind, extract, empty=0, dtype=jnp.int32):
    return jnp.asarray([empty] + [extract(row) for row in DATA[kind]], dtype)


BASE_STATS = _table('species', lambda r: [r['baseStats'][k] for k in STATS], [0]*6)
SPECIES_TYPES = _table('species', lambda r: [TYPES[to_id(t)] for t in r['types']] +
                       [0]*(2-len(r['types'])), [0, 0])
WEIGHT_HG = _table('species', lambda r: r['weighthg'])
SHEDINJA = _table('species', lambda r: r.get('maxHP') == 1, False, jnp.bool_)
GENDER = _table('species', lambda r: {'M': 1, 'F': 2, 'N': 0}.get(r.get('gender'), -1))
MOVE_TYPE = _table('moves', lambda r: TYPES[to_id(r['type'])] if to_id(r['type']) in TYPES else 0)
MOVE_POWER = _table('moves', lambda r: r['basePower'])
MOVE_ACCURACY = _table('moves', lambda r: 0 if r['accuracy'] is True else r['accuracy'])
MOVE_CATEGORY = _table('moves', lambda r: {'Status': 0, 'Physical': 1, 'Special': 2}[r['category']])
MOVE_PRIORITY = _table('moves', lambda r: r['priority'])
MOVE_PP = _table('moves', lambda r: r['pp'] if r.get('noPPBoosts') or r['id'] == 'trumpcard' else r['pp']*8//5)
NATURE_MULTIPLIERS = _table('natures', lambda r: [11 if k == r.get('plus') else 9 if k == r.get('minus')
                                               else 10 for k in STATS], [10]*6)
# Indexed [defending type, attacking type]. Native values: 0 neutral,
# 1 weakness, 2 resistance, 3 immunity. Row/column zero is the missing type.
TYPE_CHART = jnp.asarray([[0]*(len(TYPES)+1)] + [[0] + [r['damageTaken'][t['name']]
    for t in DATA['types']] for r in DATA['types']], jnp.int32)
HP_TYPE_IDS = jnp.asarray([TYPES[to_id(t)] for t in ('Fighting', 'Flying', 'Poison', 'Ground',
    'Rock', 'Bug', 'Ghost', 'Steel', 'Fire', 'Water', 'Grass', 'Electric', 'Psychic', 'Ice', 'Dragon', 'Dark')], jnp.int32)


class Sets(NamedTuple):
    species: jax.Array
    level: jax.Array
    ability: jax.Array
    item: jax.Array
    moves: jax.Array
    ivs: jax.Array
    evs: jax.Array
    nature: jax.Array
    gender: jax.Array  # -1 asks the battle initializer for the native uniform M/F draw
    shiny: jax.Array
    happiness: jax.Array
    hp_type_override: jax.Array


class SetStats(NamedTuple):
    stats: jax.Array
    types: jax.Array
    pp: jax.Array
    hp_type: jax.Array
    hp_power: jax.Array


class Construction(NamedTuple):
    sets: Sets
    values: SetStats
    rng: jax.Array
    draw_count: jax.Array
    draw_values: jax.Array
    draw_before: jax.Array
    draw_after: jax.Array


def decode_sets(teams):
    """Validate and encode supplied/generated sets, before entry effects.

    Accepted extra generator fields (role/speciesId/name) are explicitly
    metadata. Unknown fields fail; the decoder never silently discards effects.
    Full legality and generator membership belong to the reachability audit.
    """
    if len(teams) != 2 or any(not 1 <= len(team) <= 6 for team in teams):
        raise ValueError('two teams of one to six mons required')
    shape=(2, 6)
    arrays={name: np.zeros(shape + ((4,) if name=='moves' else (6,) if name in ('ivs','evs') else ()),
                           np.bool_ if name=='shiny' else np.int32) for name in Sets._fields}
    arrays['ivs'][:]=31
    arrays['nature'][:]=NATURES['hardy']
    arrays['happiness'][:]=255
    allowed={'species','moves','level','ability','item','ivs','evs','nature','gender','shiny','happiness','hpType',
             'role','speciesId','name'}
    for side,team in enumerate(teams):
        for mon,desc in enumerate(team):
            unknown=set(desc)-allowed
            if unknown: raise ValueError(f'unsupported set fields: {sorted(unknown)}')
            sp=to_id(desc['species'])
            if sp not in SPECIES: raise ValueError(f'unknown Gen 5 species: {sp}')
            row=DATA['species'][SPECIES[sp]-1]
            arrays['species'][side,mon]=SPECIES[sp]
            level=desc.get('level',100)
            if type(level) is not int or not 1 <= level <= 100: raise ValueError('invalid level')
            arrays['level'][side,mon]=level
            for key,lookup,default in (('ability',ABILITIES,''),('item',ITEMS,''),
                                       ('nature',NATURES,'Hardy')):
                value=to_id(desc.get(key,default))
                if not value and key in ('item','ability'): continue
                if value not in lookup: raise ValueError(f'unknown Gen 5 {key}: {value}')
                arrays[key][side,mon]=lookup[value]
            gender=desc.get('gender') or row.get('gender','')
            if gender not in ('','M','F','N'): raise ValueError('invalid gender')
            arrays['gender'][side,mon]={'':-1,'M':1,'F':2,'N':0}[gender]
            shiny=desc.get('shiny',False)
            if type(shiny) is not bool: raise ValueError('shiny must be bool')
            arrays['shiny'][side,mon]=shiny
            happiness=desc.get('happiness',255)
            if type(happiness) is not int or not 0 <= happiness <= 255: raise ValueError('invalid happiness')
            arrays['happiness'][side,mon]=happiness
            for key,default,limit in (('ivs',31,31),('evs',0,255)):
                values=desc.get(key,{})
                if set(values)-set(STATS): raise ValueError(f'unknown {key} stat')
                ordered=[values.get(k,default) for k in STATS]
                if any(type(v) is not int or not 0 <= v <= limit for v in ordered): raise ValueError(f'invalid {key}')
                arrays[key][side,mon]=ordered
            moves=desc['moves']
            if not 1 <= len(moves) <= 4: raise ValueError('one to four moves required')
            mids=[]
            for name in moves:
                key=to_id(name)
                if key.startswith('hiddenpower') and key!='hiddenpower':
                    type_id=key[len('hiddenpower'):]
                    if type_id not in TYPES or type_id in ('normal',): raise ValueError('invalid Hidden Power type')
                    if not arrays['hp_type_override'][side,mon]: arrays['hp_type_override'][side,mon]=TYPES[type_id]
                    key='hiddenpower'
                if key not in MOVES or key=='struggle': raise ValueError(f'unknown or forced move: {key}')
                mids.append(MOVES[key])
            if len(set(mids)) != len(mids): raise ValueError('duplicate canonical move')
            arrays['moves'][side,mon,:len(mids)]=mids
            if desc.get('hpType'):
                hp=to_id(desc['hpType'])
                if hp not in TYPES or hp=='normal': raise ValueError('invalid hpType')
                arrays['hp_type_override'][side,mon]=TYPES[hp]
    return Sets(*(jnp.asarray(arrays[name]) for name in Sets._fields))


def derive_set_stats(sets):
    """Pure JAX pre-entry values; no gender, entry hooks or scheduler draws."""
    occupied=sets.species>0
    stats=derive_stored_stats(BASE_STATS[sets.species],sets.level,sets.ivs,sets.evs,
                             NATURE_MULTIPLIERS[sets.nature],SHEDINJA[sets.species])
    hp_type,hp_power=hidden_power(sets.ivs)
    hp_type=jnp.where(sets.hp_type_override>0,sets.hp_type_override,HP_TYPE_IDS[hp_type])
    return SetStats(jnp.where(occupied[...,None],stats,0),SPECIES_TYPES[sets.species],
                    MOVE_PP[sets.moves],jnp.where(occupied,hp_type,0),jnp.where(occupied,hp_power,0))


def construct(sets,seed):
    """Native pre-entry constructor draws in p1/p2 roster order.

    Showdown samples M/F uniformly for every unspecified non-fixed gender,
    independent of the species gender ratio. Empty slots consume no draws.
    This phase ends before native entry hooks and makes no battle-ready claim.
    """
    state=(sets,seed,jnp.int32(0),jnp.zeros(12,jnp.uint32),
           jnp.zeros((12,4),jnp.uint16),jnp.zeros((12,4),jnp.uint16))
    def body(i,state):
        side,mon=i//6,i%6
        def draw(state):
            sets,rng,count,values,before,after=state
            next_rng,value=sample_bounded(rng,jnp.int32(2))
            return (sets._replace(gender=sets.gender.at[side,mon].set(value.astype(jnp.int32)+1)),
                    next_rng,count+1,values.at[count].set(value),before.at[count].set(rng),after.at[count].set(next_rng))
        sets=state[0]
        return jax.lax.cond((sets.species[side,mon]>0)&(sets.gender[side,mon]<0),draw,lambda x:x,state)
    sets,rng,count,values,before,after=jax.lax.fori_loop(0,12,body,state)
    return Construction(sets,derive_set_stats(sets),rng,count,values,before,after)


def type_matchup(move_type,defender_types):
    """Return Gen 5 exponent and immunity, with one/two defending type slots."""
    taken=TYPE_CHART[jnp.asarray(defender_types,jnp.int32),jnp.asarray(move_type,jnp.int32)[...,None]]
    exponent=jnp.sum(jnp.where(taken==1,1,jnp.where(taken==2,-1,0)),axis=-1,dtype=jnp.int32)
    return exponent,jnp.any(taken==3,axis=-1)
