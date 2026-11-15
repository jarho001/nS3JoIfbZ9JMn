"""Strict pinned confusion secondaries; silent blocked application after chance."""
import jax.numpy as jnp
from . import gen5_engine_tables as t
MOVES=('signalbeam','hurricane','dynamicpunch')
CHANCE=t.mv(lambda r:r.get('secondary',{}).get('chance',0) if r['id'] in MOVES else 0)


def supported(row):
    if row.get('id') not in MOVES:return False
    key=row['id'];chance={'signalbeam':10,'hurricane':30,'dynamicpunch':100}[key]
    if row.get('secondary')!={'chance':chance,'volatileStatus':'confusion'}:return False
    if row.get('secondaries') not in (None,[row['secondary']]):return False
    clean=dict(row);clean.pop('secondary');clean.pop('secondaries',None)
    if key=='hurricane':
        if clean.get('onModifyMove')!={'$callback':'moves:hurricane:onModifyMove'} or clean.get('basePower')!=120 or clean.get('accuracy')!=70:return False
        clean.pop('onModifyMove')
    return t.plain(clean)


def apply(c,actor,target,move,e):
    s,_=c;mon=s.active[target]
    allowed=(s.hp[target,mon]>0)&(s.volatile_state[target,mon,2,0]==0)&(e.suppression_ops.abilities(s)[target,mon]!=e.sleep_ops.OWN)
    # The chance draw has already happened. Native secondary application fails
    # silently for Own Tempo, existing confusion and fainted targets.
    return e._when(allowed,lambda c:e.effects.confuse(c,actor,target,move,e),c)
