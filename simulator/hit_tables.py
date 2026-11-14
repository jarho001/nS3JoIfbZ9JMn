"""Declarative hit-loop capabilities, separate from the exported data universe."""
import jax.numpy as jnp
from . import gen5_engine_tables as t

MIN_HITS=t.mv(lambda r:r.get('multihit',[1,1])[0] if isinstance(r.get('multihit',[1,1]),list) else r['multihit'])
MAX_HITS=t.mv(lambda r:r.get('multihit',[1,1])[-1] if isinstance(r.get('multihit',[1,1]),list) else r['multihit'])
RECOIL_NUM=t.mv(lambda r:r.get('recoil',[0,1])[0])
RECOIL_DEN=t.mv(lambda r:r.get('recoil',[0,1])[1])
DRAIN_NUM=t.mv(lambda r:r.get('drain',[0,1])[0])
DRAIN_DEN=t.mv(lambda r:r.get('drain',[0,1])[1])
SELF_DESTRUCT=t.mv(lambda r:r.get('selfdestruct')=='always').astype(jnp.bool_)
DEFROST=t.mv(lambda r:bool(r.get('flags',{}).get('defrost'))).astype(jnp.bool_)
CONTACT=t.mv(lambda r:bool(r.get('flags',{}).get('contact'))).astype(jnp.bool_)
CRASH=t.mv(lambda r:r.get('id') in ('jumpkick','highjumpkick')).astype(jnp.bool_)
HIT_COUNTS=jnp.asarray([2]*7+[3]*7+[4]*3+[5]*3,jnp.int32)
ABILITIES=('skilllink','rockhead','roughskin','ironbarbs','liquidooze','mummy','magicguard','shielddust','serenegrace')
ABILITY_IMPLEMENTED=jnp.asarray([key in ABILITIES for key in t.ABILITIES],jnp.bool_)
ITEM_IMPLEMENTED=jnp.asarray([key=='rockyhelmet' for key in t.ITEMS],jnp.bool_)


def supported(row):
    if row.get('id') in ('lowkick','grassknot','heavyslam'):
        if row.get('basePower')!=0 or row.get('basePowerCallback')!={'$callback':f"moves:{row['id']}:basePowerCallback"}:return False
        clean=dict(row);clean.pop('basePowerCallback');clean['basePower']=1
        return t.plain(clean)
    selector={'foulplay':('overrideOffensivePokemon','target'),'psyshock':('overrideDefensiveStat','def'),'psystrike':('overrideDefensiveStat','def'),'secretsword':('overrideDefensiveStat','def')}.get(row.get('id'))
    if selector:
        field,value=selector
        if row.get(field)!=value:return False
        clean=dict(row);clean.pop(field)
        return t.plain(clean)
    callback={'suckerpunch':'onTry','payback':'basePowerCallback'}.get(row.get('id'))
    if callback:
        if row.get(callback)!={'$callback':f"moves:{row['id']}:{callback}"}:return False
        clean=dict(row);clean.pop(callback)
        return t.plain(clean)
    if row.get('id') in ('jumpkick','highjumpkick'):
        if row.get('hasCrashDamage') is not True or row.get('onMoveFail')!={'$callback':f"moves:{row['id']}:onMoveFail"}:return False
        clean=dict(row);clean.pop('hasCrashDamage');clean.pop('onMoveFail')
        clean['flags']={key:value for key,value in row.get('flags',{}).items()if key!='gravity'}
        # Gravity itself stays unsupported; preserve every other capability guard.
        return t.plain(clean)
    if row.get('selfdestruct') not in (None,'always'):return False
    clean=dict(row)
    # Pinned Gen5 Scald explicitly disables the later target-thaw behavior.
    if clean.get('thawsTarget') is False:clean.pop('thawsTarget')
    # Parental Bond is post-Gen5; defrost is implemented before ModifyMove.
    clean['flags']={key:value for key,value in row.get('flags',{}).items()if key not in ('noparentalbond','defrost')}
    for key in ('multihit','recoil','drain','selfdestruct'):clean.pop(key,None)
    # Preserve every other guard, including callbacks, multiaccuracy, locks,
    # conditional power, and unimplemented secondary effects.
    return (any(row.get(key) for key in ('multihit','recoil','drain','selfdestruct')) or bool(row.get('flags',{}).get('defrost'))) and (t.plain(clean) or t.simple_secondary(clean))


MOVE_IMPLEMENTED=jnp.asarray([False]+[supported(t.MV_ROWS[key]) for key in t.MOVE_KEYS[1:]],jnp.bool_)
