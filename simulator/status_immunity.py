"""Status-immunity SetStatus/Immunity and acquisition-time Update hooks."""
import jax.numpy as jnp
from . import gen5_engine_tables as t
A=t.ABILITY_IDS
KEYS=('immunity','limber','waterveil','magmaarmor','insomnia','vitalspirit','wonderguard','waterabsorb','voltabsorb','stormdrain','lightningrod','motordrive')
ABILITY_IMPLEMENTED=jnp.asarray([k in KEYS for k in t.ABILITIES],jnp.bool_)
SLEEP=jnp.asarray([k in ('insomnia','vitalspirit')for k in t.ABILITIES],jnp.bool_)

def blocks(ability,status,e):
    return (((status==e.PSN)|(status==e.TOX))&(ability==A['immunity']))|((status==e.PAR)&(ability==A['limber']))|((status==e.BRN)&(ability==A['waterveil']))|((status==e.FRZ)&(ability==A['magmaarmor']))|((status==e.SLP)&SLEEP[ability])

def update(c,side,e):
    s,_=c;mon=s.active[side];ability=e.suppression_ops.abilities(s)[side,mon]
    def cure(c):
        c=e._event(c,e.STATUS_ABILITY,side,mon,value=ability)
        return e.effects.cure(c,side,e)
    return e._when(blocks(ability,s.status[side,mon],e)&(s.hp[side,mon]>0),cure,c)

def rest_fail(c,side,e):
    s,_=c;mon=s.active[side]
    c=e._event(c,e.REST_ABILITY,side,mon,value=e.suppression_ops.abilities(s)[side,mon])
    return e.tracking.outcome(c,side,e.i32(e.tracking.FALSE),e.REST)


def wonder_guard(s,actor,target,move,e):
    mon=s.active[target];source=s.active[actor]
    move_type=jnp.where(move==e.HIDDEN_POWER,s.hp_type[actor,source],t.MOVE_TYPE[move])
    exponent,immune=e.generic.type_matchup(move_type,s.types[target,mon])
    return (actor!=target)&(t.CATEGORY[move]>0)&(move!=e.STRUGGLE)&(e.suppression_ops.abilities(s)[target,mon]==A['wonderguard'])&((exponent<=0)|immune)


def absorbs(s,actor,target,move,e):
    mon=s.active[target];source=s.active[actor]
    ability=e.suppression_ops.abilities(s)[target,mon]
    move_type=jnp.where(move==e.HIDDEN_POWER,s.hp_type[actor,source],t.MOVE_TYPE[move])
    return (actor!=target)&((((ability==A['waterabsorb'])|(ability==A['stormdrain']))&(move_type==e.generic.TYPES['water']))|(((ability==A['voltabsorb'])|(ability==A['lightningrod'])|(ability==A['motordrive']))&(move_type==e.generic.TYPES['electric'])))


def absorb(c,actor,target,move,e):
    # Native TryHit turns the callback's null into false. No accuracy, primary
    # hit, attackedBy, secondary/contact or hit-loop Update runs for this target.
    c=e.tracking.outcome(c,actor,e.i32(e.tracking.FALSE),move)
    s,_=c;mon=s.active[target];maximum=s.stats[target,mon,0]
    ability=e.suppression_ops.abilities(s)[target,mon]
    def heal(c):
        s,tr=c;hp=jnp.minimum(maximum,s.hp[target,mon]+maximum//4)
        s=s._replace(hp=s.hp.at[target,mon].set(hp))
        return e._event((s,tr),e.HEAL,target,mon,move=jnp.where(ability==A['waterabsorb'],4,5),target=actor,targetmon=s.active[actor],value=hp,aux=maximum,status=s.status[target,mon])
    boosting=(ability==A['stormdrain'])|(ability==A['lightningrod'])|(ability==A['motordrive'])
    return e.jax.lax.cond(boosting,lambda c:absorb_boost(c,target,ability,e),lambda c:e.jax.lax.cond(c[0].hp[target,mon]<maximum,heal,lambda c:e._event(c,e.IMMUNE,target,mon,value=ability,aux=4),c),c)


def absorb_boost(c,target,ability,e):
    s,_=c;mon=s.active[target];stat=jnp.where(ability==A['motordrive'],4,2)
    old=s.boosts[target,mon,stat]
    def raise_stat(c):
        s,tr=c;s=s._replace(boosts=s.boosts.at[target,mon,stat].set(old+1))
        s=e.tracking.boosted(s,target,jnp.array(True),jnp.array(False))
        c=e._event((s,tr),e.REVEAL,target,mon,value=ability,aux=1)
        return e._event(c,e.BOOST,target,mon,value=1,aux=stat)
    # Native successful absorption reveals the ability then raises one stage;
    # at +6 it emits only immunity. Neither branch runs the hit loop.
    return e.jax.lax.cond(old<6,raise_stat,lambda c:e._event(c,e.IMMUNE,target,mon,value=ability,aux=4),c)
