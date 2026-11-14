"""Pure JAX per-hit healing/contact hooks and once-per-move recoil arithmetic."""
import jax
import jax.numpy as jnp
from . import hit_tables as h
from . import gen5_engine_tables as tables


def rounded_fraction(amount,numerator,denominator):
    # Positive native Math.round; distinct from Gen5 damage modifier rounding.
    return (2*amount*numerator+denominator)//(2*denominator)


def hit_count(c,move,actor,e):
    s,_=c;skill=e.suppression_ops.abilities(s)[actor,s.active[actor]]==tables.ABILITY_IDS['skilllink']
    minimum=h.MIN_HITS[move];maximum=h.MAX_HITS[move]
    def sample(c):
        def gen5(c):
            c,value=e._draw(c,0,20,e.MULTIHIT_COUNT)
            return c,h.HIT_COUNTS[value.astype(jnp.int32)]
        def other(c):
            c,value=e._draw(c,minimum,maximum+1,e.MULTIHIT_COUNT,2)
            return c,value.astype(jnp.int32)
        return jax.lax.cond((minimum==2)&(maximum==5),gen5,other,c)
    return jax.lax.cond((minimum!=maximum)&(~skill),sample,lambda c:(c,maximum),c)


def resolve_reactive_faints(c,actor,target,e):
    """Resolved Gen6 callbacks inherited by Gen5 request instafaint=true.

    Reverse the queued faint order when damage to the source is lethal,
    including a target KO already queued by the move's damage. This happens
    inside drain/contact, before the hit-loop Update or the next handler.
    """
    def resolve(c):
        c=e._when(~c[0].fainted[actor,c[0].active[actor]],lambda c:e._faint(c,actor),c)
        c=e._when((c[0].hp[target,c[0].active[target]]==0)&(~c[0].fainted[target,c[0].active[target]]),lambda c:e._faint(c,target),c)
        return e._when(c[0].winner<0,lambda c:e._win(c,target),c)
    return e._when(c[0].hp[actor,c[0].active[actor]]==0,resolve,c)


def drain(c,actor,target,move,dealt,e,*,substitute=False):
    def apply(c):
        s,t=c;mon=s.active[actor];tm=s.active[target]
        amount=jnp.maximum(1,(dealt*h.DRAIN_NUM[move]+h.DRAIN_DEN[move]-1)//h.DRAIN_DEN[move] if substitute else rounded_fraction(dealt,h.DRAIN_NUM[move],h.DRAIN_DEN[move]))
        def ooze(c):
            c=e._damage(c,actor,amount,cause=14,source_side=target,source_mon=tm)
            return resolve_reactive_faints(c,actor,target,e)
        def heal(c):
            s,t=c;hp=jnp.minimum(s.stats[actor,mon,0],s.hp[actor,mon]+amount)
            s=s._replace(hp=s.hp.at[actor,mon].set(hp))
            return e._event((s,t),e.HEAL,actor,mon,move=3,target=target,targetmon=tm,
                            value=hp,aux=s.stats[actor,mon,0],status=s.status[actor,mon])
        is_ooze=e.suppression_ops.abilities(s)[target,tm]==tables.ABILITY_IDS['liquidooze']
        return jax.lax.cond(is_ooze,ooze,lambda c:e._when(s.hp[actor,mon]<s.stats[actor,mon,0],heal,c),c)
    return e._when((h.DRAIN_NUM[move]>0)&(dealt>0)&(c[0].hp[actor,c[0].active[actor]]>0),apply,c)


def contact(c,actor,target,move,e):
    s,_=c;mon=s.active[actor];tm=s.active[target];ability=e.suppression_ops.abilities(s)[target,tm]
    # Native onDamagingHitOrder1 (ability), then2 (item). These orders cannot
    # tie, so there is no event-sort PRNG draw for these two handlers.
    thorn=(ability==tables.ABILITY_IDS['roughskin'])|(ability==tables.ABILITY_IDS['ironbarbs'])
    cause=jnp.where(ability==tables.ABILITY_IDS['roughskin'],11,12)
    def damage(c,amount,cause):
        c=e._damage(c,actor,amount,cause=cause,source_side=target,source_mon=tm)
        return resolve_reactive_faints(c,actor,target,e)
    c=e._when(h.CONTACT[move]&thorn&(s.hp[actor,mon]>0),
              lambda c:damage(c,s.stats[actor,mon,0]//8,cause),c)
    s,_=c
    aftermath=(ability==tables.ABILITY_IDS['aftermath'])&(s.hp[target,tm]==0)
    c=e._when(h.CONTACT[move]&aftermath&(s.hp[actor,mon]>0),lambda c:damage(c,s.stats[actor,mon,0]//4,17),c)
    s,_=c
    return e._when(h.CONTACT[move]&(s.item[target,tm]==tables.ITEM_IDS['rockyhelmet'])&(s.hp[actor,mon]>0),
                   lambda c:damage(c,s.stats[actor,mon,0]//6,13),c)


def spin(c,actor,e):
    """Shared AfterHit/AfterSubDamage hazard removal in pinned order."""
    mon=c[0].active[actor]
    for index,kind in ((1,e.SPIKES),(0,e.STEALTH_ROCK)):
        c=e._when(c[0].hazards[actor,index]>0,lambda c:e._event(c,e.SIDE_END,actor,mon,move=kind),c)
    s,trace=c
    c=(s._replace(hazards=s.hazards.at[actor].set(0),hazard_meta=s.hazard_meta.at[actor].set(0)),trace)
    return e.partial_ops.end(c,actor,e)
