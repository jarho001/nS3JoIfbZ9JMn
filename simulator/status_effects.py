"""Pure JAX status/volatile handlers used by the shared engine's scheduler.

The engine namespace supplies its numeric schema and event/RNG operations;
there is no native callback or alternate simulator here. Effect-state metadata
is causal saved state, including fields retained after Rest/Lum interaction.
"""
import jax
import jax.numpy as jnp


def empty_status():
    value=jnp.zeros((2,6,7),jnp.int32)
    return value.at[:,:,2:4].set(-1).at[:,:,5:7].set(-1)


def activate(c,side,e):
    s,t=c;m=s.active[side];order=s.effect_order
    ability=jnp.where(s.ability[side,m]>0,order,0)
    order=order+(s.ability[side,m]>0).astype(jnp.int32)
    item=jnp.where(s.item[side,m]>0,order,0)
    order=order+(s.item[side,m]>0).astype(jnp.int32)
    return s._replace(effect_order=order,ability_order=s.ability_order.at[side,m].set(ability),ability_effect_id=s.ability_effect_id.at[side,m].set(s.ability[side,m]),
                      item_order=s.item_order.at[side,m].set(item)),t


def cure(c,side,e,source=None,move=0):
    s,t=c;m=s.active[side];old=s.status[side,m]
    c=e._event(c,e.CURE,side,m,status=old,move=move);s,t=c
    source=side if source is None else source
    meta=e.i32([0,0,0,0,1,-1,-1]).at[2].set(source).at[3].set(s.active[source])
    return s._replace(status=s.status.at[side,m].set(0),toxic=s.toxic.at[side,m].set(0),
                      status_state=s.status_state.at[side,m].set(meta)),t


def lum(c,side,e,emit_item=True):
    s,t=c;m=s.active[side]
    if emit_item:c=e._event(c,e.END_ITEM,side,m)
    c=e._when((s.status[side,m]!=e.NONE)&(s.status[side,m]!=e.FNT),lambda c:cure(c,side,e),c)
    def end(c):
        c=e._event(c,e.CONF_END,side,m);s,t=c
        return s._replace(volatile_state=s.volatile_state.at[side,m,2].set(0)),t
    c=e._when(c[0].volatile_state[side,m,2,0]>0,end,c)
    s,t=c
    s=e.tracking.item_used(s,side,s.item[side,m])
    return s._replace(item=s.item.at[side,m].set(0),item_order=s.item_order.at[side,m].set(0)),t


def add_volatile(c,side,kind,source,move,e,source_kind=None):
    s,t=c;m=s.active[side];sm=s.active[source]
    present=s.volatile_state[side,m,kind,0]>0
    visited=(~present)&(s.hp[side,m]>0)&(s.weather_state[0]==2)
    s=s._replace(weather_state=s.weather_state.at[5].set(jnp.where(visited,1,s.weather_state[5])))
    source_kind=jnp.where(move>0,e.phase_ops.SOURCE_MOVE,jnp.where(move<0,e.phase_ops.SOURCE_ITEM,e.phase_ops.SOURCE_EMPTY)) if source_kind is None else e.i32(source_kind)
    row=jnp.stack((e.i32(1),s.effect_order,source,sm,move,e.i32(0),source_kind,*[e.i32(0)]*5))
    row=jnp.where(present,s.volatile_state[side,m,kind],row)
    return s._replace(volatile_state=s.volatile_state.at[side,m,kind].set(row),
                      effect_order=s.effect_order+(~present).astype(jnp.int32)),t


def sleep_switch(c,side,e):
    s,t=c;m=s.active[side]
    meta=s.status_state[side,m]
    meta=meta.at[5].set(jnp.where(s.status[side,m]==e.SLP,meta[6],meta[5]))
    return s._replace(status_state=s.status_state.at[side,m].set(meta)),t


def inflict(c,source,target,status,e,reflect=True,move=0,secondary=False,overwrite=False,source_mon=None):
    s,t=c;m=s.active[target];sm=s.active[source]if source_mon is None else source_mon;ability=e.suppression_ops.abilities(s)[target,m]
    poison=jnp.any((s.types[target,m]==e.generic.TYPES['poison'])|(s.types[target,m]==e.generic.TYPES['steel']))
    fire=jnp.any(s.types[target,m]==e.generic.TYPES['fire'])
    ice=jnp.any(s.types[target,m]==e.generic.TYPES['ice'])
    freeze_weather=(s.weather_state[0]==2)
    natural_immunity=(((status==e.TOX)|(status==e.PSN))&poison)|((status==e.BRN)&fire)|((status==e.FRZ)&ice)
    immunity=natural_immunity|e.immunity_ops.blocks(ability,status,e)|((status==e.FRZ)&freeze_weather)
    failed=(s.status[target,m]!=e.NONE)&(~jnp.asarray(overwrite))
    asleep=(s.status[target]==e.SLP)&(s.status_state[target,:,2]!=target)
    clause=(status==e.SLP)&(source!=target)&jnp.any(asleep)
    def rejected(c):
        if secondary:return c
        def same(c):return e._event(c,e.FAIL,target,m,status=status)
        def different(c):
            s,t=c
            last=jnp.max(jnp.where((jnp.arange(e.EVENTS)<t.ecount)&(t.events[:,0]==e.EV_MOVE),jnp.arange(e.EVENTS),0))
            t=t._replace(events=t.events.at[last,7].set(2))
            return e._event((s,t),e.FAIL,source,sm)
        return jax.lax.cond(s.status[target,m]==status,same,different,c)
    def clause_failure(c):
        return e._event(c,e.SLEEP_CLAUSE,value=1)
    def blocked(c):
        if secondary:return c
        return e._event(c,e.IMMUNE,target,m,value=jnp.where(e.immunity_ops.blocks(ability,status,e)&(ability!=2),ability,0),aux=jnp.where((ability==2)&((status==e.TOX)|(status==e.PSN)),1,jnp.where(e.immunity_ops.blocks(ability,status,e),4,0)))
    def apply(c):
        s,t=c
        meta=jnp.stack((status,s.effect_order,source,sm,e.i32(1),e.i32(-1),e.i32(-1)))
        s=s._replace(status=s.status.at[target,m].set(status),toxic=s.toxic.at[target,m].set(0),
                     status_state=s.status_state.at[target,m].set(meta),effect_order=s.effect_order+1)
        c=e._event((s,t),e.STATUS,target,m,move=move,status=status)
        def sleep(c):
            c,value=e._draw(c,2,5,e.SLEEP_DURATION,2);s,t=c
            return s._replace(status_state=s.status_state.at[target,m,5:7].set(value.astype(jnp.int32))),t
        c=e._when(status==e.SLP,sleep,c)
        syncable=(status==e.BRN)|(status==e.PAR)|(status==e.TOX)|(status==e.PSN)
        if reflect:
            def sync(c):
                c=e._event(c,e.ABILITY,target,m)
                return inflict(c,target,source,status,e,False)
            c=e._when((ability==1)&(source!=target)&syncable,sync,c)
        else:
            def back(c):
                c=e._event(c,e.ABILITY,target,m)
                return e._event(c,e.FAIL,source,sm,status=status)
            c=e._when((ability==1)&(source!=target)&syncable,back,c)
        return e._when(c[0].item[target,m]==2,lambda c:lum(c,target,e),c)
    def attempt(c):
        # Native trySetStatus on a secondary with existing status does nothing.
        # Sunny Day's Immunity callback is visited for every naturally allowed
        # status, not only freeze. Its effect target persists even for sleep.
        s,t=c;visited=(~failed)&(~natural_immunity)&(s.weather_state[0]==2)
        c=(s._replace(weather_state=s.weather_state.at[5].set(jnp.where(visited,1,s.weather_state[5]))),t)
        def set_status(c):
            s,t=c;c=(s._replace(sleep_clause_target=jnp.array(True)),t)
            return jax.lax.cond(clause,clause_failure,apply,c)
        return jax.lax.cond(failed,rejected,lambda c:jax.lax.cond(immunity,blocked,set_status,c),c)
    return e._when(s.hp[target,m]>0,attempt,c)


def confuse(c,actor,target,move,e,*,fatigue=False):
    s,t=c;m=s.active[target]
    def fail(c):
        s,t=c
        last=jnp.max(jnp.where((jnp.arange(e.EVENTS)<t.ecount)&(t.events[:,0]==e.EV_MOVE),jnp.arange(e.EVENTS),0))
        t=t._replace(events=t.events.at[last,7].set(2))
        return e._event((s,t),e.FAIL,actor,s.active[actor])
    def success(c):
        c=add_volatile(c,target,2,actor,move,e,source_kind=e.phase_ops.SOURCE_CONDITION if fatigue else None)
        c=e._event(c,e.CONF_START,target,m,aux=1 if fatigue else 0)
        c,value=e._draw(c,2,6,e.CONF_DURATION,2);s,t=c
        c=(s._replace(volatile_state=s.volatile_state.at[target,m,2,5].set(value.astype(jnp.int32))),t)
        return e._when(s.item[target,m]==2,lambda c:lum(c,target,e),c)
    immune=e.suppression_ops.abilities(s)[target,m]==e.sleep_ops.OWN
    def blocked(c):return c if fatigue else e._event(c,e.IMMUNE,target,m,aux=5)
    return jax.lax.cond(immune,blocked,lambda c:jax.lax.cond(s.volatile_state[target,m,2,0]>0,fail,success,c),c)


def secondary(c,actor,move,e):
    chance=jnp.maximum(e.tables.SECONDARY_CHANCE[move],e.phase_ops.FLINCH_CHANCE[move]);status=e.tables.SECONDARY_STATUS[move]
    def attempt(c):
        c,value=e._draw(c,0,100,e.SECONDARY)
        return e._when(value<chance.astype(jnp.uint32),
            lambda c:jax.lax.cond(e.phase_ops.FLINCH_CHANCE[move]>0,lambda c:e.phase_ops.flinch(c,actor,1-actor,move,e),lambda c:inflict(c,actor,1-actor,status,e,move=move,secondary=True),c),c)
    return e._when(chance>0,attempt,c)


def before_move(c,actor,move,e):
    s,t=c;m=s.active[actor]
    event_source=jnp.where(e.MOVE_ACTOR_TARGET[move],actor,1-actor)
    def sleep(c):
        s,t=c;time=s.status_state[actor,m,5]-1
        s=s._replace(status_state=s.status_state.at[actor,m,5].set(time))
        def asleep(c):return e._event(c,e.CANT,actor,m,status=e.SLP),move!=e.called_ops.SLEEP_TALK
        return jax.lax.cond(time<=0,lambda c:(cure(c,actor,e,source=event_source),jnp.array(False)),asleep,(s,t))
    c,blocked=jax.lax.cond(s.status[actor,m]==e.SLP,sleep,lambda c:(c,jnp.array(False)),c)
    def freeze(c):
        c,value=e._draw(c,0,5,e.FREEZE)
        def frozen(c):return e._event(c,e.CANT,actor,m,status=e.FRZ),jnp.array(True)
        return jax.lax.cond(value==0,lambda c:(cure(c,actor,e,source=event_source),jnp.array(False)),frozen,c)
    c,blocked=jax.lax.cond((~blocked)&(c[0].status[actor,m]==e.FRZ)&(~e.hit_tables.DEFROST[move]),freeze,lambda c:(c,blocked),c)
    c,blocked=jax.lax.cond(~blocked,lambda c:e.phase_ops.before_move(c,actor,move,e),lambda c:(c,blocked),c)
    def confusion(c):
        s,t=c;remaining=s.volatile_state[actor,m,2,5]-1
        s=s._replace(volatile_state=s.volatile_state.at[actor,m,2,5].set(remaining));c=(s,t)
        def end(c):
            c=e._event(c,e.CONF_END,actor,m);s,t=c
            return (s._replace(volatile_state=s.volatile_state.at[actor,m,2].set(0)),t),jnp.array(False)
        def active(c):
            c=e._event(c,e.CONF_ACTIVATE,actor,m)
            c,value=e._draw(c,0,2,e.CONFUSION)
            def self_hit(c):
                s,t=c
                attack=e.field_ops.scale_stat_stage(s.stats[actor,m,1],s.boosts[actor,m,0])
                defense=e.field_ops.scale_stat_stage(s.stats[actor,m,2],s.boosts[actor,m,1])
                core=e.base_damage_core(s.level[actor,m],e.i32(40),attack,defense)+2
                c,roll=e._draw(c,0,16,e.CONF_DAMAGE)
                damage=jnp.maximum(1,(core&65535)*(100-roll.astype(jnp.int32))//100)
                c=e._damage(c,actor,damage,cause=6,source_side=actor,source_mon=m);s,t=c
                c=e._when(s.hp[actor,m]==0,lambda c:e._win(e._faint(c,actor),actor),c)
                return c,jnp.array(True)
            return jax.lax.cond(value==0,lambda c:(c,jnp.array(False)),self_hit,c)
        return jax.lax.cond(remaining==0,end,active,c)
    c,blocked=jax.lax.cond((~blocked)&(c[0].volatile_state[actor,m,2,0]>0),confusion,lambda c:(c,blocked),c)
    def paralyze(c):
        c,value=e._draw(c,0,4,e.PARALYSIS)
        cant=value==0
        c=e._when(cant,lambda c:e._event(c,e.CANT,actor,m,status=e.PAR),c)
        return c,cant
    return jax.lax.cond((~blocked)&(c[0].status[actor,m]==e.PAR),paralyze,lambda c:(c,blocked),c)


def rest(c,actor,move_row,e):
    s,t=c;m=s.active[actor];hp=s.hp[actor,m];maxhp=s.stats[actor,m,0]
    def fail(c):
        s,t=c
        return e._event((s,t),e.FAIL,actor,m,aux=1)
    def success(c):
        c=e.tracking.outcome(c,actor,e.i32(e.tracking.TRUE),e.REST)
        c=inflict(c,actor,actor,e.SLP,e,move=e.REST,overwrite=True)
        s,t=c
        # Rest installs its fixed timers after the immediate Lum update.
        s=s._replace(status_state=s.status_state.at[actor,m,5:7].set(3),hp=s.hp.at[actor,m].set(maxhp))
        return e._event((s,t),e.HEAL,actor,m,value=maxhp,aux=maxhp,status=s.status[actor,m],move=-1)
    return jax.lax.cond(hp==maxhp,fail,lambda c:jax.lax.cond(e.immunity_ops.SLEEP[e.suppression_ops.abilities(c[0])[actor,m]],lambda c:e.immunity_ops.rest_fail(c,actor,e),success,c),c)
