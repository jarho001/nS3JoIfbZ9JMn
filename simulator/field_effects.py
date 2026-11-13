"""Pure JAX field, side-condition and boost operations for the shared engine."""
import jax
import jax.numpy as jnp
from mechanics.stat_stage import scale_stat_stage
from mechanics.fixedpoint import apply_fixedpoint_modifier
from . import field_tables as f


def empty_weather():
    # id, source side/mon, source move, duration, target present.
    return jnp.asarray([0,-1,-1,0,0,0],jnp.int32)


def boost(c,actor,target,move,e):
    s,t=c;m=s.active[target];change=f.BOOSTS[move]
    for step in range(7):
        k=f.BOOST_ORDER[move,step]
        def apply(c):
            s,t=c;old=s.boosts[target,m,k];new=jnp.clip(old+e.creation_ops.delta(s,target,change[k]),-6,6)
            s=s._replace(boosts=s.boosts.at[target,m,k].set(new))
            s=e.tracking.boosted(s,target,new>old,new<old)
            value=jnp.where(new!=old,e.tracking.TRUE,s.action_state[actor,s.active[actor],e.tracking.RESULT])
            s, t=e.tracking.outcome((s,t),actor,value,move)
            # Even a capped raise emits its zero-sized native boost event.
            return e._event((s,t),e.BOOST,target,m,value=new-old,aux=k,status=(e.creation_ops.delta(s,target,change[k])<0).astype(jnp.int32))
        c=e._when(change[k]!=0,apply,c)
    return e._when(c[0].action_state[actor,c[0].active[actor],e.tracking.RESULT]==e.tracking.TRUE,lambda c:e._update(e._update(c,e.HIT_UPDATE),e.HIT_UPDATE),c)


def _fail(c,actor,move_row,e):
    s,t=c;t=t._replace(events=t.events.at[move_row,7].set(2))
    return e._event((s,t),e.FAIL,actor,s.active[actor])


def move(c,actor,move,move_row,e):
    weather=f.WEATHER[move];screen=f.SCREEN[move]
    def weather_move(c):
        def start(c):
            s,t=c;m=s.active[actor]
            duration=jnp.where(s.item[actor,m]==f.WEATHER_ROCKS[weather],8,5)
            state=jnp.stack((weather,actor,m,move,duration,e.i32(0)))
            c=e._event(e.tracking.outcome((s._replace(weather_state=state),t),actor,e.i32(e.tracking.TRUE),move),e.WEATHER,value=weather)
            # Field.setWeather broadcasts WeatherChange before hit resolution.
            s,t=c;speeds=s.speed[jnp.arange(2),s.active]
            return e.form_ops.weather_change(c,e,e.WEATHER_CHANGE)
        return jax.lax.cond(c[0].weather_state[0]==weather,lambda c:_fail(c,actor,move_row,e),start,c)
    def screen_move(c):
        index=screen-1
        def start(c):
            s,t=c;m=s.active[actor];duration=jnp.where(s.item[actor,m]==f.LIGHT_CLAY,8,5)
            row=jnp.stack((duration,actor,m,move,s.effect_order))
            s=s._replace(screens=s.screens.at[actor,index].set(row),effect_order=s.effect_order+1)
            return e._event(e.tracking.outcome((s,t),actor,e.i32(e.tracking.TRUE),move),e.SIDE_START,actor,move=move)
        return jax.lax.cond(c[0].screens[actor,index,0]>0,lambda c:_fail(c,actor,move_row,e),start,c)
    return jax.lax.cond(weather>0,weather_move,screen_move,c)


def damage_parameters(c,actor,target,move,crit,physical,attack,defense,e):
    s,t=c;m=s.active[actor];tm=s.active[target]
    offensive=jnp.where(physical,0,2);defensive=jnp.where(physical,1,3)
    atkstage=s.boosts[actor,m,offensive];defstage=s.boosts[target,tm,defensive]
    atkstage=jnp.where((crit==0)&(atkstage<0),0,atkstage)
    defstage=jnp.where((crit==0)&(defstage>0),0,defstage)
    attack=scale_stat_stage(attack,atkstage);defense=scale_stat_stage(defense,defstage)
    sand=(s.weather_state[0]==3)&(~physical)&jnp.any(s.types[target,tm]==e.generic.TYPES['rock'])
    # Collecting any weather damage/stat hook sets the native effect target.
    weather=s.weather_state[0]
    present=(weather==1)|(weather==2)|((weather==3)&(~physical))
    s=s._replace(weather_state=s.weather_state.at[5].set(jnp.where(present,1,s.weather_state[5])))
    return (s,t),attack,defense


def weather_damage(core,move_type,s,e):
    weather=s.weather_state[0];water=move_type==e.generic.TYPES['water'];fire=move_type==e.generic.TYPES['fire']
    boost=((weather==1)&water)|((weather==2)&fire)
    weaken=((weather==1)&fire)|((weather==2)&water)
    modifier=jnp.where(boost,6144,jnp.where(weaken,2048,4096)).astype(jnp.int32)
    return apply_fixedpoint_modifier(core,modifier)


def accuracy(base,s,actor,target,e):
    m=s.active[actor];tm=s.active[target]
    stage=jnp.clip(s.boosts[actor,m,5]-s.boosts[target,tm,6],-6,6)
    return jnp.where(stage>=0,base*(3+stage)//3,base*3//(3-stage))


def weather_tick(c,e):
    s,t=c;weather=s.weather_state[0];remaining=jnp.maximum(0,s.weather_state[4]-1);timed=s.weather_state[4]>0
    def end(c):
        s,t=c;state=empty_weather().at[5].set(1)
        c=e._event((s._replace(weather_state=state),t),e.WEATHER,value=0)
        speeds=s.speed[jnp.arange(2),s.active]
        return e.form_ops.weather_change(c,e,e.WEATHER_CHANGE)
    def active(c):
        s,t=c;s=s._replace(weather_state=s.weather_state.at[4].set(remaining));c=(s,t)
        c=e._event(c,e.WEATHER,value=weather,aux=1)
        speeds=s.speed[jnp.arange(2),s.active];tie=(speeds[0]==speeds[1])&jnp.all(~s.fainted[jnp.arange(2),s.active])
        c,word=jax.lax.cond(tie,lambda c:e._draw(c,0,2,e.WEATHER_ORDER,2),lambda c:(c,jnp.uint32(0)),c)
        first=jnp.where(tie,word.astype(jnp.int32),jnp.argmax(speeds))
        for k in range(2):
            side=jnp.where(k==0,first,1-first);s,t=c;m=s.active[side]
            immune=jnp.any(jnp.where(weather==3,
                (s.types[side,m]==e.generic.TYPES['rock'])|(s.types[side,m]==e.generic.TYPES['ground'])|(s.types[side,m]==e.generic.TYPES['steel']),
                s.types[side,m]==e.generic.TYPES['ice']))
            ability=e.suppression_ops.abilities(s)[side,m]
            immune=immune|(s.volatile_state[side,m,e.phase_ops.VOLATILE_IDS['dig'],0]>0)|(s.volatile_state[side,m,e.phase_ops.VOLATILE_IDS['dive'],0]>0)
            immune=immune|((weather==3)&(ability==e.hook_tables.A['sandrush']))|((weather==4)&(ability==e.hook_tables.A['icebody']))
            def ice_body(c):
                s,t=c;hp=s.hp[side,m];maximum=s.stats[side,m,0];newhp=jnp.minimum(maximum,hp+jnp.maximum(1,maximum//16))
                s=s._replace(hp=s.hp.at[side,m].set(newhp))
                return e._event((s,t),e.HEAL,side,m,value=newhp,aux=maximum,status=s.status[side,m],move=2)
            c=e._when((weather==4)&(ability==e.hook_tables.A['icebody'])&(s.hp[side,m]>0)&(s.hp[side,m]<s.stats[side,m,0]),ice_body,c)
            c=e._when(((weather==3)|(weather==4))&(~immune)&(s.hp[side,m]>0),
                      lambda c:e._damage(c,side,s.stats[side,m,0]//16,cause=jnp.where(weather==3,7,8)),c)
        s,t=c
        s=s._replace(weather_state=s.weather_state.at[5].set(jnp.where((weather==3)|(weather==4),1,s.weather_state[5])))
        c=(s,t);last=first
        for k in range(2):
            side=jnp.where(k==0,first,1-first);s,t=c;m=s.active[side]
            faint=(s.hp[side,m]==0)&(~s.fainted[side,m])
            c=e._when(faint,lambda c:e._faint(c,side),c);last=jnp.where(faint,side,last)
        return e._win(c,last)
    return jax.lax.cond(timed&(remaining==0),end,active,c)


def screen_tick(c,side,index,e):
    s,t=c;left=s.screens[side,index,0]-1;move=s.screens[side,index,3]
    def end(c):
        s,t=c;s=s._replace(screens=s.screens.at[side,index].set(0))
        return e._event((s,t),e.SIDE_END,side,move=move,aux=1)
    return jax.lax.cond(left==0,end,lambda c:(c[0]._replace(screens=c[0].screens.at[side,index,0].set(left)),c[1]),c)
