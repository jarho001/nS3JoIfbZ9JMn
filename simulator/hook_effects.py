"""Pure JAX ability/item operations; native callbacks are never executed here.

These operations are used by shared schema6. Capability publication
and independent integration acceptance remain separate from this module.
"""
import jax
import jax.numpy as jnp
from . import suppression as suppression_ops
from mechanics.fixedpoint import apply_fixedpoint_modifier as modify
from . import hook_tables as h


def chain(previous,next_modifier):
    # Native chainModify rounds half up, unlike Battle.modify's half down.
    return (previous*next_modifier+2048)>>12


def speeds(s,staged,e):
    ability=suppression_ops.abilities(s);item=s.item;weather=s.weather_state[0]
    doubled=((ability==h.A['swiftswim'])&(weather==1))|((ability==h.A['chlorophyll'])&(weather==2))|((ability==h.A['sandrush'])&(weather==3))
    quick=(ability==h.A['quickfeet'])&(s.status!=e.NONE)&(s.status!=e.FNT)
    modifier=jnp.where(doubled,8192,jnp.where(quick,6144,4096)).astype(jnp.int32)
    modifier=chain(modifier,jnp.where(item==h.I['choicescarf'],6144,4096))
    speed=modify(staged,modifier)
    speed=jnp.where((s.status==e.PAR)&(~quick),speed//4,speed)
    return jnp.maximum(1,jnp.minimum(speed,10000))&8191


def priority(move,ability,e):
    return e.tables.PRIORITY[move]+((ability==h.A['prankster'])&(e.tables.CATEGORY[move]==0)).astype(jnp.int32)


def critical_denominator(move,ability,item):
    ratio=h.CRIT_RATIO[move]+(ability==h.A['superluck']).astype(jnp.int32)+(item==h.I['scopelens']).astype(jnp.int32)
    return h.CRIT_DENOMINATOR[jnp.clip(ratio,0,5)]


def critical_blocked(ability):
    return (ability==h.A['battlearmor'])|(ability==h.A['shellarmor'])


def accuracy(base,s,actor,physical):
    m=s.active[actor];ability=suppression_ops.abilities(s)[actor,m]
    modifier=jnp.where(ability==h.A['compoundeyes'],5325,jnp.where((ability==h.A['hustle'])&physical,3277,4096)).astype(jnp.int32)
    return modify(base,modifier)


def power(value,ability,item,physical,move_type,e,phase_modifier=4096):
    modifier=jnp.where((ability==h.A['technician'])&(value<=60),6144,4096).astype(jnp.int32)
    band=((item==h.I['muscleband'])&physical)|((item==h.I['wiseglasses'])&(~physical))
    typed=((item==h.I['silkscarf'])&(move_type==e.generic.TYPES['normal']))|((item==h.I['mysticwater'])&(move_type==e.generic.TYPES['water']))
    modifier=chain(modifier,jnp.where(band,4505,jnp.where(typed,4915,4096)))
    return modify(value,chain(modifier,phase_modifier))


def stats(attack,defense,s,actor,target,physical,move_type,e):
    m=s.active[actor];tm=s.active[target];ability=suppression_ops.abilities(s)[actor,m];defender=suppression_ops.abilities(s)[target,tm];item=s.item[actor,m]
    # Hustle returns a direct stat value before the queued chain modifiers.
    attack=jnp.where((ability==h.A['hustle'])&physical,modify(attack,e.i32(6144)),attack)
    huge=(ability==h.A['hugepower'])|(ability==h.A['purepower'])
    guts=(ability==h.A['guts'])&(s.status[actor,m]!=e.NONE)
    atk_modifier=jnp.where(physical&huge,8192,jnp.where(physical&guts,6144,4096)).astype(jnp.int32)
    thick=(defender==h.A['thickfat'])&((move_type==e.generic.TYPES['fire'])|(move_type==e.generic.TYPES['ice']))
    atk_modifier=chain(atk_modifier,jnp.where(thick,2048,4096))
    choice=((item==h.I['choiceband'])&physical)|((item==h.I['choicespecs'])&(~physical))
    atk_modifier=chain(atk_modifier,jnp.where(physical&e.form_ops.flower(s,actor),6144,4096))
    atk_modifier=chain(atk_modifier,jnp.where(choice,6144,4096))
    marvel=(defender==h.A['marvelscale'])&physical&(s.status[target,tm]!=e.NONE)
    def_modifier=jnp.where(marvel,6144,4096).astype(jnp.int32)
    eviolite=(s.item[target,tm]==h.I['eviolite'])&h.NFE[s.species[target,tm]]
    def_modifier=chain(def_modifier,jnp.where((~physical)&e.form_ops.flower(s,target),6144,4096))
    def_modifier=chain(def_modifier,jnp.where(eviolite,6144,4096))
    sand=(s.weather_state[0]==3)&(~physical)&jnp.any(s.types[target,tm]==e.generic.TYPES['rock'])
    def_modifier=chain(def_modifier,jnp.where(sand,6144,4096))
    return modify(attack,atk_modifier),modify(defense,def_modifier)


def damage_modifier(c,actor,target,physical,critical,exponent,e,move):
    s,t=c;m=s.active[actor];tm=s.active[target];ability=suppression_ops.abilities(s)[actor,m];defender=suppression_ops.abilities(s)[target,tm];item=s.item[actor,m]
    extra,modifier=e.phase_ops.damage_modifier(s,target,move)
    rows=[jnp.stack((modifier,s.speed[target,tm],e.i32(2)))];valid=[extra]
    for side in range(2):
        for index in range(2):
            applies=(side==target)&(index==jnp.where(physical,0,1))&(~critical)
            rows.append(jnp.stack((jnp.where(applies,2048,4096),e.i32(0),e.i32(4))))
            valid.append(s.screens[side,index,0]>0)
    offense=(ability==h.A['sniper'])|(ability==h.A['tintedlens'])
    defense=(defender==h.A['filter'])|(defender==h.A['solidrock'])
    offense_modifier=jnp.where((ability==h.A['sniper'])&critical,6144,jnp.where((ability==h.A['tintedlens'])&(exponent<0),8192,4096))
    rows.append(jnp.stack((offense_modifier,s.speed[actor,m],e.i32(7))));valid.append(offense)
    rows.append(jnp.stack((jnp.where(exponent>0,3072,4096),s.speed[target,tm],e.i32(7))));valid.append(defense)
    rows.append(jnp.stack((jnp.where(item==h.I['lifeorb'],5324,jnp.where(exponent>0,4915,4096)),s.speed[actor,m],e.i32(8))))
    valid.append((item==h.I['lifeorb'])|(item==h.I['expertbelt']))
    rows=jnp.stack(rows);valid=jnp.stack(valid);count=jnp.sum(valid).astype(jnp.int32);n=len(rows)
    order=jnp.lexsort((jnp.arange(n),rows[:,2],-rows[:,1],~valid));rows=rows[order]
    def shuffle(k,carry):
        c,rows=carry;key=rows[k,1:3]
        eq=jnp.all(rows[:,1:3]==key,axis=1)&(jnp.arange(n)<count)
        end=jnp.max(jnp.where(eq,jnp.arange(n),-1))+1
        def swap(carry):
            c,rows=carry;c,v=e._draw(c,k,end,e.DAMAGE_MODIFIERS,2);v=v.astype(jnp.int32)
            a,b=rows[k],rows[v];return c,rows.at[k].set(b).at[v].set(a)
        return jax.lax.cond((k+1<end)&(k<count),swap,lambda x:x,(c,rows))
    c,rows=jax.lax.fori_loop(0,n-1,shuffle,(c,rows))
    modifier=jax.lax.fori_loop(0,n,lambda k,v:jnp.where(k<count,chain(v,rows[k,0]),v),e.i32(4096))
    return c,modifier


def ability_start(c,side,e,initial=False):
    s,t=c;m=s.active[side];ability=suppression_ops.abilities(s)[side,m];weather=h.ABILITY_WEATHER[ability]
    c=e._when((ability==h.A['pressure'])|((ability==h.A['intimidate'])&(s.hp[1-side,s.active[1-side]]>0)),
        lambda c:e._event(c,e.REVEAL,side,m,value=ability,aux=(ability==h.A['intimidate']).astype(jnp.int32)),c)
    def frisk(c):
        c,_=e._draw(c,0,1,e.INIT if initial else e.FRISK_TARGET)
        s,t=c;item=s.item[1-side,s.active[1-side]]
        return e._when(item>0,lambda c:e._event(c,e.FRISK_ITEM,side,m,value=item),c)
    c=e._when((ability==h.A['frisk'])&(~c[0].fainted[1-side,c[0].active[1-side]]),frisk,c)
    def intimidate(c):
        s,t=c;foe=1-side;fm=s.active[foe];old=s.boosts[foe,fm,0];change=e.creation_ops.delta(s,foe,e.i32(-1));new=jnp.clip(old+change,-6,6)
        s=s._replace(boosts=s.boosts.at[foe,fm,0].set(new))
        s=e.tracking.boosted(s,foe,new>old,new<old)
        return e._event((s,t),e.BOOST,foe,fm,value=new-old,aux=0,status=(change<0).astype(jnp.int32))
    c=e._when((ability==h.A['intimidate'])&(c[0].hp[1-side,c[0].active[1-side]]>0),intimidate,c)
    def set_weather(c):
        s,t=c;state=jnp.stack((weather,side,m,e.i32(0),e.i32(0),e.i32(0)))
        c=e._event((s._replace(weather_state=state),t),e.WEATHER,side,m,move=ability,value=weather,aux=2)
        s,t=c;speeds=s.speed[jnp.arange(2),s.active]
        context=e.INIT if initial else e.WEATHER_CHANGE
        return e.form_ops.weather_change(c,e,context)
    c=e._when((weather>0)&((c[0].weather_state[0]!=weather)|(c[0].weather_state[4]>0)),set_weather,c)
    return e.form_ops.change(c,side,e)


def start(c,side,e,initial=False):
    ability=suppression_ops.abilities(c[0])[side,c[0].active[side]]
    def copy(c):return e.identity_ops.transform(c,side,1-side,e,imposter=True,initial=initial)[0]
    c=e.jax.lax.cond(ability==e.identity_ops.IMPOSTER,copy,lambda c:ability_start(c,side,e,initial=initial),c)
    return item_start(c,side,e)


def item_start(c,side,e):
    m=c[0].active[side]
    # Choice items' Start hook removes an old lock on acquisition/switch-in.
    def choice(c):
        s,t=c;return s._replace(volatile_state=s.volatile_state.at[side,m,3].set(0)),t
    return e._when(h.CHOICE[c[0].item[side,m]],choice,c)


def modify_move(c,actor,target,move,e):
    def lock(c):
        s,t=c;m=s.active[actor];old=s.volatile_state[actor,m,3];item=s.item[actor,m]
        c=e.effects.add_volatile(c,actor,3,target,-item,e);s,t=c
        return s._replace(volatile_state=s.volatile_state.at[actor,m,3,5].set(jnp.where(old[0]>0,old[5],move))),t
    return e._when(h.CHOICE[c[0].item[actor,c[0].active[actor]]],lock,c)


def choice_moves(s):
    return jnp.where(s.volatile_state[:,:,3,0]>0,s.volatile_state[:,:,3,5],0)


def disabled(s):
    locked=choice_moves(s)
    return (s.pp==0)|((locked[:,:,None]>0)&(s.moves!=locked[:,:,None]))


def normalize_locks(s):
    locked=choice_moves(s)
    valid=h.CHOICE[s.item]&jnp.any(s.moves==locked[:,:,None],axis=-1)
    return s._replace(volatile_state=s.volatile_state.at[:,:,3].set(jnp.where(valid[:,:,None],s.volatile_state[:,:,3],0)))
