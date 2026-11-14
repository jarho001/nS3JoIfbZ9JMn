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
    modifier=chain(modifier,jnp.where(e.gem_ops.speed(s,ability),8192,4096))
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


def accuracy(base,s,actor,target,category):
    m=s.active[actor];tm=s.active[target];effective=suppression_ops.abilities(s);ability=effective[actor,m];defender=effective[target,tm]
    physical=category==1
    base=jnp.where((base>0)&(category==0)&(defender==h.A['wonderskin']),50,base)
    modifier=jnp.where(ability==h.A['compoundeyes'],5325,jnp.where((ability==h.A['hustle'])&physical,3277,jnp.where(ability==h.A['victorystar'],4506,4096))).astype(jnp.int32)
    tangled=(defender==h.A['tangledfeet'])&(s.volatile_state[target,tm,2,0]>0)
    modifier=chain(modifier,jnp.where(tangled,2048,4096))
    return modify(base,modifier)


def power(value,ability,item,physical,move_type,e,phase_modifier=4096,defender_ability=0,move=0,move_modifier=4096,gem_modifier=4096):
    modifier=jnp.where((ability==h.A['technician'])&(value<=60),6144,4096).astype(jnp.int32)
    boosted=((ability==h.A['ironfist'])&h.PUNCH[move])|((ability==h.A['reckless'])&h.RECKLESS[move])
    # Iron Fist/Reckless priority23: before Dry Skin17 and item modifiers.
    modifier=chain(modifier,jnp.where(boosted,4915,4096))
    # Native SourceBasePower Dry Skin priority17: after Technician30, before items16.
    dry=(defender_ability==h.A['dryskin'])&(move_type==e.generic.TYPES['fire'])
    modifier=chain(modifier,jnp.where(dry,5120,4096))
    band=((item==h.I['muscleband'])&physical)|((item==h.I['wiseglasses'])&(~physical))
    typed=((item==h.I['silkscarf'])&(move_type==e.generic.TYPES['normal']))|((item==h.I['mysticwater'])&(move_type==e.generic.TYPES['water']))
    modifier=chain(modifier,jnp.where(band,4505,jnp.where(typed,4915,4096)))
    modifier=chain(modifier,move_modifier)
    modifier=chain(modifier,gem_modifier)
    return modify(value,chain(modifier,phase_modifier))


def stats(c,attack,defense,actor,target,physical,move_type,e,defensive_physical=None):
    defensive_physical=physical if defensive_physical is None else defensive_physical
    s,_=c;m=s.active[actor];tm=s.active[target];ability=suppression_ops.abilities(s)[actor,m];defender=suppression_ops.abilities(s)[target,tm];item=s.item[actor,m]
    # Hustle returns a direct stat value before the queued chain modifiers.
    attack=jnp.where((ability==h.A['hustle'])&physical,modify(attack,e.i32(6144)),attack)
    huge=(ability==h.A['hugepower'])|(ability==h.A['purepower'])
    guts=(ability==h.A['guts'])&(s.status[actor,m]!=e.NONE)
    atk_modifier=jnp.where(physical&huge,8192,jnp.where(physical&guts,6144,4096)).astype(jnp.int32)
    thick=(defender==h.A['thickfat'])&((move_type==e.generic.TYPES['fire'])|(move_type==e.generic.TYPES['ice']))
    atk_modifier=chain(atk_modifier,jnp.where(thick,2048,4096))
    low_hp=(h.LOW_HP_TYPE[ability]==move_type)&(s.hp[actor,m]*3<=s.stats[actor,m,0])
    atk_modifier=chain(atk_modifier,jnp.where(low_hp,6144,4096))
    defeatist=(ability==h.A['defeatist'])&(s.hp[actor,m]*2<=s.stats[actor,m,0])
    atk_modifier=chain(atk_modifier,jnp.where(defeatist,2048,4096))
    flash=s.volatile_state[actor,m,e.flash_ops.SLOT,0]>0
    flash_power=flash&(ability==e.flash_ops.ABILITY)&(move_type==e.generic.TYPES['fire'])
    atk_modifier=chain(atk_modifier,jnp.where(flash_power,6144,4096))
    choice=((item==h.I['choiceband'])&physical)|((item==h.I['choicespecs'])&(~physical))
    atk_modifier=chain(atk_modifier,jnp.where(physical&e.form_ops.flower(s,actor),6144,4096))
    atk_modifier=chain(atk_modifier,jnp.where(choice,6144,4096))
    atk_modifier=chain(atk_modifier,e.species_item_ops.offense(s,actor,physical))
    marvel=(defender==h.A['marvelscale'])&defensive_physical&(s.status[target,tm]!=e.NONE)
    def_modifier=jnp.where(marvel,6144,4096).astype(jnp.int32)
    eviolite=(s.item[target,tm]==h.I['eviolite'])&h.NFE[s.species[target,tm]]
    def_modifier=chain(def_modifier,jnp.where((~defensive_physical)&e.form_ops.flower(s,target),6144,4096))
    def_modifier=chain(def_modifier,jnp.where(eviolite,6144,4096))
    def_modifier=chain(def_modifier,e.species_item_ops.defense(s,target,defensive_physical))
    sand=(s.weather_state[0]==3)&(~defensive_physical)&jnp.any(s.types[target,tm]==e.generic.TYPES['rock'])
    def_modifier=chain(def_modifier,jnp.where(sand,6144,4096))
    # Native collects both priority5 ModifySpA handlers before checking their
    # HP/type conditions. Equal cached Speed shuffles even inactive modifiers.
    # Flash Fire is a Condition (subOrder2), not an Ability (subOrder7), so
    # it cannot fully tie with Thick Fat and must not consume that shuffle.
    raw_source=s.ability[actor,m];raw_target=s.ability[target,tm]
    tie=(~physical)&((h.LOW_HP_TYPE[raw_source]>=0)|(raw_source==h.A['defeatist']))&(raw_target==h.A['thickfat'])&(s.speed[actor,m]==s.speed[target,tm])
    c=e._when(tie,lambda c:e._draw(c,0,2,e.DAMAGE_CALC_SORT,2)[0],c)
    return c,modify(attack,atk_modifier),modify(defense,def_modifier)


def damage_modifier(c,actor,target,physical,critical,exponent,e,move):
    s,t=c;m=s.active[actor];tm=s.active[target];ability=suppression_ops.abilities(s)[actor,m];defender=suppression_ops.abilities(s)[target,tm];item=s.item[actor,m]
    extra,modifier=e.phase_ops.damage_modifier(s,target,move)
    rows=[jnp.stack((modifier,s.speed[target,tm],e.i32(2)))];valid=[extra]
    for side in range(2):
        for index in range(2):
            applies=(side==target)&(index==jnp.where(physical,0,1))&(~critical)
            rows.append(jnp.stack((jnp.where(applies,2048,4096),e.i32(0),e.i32(4))))
            valid.append(s.screens[side,index,0]>0)
    raw_source=s.ability[actor,m];raw_target=s.ability[target,tm]
    offense=(raw_source==h.A['sniper'])|(raw_source==h.A['tintedlens'])
    defense=(raw_target==h.A['filter'])|(raw_target==h.A['solidrock'])|(raw_target==h.A['multiscale'])
    offense_modifier=jnp.where((ability==h.A['sniper'])&critical,6144,jnp.where((ability==h.A['tintedlens'])&(exponent<0),8192,4096))
    rows.append(jnp.stack((offense_modifier,s.speed[actor,m],e.i32(7))));valid.append(offense)
    multiscale=(defender==h.A['multiscale'])&(s.hp[target,tm]>=s.stats[target,tm,0])
    mitigation=((defender==h.A['filter'])|(defender==h.A['solidrock']))&(exponent>0)
    defense_modifier=jnp.where(multiscale,2048,jnp.where(mitigation,3072,4096))
    rows.append(jnp.stack((defense_modifier,s.speed[target,tm],e.i32(7))));valid.append(defense)
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


def ordinary_ability_start(c,side,e,initial=False):
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
        c=e._event((s,t),e.BOOST,foe,fm,value=new-old,aux=0,status=(change<0).astype(jnp.int32))
        return e.boost_ability_ops.after_drop(c,side,foe,new-old,e)
    def try_intimidate(c):
        denied=e.stat_drop_ops.blocked(c[0],side,1-side,e.i32(-1),e)
        return e.jax.lax.cond(denied,lambda c:e.stat_drop_ops.fail(c,1-side,e),intimidate,c)
    c=e._when((ability==h.A['intimidate'])&(c[0].hp[1-side,c[0].active[1-side]]>0),try_intimidate,c)
    def set_weather(c):
        s,t=c;state=jnp.stack((weather,side,m,e.i32(0),e.i32(0),e.i32(0)))
        c=e._event((s._replace(weather_state=state),t),e.WEATHER,side,m,move=ability,value=weather,aux=2)
        s,t=c;speeds=s.speed[jnp.arange(2),s.active]
        context=e.INIT if initial else e.WEATHER_CHANGE
        return e.form_ops.weather_change(c,e,context)
    c=e._when((weather>0)&((c[0].weather_state[0]!=weather)|(c[0].weather_state[4]>0)),set_weather,c)
    c=e._when(ability==h.A['download'],lambda c:e.boost_ability_ops.download(c,side,e),c)
    return e.form_ops.change(c,side,e)


def ability_start(c,side,e,initial=False):
    s,_=c;mon=s.active[side]
    c=e._when((suppression_ops.abilities(s)[side,mon]==e.trace_ops.TRACE)&(s.hp[side,mon]>0),lambda c:e.trace_ops.start(c,side,e,initial=initial),c)
    # A Trace with no candidate has no ordinary Start effect. Factor this call
    # out of the branches so acquisition does not duplicate the dispatcher.
    return ordinary_ability_start(c,side,e,initial=initial)


def start(c,side,e,initial=False):
    ability=suppression_ops.abilities(c[0])[side,c[0].active[side]]
    def copy(c):return e.identity_ops.transform(c,side,1-side,e,imposter=True,initial=initial)[0]
    c=e.jax.lax.cond(ability==e.identity_ops.IMPOSTER,copy,lambda c:ability_start(c,side,e,initial=initial),c)
    return item_start(c,side,e,switching=True)


def item_start(c,side,e,switching=False):
    m=c[0].active[side]
    # Choice items' Start hook removes an old lock on acquisition/switch-in.
    def choice(c):
        s,t=c;return s._replace(volatile_state=s.volatile_state.at[side,m,3].set(0)),t
    c=e._when(h.CHOICE[c[0].item[side,m]],choice,c)
    return c if switching else e.stat_drop_ops.restore(c,side,e)


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
