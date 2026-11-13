"""Base/effective identity and Gen5 Transform virtual move-slot lifecycle."""
import jax.numpy as jnp
from . import gen5_engine_tables as t
FIELDS=('base_species','base_ability','base_stats','base_moves','base_pp','base_used','transformed','known_type','apparent_types','ability_effect_id')
MOVE=t.MOVE_IDS['transform'];IMPOSTER=t.ABILITY_IDS['imposter']
MOVE_IMPLEMENTED=jnp.asarray([k=='transform'for k in t.MOVE_KEYS],jnp.bool_)
ABILITY_IMPLEMENTED=jnp.asarray([k=='imposter'for k in t.ABILITIES],jnp.bool_)
BASE_PP=t.mv(lambda row:row['pp'])


def infer(fields,e):
    return dict(base_species=fields['species'],base_ability=fields['ability'],base_stats=fields['stats'],base_moves=fields['moves'],base_pp=fields['pp'],base_used=fields['action_state'][:,:,e.tracking.USED_SLOTS],transformed=jnp.zeros((2,6),jnp.bool_),known_type=fields['species']>0,apparent_types=fields['types'],ability_effect_id=fields['ability'])


def maximum_pp(s):
    return jnp.where(s.transformed[:,:,None],jnp.minimum(5,BASE_PP[s.moves]),t.PP[s.moves])


def sync(s,side,e):
    mon=s.active[side]
    def update(s):return s._replace(base_pp=s.base_pp.at[side,mon].set(s.pp[side,mon]),base_used=s.base_used.at[side,mon].set(s.action_state[side,mon,e.tracking.USED_SLOTS]))
    return e.jax.lax.cond(s.transformed[side,mon],lambda s:s,update,s)


def derived(s,species,e):
    return e.derive_stored_stats(t.BASE_STATS[species],s.level,s.ivs,s.evs,e.generic.NATURE_MULTIPLIERS[s.nature],t.SHEDINJA[species])


def restore(s,side,e):
    mon=s.active[side];base=derived(s,s.base_species,e)[side,mon]
    s=s._replace(species=s.species.at[side,mon].set(s.base_species[side,mon]),ability=s.ability.at[side,mon].set(s.base_ability[side,mon]),
      stats=s.stats.at[side,mon].set(base),base_stats=s.base_stats.at[side,mon].set(base),moves=s.moves.at[side,mon].set(s.base_moves[side,mon]),pp=s.pp.at[side,mon].set(s.base_pp[side,mon]),
      types=s.types.at[side,mon].set(t.SPECIES_TYPES[s.base_species[side,mon]]),apparent_types=s.apparent_types.at[side,mon].set(t.SPECIES_TYPES[s.base_species[side,mon]]),known_type=s.known_type.at[side,mon].set(True),transformed=s.transformed.at[side,mon].set(False))
    return e.tracking.put(s,side,e.tracking.USED_SLOTS,s.base_used[side,mon])


def transform(c,actor,target,e,imposter=False,initial=False):
    s,_=c;mon=s.active[actor];tm=s.active[target]
    allowed=(s.illusion[actor,mon]<0)&(s.illusion[target,tm]<0)&(~s.fainted[target,tm])&(~s.transformed[actor,mon])&(~s.transformed[target,tm])&(s.volatile_state[target,tm,e.sub_ops.SLOT,0]==0)
    def copy(c):
        s,tr=c;old=s.ability[actor,mon];new=s.ability[target,tm];species=s.species[target,tm]
        # setSpecies caches stats from the copier's genotype before target stats
        # replace its non-HP values; residual update later refreshes action Speed.
        own= e.derive_stored_stats(t.BASE_STATS[species],s.level[actor,mon],s.ivs[actor,mon],s.evs[actor,mon],e.generic.NATURE_MULTIPLIERS[s.nature[actor,mon]],t.SHEDINJA[species])
        stats=s.stats[actor,mon].at[1:].set(s.stats[target,tm,1:]);moves=s.moves[target,tm]
        s=s._replace(species=s.species.at[actor,mon].set(species),stats=s.stats.at[actor,mon].set(stats),speed=s.speed.at[actor,mon].set(own[5]),
          types=s.types.at[actor,mon].set(s.types[target,tm]),apparent_types=s.apparent_types.at[actor,mon].set(s.apparent_types[target,tm]),known_type=s.known_type.at[actor,mon].set(False),
          moves=s.moves.at[actor,mon].set(moves),pp=s.pp.at[actor,mon].set(jnp.minimum(5,BASE_PP[moves])),boosts=s.boosts.at[actor,mon].set(s.boosts[target,tm]),transformed=s.transformed.at[actor,mon].set(True))
        s=e.tracking.put(s,actor,e.tracking.USED_SLOTS,0);s=e.tracking.put(s,actor,e.tracking.TIMES_ATTACKED,s.action_state[target,tm,e.tracking.TIMES_ATTACKED])
        c=e._event((s,tr),e.TRANSFORM,actor,mon,target=target,targetmon=tm,aux=int(imposter))
        s,tr=c;s=s._replace(ability_seek=s.ability_seek.at[actor,mon].set(0),ability=s.ability.at[actor,mon].set(new),ability_order=s.ability_order.at[actor,mon].set(s.effect_order),ability_effect_id=s.ability_effect_id.at[actor,mon].set(new),effect_order=s.effect_order+1)
        c=e._when(old!=new,lambda c:e.hook_ops.ability_start(c,actor,e,initial=initial),(s,tr))
        return c,jnp.array(True)
    return e.jax.lax.cond(allowed,copy,lambda c:(c,jnp.array(False)),c)


def move(c,actor,target,move_row,e):
    c,success=transform(c,actor,target,e)
    def fail(c):
        s,tr=c;tr=tr._replace(events=tr.events.at[move_row,7].set(2))
        return e._event((s,tr),e.FAIL,actor,s.active[actor])
    c=e._when(~success,fail,c)
    c=e._when(success,lambda c:e.tracking.attacked(c,target,actor,MOVE,e.i32(0),e.i32(e.tracking.TRUE),e.i32(0),e),c)
    c=e.tracking.outcome(c,actor,jnp.where(success,e.tracking.TRUE,e.tracking.FALSE),MOVE)
    return e._when(success,lambda c:e._update(e._update(c,e.HIT_UPDATE),e.HIT_UPDATE),c)


def valid(s,e):
    base=jnp.clip(s.base_species,0,len(t.SPECIES)-1);actual=derived(s,s.species,e)
    occupied=s.species>0
    common=jnp.all((s.base_species>=0)&(s.base_species<len(t.SPECIES)))&jnp.all((s.base_ability>=0)&(s.base_ability<len(t.ABILITIES)))
    common=common&jnp.all((s.base_moves>=0)&(s.base_moves<len(t.MOVE_KEYS)))&jnp.all((s.base_pp>=0)&(s.base_pp<=t.PP[jnp.clip(s.base_moves,0,len(t.MOVE_KEYS)-1)]))&jnp.all((s.base_used>=0)&(s.base_used<=15))
    ordinary=((s.species==s.base_species)|e.form_ops.valid_pair(s)|e.trigger_form_ops.valid_pair(s))&jnp.all(s.stats==actual,axis=-1)&jnp.all(s.base_stats==s.stats,axis=-1)&jnp.all(s.base_moves==s.moves,axis=-1)&jnp.all(s.base_pp==s.pp,axis=-1)
    virtual=(s.stats[:,:,0]==s.base_stats[:,:,0])&jnp.all((s.stats[:,:,1:]>0)&(s.stats[:,:,1:]<=1000),axis=-1)
    return common&jnp.all(jnp.where(occupied,jnp.where(s.transformed,virtual,ordinary),~s.transformed))&jnp.all(s.apparent_types==s.types)
