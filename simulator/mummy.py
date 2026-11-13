"""Pinned Gen5 Mummy's default-order DamagingHit ability replacement."""
from . import gen5_engine_tables as t
MUMMY=t.ABILITY_IDS['mummy']

def contact(c,actor,target,move,e):
    s,_=c;mon=s.active[actor];tm=s.active[target];old=s.ability[actor,mon]
    allowed=(e.suppression_ops.abilities(s)[target,tm]==MUMMY)&e.hit_tables.CONTACT[move]&(s.hp[actor,mon]>0)&(~e.ability_ops.CANT[old])&(old!=MUMMY)
    def replace(c):
        c=e.ability_ops.end(c,actor,e);s,tr=c
        s=e.ability_ops.assign(s,actor,e.i32(MUMMY),e)
        # Resolved Mummy has no Start callback; End and activation stay immediate.
        return e._event((s,tr),e.ABILITY_CHANGE,actor,mon,target=target,targetmon=tm,value=MUMMY,aux=old,status=3)
    return e._when(allowed,replace,c)
