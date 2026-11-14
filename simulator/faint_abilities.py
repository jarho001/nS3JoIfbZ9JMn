"""Source-Move Moxie reactions after the victim's faint, before recoil."""
from . import gen5_engine_tables as tables
MOXIE=tables.ABILITY_IDS['moxie']


def moxie(c,actor,e):
    s,_=c;mon=s.active[actor]
    allowed=(s.hp[actor,mon]>0)&(e.suppression_ops.abilities(s)[actor,mon]==MOXIE)
    return e._when(allowed,lambda c:e.boost_ability_ops.raise_stat(c,actor,e.i32(0),e.i32(1),e.i32(MOXIE),e),c)
