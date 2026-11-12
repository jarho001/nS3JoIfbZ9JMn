"""Gen5 Pursuit interception queued through the single ordinary move dispatcher."""
import jax.numpy as jnp
from . import gen5_engine_tables as t
MOVE=t.MOVE_IDS['pursuit']
SLOT=22
MOVE_IMPLEMENTED=jnp.asarray([key=='pursuit'for key in t.MOVE_KEYS],jnp.bool_)

def boosted(s,target,move,e):
    return (move==MOVE)&((s.action_state[target,s.active[target],e.tracking.CALLED_BACK]>0)|s.pivot[target])


def prepare(c,row,e):
    side,mon=row[1],row[2]
    def add(c):
        c=e.effects.add_volatile(c,side,SLOT,side,MOVE,e)
        s,tr=c;return s._replace(volatile_state=s.volatile_state.at[side,mon,SLOT,5].set(1)),tr
    return e._when((c[0].active[side]==mon)&(c[0].hp[side,mon]>0),add,c)


def queued(s,side,e):
    source=1-side;sm=s.active[source]
    return (jnp.arange(e.QUEUE)<s.qcount)&(s.queue[:,0]==e.ATTACK)&(s.queue[:,1]==source)&(s.queue[:,2]==sm)


def eligible(s,side,e):
    source=1-side;sm=s.active[source];mon=s.active[side]
    encore=s.volatile_state[source,sm,e.phase_ops.VOLATILE_IDS['encore']]
    return (s.hp[side,mon]>0)&(s.action_state[side,mon,e.tracking.SKIP_PURSUIT]==0)&(s.volatile_state[source,sm,SLOT,0]>0)&(s.hp[source,sm]>0)&jnp.any(queued(s,side,e))&((encore[0]==0)|(encore[7]==MOVE))


def complete(c,side,e):
    s,tr=c;mon=s.active[side];continuing=(s.hp[side,mon]>0)&(~s.pivot[side])
    s=e.tracking.put(s,side,e.tracking.PURSUIT_ACTIVE,jnp.where(continuing,0,s.action_state[side,mon,e.tracking.PURSUIT_ACTIVE]))
    s=e.tracking.put(s,side,e.tracking.SKIP_PURSUIT,(~continuing).astype(jnp.int32))
    return s._replace(pivot=s.pivot.at[side].set(s.pivot[side]&(s.hp[side,mon]>0))),tr


def schedule(c,side,e,finish_row=None):
    s,tr=c;mon=s.active[side];qmask=queued(s,side,e);ix=jnp.argmax(qmask)
    attack=s.queue[ix].at[6].set(MOVE)
    keep=(jnp.arange(e.QUEUE)<s.qcount)&(~qmask);indices=jnp.nonzero(keep,size=e.QUEUE,fill_value=0)[0];count=jnp.sum(keep).astype(jnp.int32)
    tail=jnp.where((jnp.arange(e.QUEUE)<count)[:,None],s.queue[indices],0)
    head=attack[None,:]if finish_row is None else jnp.stack((attack,finish_row.at[0].set(e.FINISH_SWITCH)))
    size=head.shape[0];queue=jnp.concatenate((head,tail),axis=0)[:e.QUEUE]
    s=s._replace(queue=queue,qcount=count+size)
    c=e._when(s.action_state[side,mon,e.tracking.PURSUIT_ACTIVE]==0,lambda c:e._event(c,e.ACTIVATE,side,mon,move=MOVE),(s,tr))
    return e.tracking.put(c[0],side,e.tracking.PURSUIT_ACTIVE,1),c[1]._replace(overflow=c[1].overflow|(count+size>e.QUEUE))


def before_switch(c,row,e):
    side=row[1];mon=c[0].active[side]
    c=(e.tracking.put(c[0],side,e.tracking.CALLED_BACK,1),c[1])
    def ordinary(c):
        c=e._when(c[0].action_state[side,mon,e.tracking.SKIP_PURSUIT]==0,lambda c:complete(e._update(c,e.UPDATE),side,e),c)
        return e._switch_entry(c,row)
    return e.jax.lax.cond(eligible(c[0],side,e),lambda c:schedule(c,side,e,finish_row=row),ordinary,c)


def pivot(c,side,e):
    mon=c[0].active[side]
    def start(c):
        return e.jax.lax.cond(eligible(c[0],side,e),lambda c:schedule(c,side,e),lambda c:complete(e._update(c,e.UPDATE),side,e),c)
    return e._when(c[0].action_state[side,mon,e.tracking.SKIP_PURSUIT]==0,start,c)
