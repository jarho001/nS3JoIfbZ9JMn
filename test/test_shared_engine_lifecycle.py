import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import jax
import jax.numpy as jnp
import numpy as np
from simulator import shared_engine as e
from simulator.shared_engine_io import player_view
from test.shared_engine_support import native_cases,initialize_case,choices_array,assert_step
from test.test_shared_engine_cases import COMPILED

BATCH=jax.jit(jax.vmap(e.advance))
RESET=jax.jit(e.reset)

def equal(test,a,b):
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b),strict=True):np.testing.assert_array_equal(x,y)

def stack(items):return jax.tree.map(lambda *x:jnp.stack(x),*items)
def lane(tree,k):return jax.tree.map(lambda x:x[k],tree)

class SharedEngineLifecycleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases=native_cases();cls.by_id={c['id']:c for c in cls.cases}

    def test_eager_and_compiled_boundary_equivalence(self):
        for case in self.cases:
            if case['setup'].get('fullBattle'):continue
            name=case['id'];s,_=initialize_case(case)
            for step in case['steps']:
                choices=choices_array(s,step['choices'])
                with jax.disable_jit():result=e.advance(s,choices)
                compiled=COMPILED(s,choices);equal(self,result,compiled)
                assert_step(self,s,result,step,name+' eager');s=result.state

    def test_catalog_boundaries_in_heterogeneous_batches(self):
        inputs=[]
        for case in self.cases:
            if case['setup'].get('fullBattle'):continue
            s,_=initialize_case(case)
            for step in case['steps']:
                action=choices_array(s,step['choices']);inputs.append((s,action))
                s=COMPILED(s,action).state
        self.assertGreater(len(inputs),100)
        for offset in range(0,len(inputs),32):
            chunk=inputs[offset:offset+32]
            chunk+=chunk[-1:]*(32-len(chunk))
            result=BATCH(stack([x[0] for x in chunk]),jnp.stack([x[1] for x in chunk]))
            for k,(s,action) in enumerate(chunk):equal(self,lane(result,k),COMPILED(s,action))
        print(f'CPU heterogeneous catalog agreement: {len(inputs)} native boundaries, 32 lanes',flush=True)

    def test_complete_battles_in_heterogeneous_batches(self):
        battles=[c for c in self.cases if c['setup'].get('fullBattle')]
        self.assertGreaterEqual(len(battles),4)
        for size in (1,8,32):
            cases=[battles[k%len(battles)] for k in range(size)]
            scalars=[initialize_case(c)[0] for c in cases];batched=stack(scalars)
            for boundary in range(max(len(c['steps']) for c in cases)):
                choices=[choices_array(s,c['steps'][boundary]['choices']) if boundary<len(c['steps']) else e.i32([-1,-1]) for s,c in zip(scalars,cases)]
                result=BATCH(batched,jnp.stack(choices))
                for k,(s,action,c) in enumerate(zip(scalars,choices,cases)):
                    expected=COMPILED(s,action);actual=lane(result,k)
                    equal(self,actual,expected)
                    scalars[k]=expected.state
                batched=result.state
            self.assertTrue(np.all(np.asarray(batched.winner)>=0))
        print('CPU heterogeneous complete-trace scalar/JIT/vmap agreement: batch sizes 1, 8, 32',flush=True)

    def test_serialized_continuation_private_commit_and_reset(self):
        # Splits at all request boundaries, including pending, pivot and forced.
        for name in ('commit-replace','commit-cancel','pivot-resume','forced-replacement','F03-p1'):
            case=self.by_id[name];s,_=initialize_case(case)
            for step in case['steps']:
                with tempfile.TemporaryDirectory() as directory:
                    path=Path(directory)/'state.npz'
                    np.savez(path,**dict(zip(e.State._fields,jax.device_get(s),strict=True)))
                    with np.load(path,allow_pickle=False) as saved:
                        restored=e.State(*(jnp.asarray(saved[name]) for name in e.State._fields))
                    equal(self,s,restored)
                action=choices_array(s,step['choices'])
                a=COMPILED(s,action);b=COMPILED(restored,action);equal(self,a,b)
                if int(a.code)==e.PENDING:
                    for side in range(2):self.assertEqual(player_view(s,side),player_view(a.state,side))
                    self.assertEqual(int(a.trace.ecount),0);self.assertEqual(int(a.trace.dcount),0)
                s=a.state
        replacement,_=initialize_case(self.by_id['wisp-hit'])
        terminal,_=initialize_case(self.by_id['struggle-double-ko'])
        terminal=COMPILED(terminal,e.i32([10,10])).state
        token=jnp.stack((terminal.generation,terminal.episode))
        r=RESET(terminal,replacement,token);self.assertEqual(int(r.code),e.OK)
        equal(self,r.state,replacement._replace(generation=terminal.generation+1))
        live=RESET(replacement,replacement,jnp.stack((replacement.generation,replacement.episode)))
        self.assertEqual(int(live.code),e.INVALID);equal(self,live.state,replacement)
        stale=RESET(terminal,replacement,token+1);self.assertNotEqual(int(stale.code),e.OK);equal(self,stale.state,terminal)
        mixed=stack([terminal,replacement]);replacements=stack([replacement,replacement]);tokens=jnp.stack([token,jnp.stack((replacement.generation,replacement.episode))])
        out=jax.jit(jax.vmap(e.reset))(mixed,replacements,tokens)
        equal(self,lane(out,0),r);equal(self,lane(out,1),live)

    def test_information_equivalence_and_submission_orders(self):
        case=self.by_id['F01-p1'];a,_=initialize_case(case)
        # Native constructed alternatives change hidden reserves and held item,
        # preserving the active reveal. These histories are indistinguishable.
        alt=json.loads(json.dumps(case));alt['setup']['p2'][1]['item']='Lum Berry'
        alt['setup']['p2'][1]['moves']=['Recover','Protect']
        b,_=initialize_case(alt)
        self.assertEqual(player_view(a,0),player_view(b,0))
        # Observe the same native HP interval through two real heal histories.
        pairs=[]
        for name in ('privacy-hp-a','privacy-hp-b'):
            case=self.by_id[name];state,_=initialize_case(case)
            for step in case['steps']:state=COMPILED(state,choices_array(state,step['choices'])).state
            pairs.append(state)
        self.assertEqual(player_view(pairs[0],0),player_view(pairs[1],0))
        self.assertNotEqual(player_view(pairs[0],1),player_view(pairs[1],1))
        # Different opposing pending choices never enter p1's view.
        c=COMPILED(a,e.i32([-1,0])).state;d=COMPILED(a,e.i32([-1,1])).state
        self.assertEqual(player_view(c,0),player_view(d,0))
        # Opposing exact HP inside a common public interval stays concealed.
        row=e.i32([e.DAMAGE,1,0,0,0,0,230,235,0,0])
        c=a._replace(hp=a.hp.at[1,0].set(230),history=a.history.at[a.hcount].set(row),hcount=a.hcount+1)
        d=c._replace(hp=c.hp.at[1,0].set(229),history=c.history.at[a.hcount,6].set(229))
        self.assertEqual(player_view(c,0),player_view(d,0))
        self.assertNotEqual(player_view(c,1),player_view(d,1))
        # Either submission order resolves the same simultaneous commitments.
        x=COMPILED(COMPILED(a,e.i32([0,-1])).state,e.i32([-1,0]))
        y=COMPILED(COMPILED(a,e.i32([-1,0])).state,e.i32([0,-1]))
        z=COMPILED(a,e.i32([0,0]));equal(self,x,y);equal(self,x,z)
        # Replacement/undo erase the previous unexecuted choice.
        p=COMPILED(a,e.i32([1,-1])).state
        p=COMPILED(p,e.i32([-2,-1])).state
        equal(self,COMPILED(p,e.i32([0,0])),z)

    def test_mixed_phases_rejections_and_independent_rng(self):
        normal,_=initialize_case(self.by_id['wisp-hit'])
        pending=COMPILED(normal,e.i32([0,-1])).state
        pivot,_=initialize_case(self.by_id['pivot-resume']);pivot=COMPILED(pivot,e.i32([0,1])).state
        forced,_=initialize_case(self.by_id['forced-replacement']);forced=COMPILED(forced,e.i32([0,1])).state
        terminal,_=initialize_case(self.by_id['struggle-double-ko']);terminal=COMPILED(terminal,e.i32([10,10])).state
        unsupported=normal._replace(moves=normal.moves.at[0,0,0].set(99))
        over=normal._replace(hcount=e.i32(e.HISTORY))
        states=stack([normal,normal,pending,pivot,forced,terminal,unsupported,over])
        actions=e.i32([[0,0],[4,0],[-1,0],[5,-1],[-1,5],[0,0],[0,0],[0,0]])
        result=BATCH(states,actions)
        self.assertEqual(np.asarray(result.code).tolist(),[e.OK,e.INVALID,e.OK,e.OK,e.OK,e.TERMINAL,e.UNSUPPORTED,e.CAPACITY])
        for k in range(8):equal(self,lane(result,k),COMPILED(lane(states,k),actions[k]))
        changed=states._replace(rng=states.rng.at[0].set(jnp.array([7,8,9,10],jnp.uint16)))
        other=BATCH(changed,actions)
        for k in range(1,8):equal(self,lane(result,k),lane(other,k))

    def test_transient_buffer_overflow_rolls_back(self):
        s,_=initialize_case(self.by_id['wisp-hit'])
        # Synthetic smaller physical buffers exercise the exact production
        # overflow path; no expected state/event assertion is relaxed.
        for field in ('EVENTS','DRAWS'):
            with patch.object(e,field,1):
                result=jax.jit(lambda state:e.advance(state,e.i32([0,0])))(s)
                self.assertEqual(int(result.code),e.CAPACITY);equal(self,result.state,s)
                self.assertEqual(int(result.trace.ecount),0);self.assertEqual(int(result.trace.dcount),0)
                self.assertFalse(np.any(np.asarray(result.trace.events)))

    def test_atomic_invalid_unsupported_stale_and_capacity(self):
        s,_=initialize_case(self.by_id['wisp-hit'])
        for action in (e.i32([4,0]),e.i32([11,0]),e.i32([-3,0])):
            r=COMPILED(s,action);self.assertEqual(int(r.code),e.INVALID);equal(self,s,r.state)
            self.assertEqual(int(r.trace.ecount),0);self.assertEqual(int(r.trace.dcount),0)
        stale=jax.jit(e.advance)(s,e.i32([0,0]),e.i32([1,0]));self.assertEqual(int(stale.code),e.STALE);equal(self,s,stale.state)
        excluded=s._replace(moves=s.moves.at[0,0,0].set(99))
        r=COMPILED(excluded,e.i32([0,0]));self.assertEqual(int(r.code),e.UNSUPPORTED);equal(self,excluded,r.state)
        arbitrary=s._replace(stats=s.stats.at[0,0,1].set(999))
        r=COMPILED(arbitrary,e.i32([0,0]));self.assertEqual(int(r.code),e.UNSUPPORTED);equal(self,arbitrary,r.state)
        occupied=s._replace(species=s.species.at[0,:4].set(e.MEW))
        r=COMPILED(occupied,e.i32([0,0]));self.assertEqual(int(r.code),e.UNSUPPORTED);equal(self,occupied,r.state)
        for over in (s._replace(hcount=e.i32(e.HISTORY)),s._replace(episode=e.i32(2**31-1))):
            r=COMPILED(over,e.i32([0,0]));self.assertEqual(int(r.code),e.CAPACITY);equal(self,over,r.state)
            self.assertEqual(int(r.trace.ecount),0);self.assertEqual(int(r.trace.dcount),0)
        pivot,_=initialize_case(self.by_id['pivot-resume']);pivot=COMPILED(pivot,e.i32([0,1])).state
        over=pivot._replace(qcount=e.i32(e.QUEUE),queue=jnp.tile(pivot.queue[0],(e.QUEUE,1)))
        r=COMPILED(over,e.i32([5,-1]));self.assertEqual(int(r.code),e.CAPACITY);equal(self,over,r.state)
        for action in (jnp.array([0,0],jnp.float32),jnp.array([0],jnp.int32)):
            with self.assertRaises(ValueError):e.advance(s,action)
        with self.assertRaises(ValueError):e.advance(s._replace(hp=s.hp[:,0]),e.i32([0,0]))
        for teams in ([[{'species':'Mew','moves':['Tackle'],'ability':'Pressure'}]]*2, [[{'species':'Mew','moves':['Sleep Talk']}]]*2, [[{'species':'Mew','moves':['Tackle'],'level':50.5}]]*2):
            with self.assertRaises(ValueError):e.make_state(teams)

if __name__=='__main__':unittest.main()
