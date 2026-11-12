"""Same-kernel core parity, lossless histories and checkpoint corruption."""
import json,os,tempfile,unittest
from pathlib import Path
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:3])
import jax
import numpy as np
from simulator import shared_engine as e,journal as j
from simulator.shared_engine_io import player_view
from test.shared_engine_support import initialize_case,choices_array,assert_step

class CoreJournalTest(unittest.TestCase):
    def test_full_core_native_parity_journal_resume_and_atomic_failures(self):
        cases=json.loads(Path(os.environ['LOCK_NATIVE_CASES']).read_text())['cases']
        case=next(c for c in cases if c['id']=='G4-lock-complete-1')
        full,_=initialize_case(case);checkpoint=j.from_state(full)
        core_step=jax.jit(e.advance_core);full_step=jax.jit(e.advance)
        for n,expected in enumerate(case['steps']):
            choices=choices_array(full,expected['choices']);before=full
            a=full_step(full,choices);checkpoint,b=j.advance(checkpoint,choices,step=core_step)
            assert_step(self,before,a,expected,f'boundary{n}')
            for name in e.CoreState._fields:np.testing.assert_array_equal(getattr(a.state,name),getattr(b.state,name),err_msg=name)
            for x,y in zip(jax.tree.leaves(a.trace),jax.tree.leaves(b.trace)):np.testing.assert_array_equal(x,y)
            full=a.state
            for side in range(2):self.assertEqual(j.player_view(checkpoint,side),player_view(full,side))
            if n==12:
                with tempfile.TemporaryDirectory()as tmp:
                    path=Path(tmp)/'journal.npz';j.save(path,checkpoint);checkpoint=j.load(path)
                    with np.load(path,allow_pickle=False)as data:fields={k:data[k]for k in data.files}
                    meta=json.loads(str(fields['metadata']));meta['pages'].pop();fields['metadata']=np.asarray(json.dumps(meta));np.savez(path,**fields)
                    with self.assertRaisesRegex(ValueError,'checksum'):j.load(path)
        prior=checkpoint;checkpoint,r=j.advance(checkpoint,e.i32([0,0]),step=core_step)
        self.assertEqual(int(r.code),e.TERMINAL);self.assertIs(checkpoint,prior)
        live,_=initialize_case(cases[0]);checkpoint=j.from_state(live)
        for choices,token,code in [(e.i32([99,-1]),None,e.INVALID),(e.i32([0,0]),e.i32([-1,-1]),e.STALE)]:
            after,r=j.advance(checkpoint,choices,token,step=core_step);self.assertEqual(int(r.code),code);self.assertIs(after,checkpoint)
        # Full compatibility history capacity is independent of core mechanics.
        filled=live._replace(hcount=e.i32(e.HISTORY));choice=e.i32([0,0])
        self.assertEqual(int(full_step(filled,choice).code),e.CAPACITY)
        self.assertEqual(int(core_step(e.core_state(filled),choice).code),e.OK)
        self.assertNotIn('history',e.CoreState._fields);self.assertNotIn('hcount',e.CoreState._fields)
        self.assertGreaterEqual(len(prior.pages),len(case['steps']))


class TurnLimitJournalTest(unittest.TestCase):
    def test_native_long_game_every_boundary_and_lossless_history(self):
        from test.shared_engine_support import assert_state
        from simulator.shared_engine_io import draw_rows,project_events
        case=json.loads(Path(os.environ['TURN_LIMIT_NATIVE_CASE']).read_text())
        full,trace=initialize_case(case);self.assertEqual(draw_rows(trace),case['initializationDraws'])
        checkpoint=j.from_state(full);step=jax.jit(e.advance_core)
        for n,expected in enumerate(case['steps']):
            choices=choices_array(checkpoint.core,expected['choices'])
            checkpoint,r=j.advance(checkpoint,choices,step=step)
            self.assertEqual(int(r.code),e.OK,f'boundary{n}')
            self.assertEqual(draw_rows(r.trace),[{'args':[d['bound'],d['to']] if 'to' in d else [d['bound']], 'value':d['value'],'before':d['before'],'after':d['after'],'context':d['semanticContext']}for d in expected['draws']],f'draws{n}')
            assert_state(self,r.state,expected['after'],f'boundary{n}')
            for side in range(2):self.assertEqual(project_events(r.state,r.trace,side),expected['events'][f'p{side+1}'],f'events{n}')
            if n==500:
                with tempfile.TemporaryDirectory()as tmp:
                    p=Path(tmp)/'resume.npz';j.save(p,checkpoint);checkpoint=j.load(p)
        self.assertEqual(int(checkpoint.core.winner),0)
        self.assertEqual(int(checkpoint.core.turn),1001)
        for side in range(2):
            expected=list(case['initialEvents'][f'p{side+1}'])
            for s in case['steps']:expected.extend(s['events'][f'p{side+1}'])
            self.assertEqual(j.player_view(checkpoint,side)['events'],expected)
            self.assertGreater(len(expected),e.HISTORY)



class CoreExecutionTest(unittest.TestCase):
    def test_mixed_core_batches_chunk_rollout_and_reset(self):
        from simulator.rollout import rollout
        import jax.numpy as npj
        cases=json.loads(Path(os.environ['LOCK_NATIVE_CASES']).read_text())['cases']
        starts=[e.core_state(initialize_case(c)[0])for c in cases if c['setup'].get('fullBattle')]
        step=jax.jit(e.advance_core);actions=e.i32([0,0])
        expected=[step(s,actions)for s in starts]
        stack=lambda xs:jax.tree.map(lambda *v:npj.stack(v),*xs)
        for size in (1,8,32):
            indices=[n%len(starts)for n in range(size)]
            actual=jax.jit(jax.vmap(e.advance_core))(stack([starts[n]for n in indices]),npj.tile(actions,(size,1)))
            for x,y in zip(jax.tree.leaves(actual),jax.tree.leaves(stack([expected[n]for n in indices]))):np.testing.assert_array_equal(x,y)
            jax.clear_caches()
        # Chunk execution retains no full-history arrays in either mode.
        plan=npj.tile(actions,(2,1))
        lean=jax.jit(rollout)(starts[0],plan)
        full=jax.jit(lambda s,p:rollout(s,p,retain=('trace','codes')))(starts[0],plan)
        for x,y in zip(jax.tree.leaves(lean.state),jax.tree.leaves(full.state)):np.testing.assert_array_equal(x,y)
        terminal=starts[0]._replace(winner=e.i32(0),request=e.i32([e.END,e.END]))
        reset=jax.jit(e.reset)(terminal,starts[0],e.i32([terminal.generation,terminal.episode]))
        self.assertEqual(int(reset.code),e.OK);self.assertEqual(int(reset.state.generation),int(terminal.generation)+1)

if __name__=='__main__':unittest.main()
