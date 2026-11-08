"""Independent pinned native pre-entry sets, HP bitplanes and type chart."""
import json
import subprocess
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mechanics.hidden_power import hidden_power
from simulator import gen5_tables as g


class Gen5TablesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native=json.loads(subprocess.check_output(['node','reference/gen5-data-cases.cjs'],text=True,timeout=120))

    def test_pre_entry_generated_set_values(self):
        records=self.native['records']
        self.assertEqual(self.native['pin'],g.PIN)
        self.assertEqual(sum(r['id'].startswith('generated-set:') for r in records),388)
        compiled=jax.jit(g.derive_set_stats)
        for offset in range(0,len(records),12):
            chunk=records[offset:offset+12]
            # Padding reuses an independent expectation, never a JAX result.
            chunk=chunk+[records[0]]*(12-len(chunk))
            sets=g.decode_sets([[r['set'] for r in chunk[:6]],[r['set'] for r in chunk[6:]]])
            result=jax.device_get(compiled(sets))
            for n,record in enumerate(chunk):
                side,mon=divmod(n,6);expected=record['expected'];label=record['id']
                self.assertEqual(int(sets.species[side,mon]),g.SPECIES[expected['species']],label)
                np.testing.assert_array_equal(result.stats[side,mon],expected['stats'],err_msg=label)
                mids=[g.MOVES[m['id']] for m in expected['moves']]
                np.testing.assert_array_equal(sets.moves[side,mon,:len(mids)],mids,err_msg=label)
                np.testing.assert_array_equal(result.pp[side,mon,:len(mids)],[m['maxpp'] for m in expected['moves']],err_msg=label)
                self.assertEqual(int(result.hp_type[side,mon]),g.TYPES[g.to_id(expected['hpType'])],label)
                self.assertEqual(int(result.hp_power[side,mon]),expected['hpPower'],label)
                self.assertEqual(int(sets.item[side,mon]),g.ITEMS.get(expected['item'],0),label)
                self.assertEqual(int(sets.ability[side,mon]),g.ABILITIES.get(expected['ability'],0),label)
                np.testing.assert_array_equal(sets.ivs[side,mon],expected['ivs'],err_msg=label)
                np.testing.assert_array_equal(sets.evs[side,mon],expected['evs'],err_msg=label)

    def test_every_hidden_power_bitplane(self):
        records=self.native['hiddenPower']
        ivs=jnp.asarray([r['ivs'] for r in records],jnp.int32)
        expected_type=np.asarray([g.TYPES[g.to_id(r['type'])] for r in records])
        expected_power=np.asarray([r['power'] for r in records])
        for run in (hidden_power,jax.jit(hidden_power),jax.jit(jax.vmap(hidden_power))):
            kind,power=run(ivs)
            np.testing.assert_array_equal(g.HP_TYPE_IDS[kind],expected_type)
            np.testing.assert_array_equal(power,expected_power)

    def test_native_type_chart_and_dual_types(self):
        types=self.native['types'];attack=jnp.asarray([g.TYPES[g.to_id(t)] for t in types],jnp.int32)
        defense=jnp.stack((attack,jnp.zeros_like(attack)),axis=-1)
        exponent,immune=jax.jit(g.type_matchup)(attack[:,None],defense[None,:,:])
        np.testing.assert_array_equal(exponent,[[x['exponent'] for x in row] for row in self.native['chart']])
        np.testing.assert_array_equal(immune,[[x['immune'] for x in row] for row in self.native['chart']])
        # Gen 5 Steel still resists Ghost and Dark; Ground immunities remain
        # separate from exponents so a weak second type cannot cancel immunity.
        kind,blocked=g.type_matchup(jnp.asarray(g.TYPES['ground']),jnp.asarray([g.TYPES['flying'],g.TYPES['electric']]))
        self.assertTrue(bool(blocked));self.assertEqual(int(kind),1)
        for move in ('ghost','dark'):
            kind,blocked=g.type_matchup(jnp.asarray(g.TYPES[move]),jnp.asarray([g.TYPES['steel'],0]))
            self.assertEqual(int(kind),-1);self.assertFalse(bool(blocked))

    def test_mixed_batch_eager_and_compiled_set_derivation(self):
        records=self.native['records']
        one=g.decode_sets([[records[0]['set']],[records[-1]['set']]])
        eager=g.derive_set_stats(one)
        for size in (1,8,32):
            batch=jax.tree.map(lambda x:jnp.stack([x]*size),one)
            result=jax.jit(jax.vmap(g.derive_set_stats))(batch)
            for actual,expected in zip(jax.tree.leaves(result),jax.tree.leaves(eager),strict=True):
                np.testing.assert_array_equal(actual,np.broadcast_to(expected,(size,)+expected.shape))
        self.assertTrue(np.all(np.asarray(eager.stats)[0,1:]==0))
        self.assertTrue(np.all(np.asarray(eager.pp)[0,1:]==0))

    def test_native_constructor_gender_draws_and_saved_sets(self):
        compiled=jax.jit(g.construct)
        for record in self.native['records']:
            sets=g.decode_sets([[record['set']],[{'species':'Mew','moves':['Tackle']} ]])
            result=jax.device_get(compiled(sets,jnp.asarray(record['seed'],jnp.uint16)))
            expected=record['expected']
            self.assertEqual(int(result.sets.gender[0,0]),{'':0,'M':1,'F':2}[expected['gender']],record['id'])
            self.assertEqual(result.rng.tolist(),record['rng'],record['id'])
            self.assertEqual(int(result.draw_count),len(record['draws']),record['id'])
            for i,draw in enumerate(record['draws']):
                self.assertEqual(draw['args'],[2])
                self.assertEqual(int(result.draw_values[i]),draw['value'])
                self.assertEqual(result.draw_before[i].tolist(),draw['before'])
                self.assertEqual(result.draw_after[i].tolist(),draw['after'])
        # Vmap preserves lane-local draw counts and independent native seeds.
        records=self.native['records'][:32]
        states=[g.decode_sets([[r['set']],[{'species':'Mew','moves':['Tackle']}]]) for r in records]
        inputs=jax.tree.map(lambda *xs:jnp.stack(xs),*states)
        seeds=jnp.asarray([r['seed'] for r in records],jnp.uint16)
        result=jax.device_get(jax.jit(jax.vmap(g.construct))(inputs,seeds))
        for n,record in enumerate(records):
            self.assertEqual(result.rng[n].tolist(),record['rng'])
            self.assertEqual(int(result.draw_count[n]),len(record['draws']))

    def test_unknown_set_effects_rejected(self):
        base={'species':'Mew','moves':['Tackle']}
        for extra in ({'teraType':'Fire'},{'evs':{'hp':256}},{'ivs':{'spe':-1}},
                      {'level':True},{'moves':['Hidden Power Ice','Hidden Power Fire']},
                      {'moves':['Struggle']},{'ability':'Hadron Engine'},{'item':'Booster Energy'}):
            with self.subTest(extra=extra),self.assertRaises(ValueError):g.decode_sets([[base|extra],[base]])
        with self.assertRaises(TypeError):hidden_power(jnp.ones(6,jnp.float32))
        with self.assertRaises(ValueError):hidden_power(jnp.ones(5,jnp.int32))


if __name__=='__main__':unittest.main()
