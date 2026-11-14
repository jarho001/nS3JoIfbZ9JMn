'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),zlib=require('node:zlib');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native-only Counter and Mirror Coat record per-hit damage in pre-turn conditions and skip called-move setup',()=>{
 require('../reference/neutral-turn-oracle.cjs').assertReady();
 const ps=require('../reference/vendor/pokemon-showdown/dist/sim'),{runCase}=require('../reference/shared-engine-cases.cjs'),spec=require('../reference/retaliation-cases.cjs');
 const actual={pin:require('../reference-lock.json').commit,witnesses:spec.membershipWitnesses(),cases:spec().map(c=>runCase(ps,c))};
 assert.deepEqual(normalizeStacks(JSON.parse(JSON.stringify(actual))),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/retaliation-precheck.json.gz')))));
 assert.equal(actual.cases.length,39);assert.equal(actual.cases.reduce((n,c)=>n+c.steps.length,0),55);assert.ok(actual.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
 for(const c of actual.cases)for(const side of ['p1','p2'])for(const mon of c.setup[side])assert.equal(new Set(mon.moves.map(m=>ps.Dex.toID(m))).size,mon.moves.length,c.id+' unique moves');
 const c=id=>actual.cases.find(c=>c.id==='G4-retaliation-'+id),p=(id,i=0,side=0)=>c(id).steps[i].after.sides[side].pokemon[0];
 for(const [key,move]of [['counter','Counter'],['mirrorcoat','Mirror Coat']]){
  assert.equal(p(key,0,1).hp,175-2*(175-p(key).hp));
  assert.equal(p(key+'-wrong-category',0,1).hp,175);assert.equal(p(key+'-no-damage',0,1).hp,175);
  const row=p(key+'-terminal').volatiles[key];assert.equal(row.duration,1);assert.equal(row.name,move);assert.equal(row.slot,'p2a');assert.deepEqual(row.sourceEffect,{id:''});assert.equal(row.damage,2*(175-p(key+'-terminal').hp));
  assert.equal(p(key).volatiles[key],undefined);assert.equal(p(key+'-one-hp').hp,0);
  assert.equal(p(key+'-sash-unburden').hp,1);assert.ok(p(key+'-sash-unburden').volatiles.unburden);
  const sleep=c(key+'-sleep-talk').steps[1];assert.ok(sleep.events.p1.some(e=>e.message.includes('|'+move+'|')&&e.message.includes('[from] move: Sleep Talk')));assert.ok(sleep.events.p1.some(e=>e.message==='|-fail|p1a: Mew'));assert.equal(p(key+'-sleep-talk',1,1).hp,175);
  const draws=c(key+'-both').steps[0].draws;assert.ok(draws.some(d=>d.semanticContext==='before-turn'));assert.equal(draws.filter(d=>d.semanticContext==='queue-sort').length,2);
 }
 assert.equal(p('counter-doublekick').hp,163);assert.equal(p('counter-doublekick',0,1).hp,163);
 assert.equal(p('counter-mummy').ability,'mummy');assert.equal(p('mirrorcoat-mummy').ability,'technician');
 assert.deepEqual(actual.witnesses.map(w=>w.set.species),['Wobbuffet','Wobbuffet']);
});
