'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),zlib=require('node:zlib');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native confusion secondaries preserve chance, duration, silent blocks and source state',()=>{
 require('../reference/neutral-turn-oracle.cjs').assertReady();
 const ps=require('../reference/vendor/pokemon-showdown/dist/sim'),{runCase}=require('../reference/shared-engine-cases.cjs'),spec=require('../reference/confusion-secondary-cases.cjs');
 const actual=JSON.parse(require('node:child_process').execFileSync('node',['reference/shared-engine-cases.cjs','--batch','confusionsecondary'],{maxBuffer:64*1024*1024}));
 assert.deepEqual(normalizeStacks(JSON.parse(JSON.stringify(actual))),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/confusion-secondary.json.gz')))));
 assert.equal(actual.cases.length,44);assert.equal(actual.cases.reduce((n,c)=>n+c.steps.length,0),62);assert.ok(actual.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
 for(const c of actual.cases)for(const side of ['p1','p2'])for(const mon of c.setup[side])assert.equal(new Set(mon.moves.map(m=>ps.Dex.toID(m))).size,mon.moves.length,c.id+' unique moves');
 const c=id=>actual.cases.find(c=>c.id==='G4-confusion-secondary-'+id),events=(id,i=0)=>c(id).steps[i].events.p1.map(e=>e.message),draws=(id,i=0)=>c(id).steps[i].draws,started=(id,i=0)=>events(id,i).some(e=>e.startsWith('|-start|p2a:')&&e.includes('|confusion'));
 for(const [key,move]of [['signal','Signal Beam'],['hurricane','Hurricane'],['punch','Dynamic Punch']]){
  assert.ok(started(key+'-positive'));assert.ok(draws(key+'-positive').some(d=>d.semanticContext==='confusion-duration'));
  for(const suffix of ['owntempo','shielddust','protect','terminal'])assert.ok(!started(key+'-'+suffix));
  assert.ok(!events(key+'-owntempo').some(e=>e.includes('|-immune|')));assert.ok(!draws(key+'-shielddust').some(d=>d.semanticContext==='secondary'));
  assert.ok(!draws(key+'-existing',1).some(d=>d.semanticContext==='confusion-duration'));assert.ok(!events(key+'-existing',1).some(e=>e.startsWith('|-fail|')));
  assert.ok(!started(key+'-substitute',1));assert.equal(c(key+'-terminal').steps[0].after.terminal,true);
  assert.ok(events(key+'-sleep-talk',1).some(e=>e.includes('|'+move+'|')&&e.includes('[from] move: Sleep Talk')));
 }
 assert.ok(draws('punch-terminal').some(d=>d.semanticContext==='secondary'));assert.ok(!draws('punch-terminal').some(d=>d.semanticContext==='confusion-duration'));
 assert.ok(!draws('punch-ghost').some(d=>d.semanticContext==='secondary'));
 assert.deepEqual(spec.membershipWitnesses().map(w=>w.set.species),['Beheeyem','Moltres','Golurk']);
});
