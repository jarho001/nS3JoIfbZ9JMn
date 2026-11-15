'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),zlib=require('node:zlib');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native Big Pecks and Hyper Cutter filter external drops by stat',()=>{
 require('../reference/neutral-turn-oracle.cjs').assertReady();
 const ps=require('../reference/vendor/pokemon-showdown/dist/sim'),{runCase}=require('../reference/shared-engine-cases.cjs'),spec=require('../reference/stat-guard-cases.cjs');
 const actual=JSON.parse(require('node:child_process').execFileSync('node',['reference/shared-engine-cases.cjs','--batch','statguard'],{maxBuffer:64*1024*1024}));
 assert.deepEqual(normalizeStacks(JSON.parse(JSON.stringify(actual))),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/stat-guard.json.gz')))));
 assert.equal(actual.cases.length,35);assert.equal(actual.cases.reduce((n,c)=>n+c.steps.length,0),39);assert.ok(actual.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
 for(const c of actual.cases)for(const side of ['p1','p2'])for(const mon of c.setup[side])assert.equal(new Set(mon.moves.map(m=>ps.Dex.toID(m))).size,mon.moves.length,c.id+' unique moves');
 const c=id=>actual.cases.find(c=>c.id==='G4-stat-guard-'+id),p=(id,side=1,i=0)=>c(id).steps[i].after.sides[side].pokemon[0];
 assert.equal(p('pecks-tickle').boosts.atk,-1);assert.equal(p('pecks-tickle').boosts.def,0);
 assert.equal(p('cutter-tickle').boosts.atk,0);assert.equal(p('cutter-tickle').boosts.def,-1);
 assert.equal(c('pecks-intimidate').initial.sides[1].pokemon[0].boosts.atk,-1);assert.equal(c('cutter-intimidate').initial.sides[1].pokemon[0].boosts.atk,0);
 for(const [key,stat,other,name]of [['pecks','def','atk','Big Pecks'],['cutter','atk','def','Hyper Cutter']]){
  assert.equal(p(key+'-self').boosts.def,-1);assert.equal(p(key+'-self').boosts.spd,-1);
  assert.equal(p(key+'-suppressed',1,1).boosts.atk,-1);assert.equal(p(key+'-suppressed',1,1).boosts.def,-1);
  for(const suffix of ['roleplay','swap','trace','reflected']){assert.equal(p(key+'-'+suffix,0).boosts[stat],0);assert.equal(p(key+'-'+suffix,0).boosts[other],-1);}
  assert.ok(c(key+'-tickle').steps[0].events.p1.some(e=>e.message.includes('|unboost|'+stat+'|[from] ability: '+name)));
 }
 assert.deepEqual(spec.membershipWitnesses().map(w=>w.set.species),['Pidgeot','Kingler']);
});
