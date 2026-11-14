'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),zlib=require('node:zlib');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native Counter and Mirror Coat record per-hit damage in pre-turn conditions and skip called-move setup',()=>{
 require('../reference/neutral-turn-oracle.cjs').assertReady();
 const ps=require('../reference/vendor/pokemon-showdown/dist/sim'),{runCase}=require('../reference/shared-engine-cases.cjs'),spec=require('../reference/retaliation-cases.cjs');
 const actual=JSON.parse(require('node:child_process').execFileSync('node',['reference/shared-engine-cases.cjs','--batch','retaliation'],{maxBuffer:64*1024*1024}));
 assert.deepEqual(normalizeStacks(JSON.parse(JSON.stringify(actual))),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/retaliation.json.gz')))));
 assert.equal(actual.cases.length,52);assert.equal(actual.cases.reduce((n,c)=>n+c.steps.length,0),70);assert.ok(actual.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
 for(const c of actual.cases)for(const side of ['p1','p2'])for(const mon of c.setup[side])assert.equal(new Set(mon.moves.map(m=>ps.Dex.toID(m))).size,mon.moves.length,c.id+' unique moves');
 const c=id=>actual.cases.find(c=>c.id==='G4-retaliation-'+id),p=(id,i=0,side=0)=>c(id).steps[i].after.sides[side].pokemon[0];
 for(const [key,move]of [['counter','Counter'],['mirrorcoat','Mirror Coat']]){
  const exhausted=c(key+'-struggle-slot4');assert.ok(exhausted.steps[0].events.p1.some(e=>e.message.includes('|Struggle|')));assert.equal(exhausted.steps[0].after.effectOrder,exhausted.initial.effectOrder);
  assert.equal(p(key).moves[0].pp,30);assert.equal(p(key+'-no-damage').moves[0].pp,30);
  assert.equal(p(key,0,1).hp,175-2*(175-p(key).hp));
  assert.equal(p(key+'-wrong-category',0,1).hp,175);assert.equal(p(key+'-no-damage',0,1).hp,175);
  const row=p(key+'-terminal').volatiles[key];assert.equal(row.duration,1);assert.equal(row.name,move);assert.equal(row.slot,'p2a');assert.deepEqual(row.sourceEffect,{id:''});assert.equal(row.damage,2*(175-p(key+'-terminal').hp));
  const reactive=p(key+'-helmet-terminal').volatiles[key];assert.ok(reactive.damage>0);assert.equal(reactive.slot,'p2a');assert.equal(p(key+'-helmet-terminal',0,1).hp,0);
  const blank=p(key+'-blank-terminal').volatiles[key];assert.equal(blank.duration,1);assert.equal(blank.slot,null);assert.equal(blank.damage,0);
  assert.equal(p(key+'-frozen').status,'');assert.equal(p(key+'-fire-thaw').status,'');
  assert.equal(p(key).volatiles[key],undefined);assert.equal(p(key+'-one-hp').hp,0);
  assert.equal(p(key+'-sash-unburden').hp,1);assert.ok(p(key+'-sash-unburden').volatiles.unburden);
  const sleep=c(key+'-sleep-talk').steps[1];assert.ok(sleep.events.p1.some(e=>e.message.includes('|'+move+'|')&&e.message.includes('[from] move: Sleep Talk')));assert.ok(sleep.events.p1.some(e=>e.message==='|-fail|p1a: Mew'));assert.equal(p(key+'-sleep-talk',1,1).hp,175);
  const draws=c(key+'-both').steps[0].draws;assert.ok(draws.some(d=>d.semanticContext==='before-turn'));assert.equal(draws.filter(d=>d.semanticContext==='queue-sort').length,2);
 }
 for(const target of ['mew','gengar']){const row=p('counter-pivot-'+target).volatiles.counter;assert.equal(row.duration,1);assert.equal(row.slot,'p2a');assert.ok(row.damage>0);}
 assert.ok(c('counter-pursuit-tie').steps[0].draws.some(d=>d.semanticContext==='queue-sort'));
 assert.equal(p('counter-doublekick').hp,163);assert.equal(p('counter-doublekick',0,1).hp,163);
 assert.equal(p('counter-mummy').ability,'mummy');assert.equal(p('mirrorcoat-mummy').ability,'technician');
 assert.deepEqual(spec.membershipWitnesses().map(w=>w.set.species),['Wobbuffet','Wobbuffet']);
});
