'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),zlib=require('node:zlib');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native Super Fang and Seismic Toss preserve fixed damage, hit lifecycle and skipped damage RNG',()=>{
 require('../reference/neutral-turn-oracle.cjs').assertReady();
 const ps=require('../reference/vendor/pokemon-showdown/dist/sim'),{runCase}=require('../reference/shared-engine-cases.cjs'),spec=require('../reference/fixed-damage-cases.cjs');
 const actual=JSON.parse(require('node:child_process').execFileSync('node',['reference/shared-engine-cases.cjs','--batch','fixeddamage'],{maxBuffer:64*1024*1024}));
 assert.deepEqual(normalizeStacks(JSON.parse(JSON.stringify(actual))),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/fixed-damage.json.gz')))));
 assert.equal(actual.cases.length,48);assert.equal(actual.cases.reduce((n,c)=>n+c.steps.length,0),66);assert.ok(actual.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
 for(const c of actual.cases)for(const side of ['p1','p2'])for(const mon of c.setup[side])assert.equal(new Set(mon.moves.map(m=>ps.Dex.toID(m))).size,mon.moves.length,c.id+' unique moves');
 const c=id=>actual.cases.find(c=>c.id==='G4-fixed-damage-'+id),p=(id,i=0,side=1)=>c(id).steps[i].after.sides[side].pokemon[0];
 assert.equal(p('superfang').hp,88);assert.equal(p('superfang',1).hp,44);assert.equal(p('superfang-partial').hp,51);
 assert.equal(p('seismictoss').hp,125);assert.equal(p('seismictoss',1).hp,75);
 for(const level of [1,100]){
  const a=c('seismictoss-level'+level);assert.equal(p('seismictoss-level'+level).hp,a.initial.sides[1].pokemon[0].hp-level);
  assert.ok(!a.steps[0].draws.some(d=>['critical','damage'].includes(d.semanticContext)));
 }
 for(const key of ['superfang','seismictoss']){
  assert.ok(c(key+'-ghost').steps[0].events.p1.some(e=>e.message.startsWith('|-immune|p2a:')));
  assert.ok(!c(key+'-onehp').steps[0].draws.some(d=>['critical','damage'].includes(d.semanticContext)));
  assert.equal(c(key).steps[0].draws.filter(d=>d.semanticContext==='critical').length,1);
  assert.equal(c(key).steps[0].draws.filter(d=>d.semanticContext==='damage').length,1);
 }
 assert.ok(c('superfang-miss').steps[0].events.p1.some(e=>e.message.startsWith('|-miss|p1a:')));
 assert.equal(p('seismictoss-sash').hp,1);assert.equal(p('seismictoss-sturdy').hp,1);
 assert.equal(p('seismictoss-gem-unburden',0,0).item,'');assert.equal(p('seismictoss-gem-unburden',0,0).speed,304);
 assert.equal(p('seismictoss-gem-ghost',0,0).item,'fightinggem');
 assert.equal(p('seismictoss-gem-terminal',0,0).volatiles.gem.duration,1);
 assert.equal(p('seismictoss-moxie-terminal',0,0).boosts.atk,1);
 assert.ok(c('superfang-wonderguard').steps[0].events.p1.some(e=>e.message.includes('ability: Wonder Guard')));
 assert.equal(p('seismictoss-wonderguard-super').hp,105);
 assert.ok(c('superfang-static').steps[0].draws.some(d=>d.semanticContext==='contact-status'));
 assert.equal(p('seismictoss-aftermath-terminal',0,0).hp,0);assert.equal(p('seismictoss-aftermath-terminal').hp,0);
 assert.ok(c('superfang-sleep-talk').steps[1].events.p1.some(e=>e.message.startsWith('|move|p1a:')&&e.message.includes('|Super Fang|')));
 assert.deepEqual(spec.membershipWitnesses().map(w=>w.set.species),['Crobat','Blissey']);
});
