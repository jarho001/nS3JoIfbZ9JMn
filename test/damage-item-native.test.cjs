'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),zlib=require('node:zlib');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native Eviolite and Expert Belt preserve base-species defense and effective-hit damage',()=>{
 require('../reference/neutral-turn-oracle.cjs').assertReady();
 const ps=require('../reference/vendor/pokemon-showdown/dist/sim'),{runCase}=require('../reference/shared-engine-cases.cjs'),spec=require('../reference/damage-item-cases.cjs');
 const actual=JSON.parse(require('node:child_process').execFileSync('node',['reference/shared-engine-cases.cjs','--batch','damageitem'],{maxBuffer:64*1024*1024}));
 assert.deepEqual(normalizeStacks(JSON.parse(JSON.stringify(actual))),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/damage-item.json.gz')))));
 assert.equal(actual.cases.length,39);assert.equal(actual.cases.reduce((n,c)=>n+c.steps.length,0),46);assert.ok(actual.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
 for(const c of actual.cases)for(const side of ['p1','p2'])for(const mon of c.setup[side])assert.equal(new Set(mon.moves.map(m=>ps.Dex.toID(m))).size,mon.moves.length,c.id+' unique moves');
 const c=id=>actual.cases.find(c=>c.id==='G4-damage-item-'+id),hp=id=>c(id).steps[0].after.sides[1].pokemon[0].hp;
 for(const category of ['physical','special']){
  assert.ok(hp('eviolite-'+category+'-nfe-held')>hp('eviolite-'+category+'-nfe-empty'));
  assert.equal(hp('eviolite-'+category+'-fully-evolved-held'),hp('eviolite-'+category+'-fully-evolved-empty'));
 }
 for(const key of ['super','quad','physical'])assert.ok(hp('expertbelt-'+key+'-held')<hp('expertbelt-'+key+'-empty'));
 for(const key of ['neutral','resist','immune','net-neutral'])assert.equal(hp('expertbelt-'+key+'-held'),hp('expertbelt-'+key+'-empty'));
 for(const id of ['eviolite-critical','expertbelt-critical'])assert.ok(c(id).steps[0].events.p1.some(e=>e.message.startsWith('|-crit|p2a:')),id+' native critical');
 assert.equal(c('eviolite-knockoff').steps[0].after.sides[1].pokemon[0].item,'');
 assert.equal(c('expertbelt-trick').steps[0].after.sides[1].pokemon[0].item,'expertbelt');
 for(const [name,target]of [['vigoroth','Mew'],['mew','Vigoroth']])assert.equal(c('eviolite-transform-'+name).steps[0].after.sides[0].pokemon[0].species,target);
 assert.deepEqual(spec.membershipWitnesses().map(w=>w.set.species),['Seaking','Vigoroth']);
});
