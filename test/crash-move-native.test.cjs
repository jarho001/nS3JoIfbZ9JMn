'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),zlib=require('node:zlib');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native Jump Kick and High Jump Kick preserve crash triggers, rounding and faint requests',()=>{
 require('../reference/neutral-turn-oracle.cjs').assertReady();
 const spec=require('../reference/crash-move-cases.cjs');
 const actual=JSON.parse(require('node:child_process').execFileSync('node',['reference/shared-engine-cases.cjs','--batch','crashmove'],{maxBuffer:32*1024*1024}));
 assert.deepEqual(normalizeStacks(actual),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/crash-move.json.gz')))));
 assert.equal(actual.cases.length,44);assert.equal(actual.cases.reduce((n,c)=>n+c.steps.length,0),50);assert.ok(actual.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
 const c=id=>actual.cases.find(c=>c.id==='G4-crash-'+id),first=id=>c(id).steps[0],hp=id=>first(id).after.sides[0].pokemon[0].hp;
 for(const key of ['jumpkick','highjumpkick']){
  for(const trigger of ['protect','ghost','rockhead','reckless','noguard','wonderguard','seed-15'])assert.equal(hp(key+'-'+trigger),88,key+' '+trigger);
  assert.equal(hp(key+'-magicguard'),175);assert.equal(hp(key+'-one-hp-sash'),0);
  assert.ok(!first(key+'-no-target').events.p1.some(e=>e.message.includes('[from] '+key)));
  assert.ok(c(key+'-sleep-talk').steps[1].events.p1.some(e=>e.message.includes('[from] '+key)));
  assert.ok(!first(key+'-noguard-hit').events.p1.some(e=>e.message.startsWith('|-miss|')));assert.equal(hp(key+'-even-hp'),88);
  assert.ok(first(key+'-low-hp').events.p1.some(e=>e.message==='|win|p2'));
  assert.deepEqual(first(key+'-reserve').after.sides[0].request.forceSwitch,[true]);
  assert.ok(first(key+'-seed-15').events.p1.some(e=>e.message.startsWith('|-miss|')));
 }
 assert.deepEqual(spec.membershipWitnesses().map(w=>w.set.species),['Lopunny','Mienshao']);
});
