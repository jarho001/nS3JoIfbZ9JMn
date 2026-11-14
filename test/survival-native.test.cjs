'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),zlib=require('node:zlib');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native Sturdy and Focus Sash precheck preserves full HP, priority, suppression and multihit branches',()=>{
 require('../reference/neutral-turn-oracle.cjs').assertReady();
 const spec=require('../reference/survival-cases.cjs');
 const actual=JSON.parse(require('node:child_process').execFileSync('node',['reference/shared-engine-cases.cjs','--batch','survival'],{maxBuffer:16*1024*1024}));
 assert.deepEqual(normalizeStacks(JSON.parse(JSON.stringify(actual))),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/survival.json.gz')))));
 assert.equal(actual.cases.length,32);assert.equal(actual.cases.reduce((n,c)=>n+c.steps.length,0),39);
 assert.ok(actual.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
 const c=id=>actual.cases.find(c=>c.id==='G4-survival-'+id),mon=(id,i=0)=>c(id).steps[i].after.sides[1].pokemon[0],events=id=>c(id).steps[0].events.p1.map(e=>e.message);
 for(const key of ['sturdy','sash','both']){
  assert.ok(mon(key+'-full').hp>0);assert.equal(mon(key+'-full',1).hp,0);
  assert.equal(mon(key+'-chipped').hp,0);assert.equal(mon(key+'-multi').hp,0);assert.equal(mon(key+'-burn').hp,0);
 }
 assert.equal(mon('sash-full').item,'');assert.equal(mon('both-full').item,'focussash');
 assert.ok(events('sturdy-full').some(e=>e.includes('|Sturdy')));assert.ok(events('sash-full').some(e=>e==='|-enditem|p2a: Mew|Focus Sash'));
 assert.equal(mon('sturdy-suppressed',1).hp,0);assert.ok(mon('both-suppressed',1).hp>0);assert.equal(mon('both-suppressed',1).item,'');
 for(const key of ['sturdy','sash'])assert.equal(mon(key+'-one-hp').hp,1);
 assert.deepEqual(spec.membershipWitnesses().map(w=>w.set.species),['Crustle','Qwilfish']);
});
