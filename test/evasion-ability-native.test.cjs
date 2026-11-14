'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),zlib=require('node:zlib');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native Tangled Feet and Wonder Skin precheck preserves target modifiers and rounding',()=>{
 require('../reference/neutral-turn-oracle.cjs').assertReady();
 const spec=require('../reference/evasion-ability-cases.cjs');
 const actual=JSON.parse(require('node:child_process').execFileSync('node',['reference/shared-engine-cases.cjs','--batch','evasionability'],{maxBuffer:24*1024*1024}));
 assert.deepEqual(normalizeStacks(JSON.parse(JSON.stringify(actual))),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/evasion-ability.json.gz')))));
 assert.equal(actual.cases.length,32);assert.equal(actual.cases.reduce((n,c)=>n+c.steps.length,0),59);
 assert.ok(actual.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
 const c=id=>actual.cases.find(c=>c.id==='G4-evasion-ability-'+id),events=(id,i=0)=>c(id).steps[i].events.p1.map(e=>e.message);
 assert.ok(events('wonderskin-threshold').some(e=>e.startsWith('|-miss|')));
 assert.ok(events('control-threshold').some(e=>e.startsWith('|-status|')));
 assert.equal(c('tangledfeet-rounding').steps[1].draws.find(d=>d.semanticContext==='accuracy').value,58);
 assert.ok(!events('tangledfeet-rounding',1).some(e=>e.startsWith('|-miss|')));
 assert.deepEqual(spec.membershipWitnesses().map(w=>w.set.species),['Chatot','Delcatty']);
});
