'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),zlib=require('node:zlib');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native-only Tangled Feet and Wonder Skin precheck preserves target modifiers and rounding',()=>{
 require('../reference/neutral-turn-oracle.cjs').assertReady();
 const ps=require('../reference/vendor/pokemon-showdown/dist/sim'),{runCase}=require('../reference/shared-engine-cases.cjs'),spec=require('../reference/evasion-ability-cases.cjs');
 const actual={pin:require('../reference-lock.json').commit,witnesses:spec.membershipWitnesses(),cases:spec().map(c=>runCase(ps,c))};
 assert.deepEqual(normalizeStacks(JSON.parse(JSON.stringify(actual))),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/evasion-ability-precheck.json.gz')))));
 assert.equal(actual.cases.length,32);assert.equal(actual.cases.reduce((n,c)=>n+c.steps.length,0),59);
 assert.ok(actual.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
 const c=id=>actual.cases.find(c=>c.id==='G4-evasion-ability-'+id),events=(id,i=0)=>c(id).steps[i].events.p1.map(e=>e.message);
 assert.ok(events('wonderskin-threshold').some(e=>e.startsWith('|-miss|')));
 assert.ok(events('control-threshold').some(e=>e.startsWith('|-status|')));
 assert.equal(c('tangledfeet-rounding').steps[1].draws.find(d=>d.semanticContext==='accuracy').value,58);
 assert.ok(!events('tangledfeet-rounding',1).some(e=>e.startsWith('|-miss|')));
 assert.deepEqual(actual.witnesses.map(w=>w.set.species),['Chatot','Delcatty']);
});
