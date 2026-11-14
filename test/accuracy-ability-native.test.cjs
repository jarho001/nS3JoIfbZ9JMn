'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),zlib=require('node:zlib');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native-only No Guard and Victory Star preliminary branches and membership reproduce',()=>{
 require('../reference/neutral-turn-oracle.cjs').assertReady();
 const ps=require('../reference/vendor/pokemon-showdown/dist/sim'),{runCase}=require('../reference/shared-engine-cases.cjs'),spec=require('../reference/accuracy-ability-cases.cjs');
 const actual={pin:require('../reference-lock.json').commit,witnesses:spec.membershipWitnesses(),cases:spec().map(c=>runCase(ps,c))};
 assert.deepEqual(normalizeStacks(JSON.parse(JSON.stringify(actual))),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/accuracy-ability-precheck.json.gz')))));
 assert.equal(actual.cases.length,21);assert.equal(actual.cases.reduce((n,c)=>n+c.steps.length,0),25);
 assert.ok(actual.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
 const c=id=>actual.cases.find(c=>c.id==='G4-accuracy-ability-'+id),draws=id=>c(id).steps[0].draws;
 assert.ok(!draws('noguard-source-megapunch').some(d=>d.semanticContext==='accuracy'));
 assert.ok(!draws('noguard-target-megapunch').some(d=>d.semanticContext==='accuracy'));
 assert.ok(draws('victorystar-source-megapunch').some(d=>d.semanticContext==='accuracy'));
 assert.ok(c('noguard-ghost').steps[0].events.p1.some(e=>e.message.startsWith('|-immune|')));
 assert.deepEqual(actual.witnesses.map(w=>w.set.species),['Golurk','Victini']);
});
