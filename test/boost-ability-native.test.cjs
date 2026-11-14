'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),zlib=require('node:zlib');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native-only Download and Defiant precheck preserves acquisition, per-stat drops and caps',()=>{
 require('../reference/neutral-turn-oracle.cjs').assertReady();
 const ps=require('../reference/vendor/pokemon-showdown/dist/sim'),{runCase}=require('../reference/shared-engine-cases.cjs'),spec=require('../reference/boost-ability-cases.cjs');
 const actual={pin:require('../reference-lock.json').commit,witnesses:spec.membershipWitnesses(),cases:spec().map(c=>runCase(ps,c))};
 assert.deepEqual(normalizeStacks(JSON.parse(JSON.stringify(actual))),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/boost-ability-precheck.json.gz')))));
 assert.equal(actual.cases.length,29);assert.equal(actual.cases.reduce((n,c)=>n+c.steps.length,0),36);
 assert.ok(actual.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
 const c=id=>actual.cases.find(c=>c.id==='G4-boost-ability-'+id),mon=id=>c(id).steps[0].after.sides[1].pokemon[0];
 assert.equal(mon('defiant-growl').boosts.atk,1);assert.equal(mon('defiant-tickle').boosts.atk,3);
 assert.equal(mon('defiant-cap--6').boosts.atk,-6);assert.equal(mon('defiant-herb').item,'whiteherb');
 assert.equal(c('download-pressure-hardy').initial.sides[0].pokemon[0].boosts.spa,1);
 assert.equal(c('download-pressure-calm').initial.sides[0].pokemon[0].boosts.atk,1);
 assert.equal(c('download-ignore-marvelscale').steps[0].after.sides[0].pokemon[1].boosts.atk,1);
 assert.deepEqual(actual.witnesses.map(w=>w.set.species),['Farfetch’d','Genesect']);
});
