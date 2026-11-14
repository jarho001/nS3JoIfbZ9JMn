'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),zlib=require('node:zlib');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native Download and Defiant precheck preserves acquisition, per-stat drops and caps',()=>{
 require('../reference/neutral-turn-oracle.cjs').assertReady();
 const spec=require('../reference/boost-ability-cases.cjs');
 const actual=JSON.parse(require('node:child_process').execFileSync('node',['reference/shared-engine-cases.cjs','--batch','boostability'],{maxBuffer:24*1024*1024}));
 assert.deepEqual(normalizeStacks(JSON.parse(JSON.stringify(actual))),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/boost-ability.json.gz')))));
 assert.equal(actual.cases.length,30);assert.equal(actual.cases.reduce((n,c)=>n+c.steps.length,0),37);
 assert.ok(actual.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
 const c=id=>actual.cases.find(c=>c.id==='G4-boost-ability-'+id),mon=id=>c(id).steps[0].after.sides[1].pokemon[0];
 assert.equal(mon('defiant-growl').boosts.atk,1);assert.equal(mon('defiant-tickle').boosts.atk,3);
 assert.equal(mon('defiant-cap--6').boosts.atk,-6);assert.equal(mon('defiant-herb').item,'whiteherb');
 assert.equal(c('download-pressure-hardy').initial.sides[0].pokemon[0].boosts.spa,1);
 assert.equal(c('download-pressure-calm').initial.sides[0].pokemon[0].boosts.atk,1);
 assert.equal(c('download-ignore-marvelscale').steps[0].after.sides[0].pokemon[1].boosts.atk,1);
 assert.deepEqual(spec.membershipWitnesses().map(w=>w.set.species),['Farfetch’d','Genesect']);
});
