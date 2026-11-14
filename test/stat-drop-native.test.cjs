'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),zlib=require('node:zlib');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native-only White Herb and Clear Body precheck reproduces pinned branches and witnesses',()=>{
 require('../reference/neutral-turn-oracle.cjs').assertReady();
 const ps=require('../reference/vendor/pokemon-showdown/dist/sim'),{runCase}=require('../reference/shared-engine-cases.cjs'),spec=require('../reference/stat-drop-cases.cjs');
 const cases=spec().map(c=>runCase(ps,c)),actual={pin:require('../reference-lock.json').commit,witnesses:spec.membershipWitnesses(),cases};
 assert.deepEqual(normalizeStacks(JSON.parse(JSON.stringify(actual))),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/stat-drop-precheck.json.gz')))));
 assert.equal(cases.length,29);assert.equal(cases.reduce((n,c)=>n+c.steps.length,0),65);
 assert.ok(cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
 const mon=id=>cases.find(c=>c.id==='G4-stat-drop-'+id).steps[0].after.sides[1].pokemon[0];
 assert.equal(mon('clear-growl').boosts.atk,0);assert.equal(mon('clear-self').boosts.def,-1);
 assert.equal(mon('herb-self').boosts.def,0);assert.equal(mon('herb-self').boosts.atk,2);assert.equal(mon('herb-self').item,'');
 assert.equal(mon('herb-self').phaseState.lastItem,'whiteherb');assert.equal(mon('herb-self').phaseState.ateBerry,false);
 assert.equal(mon('herb-simple').boosts.atk,4);assert.equal(mon('herb-contrary').boosts.atk,0);assert.equal(mon('herb-contrary').boosts.def,1);
 assert.deepEqual(actual.witnesses.map(w=>w.set.species),['Klinklang','Crustle']);
});
