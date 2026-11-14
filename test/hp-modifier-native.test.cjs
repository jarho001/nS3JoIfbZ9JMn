'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),zlib=require('node:zlib');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native-only Multiscale and Defeatist precheck preserves HP boundaries and raw handler draws',()=>{
 require('../reference/neutral-turn-oracle.cjs').assertReady();
 const ps=require('../reference/vendor/pokemon-showdown/dist/sim'),{runCase}=require('../reference/shared-engine-cases.cjs'),spec=require('../reference/hp-modifier-cases.cjs');
 const actual={pin:require('../reference-lock.json').commit,witnesses:spec.membershipWitnesses(),cases:spec().map(c=>runCase(ps,c))};
 assert.deepEqual(normalizeStacks(JSON.parse(JSON.stringify(actual))),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/hp-modifier-precheck.json.gz')))));
 assert.equal(actual.cases.length,29);assert.equal(actual.cases.reduce((n,c)=>n+c.steps.length,0),40);
 assert.ok(actual.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
 const c=id=>actual.cases.find(c=>c.id==='G4-hp-modifier-'+id),hp=id=>c(id).steps[0].after.sides[1].pokemon[0].hp;
 for(const move of ['tackle','surf']){assert.equal(hp('defeatist-'+move+'-87'),hp('defeatist-'+move+'-88'));assert.ok(hp('defeatist-'+move+'-88')>hp('defeatist-'+move+'-89'));assert.ok(175-hp('multiscale-'+move)<174-hp('multiscale-'+move+'-chipped'));}
 assert.ok(c('multiscale-critical').steps[0].events.p1.some(e=>e.message.startsWith('|-crit|')));
 assert.ok(c('multiscale-suppressed-tie').steps[1].draws.some(d=>d.semanticContext==='damage-modifiers'));
 assert.ok(c('defeatist-suppressed-tie').steps[1].draws.some(d=>d.semanticContext==='damage-calculation-sort'));
 assert.equal(c('defeatist-multi-mummy').steps[0].after.sides[0].pokemon[0].ability,'mummy');
 assert.deepEqual(actual.witnesses.map(w=>w.set.species),['Dragonite','Archeops']);
});
