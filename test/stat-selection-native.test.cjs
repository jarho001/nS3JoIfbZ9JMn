'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),zlib=require('node:zlib');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native-only stat-selection moves preserve borrowed stats, source modifiers and defensive-stat hooks',()=>{
 require('../reference/neutral-turn-oracle.cjs').assertReady();
 const ps=require('../reference/vendor/pokemon-showdown/dist/sim'),{runCase}=require('../reference/shared-engine-cases.cjs'),spec=require('../reference/stat-selection-cases.cjs');
 const actual={pin:require('../reference-lock.json').commit,witnesses:spec.membershipWitnesses(),cases:spec().map(c=>runCase(ps,c))};
 assert.deepEqual(normalizeStacks(JSON.parse(JSON.stringify(actual))),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/stat-selection-precheck.json.gz')))));
 assert.equal(actual.cases.length,66);assert.equal(actual.cases.reduce((n,c)=>n+c.steps.length,0),76);assert.ok(actual.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
 const c=id=>actual.cases.find(c=>c.id==='G4-stat-selection-'+id),hp=id=>Number(c(id).steps.at(-1).events.p2.find(e=>e.message.startsWith('|-damage|p2a:')).message.split('|')[3].split(/[ /]/)[0]);
 for(const suffix of ['actor-attack','target-hugepower','target-band','burn-target','target-guts'])assert.equal(hp('foulplay-'+suffix),hp('foulplay-base'),suffix);
 assert.ok(hp('foulplay-target-attack')<hp('foulplay-base'));assert.ok(hp('foulplay-source-band')<hp('foulplay-base'));assert.ok(hp('foulplay-burn-source')>hp('foulplay-base'));
 for(const key of ['psyshock','psystrike','secretsword']){
  assert.equal(hp(key+'-special-defense'),hp(key+'-base'));assert.ok(hp(key+'-defense')>hp(key+'-base'));
  assert.equal(hp(key+'-reflect'),hp(key+'-base'));assert.ok(hp(key+'-lightscreen')>hp(key+'-base'));
  assert.equal(hp(key+'-choiceband'),hp(key+'-base'));assert.ok(hp(key+'-choicespecs')<hp(key+'-base'));
  assert.ok(c(key+'-critical').steps[0].events.p1.some(e=>e.message.startsWith('|-crit|')));
 }
 assert.ok(c('foulplay-critical').steps[0].events.p1.some(e=>e.message.startsWith('|-crit|')));
 assert.deepEqual(actual.witnesses.map(w=>w.set.species),['Latios','Amoonguss','Mewtwo','Keldeo']);
});
