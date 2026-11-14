'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),zlib=require('node:zlib');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native-only Unaware precheck preserves directional boosts and global stat-query draws',()=>{
 require('../reference/neutral-turn-oracle.cjs').assertReady();
 const ps=require('../reference/vendor/pokemon-showdown/dist/sim'),{runCase}=require('../reference/shared-engine-cases.cjs'),spec=require('../reference/unaware-cases.cjs');
 const actual={pin:require('../reference-lock.json').commit,witnesses:spec.membershipWitnesses(),cases:spec().map(c=>runCase(ps,c))};
 assert.deepEqual(normalizeStacks(JSON.parse(JSON.stringify(actual))),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/unaware-precheck.json.gz')))));
 assert.equal(actual.cases.length,28);assert.equal(actual.cases.reduce((n,c)=>n+c.steps.length,0),37);
 assert.ok(actual.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
 const c=id=>actual.cases.find(c=>c.id==='G4-unaware-'+id),hp=id=>Number(c(id).steps[0].events.p2.find(e=>e.message.startsWith('|-damage|p2a: Mew|')).message.split('|')[3].split('/')[0]);
 assert.ok(hp('source-3-1')<hp('target-3-1'));assert.ok(hp('source--3-1')>hp('target--3-1'));
 assert.equal(c('dual-tie').initializationDraws.length,5);assert.equal(c('dual-fast-source').initializationDraws.length,0);
 assert.deepEqual(c('dual-tie').steps[0].draws.slice(0,4).map(d=>d.semanticContext),Array(4).fill('modify-boost-sort'));
 assert.equal(c('dual-forced').steps[1].draws.filter(d=>d.semanticContext==='modify-boost-sort').length,2);
 assert.equal(actual.witnesses[0].set.species,'Quagsire');
});
