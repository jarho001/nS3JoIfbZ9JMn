'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),zlib=require('node:zlib');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native Unaware precheck preserves directional boosts and global stat-query draws',()=>{
 require('../reference/neutral-turn-oracle.cjs').assertReady();
 const spec=require('../reference/unaware-cases.cjs');
 const actual=JSON.parse(require('node:child_process').execFileSync('node',['reference/shared-engine-cases.cjs','--batch','unaware'],{maxBuffer:24*1024*1024}));
 assert.deepEqual(normalizeStacks(JSON.parse(JSON.stringify(actual))),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/unaware.json.gz')))));
 assert.equal(actual.cases.length,32);assert.equal(actual.cases.reduce((n,c)=>n+c.steps.length,0),46);
 assert.ok(actual.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
 const c=id=>actual.cases.find(c=>c.id==='G4-unaware-'+id),hp=id=>Number(c(id).steps[0].events.p2.find(e=>e.message.startsWith('|-damage|p2a: Mew|')).message.split('|')[3].split('/')[0]);
 assert.ok(hp('source-3-1')<hp('target-3-1'));assert.ok(hp('source--3-1')>hp('target--3-1'));
 assert.equal(c('dual-tie').initializationDraws.length,5);assert.equal(c('dual-fast-source').initializationDraws.length,0);
 assert.deepEqual(c('dual-tie').steps[0].draws.slice(0,4).map(d=>d.semanticContext),Array(4).fill('modify-boost-sort'));
 assert.equal(c('dual-forced').steps[1].draws.filter(d=>d.semanticContext==='modify-boost-sort').length,2);
 assert.equal(spec.membershipWitnesses()[0].set.species,'Quagsire');
});
