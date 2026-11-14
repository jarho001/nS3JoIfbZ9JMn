'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),zlib=require('node:zlib');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native-only Sucker Punch and Payback preserve queued-action predicates and dynamic power',()=>{
 require('../reference/neutral-turn-oracle.cjs').assertReady();
 const ps=require('../reference/vendor/pokemon-showdown/dist/sim'),{runCase}=require('../reference/shared-engine-cases.cjs'),spec=require('../reference/conditional-attack-cases.cjs');
 const actual={pin:require('../reference-lock.json').commit,witnesses:spec.membershipWitnesses(),cases:spec().map(c=>runCase(ps,c))};
 assert.deepEqual(normalizeStacks(JSON.parse(JSON.stringify(actual))),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/conditional-attack-precheck.json.gz')))));
 assert.equal(actual.cases.length,38);assert.equal(actual.cases.reduce((n,c)=>n+c.steps.length,0),48);assert.ok(actual.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
 const c=id=>actual.cases.find(c=>c.id==='G4-conditional-attack-'+id),lines=id=>c(id).steps.at(-1).events.p1.map(e=>e.message);
 for(const suffix of ['vs-recover','vs-protect','priority-first','switch','recharge','slow-sleep-talk'])assert.ok(lines('suckerpunch-'+suffix).includes('|-fail|p1a: Mew'),suffix);
 assert.ok(!lines('suckerpunch-vs-tackle').includes('|-fail|p1a: Mew'));
 const hp=id=>Number(c(id).steps[0].events.p2.find(e=>e.message.startsWith('|-damage|p2a:')).message.split('|')[3].split('/')[0]);
 assert.ok(hp('payback-slow')<hp('payback-vs-tackle'));
 assert.equal(hp('payback-slow'),hp('payback-slow-technician'));
 assert.ok(hp('payback-technician')<hp('payback-vs-tackle'));
 assert.deepEqual(actual.witnesses.map(w=>w.set.species),['Arbok','Ambipom']);
});
