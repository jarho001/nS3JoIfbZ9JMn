'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),zlib=require('node:zlib');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native Sucker Punch and Payback preserve queued-action predicates and dynamic power',()=>{
 require('../reference/neutral-turn-oracle.cjs').assertReady();
 const spec=require('../reference/conditional-attack-cases.cjs');
 const actual=JSON.parse(require('node:child_process').execFileSync('node',['reference/shared-engine-cases.cjs','--batch','conditionalattack'],{maxBuffer:32*1024*1024}));
 assert.deepEqual(normalizeStacks(actual),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/conditional-attack.json.gz')))));
 assert.equal(actual.cases.length,38);assert.equal(actual.cases.reduce((n,c)=>n+c.steps.length,0),48);assert.ok(actual.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
 for(const c of actual.cases)for(const side of ['p1','p2'])for(const mon of c.setup[side])assert.equal(new Set(mon.moves.map(m=>m.toLowerCase().replace(/[^a-z0-9]/g,''))).size,mon.moves.length,c.id+' unique moves');
 const c=id=>actual.cases.find(c=>c.id==='G4-conditional-attack-'+id),lines=id=>c(id).steps.at(-1).events.p1.map(e=>e.message);
 for(const suffix of ['vs-recover','vs-protect','priority-first','switch','recharge','slow-sleep-talk'])assert.ok(lines('suckerpunch-'+suffix).includes('|-fail|p1a: Mew'),suffix);
 assert.ok(!lines('suckerpunch-vs-tackle').includes('|-fail|p1a: Mew'));
 const hp=id=>Number(c(id).steps[0].events.p2.find(e=>e.message.startsWith('|-damage|p2a:')).message.split('|')[3].split('/')[0]);
 assert.ok(hp('payback-slow')<hp('payback-vs-tackle'));
 assert.equal(hp('payback-slow'),hp('payback-slow-technician'));
 assert.ok(hp('payback-technician')<hp('payback-vs-tackle'));
 assert.deepEqual(spec.membershipWitnesses().map(w=>w.set.species),['Arbok','Ambipom']);
});
