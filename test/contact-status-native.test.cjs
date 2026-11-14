'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),zlib=require('node:zlib');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native-only Static and Flame Body preserve contact chance, status attribution and post-terminal draws',()=>{
 require('../reference/neutral-turn-oracle.cjs').assertReady();
 const ps=require('../reference/vendor/pokemon-showdown/dist/sim'),{runCase}=require('../reference/shared-engine-cases.cjs'),spec=require('../reference/contact-status-cases.cjs');
 const actual={pin:require('../reference-lock.json').commit,witnesses:spec.membershipWitnesses(),cases:spec().map(c=>runCase(ps,c))};
 assert.deepEqual(normalizeStacks(JSON.parse(JSON.stringify(actual))),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/contact-status-precheck.json.gz')))));
 assert.equal(actual.cases.length,40);assert.equal(actual.cases.reduce((n,c)=>n+c.steps.length,0),52);
 assert.ok(actual.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
 const first=id=>actual.cases.find(c=>c.id==='G4-contact-status-'+id).steps[0],draws=id=>first(id).draws.filter(d=>d.semanticContext==='contact-status');
 for(const [key,status,name]of [['static','par','Static'],['flamebody','brn','Flame Body']]){
  const hit=first(key+'-tackle');assert.equal(hit.after.sides[0].pokemon[0].status,status);
  assert.ok(hit.events.p1.some(e=>e.message==='|-status|p1a: Mew|'+status+'|[from] ability: '+name+'|[of] p2a: Mew'));
  assert.equal(draws(key+'-surf').length,0);assert.equal(draws(key+'-earthquake').length,0);
  assert.equal(draws(key+'-immune').length,1);assert.equal(first(key+'-immune').after.sides[0].pokemon[0].status,'');
  assert.equal(draws(key+'-helmet-source-ko').length,1);assert.ok(first(key+'-helmet-source-ko').events.p1.some(e=>e.message==='|win|p2'));
  assert.equal(first(key+'-lum').after.sides[0].pokemon[0].status,'');assert.equal(first(key+'-lum').after.sides[0].pokemon[0].item,'');
  assert.equal(first(key+'-synchronize').after.sides[1].pokemon[0].status,status);
 }
 assert.equal(first('static-electric').after.sides[0].pokemon[0].status,'par');
 assert.equal(first('flamebody-fire').after.sides[0].pokemon[0].status,'');
 assert.deepEqual(actual.witnesses.map(w=>w.set.species),['Ampharos','Magmortar']);
});
