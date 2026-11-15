'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),zlib=require('node:zlib');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native-only Haze and Clear Smog clear raw boosts with distinct target and hit gates',()=>{
 require('../reference/neutral-turn-oracle.cjs').assertReady();
 const ps=require('../reference/vendor/pokemon-showdown/dist/sim'),{runCase}=require('../reference/shared-engine-cases.cjs'),spec=require('../reference/boost-reset-cases.cjs');
 const actual={pin:require('../reference-lock.json').commit,witnesses:spec.membershipWitnesses(),cases:spec().map(c=>runCase(ps,c))};
 assert.deepEqual(normalizeStacks(JSON.parse(JSON.stringify(actual))),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/boost-reset-precheck.json.gz')))));
 assert.equal(actual.cases.length,35);assert.equal(actual.cases.reduce((n,c)=>n+c.steps.length,0),41);assert.ok(actual.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
 for(const c of actual.cases)for(const side of ['p1','p2'])for(const mon of c.setup[side])assert.equal(new Set(mon.moves.map(m=>ps.Dex.toID(m))).size,mon.moves.length,c.id+' unique moves');
 const c=id=>actual.cases.find(c=>c.id==='G4-boost-reset-'+id),p=(id,side=1,i=0)=>c(id).steps[i].after.sides[side].pokemon[0],zero=b=>Object.values(b).every(v=>v===0);
 for(const key of ['haze','smog'])for(const suffix of ['both','simple','contrary','defiant','clearbody','unaware'])assert.ok(zero(p(key+'-'+suffix).boosts),key+'-'+suffix);
 assert.ok(zero(p('haze-both',0).boosts));assert.ok(!zero(p('smog-both',0).boosts));
 for(const suffix of ['protect','steel']){assert.ok(zero(p('haze-'+suffix).boosts));assert.ok(!zero(p('smog-'+suffix).boosts));}
 assert.ok(zero(p('haze-substitute',1,1).boosts));assert.equal(p('smog-substitute',1,1).boosts.spd,2);
 for(const key of ['haze','smog']){assert.ok(zero(p(key+'-midturn',1,1).boosts));assert.ok(zero(p(key+'-sleep-talk',1,1).boosts));assert.ok(c(key+'-sleep-talk').steps[1].events.p1.some(e=>e.message.includes('[from] move: Sleep Talk')));}
 for(const key of ['haze','smog'])for(const side of [0,1])assert.equal(c(key+'-reserves').steps[0].after.sides[side].pokemon[1].boosts.spa,3);
 assert.equal(c('smog-terminal').steps[0].after.terminal,true);assert.ok(zero(p('smog-terminal').boosts));
 assert.deepEqual(actual.witnesses.map(w=>w.set.species),['Amoonguss','Tentacruel']);
});
