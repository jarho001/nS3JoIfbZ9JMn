'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),zlib=require('node:zlib');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native weight moves preserve hectogram thresholds, ratio boundaries and transformed species',()=>{
 require('../reference/neutral-turn-oracle.cjs').assertReady();
 const ps=require('../reference/vendor/pokemon-showdown/dist/sim'),spec=require('../reference/weight-based-cases.cjs');
 const actual=JSON.parse(require('node:child_process').execFileSync('node',['reference/shared-engine-cases.cjs','--batch','weightbased'],{maxBuffer:48*1024*1024}));
 assert.deepEqual(normalizeStacks(actual),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/weight-based.json.gz')))));
 assert.equal(actual.cases.length,40);assert.equal(actual.cases.reduce((n,c)=>n+c.steps.length,0),48);assert.ok(actual.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
 for(const c of actual.cases)for(const side of ['p1','p2'])for(const mon of c.setup[side])assert.equal(new Set(mon.moves.map(m=>ps.Dex.toID(m))).size,mon.moves.length,c.id+' unique moves');
 const dex=ps.Dex.mod('gen5'),invoke=(id,source,target)=>dex.moves.get(id).basePowerCallback.call({debug(){}},{getWeight:()=>source},{getWeight:()=>target});
 for(const key of ['lowkick','grassknot'])for(const [weight,power]of [[99,20],[100,40],[249,40],[250,60],[499,60],[500,80],[999,80],[1000,100],[1999,100],[2000,120]])assert.equal(invoke(key,40,weight),power,key+' '+weight+'hg');
 for(const [weight,power]of [[199,40],[200,60],[299,60],[300,80],[399,80],[400,100],[499,100],[500,120]])assert.equal(invoke('heavyslam',weight,100),power,'ratio '+weight+'/100');
 assert.equal(dex.species.get('Mew').weighthg,40);
 const c=id=>actual.cases.find(c=>c.id==='G4-weight-based-'+id),hp=id=>Number(c(id).steps[0].events.p2.find(e=>e.message.startsWith('|-damage|p2a:')).message.split('|')[3].split(/[ /]/)[0]);
 for(const key of ['lowkick','grassknot']){
  assert.ok(hp(key+'-technician-vulpix')<hp(key+'-vulpix'));assert.ok(hp(key+'-technician-flareon')<hp(key+'-flareon'));assert.equal(hp(key+'-technician-zweilous'),hp(key+'-zweilous'));
  assert.ok(c(key+'-roughskin').steps[0].events.p1.some(e=>e.message.includes('[from] ability: Rough Skin')));
 }
 assert.deepEqual(spec.membershipWitnesses().map(w=>w.set.species),['Aggron','Emboar','Bisharp']);
});
