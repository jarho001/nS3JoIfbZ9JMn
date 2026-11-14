'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),zlib=require('node:zlib'),{execFileSync}=require('node:child_process');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native Light Ball, Thick Club and Soul Dew preserve species and stat-chain guards',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'g4-species-items-'));
 try{
  const out=path.join(dir,'native.json');execFileSync('node',['reference/shared-engine-cases.cjs','--batch','speciesitems','--output',out]);
  const x=JSON.parse(fs.readFileSync(out));assert.deepEqual(normalizeStacks(x),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/species-items.json.gz')))));
  assert.equal(x.cases.length,41);assert.equal(x.cases.reduce((n,c)=>n+c.steps.length,0),94);
  const c=id=>x.cases.find(c=>c.id==='G4-species-item-'+id),hp=(id,i=0)=>c(id).steps[i].after.sides[1].pokemon[0].hp;
  assert.ok(hp('ball-ordinary')<hp('ball-no-item'));assert.ok(hp('club-ordinary')<hp('club-no-item'));
  assert.equal(hp('dew-ordinary'),hp('dew-no-item'));assert.ok(hp('dew-ordinary',1)<hp('dew-no-item',1));
  assert.ok(x.cases.filter(c=>c.id.endsWith('-critical')).every(c=>c.steps.some(s=>s.events.p1.some(e=>e.message.startsWith('|-crit|')))));
  for(const key of ['ball','club','dew'])for(const suffix of ['owner','nonowner'])assert.equal(c(key+'-transform-'+suffix).steps[0].after.sides[0].pokemon[0].identityState.transformed,true);
  assert.ok(x.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
  assert.deepEqual(require('../reference/species-item-cases.cjs').membershipWitnesses().map(w=>w.set.species),['Pikachu','Marowak','Latios']);
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
