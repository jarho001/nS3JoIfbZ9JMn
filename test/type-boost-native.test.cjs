'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),zlib=require('node:zlib'),{execFileSync}=require('node:child_process');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native type interception boosts retain caps, ordering and lifecycle boundaries',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'g4-type-boost-'));
 try{
  const out=path.join(dir,'native.json');execFileSync('node',['reference/shared-engine-cases.cjs','--batch','typeboost','--output',out]);
  const x=JSON.parse(fs.readFileSync(out));assert.deepEqual(normalizeStacks(x),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/type-boost.json.gz')))));
  assert.equal(x.cases.length,54);assert.equal(x.cases.reduce((n,c)=>n+c.steps.length,0),87);
  const c=id=>x.cases.find(c=>c.id==='G4-type-boost-'+id),events=(id,i=0)=>c(id).steps[i].events.p1.map(e=>e.message);
  for(const key of ['storm','rod','motor']){
   assert.ok(events(key+'-ordinary').some(s=>s.includes('|-boost|')));
   assert.ok(events(key+'-cap',6).some(s=>s.includes('|-immune|')));
   assert.ok(!events(key+'-cap',6).some(s=>s.includes('|-boost|')));
   assert.ok(events(key+'-protect').some(s=>s.includes('|-activate|')));
  }
  assert.ok(!events('rod-ground').some(s=>s.includes('Lightning Rod')));
  assert.ok(x.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
  assert.deepEqual(require('../reference/type-boost-cases.cjs').membershipWitnesses().map(w=>w.ability),['lightningrod','motordrive','stormdrain']);
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
