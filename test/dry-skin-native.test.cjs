'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),zlib=require('node:zlib'),{execFileSync}=require('node:child_process');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native Gen5 Dry Skin retains absorption, fire power and weather ordering',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'g4-dry-skin-'));
 try{
  const out=path.join(dir,'native.json');execFileSync('node',['reference/shared-engine-cases.cjs','--batch','dryskin','--output',out]);
  const x=JSON.parse(fs.readFileSync(out));assert.deepEqual(normalizeStacks(x),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/dry-skin.json.gz')))));
  assert.equal(x.cases.length,34);assert.equal(x.cases.reduce((n,c)=>n+c.steps.length,0),68);
  const c=id=>x.cases.find(c=>c.id==='G4-dry-skin-'+id),events=(id,i=0)=>c(id).steps[i].events.p1.map(e=>e.message);
  assert.equal(c('water-partial').steps[0].after.sides[1].pokemon[0].hp,103);
  assert.equal(c('rain').steps[0].after.sides[1].pokemon[0].hp,101);
  assert.equal(c('sun').steps[0].after.sides[1].pokemon[0].hp,59);
  assert.ok(events('sun').some(s=>s.includes('ability: Dry Skin|[of] p2a: Mew')));
  for(const id of ['rain-expiry','sun-expiry'])assert.ok(events(id,4).includes('|-weather|none'));
  assert.ok(x.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
  assert.equal(require('../reference/dry-skin-cases.cjs').membershipWitnesses()[0].set.species,'Jynx');
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
