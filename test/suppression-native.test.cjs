'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),zlib=require('node:zlib'),{execFileSync}=require('node:child_process');
test('native Gastro Acid suppresses effects while retaining stored ability',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'g4-suppression-'));
 try{
  const out=path.join(dir,'native.json');execFileSync('node',['reference/shared-engine-cases.cjs','--batch','suppression','--output',out]);
  const x=JSON.parse(fs.readFileSync(out));assert.deepEqual(x,JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/suppression.json.gz'))));assert.equal(x.cases.length,23);assert.equal(x.cases.reduce((n,c)=>n+c.steps.length,0),47);
  const c=x.cases.find(c=>c.id==='G4-suppression-pressure');
  assert.equal(c.steps[0].after.sides[1].pokemon[0].ability,'pressure');
  assert.equal(c.steps[0].after.sides[1].pokemon[0].volatiles.gastroacid.name,'Gastro Acid');
  const w=x.cases.find(c=>c.id==='G4-suppression-start-suppressed');
  assert.equal(w.steps[1].after.sides[1].pokemon[0].ability,'drought');
  assert.equal(w.steps[1].after.field.weather,'raindance');
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
