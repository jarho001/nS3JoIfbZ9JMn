'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),zlib=require('node:zlib'),{execFileSync}=require('node:child_process');
test('native Forecast and Flower Gift follow weather with original identity',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'g4-forms-'));
 try{
  const out=path.join(dir,'native.json');execFileSync('node',['reference/shared-engine-cases.cjs','--batch','forms','--output',out]);
  const x=JSON.parse(fs.readFileSync(out));assert.deepEqual(x,JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/forms.json.gz'))));assert.equal(x.cases.length,16);assert.equal(x.cases.reduce((n,c)=>n+c.steps.length,0),43);
  const c=x.cases.find(c=>c.id==='G4-forms-cycle');
  assert.deepEqual(c.steps.map(s=>s.after.sides[0].pokemon[0].species),['Castform-Sunny','Castform-Rainy','Castform-Snowy','Castform']);
  assert.ok(c.steps.every(s=>s.after.sides[0].pokemon[0].identityState.baseSpecies==='castform'));
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
