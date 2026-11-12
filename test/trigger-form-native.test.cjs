'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),zlib=require('node:zlib'),{execFileSync}=require('node:child_process');
test('native Relic Song and Zen Mode preserve form lifecycle',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'g4-trigger-forms-'));
 try{
  const out=path.join(dir,'native.json');execFileSync('node',['reference/shared-engine-cases.cjs','--batch','triggerforms','--output',out]);
  const x=JSON.parse(fs.readFileSync(out));assert.deepEqual(x,JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/trigger-forms.json.gz'))));assert.equal(x.cases.length,17);assert.equal(x.cases.reduce((n,c)=>n+c.steps.length,0),30);
  const c=x.cases.find(c=>c.id==='G4-trigger-forms-zen-low');
  assert.equal(c.steps[0].after.sides[0].pokemon[0].species,'Darmanitan-Zen');
  assert.equal(c.steps[1].after.sides[0].pokemon[0].species,'Darmanitan');
  assert.equal(c.steps[0].after.sides[0].pokemon[0].volatiles.zenmode.sourceEffect.type,'Ability');
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
