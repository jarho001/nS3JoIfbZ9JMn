'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),zlib=require('node:zlib'),{execFileSync}=require('node:child_process');
test('native ability exchange preserves base ability, provenance and Start semantics',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'g4-ability-change-'));
 try{
  const out=path.join(dir,'native.json');execFileSync('node',['reference/shared-engine-cases.cjs','--batch','abilitychange','--output',out]);
  const x=JSON.parse(fs.readFileSync(out));assert.deepEqual(x,JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/ability-change.json.gz'))));assert.equal(x.cases.length,29);assert.equal(x.cases.reduce((n,c)=>n+c.steps.length,0),58);
  const c=x.cases.find(c=>c.id==='G4-ability-change-swap-pressure');
  assert.equal(c.steps[0].after.sides[0].pokemon[0].ability,'pressure');
  assert.equal(c.steps[0].after.sides[0].pokemon[0].identityState.baseAbility,'synchronize');
  assert.equal(c.steps[0].after.sides[1].pokemon[0].abilityChangeState.volatileStaleness,'external');
  const a=x.cases.find(c=>c.id==='G4-ability-change-swap-imposter-start');
  assert.equal(a.steps[2].after.sides[0].pokemon[0].identityState.transformed,false);
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
