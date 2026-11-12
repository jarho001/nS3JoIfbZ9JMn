'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),zlib=require('node:zlib'),{execFileSync}=require('node:child_process');
test('native ability creation and boost changes retain original identity',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'g4-ability-create-'));
 try{
  const out=path.join(dir,'native.json');execFileSync('node',['reference/shared-engine-cases.cjs','--batch','abilitycreate','--output',out]);
  const x=JSON.parse(fs.readFileSync(out));assert.deepEqual(x,JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/ability-create.json.gz'))));assert.equal(x.cases.length,23);assert.equal(x.cases.reduce((n,c)=>n+c.steps.length,0),48);
  const c=x.cases.find(c=>c.id==='G4-ability-create-simple-beam');
  assert.equal(c.steps[0].after.sides[1].pokemon[0].boosts.spa,2);
  assert.equal(c.steps[0].after.sides[1].pokemon[0].identityState.baseAbility,'pressure');
  const w=x.cases.find(c=>c.id==='G4-ability-create-worry-sleep');
  assert.equal(w.steps[0].after.sides[1].pokemon[0].status,'slp');
  assert.equal(w.steps[1].after.sides[1].pokemon[0].status,'');
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
