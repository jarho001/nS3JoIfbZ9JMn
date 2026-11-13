'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),zlib=require('node:zlib'),{execFileSync}=require('node:child_process');
test('native Gen5 Nature Power calls Earthquake with caller PP',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'g4-nature-power-'));
 try{
  const out=path.join(dir,'native.json');execFileSync('node',['reference/shared-engine-cases.cjs','--batch','naturepower','--output',out]);
  const x=JSON.parse(fs.readFileSync(out));assert.deepEqual(x,JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/nature-power.json.gz'))));assert.equal(x.cases.length,14);assert.equal(x.cases.reduce((n,c)=>n+c.steps.length,0),19);
  const c=x.cases.find(c=>c.id==='G4-nature-power-ordinary');
  assert.equal(c.steps[0].after.sides[0].pokemon[0].phaseState.lastMove.id,'naturepower');
  assert.equal(c.steps[0].after.sides[0].pokemon[0].phaseState.lastMoveUsed.id,'earthquake');
  assert.equal(c.steps[0].after.sides[0].pokemon[0].moves[0].pp,30);
  assert.ok(x.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
