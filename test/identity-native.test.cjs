'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),zlib=require('node:zlib'),{execFileSync}=require('node:child_process');
test('native Transform and Imposter preserve original requests and virtual identity',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'g4-identity-'));
 try{
  const out=path.join(dir,'native.json');execFileSync('node',['reference/shared-engine-cases.cjs','--batch','identity','--output',out]);
  const x=JSON.parse(fs.readFileSync(out));assert.deepEqual(x,JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/identity.json.gz'))));assert.equal(x.cases.length,17);assert.equal(x.cases.reduce((n,c)=>n+c.steps.length,0),36);
  const ordinary=x.cases.find(c=>c.id==='G4-identity-ordinary');
  const mon=ordinary.steps[0].after.sides[0].pokemon[0];
  assert.equal(mon.species,'Gengar');assert.equal(mon.identityState.baseSpecies,'mew');
  assert.equal(mon.identityState.transformed,true);assert.ok(mon.identityState.virtualMoves.every(Boolean));
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
