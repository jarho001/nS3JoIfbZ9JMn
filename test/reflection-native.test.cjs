'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),zlib=require('node:zlib'),{execFileSync}=require('node:child_process');
test('native Gen5 reflection preserves source, return and one-bounce semantics',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'g4-reflection-'));
 try{
  const out=path.join(dir,'native.json');execFileSync('node',['reference/shared-engine-cases.cjs','--batch','reflection','--output',out]);
  const x=JSON.parse(fs.readFileSync(out));assert.deepEqual(x,JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/reflection.json.gz'))));assert.equal(x.cases.length,22);assert.equal(x.cases.reduce((n,c)=>n+c.steps.length,0),26);
  const called=x.cases.find(c=>c.id==='G4-reflection-called');
  assert.ok(called.steps[0].events.p1.some(e=>e.message.includes('[from] ability: Magic Bounce')));
  assert.equal(called.steps[0].after.sides[0].pokemon[0].phaseState.moveLastTurnResult,false);
  const protect=x.cases.find(c=>c.id==='G4-reflection-protected');
  assert.ok(!protect.steps[0].events.p1.some(e=>e.message.includes('[from] ability: Magic Bounce')));
  assert.ok(x.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
