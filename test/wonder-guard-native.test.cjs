'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),zlib=require('node:zlib'),{execFileSync}=require('node:child_process');
test('native Gen5 Wonder Guard preserves direct immunities and indirect bypasses',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'g4-wonder-guard-'));
 try{
  const out=path.join(dir,'native.json');execFileSync('node',['reference/shared-engine-cases.cjs','--batch','wonderguard','--output',out]);
  const x=JSON.parse(fs.readFileSync(out));assert.deepEqual(x,JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/wonder-guard.json.gz'))));assert.equal(x.cases.length,18);assert.equal(x.cases.reduce((n,c)=>n+c.steps.length,0),22);
  const neutral=x.cases.find(c=>c.id==='G4-wonder-guard-neutral');
  assert.ok(neutral.steps[0].events.p1.some(e=>e.message.includes('[from] ability: Wonder Guard')));
  const typed=x.cases.find(c=>c.id==='G4-wonder-guard-type-immunity');
  assert.ok(typed.steps[0].events.p1.some(e=>e.message==='|-immune|p2a: Shedinja'));
  const struggle=x.cases.find(c=>c.id==='G4-wonder-guard-struggle');
  assert.ok(!struggle.steps[0].events.p1.some(e=>e.message.includes('[from] ability: Wonder Guard')));
  assert.ok(x.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
