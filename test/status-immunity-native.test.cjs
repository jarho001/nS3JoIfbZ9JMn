'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),zlib=require('node:zlib'),{execFileSync}=require('node:child_process');
test('native status immunity prevents ailments and cures them on ability acquisition',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'g4-status-immunity-'));
 try{
  const out=path.join(dir,'native.json');execFileSync('node',['reference/shared-engine-cases.cjs','--batch','statusimmunity','--output',out]);
  const x=JSON.parse(fs.readFileSync(out));assert.deepEqual(x,JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/status-immunity.json.gz'))));assert.equal(x.cases.length,18);assert.equal(x.cases.reduce((n,c)=>n+c.steps.length,0),39);
  const c=x.cases.find(c=>c.id==='G4-status-immunity-acquire-Insomnia-slp');
  assert.equal(c.steps[0].after.sides[1].pokemon[0].status,'slp');
  assert.equal(c.steps[1].after.sides[1].pokemon[0].status,'');
  const f=x.cases.find(c=>c.id==='G4-status-immunity-freeze-secondary');
  assert.ok(f.steps.flatMap(s=>s.draws).some(d=>d.semanticContext==='secondary'&&d.value<10));
  assert.ok(f.steps.every(s=>!s.after.sides[1].pokemon[0].status));
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
