'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),zlib=require('node:zlib'),{execFileSync}=require('node:child_process');
test('native Illusion displays disguises until the damaging-hit reveal',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'g4-illusion-'));
 try{
  const out=path.join(dir,'native.json');execFileSync('node',['reference/shared-engine-cases.cjs','--batch','illusion','--output',out]);
  const x=JSON.parse(fs.readFileSync(out));assert.deepEqual(x,JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/illusion.json.gz'))));assert.equal(x.cases.length,12);assert.equal(x.cases.reduce((n,c)=>n+c.steps.length,0),25);
  const c=x.cases.find(c=>c.id==='G4-illusion-multihit');
  const events=c.steps[0].events.p2.map(e=>e.message);
  assert.ok(events.some(e=>e.startsWith('|-damage|p1a: Snorlax|')));
  assert.ok(events.some(e=>e.startsWith('|-damage|p1a: Zoroark|')));
  assert.ok(events.some(e=>e.startsWith('|replace|p1a: Zoroark|')));
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
