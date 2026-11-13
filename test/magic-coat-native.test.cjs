'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),zlib=require('node:zlib'),{execFileSync}=require('node:child_process');
test('native Gen5 Magic Coat preserves condition source, priority flag and expiry',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'g4-magic-coat-'));
 try{
  const out=path.join(dir,'native.json');execFileSync('node',['reference/shared-engine-cases.cjs','--batch','magiccoat','--output',out]);
  const x=JSON.parse(fs.readFileSync(out));assert.deepEqual(x,JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/magic-coat.json.gz'))));assert.equal(x.cases.length,18);assert.equal(x.cases.reduce((n,c)=>n+c.steps.length,0),23);
  const pivot=x.cases.find(c=>c.id==='G4-magic-coat-prankster-pivot');
  assert.equal(pivot.steps[0].after.sides[1].pokemon[0].volatiles.magiccoat.pranksterBoosted,true);
  assert.ok(!('magiccoat' in pivot.steps[1].after.sides[1].pokemon[0].volatiles));
  const both=x.cases.find(c=>c.id==='G4-magic-coat-both-bounce');
  assert.ok(both.steps[0].events.p1.some(e=>e.message.includes('[from] Magic Coat')));
  assert.ok(!both.steps[0].events.p1.some(e=>e.message.includes('[from] ability: Magic Bounce')));
  assert.ok(x.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
