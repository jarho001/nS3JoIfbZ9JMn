'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),zlib=require('node:zlib'),{execFileSync}=require('node:child_process');
test('native Gen5 Copycat and Mirror Move preserve caller and forced-move semantics',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'g4-copy-moves-'));
 try{
  const out=path.join(dir,'native.json');execFileSync('node',['reference/shared-engine-cases.cjs','--batch','copymoves','--output',out]);
  const x=JSON.parse(fs.readFileSync(out));assert.deepEqual(x,JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/copy-moves.json.gz'))));assert.equal(x.cases.length,24);assert.equal(x.cases.reduce((n,c)=>n+c.steps.length,0),48);
  const mirror=x.cases.find(c=>c.id==='G4-copy-moves-mirror');
  assert.equal(mirror.steps[0].after.sides[0].pokemon[0].moves[0].pp,29);
  const charge=x.cases.find(c=>c.id==='G4-copy-moves-copycat-charge-slot0');
  assert.equal(charge.steps[1].after.sides[0].pokemon[0].phaseState.moveSlotsUsed[1],false);
  assert.ok(x.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
