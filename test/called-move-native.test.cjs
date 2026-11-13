'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),zlib=require('node:zlib'),{execFileSync}=require('node:child_process');
test('native Sleep Talk distinguishes root and called move accounting',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'g4-called-'));
 try{
  const out=path.join(dir,'native.json');execFileSync('node',['reference/shared-engine-cases.cjs','--batch','calledmoves','--output',out]);
  const x=JSON.parse(fs.readFileSync(out));assert.deepEqual(x,JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/called-moves.json.gz'))));assert.equal(x.cases.length,20);assert.equal(x.cases.reduce((n,c)=>n+c.steps.length,0),51);
  const c=x.cases.find(c=>c.id==='G4-called-ordinary');
  assert.equal(c.steps[1].after.sides[0].pokemon[0].phaseState.lastMove.id,'sleeptalk');
  assert.equal(c.steps[1].after.sides[0].pokemon[0].phaseState.lastMoveUsed.id,'tackle');
  assert.equal(c.steps[1].after.sides[0].pokemon[0].moves[2].pp,56);
  assert.equal(c.steps[1].after.sides[0].pokemon[0].moves[1].pp,14);
  assert.ok(x.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
