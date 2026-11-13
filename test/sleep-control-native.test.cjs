'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),zlib=require('node:zlib'),{execFileSync}=require('node:child_process');
test('native Yawn and Own Tempo retain delayed source and confusion immunity',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'g4-sleep-control-'));
 try{
  const out=path.join(dir,'native.json');execFileSync('node',['reference/shared-engine-cases.cjs','--batch','sleepcontrol','--output',out]);
  const x=JSON.parse(fs.readFileSync(out));assert.deepEqual(x,JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/sleep-control.json.gz'))));assert.equal(x.cases.length,18);assert.equal(x.cases.reduce((n,c)=>n+c.steps.length,0),36);
  const c=x.cases.find(c=>c.id==='G4-sleep-control-source-switch');
  assert.equal(c.steps[1].after.sides[1].pokemon[0].status,'slp');
  assert.deepEqual(c.steps[1].after.sides[1].pokemon[0].statusState.source,{$pokemon:{side:0,mon:0}});
  const w=x.cases.find(c=>c.id==='G4-sleep-control-own-block');
  assert.equal(w.steps[0].after.sides[1].pokemon[0].volatiles.confusion,undefined);
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
