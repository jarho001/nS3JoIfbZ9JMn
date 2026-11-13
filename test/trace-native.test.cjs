'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),zlib=require('node:zlib'),{execFileSync}=require('node:child_process');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native Gen5 Trace preserves seek, suppression and delayed copying',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'g4-trace-'));
 try{
  const out=path.join(dir,'native.json');execFileSync('node',['reference/shared-engine-cases.cjs','--batch','trace','--output',out]);
  const x=JSON.parse(fs.readFileSync(out));assert.deepEqual(normalizeStacks(x),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/trace.json.gz')))));assert.equal(x.cases.length,21);assert.equal(x.cases.reduce((n,c)=>n+c.steps.length,0),26);
  const waiting=x.cases.find(c=>c.id==='G4-trace-blocked-trace');
  assert.equal(waiting.initial.sides[0].pokemon[0].abilityState.seek,true);
  const acquire=x.cases.find(c=>c.id==='G4-trace-suppressed-acquire');
  assert.ok(!('seek' in acquire.steps[1].after.sides[0].pokemon[0].abilityState));
  const delayed=x.cases.find(c=>c.id==='G4-trace-delayed');
  assert.equal(delayed.steps[0].after.sides[0].pokemon[0].ability,'insomnia');
  const faint=x.cases.find(c=>c.id==='G4-trace-faint-seek');
  assert.equal(faint.steps[0].after.sides[0].pokemon[0].abilityState.seek,true);
  assert.ok(x.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
