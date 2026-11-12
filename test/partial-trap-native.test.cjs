'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),zlib=require('node:zlib'),{execFileSync}=require('node:child_process');
test('pinned partial trapping fixes duration, cleanup, residual order and disclosed retries',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'g4-partial-trap-'));
 try{
  const out=path.join(dir,'native.json');execFileSync('node',['reference/shared-engine-cases.cjs','--batch','partialtrap','--output',out]);
  const actual=JSON.parse(fs.readFileSync(out));assert.deepEqual(actual,JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/partial-trap.json.gz'))));
  assert.equal(actual.cases.length,27);assert.equal(actual.cases.reduce((n,c)=>n+c.steps.length,0),105);
  const find=id=>actual.cases.find(c=>c.id==='G4-partial-trap-'+id),v=id=>find(id).steps[0].after.sides[1].pokemon[0].volatiles.partiallytrapped;
  assert.equal(v('expiry-ordinary').boundDivisor,16);assert.equal(v('expiry-Binding Band').boundDivisor,8);assert.equal(v('expiry-Grip Claw').duration,7);
  const durationDraws=find('tied-both').steps.flatMap(s=>s.draws.filter(d=>d.semanticContext==='volatile-duration').map(d=>d.value));assert.deepEqual(durationDraws.sort(),[5,6]);
  const req=find('known-trap-retry');assert.equal(req.steps[0].after.sides[1].request.active[0].trapped,true);assert.equal(req.steps[1].accepted[0].accepted,false);assert.deepEqual(req.steps[1].before.rng,req.steps[1].after.rng);
  assert.equal(find('substitute-blocks').steps[1].after.sides[1].pokemon[0].volatiles.partiallytrapped,undefined);
  for(const id of ['substitute-clears','source-switch','source-ko'])assert.ok(find(id).steps.some(s=>s.events.p1.some(e=>e.message.endsWith('|Bind|[partiallytrapped]|[silent]'))),id);
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
