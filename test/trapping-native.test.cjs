'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),zlib=require('node:zlib'),{execFileSync}=require('node:child_process');
test('native trapping corpus fixes disclosure, cantUndo, escapes and seat mirrors',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'g4-trapping-'));
 try{
  const out=path.join(dir,'native.json');execFileSync('node',['reference/shared-engine-cases.cjs','--batch','trapping','--output',out]);
  const actual=JSON.parse(fs.readFileSync(out));assert.deepEqual(actual,JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/trapping.json.gz'))));
  assert.equal(actual.cases.length,31);assert.equal(actual.cases.reduce((n,c)=>n+c.steps.length,0),215);
  const find=id=>actual.cases.find(c=>c.id==='G4-trap-'+id);
  const hidden=find('shadow-opponent-pending'),reject=hidden.steps[1];
  assert.equal(reject.accepted[0].accepted,false);assert.deepEqual(reject.before.rng,reject.after.rng);assert.deepEqual(reject.before.sides[1].pending,reject.after.sides[1].pending);
  assert.equal(hidden.initial.sides[0].request.active[0].maybeTrapped,true);assert.equal(reject.after.sides[0].request.active[0].trapped,true);
  const query=find('query-frisk-undo');assert.equal(query.steps[0].after.sides[0].cantUndo,true);assert.equal(query.steps[1].accepted[0].accepted,false);
  assert.equal(find('ghost-shadow').initial.sides[0].pokemon[0].phaseState.trapped,'hidden');
  for(const c of actual.cases.filter(c=>c.setup.fullBattle))assert.equal(c.steps.at(-1).after.terminal,true);
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
