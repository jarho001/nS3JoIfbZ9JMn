'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),zlib=require('node:zlib'),{execFileSync}=require('node:child_process');
test('native exchange and restoration retain initial item, consumption lineage and Choice provenance',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'g4-items-'));
 try{
  const out=path.join(dir,'native.json');execFileSync('node',['reference/shared-engine-cases.cjs','--batch','items','--output',out]);
  const x=JSON.parse(fs.readFileSync(out));assert.deepEqual(x,JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/items.json.gz'))));assert.equal(x.cases.length,28);assert.equal(x.cases.reduce((n,c)=>n+c.steps.length,0),44);
  const find=id=>x.cases.find(c=>c.id==='G4-items-'+id),first=id=>find(id).steps[0].after.sides;
  assert.equal(first('Trick-both')[0].pokemon[0].phaseState.initialItem,'lifeorb');assert.equal(first('Trick-both')[0].pokemon[0].item,'leftovers');
  assert.equal(first('Trick-both')[0].pokemon[0].phaseState.pendingStaleness,'internal');assert.equal(first('knock-plain')[1].pokemon[0].phaseState.lastItem,'');
  assert.equal(find('herb-recycle').steps[1].after.sides[0].pokemon[0].item,'powerherb');
  for(const c of x.cases)assert.ok(c.steps.every(s=>s.accepted.every(a=>a.accepted)),c.id);
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
