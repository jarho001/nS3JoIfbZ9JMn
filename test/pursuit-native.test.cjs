'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),zlib=require('node:zlib'),{execFileSync}=require('node:child_process');
test('native Pursuit preserves switch/pivot interception and private causal flags',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'g4-pursuit-'));
 try{
  const out=path.join(dir,'native.json');execFileSync('node',['reference/shared-engine-cases.cjs','--batch','pursuit','--output',out]);
  const x=JSON.parse(fs.readFileSync(out));assert.deepEqual(x,JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/pursuit.json.gz'))));assert.equal(x.cases.length,23);assert.equal(x.cases.reduce((n,c)=>n+c.steps.length,0),41);
  const find=id=>x.cases.find(c=>c.id==='G4-pursuit-'+id);
  for(const id of ['switch','victim-ko','pivot-slower'])assert.ok(find(id).steps[0].events.p1.some(e=>e.message.endsWith('|[from] move: Pursuit')));
  assert.equal(find('victim-ko').steps[0].after.sides[0].pokemon[0].phaseState.skipPursuit,true);
  assert.equal(find('pivot-slower').steps[0].after.sides[0].pokemon[0].phaseState.pursuitActivated,true);
  assert.ok(find('phazing').steps[0].events.p1.some(e=>e.message.startsWith('|drag|')));
  for(const c of x.cases)assert.ok(c.steps.every(s=>s.accepted.every(a=>a.accepted)),c.id);
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
