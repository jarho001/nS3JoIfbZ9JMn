'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),zlib=require('node:zlib'),{execFileSync}=require('node:child_process');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native Gen5 Flash Fire retains activation, modifiers and cleanup',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'g4-flash-fire-'));
 try{
  const out=path.join(dir,'native.json');execFileSync('node',['reference/shared-engine-cases.cjs','--batch','flashfire','--output',out]);
  const x=JSON.parse(fs.readFileSync(out));assert.deepEqual(normalizeStacks(x),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/flash-fire.json.gz')))));
  assert.equal(x.cases.length,29);assert.equal(x.cases.reduce((n,c)=>n+c.steps.length,0),43);
  const c=id=>x.cases.find(c=>c.id==='G4-flash-fire-'+id),events=(id,i=0)=>c(id).steps[i].events.p1.map(e=>e.message);
  const row=c('ordinary').steps[0].after.sides[1].pokemon[0].volatiles.flashfire;
  assert.deepEqual(row.sourceEffect,{$effect:'flashfire',type:'Ability'});
  assert.ok(events('repeat',1).some(s=>s.includes('|-immune|')));
  for(const id of ['gas-end','replace-end','switch-end','transform-end','mummy-end'])assert.ok(events(id,1).some(s=>s.includes('|-end|')&&s.includes('Flash Fire|[silent]')));
  assert.ok(!events('faint-end',1).some(s=>s.includes('|-end|')));
  assert.ok(c('pivot').steps[0].after.sides[0].pokemon[0].volatiles.flashfire);
  assert.ok(x.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
  assert.equal(require('../reference/flash-fire-cases.cjs').membershipWitnesses()[0].set.species,'Chandelure');
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
