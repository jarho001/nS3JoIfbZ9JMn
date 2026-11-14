'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),zlib=require('node:zlib'),{execFileSync}=require('node:child_process');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native Gen5 switch recovery retains cure, silent healing and Pursuit order',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'g4-switch-recovery-'));
 try{
  const out=path.join(dir,'native.json');execFileSync('node',['reference/shared-engine-cases.cjs','--batch','switchrecovery','--output',out]);
  const x=JSON.parse(fs.readFileSync(out));assert.deepEqual(normalizeStacks(x),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/switch-recovery.json.gz')))));
  assert.equal(x.cases.length,22);assert.equal(x.cases.reduce((n,c)=>n+c.steps.length,0),30);
  const c=id=>x.cases.find(c=>c.id==='G4-switch-recovery-'+id),events=(id,i=0)=>c(id).steps[i].events.p1.map(e=>e.message);
  assert.equal(c('cure-tox').steps[0].after.sides[0].pokemon[0].status,'');
  assert.ok(events('cure-tox').some(s=>s.includes('ability: Natural Cure|[silent]')));
  assert.equal(c('regen-60').steps[0].after.sides[0].pokemon[0].hp,118);
  assert.equal(c('regen-pursuit').steps[0].after.sides[0].pokemon[0].hp,146);
  assert.equal(c('regen-pursuit-ko').steps[0].after.sides[0].pokemon[0].hp,0);
  assert.ok(!events('regen-60').some(s=>s.startsWith('|-heal')));
  assert.ok(x.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
  assert.deepEqual(require('../reference/switch-recovery-cases.cjs').membershipWitnesses().map(w=>w.set.species),['Slowking','Celebi']);
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
