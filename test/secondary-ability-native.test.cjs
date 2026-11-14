'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),zlib=require('node:zlib'),{execFileSync}=require('node:child_process');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native Gen5 secondary abilities retain target filters and modified chance',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'g4-secondary-ability-'));
 try{
  const out=path.join(dir,'native.json');execFileSync('node',['reference/shared-engine-cases.cjs','--batch','secondaryability','--output',out]);
  const x=JSON.parse(fs.readFileSync(out));assert.deepEqual(normalizeStacks(x),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/secondary-ability.json.gz')))));
  assert.equal(x.cases.length,29);assert.equal(x.cases.reduce((n,c)=>n+c.steps.length,0),33);
  const c=id=>x.cases.find(c=>c.id==='G4-secondary-ability-'+id),events=(id,i=0)=>c(id).steps[i].events.p1.map(e=>e.message);
  assert.ok(!c('dust-flamethrower-1').steps[0].draws.some(d=>d.semanticContext==='secondary'));
  assert.ok(events('dust-primary').some(s=>s.includes('|tox')));
  assert.ok(events('dust-thaw').some(s=>s.includes('|-curestatus|')));
  const mummy=events('grace-mummy');assert.ok(mummy.findIndex(s=>s.includes('|-status|'))<mummy.findIndex(s=>s.includes('ability: Mummy')));
  assert.ok(x.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
  assert.deepEqual(require('../reference/secondary-ability-cases.cjs').membershipWitnesses().map(w=>w.set.species),['Dunsparce','Dustox']);
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
