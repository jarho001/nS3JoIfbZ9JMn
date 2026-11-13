'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),zlib=require('node:zlib'),{execFileSync}=require('node:child_process');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native Mummy retains contact, End and ordered Helmet/thaw boundaries',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'g4-mummy-'));
 try{
  const out=path.join(dir,'native.json');execFileSync('node',['reference/shared-engine-cases.cjs','--batch','mummy','--output',out]);
  const x=JSON.parse(fs.readFileSync(out));assert.deepEqual(normalizeStacks(x),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/mummy.json.gz')))));
  assert.equal(x.cases.length,22);assert.equal(x.cases.reduce((n,c)=>n+c.steps.length,0),27);
  const events=id=>x.cases.find(c=>c.id==='G4-mummy-'+id).steps[0].events.p1.map(e=>e.message);
  const ordered=events('frozen-helmet');
  assert.ok(ordered.findIndex(e=>e.includes('Rocky Helmet'))<ordered.findIndex(e=>e.includes('-curestatus')));
  assert.ok(ordered.findIndex(e=>e.includes('-curestatus'))<ordered.findIndex(e=>e.includes('ability: Mummy')));
  assert.ok(!events('helmet-ko').some(e=>e.includes('ability: Mummy')));
  assert.ok(events('target-ko').some(e=>e.includes('ability: Mummy')));
  assert.ok(events('illusion').findIndex(e=>e.includes('|Illusion'))<events('illusion').findIndex(e=>e.includes('ability: Mummy')));
  for(const id of ['non-contact','already-mummy','protected','substitute','suppressed'])assert.ok(!events(id).some(e=>e.includes('ability: Mummy')));
  assert.ok(x.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
