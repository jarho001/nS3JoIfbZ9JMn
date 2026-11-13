'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),zlib=require('node:zlib'),{execFileSync}=require('node:child_process');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native Magic Guard preserves damage sources, direct exceptions and advancing counters',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'g4-magic-guard-'));
 try{
  const out=path.join(dir,'native.json');execFileSync('node',['reference/shared-engine-cases.cjs','--batch','magicguard','--output',out]);
  const x=JSON.parse(fs.readFileSync(out));assert.deepEqual(normalizeStacks(x),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/magic-guard.json.gz')))));
  assert.equal(x.cases.length,26);assert.equal(x.cases.reduce((n,c)=>n+c.steps.length,0),42);
  const c=id=>x.cases.find(c=>c.id==='G4-magic-guard-'+id),events=id=>c(id).steps.flatMap(s=>s.events.p1.map(e=>e.message));
  for(const id of ['brn','psn','tox','toxic-counter','roughskin','ironbarbs','helmet','ooze','life-orb','recoil','substitute-recoil','partial-trap','hazards','acquired','trace']) {
   assert.ok(!events(id).some(e=>e.startsWith('|-damage|p1')&&!['substitute-recoil','partial-trap'].includes(id)),id);
   assert.ok(!events(id).some(e=>/ability: (Rough Skin|Iron Barbs|Liquid Ooze)/.test(e)),id+' silent prevention');
  }
  assert.ok(events('struggle').some(e=>e.includes('0 fnt|[from] recoil')));
  assert.ok(events('confusion').some(e=>e.includes('[from] confusion')));
  assert.ok(events('substitute-cost').some(e=>e.startsWith('|-damage|p1')));
  assert.ok(events('self-destruct').some(e=>e.startsWith('|faint|p1')));
  for(const id of ['suppressed','lost','mummy'])assert.ok(events(id).some(e=>e.includes('[from] psn')||e.includes('[from] brn')),id);
  assert.ok(x.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
