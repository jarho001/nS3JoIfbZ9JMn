'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),zlib=require('node:zlib'),{execFileSync}=require('node:child_process');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native Shed Skin and Poison Heal preserve status draws, counters and damage replacement',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'g4-residual-recovery-'));
 try{
  const out=path.join(dir,'native.json');execFileSync('node',['reference/shared-engine-cases.cjs','--batch','residualrecovery','--output',out]);
  const x=JSON.parse(fs.readFileSync(out));assert.deepEqual(normalizeStacks(x),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/residual-recovery.json.gz')))));
  assert.equal(x.cases.length,35);assert.equal(x.cases.reduce((n,c)=>n+c.steps.length,0),78);
  const c=id=>x.cases.find(c=>c.id==='G4-residual-recovery-'+id),mon=(id,i=0)=>c(id).steps[i].after.sides[1].pokemon[0];
  assert.equal(mon('poison-tox').hp,101);assert.equal(mon('poison-tox').toxicStage,1);
  assert.equal(mon('poison-tox',1).toxicStage,2);assert.equal(mon('poison-cap').hp,175);
  assert.equal(mon('poison-full',1).hp,175);assert.equal(mon('poison-stage-cap',2).toxicStage,15);
  assert.equal(mon('poison-gas').hp,70);assert.equal(mon('poison-brn').hp,59);
  const shed=x.cases.filter(c=>c.id.includes('-shed-')),steps=shed.flatMap(c=>c.steps),messages=steps.flatMap(s=>s.events.p1.map(e=>e.message));
  assert.ok(messages.some(s=>s.includes('ability: Shed Skin')));assert.ok(messages.some(s=>s.startsWith('|-damage|')));
  const draws=steps.flatMap(s=>s.draws).filter(d=>d.effectId==='shedskin');assert.ok(draws.length);assert.ok(draws.some(d=>d.value<33));assert.ok(draws.some(d=>d.value>=33));assert.ok(draws.every(d=>d.bound===100&&d.to===undefined&&d.semanticContext==='residual-handlers'));
  assert.ok(x.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
  assert.deepEqual(require('../reference/residual-recovery-cases.cjs').membershipWitnesses().map(w=>w.set.species),['Arbok','Breloom']);
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
