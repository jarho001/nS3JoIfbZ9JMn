'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),zlib=require('node:zlib'),{execFileSync}=require('node:child_process');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native rain recovery group preserves healing, cure ordering, expiry and membership',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'g4-rain-recovery-'));
 try{
  const out=path.join(dir,'native.json');execFileSync('node',['reference/shared-engine-cases.cjs','--batch','rainrecovery','--output',out]);
  const x=JSON.parse(fs.readFileSync(out));assert.deepEqual(normalizeStacks(x),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/rain-recovery.json.gz')))));
  assert.equal(x.cases.length,33);assert.equal(x.cases.reduce((n,c)=>n+c.steps.length,0),76);
  const c=id=>x.cases.find(c=>c.id==='G4-rain-recovery-'+id),mon=(id,i=0)=>c(id).steps[i].after.sides[1].pokemon[0];
  assert.equal(mon('dish-partial').hp,90);assert.equal(mon('dish-cap').hp,175);
  assert.equal(mon('hydration-tox').hp,80);assert.equal(mon('hydration-tox').status,'');
  assert.equal(mon('hydration-expiry-status',4).status,'par');
  assert.equal(mon('hydration-gas-before',1).status,'tox');
  const events=c('hydration-leftovers-toxic').steps[0].events.p1.map(e=>e.message);
  assert.ok(events.findIndex(s=>s.includes('ability: Hydration'))<events.findIndex(s=>s.includes('Leftovers')));
  assert.ok(!events.some(s=>s.startsWith('|-damage|')));
  assert.ok(x.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
  assert.deepEqual(require('../reference/rain-recovery-cases.cjs').membershipWitnesses().map(w=>w.set.species),['Pelipper','Swanna']);
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
