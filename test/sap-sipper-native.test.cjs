'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),zlib=require('node:zlib'),{execFileSync}=require('node:child_process');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native Gen5 Sap Sipper retains grass status/damage interception and caps',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'g4-sap-sipper-'));
 try{
  const out=path.join(dir,'native.json');execFileSync('node',['reference/shared-engine-cases.cjs','--batch','sapsipper','--output',out]);
  const x=JSON.parse(fs.readFileSync(out));assert.deepEqual(normalizeStacks(x),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/sap-sipper.json.gz')))));
  assert.equal(x.cases.length,17);assert.equal(x.cases.reduce((n,c)=>n+c.steps.length,0),29);
  const c=id=>x.cases.find(c=>c.id==='G4-sap-sipper-'+id),events=(id,i=0)=>c(id).steps[i].events.p1.map(e=>e.message);
  assert.ok(events('spore').some(s=>s.includes('|atk|1')));
  assert.ok(events('cap',6).some(s=>s.includes('|-immune|')));
  assert.ok(!events('cap',6).some(s=>s.includes('|-boost|')));
  assert.equal(c('drain-no-heal').steps[0].after.sides[0].pokemon[0].hp,50);
  assert.ok(!require('../reference/vendor/pokemon-showdown/dist/sim').Dex.mod('gen5').abilities.get('sapsipper').onAllyTryHitSide);
  assert.ok(x.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
  assert.equal(require('../reference/sap-sipper-cases.cjs').membershipWitnesses()[0].set.species,'Girafarig');
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
