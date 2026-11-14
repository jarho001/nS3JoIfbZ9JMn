'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),zlib=require('node:zlib'),{execFileSync}=require('node:child_process');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native Chesto and Sitrus preserve consumption, threshold timing and restoration',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'g4-restorative-berries-'));
 try{
  const out=path.join(dir,'native.json');execFileSync('node',['reference/shared-engine-cases.cjs','--batch','restorativeberries','--output',out]);
  const x=JSON.parse(fs.readFileSync(out));assert.deepEqual(normalizeStacks(x),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/restorative-berries.json.gz')))));
  assert.equal(x.cases.length,35);assert.equal(x.cases.reduce((n,c)=>n+c.steps.length,0),77);
  const c=id=>x.cases.find(c=>c.id==='G4-restorative-berry-'+id),mon=(id,i=0)=>c(id).steps[i].after.sides[1].pokemon[0];
  assert.equal(mon('chesto-rest').status,'');assert.equal(mon('chesto-rest').item,'');assert.equal(mon('chesto-rest').phaseState.ateBerry,true);
  assert.equal(mon('chesto-spore').status,'');assert.equal(mon('chesto-brn').status,'brn');assert.equal(mon('chesto-brn').item,'chestoberry');
  assert.equal(mon('sitrus-hp-87').hp,130);assert.equal(mon('sitrus-hp-87').item,'');assert.equal(mon('sitrus-hp-88').hp,88);assert.equal(mon('sitrus-hp-88').item,'sitrusberry');
  assert.equal(mon('sitrus-rounding').hp,132);assert.equal(mon('sitrus-faint').hp,0);assert.equal(mon('sitrus-faint').item,'sitrusberry');
  assert.equal(mon('sitrus-knockoff').item,'');assert.equal(mon('sitrus-knockoff').phaseState.ateBerry,false);
  const messages=c('chesto-rest').steps[0].events.p1.map(e=>e.message);assert.ok(messages.findIndex(s=>s.startsWith('|-heal|'))<messages.findIndex(s=>s.startsWith('|-enditem|')));
  assert.ok(x.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
  assert.deepEqual(require('../reference/restorative-berry-cases.cjs').membershipWitnesses().map(w=>w.set.species),['Regirock','Tropius']);
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
