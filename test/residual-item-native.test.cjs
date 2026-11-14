'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),zlib=require('node:zlib'),{execFileSync}=require('node:child_process');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native Black Sludge and status Orbs preserve residual order, status sources and interactions',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'g4-residual-items-'));
 try{
  const out=path.join(dir,'native.json');execFileSync('node',['reference/shared-engine-cases.cjs','--batch','residualitems','--output',out]);
  const x=JSON.parse(fs.readFileSync(out));assert.deepEqual(normalizeStacks(x),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/residual-items.json.gz')))));
  assert.equal(x.cases.length,59);assert.equal(x.cases.reduce((n,c)=>n+c.steps.length,0),177);
  const c=id=>x.cases.find(c=>c.id==='G4-residual-item-'+id),mon=(id,i=0)=>c(id).steps[i].after.sides[1].pokemon[0];
  assert.equal(mon('sludge-ordinary').hp,154);assert.equal(mon('sludge-poison').hp,88);assert.equal(mon('sludge-cap').hp,135);
  assert.equal(mon('sludge-magicguard').hp,80);assert.equal(mon('sludge-faint').hp,0);assert.equal(mon('sludge-minimum').hp,0);
  assert.equal(mon('toxic-ordinary').toxicStage,0);assert.equal(mon('toxic-ordinary',1).toxicStage,1);assert.equal(mon('toxic-ordinary',1).hp,165);
  assert.equal(mon('toxic-poisonheal',1).hp,101);assert.equal(mon('flame-ordinary',1).hp,154);
  for(const id of ['toxic-poison','toxic-steel','toxic-immunity','flame-fire','flame-waterveil'])assert.equal(mon(id).status,'',id);
  assert.equal(mon('toxic-gas').status,'tox');assert.equal(mon('flame-gas').status,'brn');
  assert.ok(c('toxic-ordinary').steps[0].events.p1.some(e=>e.message==='|-status|p2a: Mew|tox|[from] item: Toxic Orb'));
  assert.ok(c('flame-ordinary').steps[0].events.p1.some(e=>e.message==='|-status|p2a: Mew|brn|[from] item: Flame Orb'));
  assert.ok(x.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
  assert.deepEqual(require('../reference/residual-item-cases.cjs').membershipWitnesses().map(w=>w.set.species),['Arbok','Raticate','Sigilyph']);
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
