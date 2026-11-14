'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),zlib=require('node:zlib'),{execFileSync}=require('node:child_process');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native type absorption retains healing, interception and lifecycle boundaries',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'g4-absorption-'));
 try{
  const out=path.join(dir,'native.json');execFileSync('node',['reference/shared-engine-cases.cjs','--batch','absorption','--output',out]);
  const x=JSON.parse(fs.readFileSync(out));assert.deepEqual(normalizeStacks(x),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/absorption.json.gz')))));
  assert.equal(x.cases.length,54);assert.equal(x.cases.reduce((n,c)=>n+c.steps.length,0),71);
  const c=id=>x.cases.find(c=>c.id==='G4-absorption-'+id),events=(id,i=0)=>c(id).steps[i].events.p1.map(e=>e.message),hp=(id,i=0)=>c(id).steps[i].after.sides[1].pokemon[0].hp;
  for(const [key,ability]of [['water','Water Absorb'],['electric','Volt Absorb']]){
   assert.equal(hp(key+'-partial'),103);assert.equal(hp(key+'-rounding'),104);assert.equal(hp(key+'-cap'),175);
   assert.ok(events(key+'-full').some(s=>s.includes('-immune')&&s.includes('ability: '+ability)));
   assert.ok(events(key+'-partial').some(s=>s.includes('-heal')&&s.endsWith('[from] ability: '+ability+'|[of] p1a: Mew')));
   assert.equal(c(key+'-life-orb').steps[0].after.sides[0].pokemon[0].hp,1);
   assert.ok(!events(key+'-protect').some(s=>s.includes('ability: '+ability)));
   assert.equal(c(key+'-partial').steps[0].after.sides[0].pokemon[0].phaseState.moveLastTurnResult,false);
   assert.equal(c(key+'-inaccurate').steps[0].draws.filter(d=>d.semanticContext==='accuracy').length,1,'only opposing Growl draws accuracy');
   assert.ok(c(key+'-substitute').steps[1].after.sides[1].pokemon[0].volatiles.substitute);
   assert.deepEqual(c(key+'-tied-60').steps[0].draws,c(key+'-tied-175').steps[0].draws,'cancelled full/partial HP preserve handler ordering');
  }
  assert.equal(hp('water-transform-base-hp',1),90,'Transform retains Ditto base maximum HP for quarter heal');
  assert.ok(events('water-fly').some(s=>s.includes('|-miss|')));
  assert.ok(events('water-dive').some(s=>s.includes('ability: Water Absorb')));
  for(const id of ['water-sleep-talk','water-sleep-talk-tied'])assert.ok(events(id,1).some(s=>s.includes('[from] move: Sleep Talk')));
  assert.ok(events('electric-reflected').some(s=>s.includes('ability: Volt Absorb|[of] p2a: Mew')));
  assert.ok(!events('electric-ground-protect').some(s=>s.includes('ability: Volt Absorb')||s.includes('-activate')));
  assert.ok(x.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
  assert.deepEqual(require('../reference/absorption-cases.cjs').membershipWitnesses().map(w=>[w.ability,w.seed,w.set.species]),[['waterabsorb',[1,2,3,4],'Maractus'],['voltabsorb',[2,2,3,4],'Thundurus-Therian']]);
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
