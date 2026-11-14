'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),zlib=require('node:zlib'),{execFileSync}=require('node:child_process');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native low-HP offensive stat hooks retain thresholds, rounding and lifecycle',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'g4-low-hp-'));
 try{
  const out=path.join(dir,'native.json');execFileSync('node',['reference/shared-engine-cases.cjs','--batch','lowhp','--output',out]);
  const x=JSON.parse(fs.readFileSync(out));assert.deepEqual(normalizeStacks(x),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/low-hp.json.gz')))));
  assert.equal(x.cases.length,40);assert.equal(x.cases.reduce((n,c)=>n+c.steps.length,0),48);
  const c=id=>x.cases.find(c=>c.id==='G4-low-hp-'+id),damage=(id,step=0)=>c(id).steps[step].after.sides[1].pokemon[0].phaseState.attackedBy.at(-1).damage;
  for(const ability of ['overgrow','blaze','torrent','swarm'])assert.ok(damage(ability+'-special')>damage(ability+'-above'),ability);
  const exact=c('exact-third').initial.sides[0].pokemon[0];assert.equal(exact.hp*3,exact.maxhp);
  assert.equal(damage('exact-third'),damage('overgrow-special'));
  assert.ok(damage('drain-crossing')>damage('drain-crossing',1));
  assert.equal(damage('odd-stat-rounding'),44);
  assert.ok(c('critical').steps[0].events.p1.some(e=>e.message.startsWith('|-crit|')));
  for(const id of ['tied-special','tied-above','tied-wrong-type'])assert.equal(c(id).steps[0].draws.filter(d=>d.semanticContext==='damage-calculation-sort').length,1,id+' collects before callback guards');
  assert.ok(x.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
  const witnesses=require('../reference/low-hp-cases.cjs').membershipWitnesses();
  assert.deepEqual(witnesses.map(w=>[w.ability,w.seed,w.set.species]),[['overgrow',[1,2,3,4],'Sceptile'],['blaze',[7,2,3,4],'Emboar'],['torrent',[8,2,3,4],'Feraligatr'],['swarm',[31,2,3,4],'Scolipede']]);
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});

test('new ability families preserve the existing generic corpus selection',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'g4-low-hp-isolation-'));
 try{
  const out=path.join(dir,'generic.json');execFileSync('node',['reference/shared-engine-cases.cjs','--batch','generic','--output',out]);
  const actual=JSON.parse(fs.readFileSync(out)),expected=JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/generic.json.gz')));
  assert.deepEqual(actual.cases.map(c=>c.id),expected.cases.map(c=>c.id));
  assert.equal(actual.cases.length,24);assert.equal(actual.cases.reduce((n,c)=>n+c.steps.length,0),187);
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
