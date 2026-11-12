'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),zlib=require('node:zlib');
const {execFileSync}=require('node:child_process');
const read=name=>JSON.parse(zlib.gunzipSync(fs.readFileSync(`test-logs/g4-cpu-conformance/${name}.json.gz`)));
test('design source frontier reproduces with typed query and mutation obligations',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'g4-frontier-'));
 try{
  const out=path.join(dir,'audit.json');execFileSync('node',['reference/gen5-closure-audit.cjs',out]);
  const actual=JSON.parse(fs.readFileSync(out));assert.deepEqual(actual,read('g4-design-frontier'));
  assert.equal(actual.designFrontier.sleepTalkEligibility.naturepower,false);
  assert.equal(actual.designFrontier.sleepTalkEligibility.taunt,true);
  assert.equal(actual.rootMoveIds.length,223);assert.equal(actual.initialItemReturnBound.length,45);
  assert.deepEqual([...new Set(actual.designFrontier.queryAbilities.map(r=>r.ability))].sort(),['arenatrap','magnetpull','shadowtag']);
  assert.ok(actual.designFrontier.mutationSites.some(r=>r.owner==='moves:sleeptalk'&&r.method==='useMove'));
  assert.equal(actual.designFrontier.status,'source_frontier_not_complete_closure_proof');
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
test('native witnesses preserve hidden-trap retry disclosure and Transform identity',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'g4-design-'));
 try{
  const out=path.join(dir,'native.json');execFileSync('node',['reference/g4-design-cases.cjs',out]);
  const actual=JSON.parse(fs.readFileSync(out));assert.deepEqual(actual,read('g4-design-native'));
  const [query,retry,ghost,transform]=actual.cases;
  assert.equal(query.initial.sides[0].request.active[0].maybeTrapped,true);
  assert.equal(query.initial.sides[1].pokemon[0].ability,'frisk');
  for(const c of [retry,ghost]){
   const step=c.steps[0];assert.equal(step.accepted[0].accepted,false);
   assert.deepEqual(step.after.rng,c.initial.rng);assert.deepEqual(step.draws,[]);
   assert.equal(c.initial.sides[0].request.active[0].maybeTrapped,true);
   assert.equal(step.after.sides[0].request.active[0].trapped,true);
   assert.notDeepEqual(step.after.sides[0].request,c.initial.sides[0].request);
   assert.equal(step.after.sides[0].pokemon[0].phaseState.trapped,'hidden');
   assert.ok(step.privatePackets.some(s=>s.includes('"update":true')));
  }
  const p=transform.steps[0].after.sides[0].pokemon[0];
  assert.equal(p.species,'Snorlax');assert.equal(p.maxhp,175);assert.equal(p.ability,'immunity');assert.equal(p.baseAbility,'synchronize');
  assert.equal(p.hpType,transform.initial.sides[0].pokemon[0].hpType);
  assert.ok(p.moves.every(m=>m.pp===5&&m.maxpp===5));
  const chain=actual.cases[4],step=chain.steps[0];
  const messages=step.events.p1.map(e=>e.message);
  assert.ok(messages.some(s=>s.includes('[from] move: Sleep Talk')));
  assert.ok(messages.some(s=>s.includes('[from] ability: Magic Bounce')));
  const slots=step.after.sides[0].pokemon[0].moves;
  assert.equal(slots[0].pp,15);assert.equal(slots[1].pp,32);
  assert.equal(step.draws.filter(d=>d.semanticContext==='sleep-talk-selection').length,1);
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
