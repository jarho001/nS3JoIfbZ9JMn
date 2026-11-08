'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const {assertReady,withObservedBattleDraws}=require('../reference/neutral-turn-oracle.cjs');
const {cases,runCase}=require('../reference/shared-engine-cases.cjs');

test('status duration family captures native counters, sources, clauses and raw draws',()=>{
  assertReady();
  const ps=require('../reference/vendor/pokemon-showdown/dist/sim');
  const pack=cases.filter(c=>c.statusFamily).map(c=>runCase(ps,c));
  assert.equal(pack.length,48);
  const find=id=>pack.find(c=>c.id===id);
  for(const c of pack)for(const step of c.steps) {
    assert.ok(step.accepted.every(a=>a.accepted),c.id);
    assert.ok(Number.isInteger(step.after.effectOrder));
    for(const side of step.after.sides)for(const p of side.pokemon) {
      assert.ok(p.statusState);assert.ok(p.volatiles);assert.ok(p.abilityState);assert.ok(p.itemState);
    }
  }
  const sleep=find('G4-status-sleep-switch-reset');
  const first=sleep.steps[0].after.sides[1].pokemon[0].statusState;
  const returned=sleep.steps[3].after.sides[1].pokemon[0].statusState;
  assert.equal(first.time,first.startTime-1);
  assert.equal(returned.time,first.startTime);
  assert.deepEqual(first.source,{$pokemon:{side:0,mon:0}});
  const clause=find('G4-status-sleep-clause');
  assert.equal(clause.steps[1].after.sides[1].pokemon[1].status,'');
  assert.ok(clause.steps[1].events.p1.some(e=>e.message==='|-message|Sleep Clause Mod activated.'));
  const rest=find('G4-status-rest-exempt-clause').steps.at(-1).after.sides[1].pokemon;
  assert.ok(rest.every(p=>p.status==='slp'));
  assert.equal(rest[0].statusState.source.$pokemon.side,1);
  assert.equal(rest[1].statusState.source.$pokemon.side,0);
  // Rest overwrites the native statusState timers after Lum has already cured
  // it. Empty status does not justify normalizing these retained fields away.
  const lum=find('G4-status-rest-lum').steps[0].after.sides[0].pokemon[0];
  assert.equal(lum.status,'');assert.equal(lum.item,'');
  assert.equal(lum.statusState.time,3);assert.equal(lum.statusState.startTime,3);
  const rows=pack.flatMap(c=>c.steps.flatMap(s=>s.draws));
  for(const kind of ['sleep-duration','freeze','confusion-duration','confusion','confusion-damage','secondary']) {
    assert.ok(rows.some(d=>d.semanticContext===kind),kind);
  }
  assert.ok(rows.filter(d=>d.semanticContext==='sleep-duration').every(d=>d.bound===2&&d.to===5));
  assert.ok(rows.filter(d=>d.semanticContext==='confusion-duration').every(d=>d.bound===2&&d.to===6));
  assert.ok(rows.filter(d=>d.semanticContext==='freeze').every(d=>d.bound===5&&d.effectId==='frz'));
  assert.ok(rows.filter(d=>d.semanticContext==='secondary').every(d=>d.bound===100));
  const ko=find('G4-status-secondary-ko').steps[0];
  assert.equal(ko.after.sides[1].pokemon[0].hp,0);
  assert.ok(ko.draws.some(d=>d.semanticContext==='secondary'));
  assert.ok(!find('G4-status-secondary-protected').steps[0].draws.some(d=>d.semanticContext==='secondary'));
});

test('optional callback annotations leave original PRNG results and helper API intact',()=>{
  assertReady();
  const ps=require('../reference/vendor/pokemon-showdown/dist/sim');
  const scenario=cases.find(c=>c.id==='G4-status-sleep-0-false');
  function observed(includeEffects) {
    const b=new ps.Battle({formatid:'gen5randombattle',seed:[0,2,3,4],
      p1:{name:'p1',team:scenario.p1},p2:{name:'p2',team:scenario.p2}});
    try {
      const result=withObservedBattleDraws(ps,b,()=>b.makeChoices('move 1','move 1'),{includeEffects});
      return {rng:b.prng.getSeed(),draws:result.draws};
    }finally{b.destroy();}
  }
  const original=observed(false),annotated=observed(true);
  assert.equal(original.rng,annotated.rng);
  assert.ok(original.draws.every(d=>!Object.hasOwn(d,'effectId')));
  assert.ok(annotated.draws.some(d=>d.effectId==='slp'));
  const numeric=pack=>pack.draws.map(d=>({bound:d.bound,to:d.to,value:d.value,before:d.before,after:d.after}));
  assert.deepEqual(numeric(original),numeric(annotated));
});
