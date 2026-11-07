'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const {assertReady}=require('../reference/neutral-turn-oracle.cjs');
const {cases,runCase}=require('../reference/shared-engine-cases.cjs');
const ps=require('../reference/vendor/pokemon-showdown/dist/sim');
const {BattleStream}=require('../reference/vendor/pokemon-showdown/dist/sim/battle-stream');
const {projectPackets}=require('../environment/protocol.cjs');

test('native BattleStream replacement/undo and direct snapshots share recipient boundaries',()=>{
 assertReady();
 for(const id of ['commit-replace','commit-cancel','pivot-resume','forced-replacement','G4-place-Spikes','G4-order-false','G4-order-true','G4-spin-ghost','G4-double-entry']) {
  const scenario=cases.find(c=>c.id===id);const expected=runCase(ps,scenario);
  const packets=[];const stream=new BattleStream({keepAlive:true,noCatch:true});
  stream.pushMessage=(type,data)=>packets.push(type+'\n'+data);
  try {
   stream._write('>start '+JSON.stringify({formatid:'gen5randombattle',seed:[scenario.seed,2,3,4],p1:{name:'p1',team:scenario.p1},p2:{name:'p2',team:scenario.p2}}));
   for(const side of ['p1','p2']) {
    const view=projectPackets(packets,side).records;
    assert.deepEqual(view.filter(x=>x.type==='event'),expected.initialEvents[side]);
    assert.equal(view.filter(x=>x.type==='request').length,1);
   }
   for(const [n,choices] of scenario.steps.entries()) {
    packets.length=0;
    stream._write(Object.entries(choices).map(([side,choice])=>`>${side} ${choice}`).join('\n'));
    for(const [sideIndex,side] of ['p1','p2'].entries()) {
     const view=projectPackets(packets,side).records;
     assert.deepEqual(view.filter(x=>x.type==='event'),expected.steps[n].events[side]);
     const requests=view.filter(x=>x.type==='request');
     if(expected.steps[n].events[side].length)assert.deepEqual(requests.map(x=>x.request),[expected.steps[n].after.sides[sideIndex].request]);
     else assert.deepEqual(view,[],'pending/undo must not invent acknowledgements or public information');
    }
   }
  } finally {stream.battle?.destroy();}
 }
});

test('native paired histories hide reserves, items and pending opposing choice',()=>{
 const a=runCase(ps,cases.find(c=>c.id==='privacy-a'));
 const b=runCase(ps,cases.find(c=>c.id==='privacy-b'));
 assert.deepEqual(a.initialEvents.p1,b.initialEvents.p1);
 assert.deepEqual(a.initial.sides[0].request,b.initial.sides[0].request);
 assert.deepEqual(a.steps[0].events.p1,b.steps[0].events.p1);
 assert.deepEqual(a.steps[0].after.sides[0].request,b.steps[0].after.sides[0].request);
 assert.notDeepEqual(a.steps[0].after.sides[1].pending,b.steps[0].after.sides[1].pending);
});
