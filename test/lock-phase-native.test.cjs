'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const {assertReady}=require('../reference/neutral-turn-oracle.cjs');
const {cases,runCase}=require('../reference/shared-engine-cases.cjs');
test('native phase corpus fixes forced requests, causal fields and complete games',()=>{
 assertReady();const ps=require('../reference/vendor/pokemon-showdown/dist/sim');
 const records=cases.filter(c=>c.lockFamily).map(c=>runCase(ps,c));
 assert.equal(records.length,73);assert.equal(records.reduce((n,c)=>n+c.steps.length,0),371);
 for(const c of records){
  assert.ok(c.initial.sides[0].pokemon[0].phaseState);
  for(const step of c.steps)for(const side of step.after.sides){
   if(side.request?.active)assert.equal(side.request.update,undefined);
  }
 }
 const games=records.filter(c=>c.setup.fullBattle);assert.equal(games.length,2);
 for(const c of games){assert.equal(c.steps.at(-1).after.terminal,true);assert.equal(c.setup.p1.length,6);}
 const fatigue=records.find(c=>c.id==='G4-lock-choice-fatigue');
 const packets=fatigue.steps[2].privatePackets.join('\n');
 assert.ok(packets.includes('"update":true'));assert.ok(packets.includes('"disabledSource":""'));
 assert.ok(packets.includes('[Unavailable choice]'));
 assert.equal(fatigue.steps[1].after.sides[0].request.update,undefined);
 assert.equal(fatigue.steps[1].after.sides[0].request.active[0].moves[0].disabledSource,undefined);
 const rejected=records.flatMap(c=>c.steps).filter(s=>s.accepted.some(a=>!a.accepted));
 assert.ok(rejected.length);for(const s of rejected)assert.equal(s.accepted.length,1);
});
