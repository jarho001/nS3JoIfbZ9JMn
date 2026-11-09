'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const {assertReady}=require('../reference/neutral-turn-oracle.cjs');
const {cases,runCase}=require('../reference/shared-engine-cases.cjs');

test('native-only hook inputs preserve ability startup, PP, locks, speed and permanent weather',()=>{
 assertReady();const ps=require('../reference/vendor/pokemon-showdown/dist/sim');
 const selected=cases.filter(c=>c.hookFamily);assert.equal(selected.length,41);
 const records=selected.map(c=>runCase(ps,c));
 assert.equal(records.reduce((n,c)=>n+c.steps.length,0),144);
 for(const c of records)for(const step of c.steps){
  assert.ok(step.accepted.every(a=>a.accepted),c.id);
  assert.ok(step.after.field);assert.ok(step.after.sides.every(s=>s.sideConditions));
 }
 const find=id=>records.find(c=>c.id===`G4-hooks-${id}`);
 const pressure=find('ability-Pressure');
 assert.equal(pressure.steps[0].after.sides[1].pokemon[0].moves[0].pp,54);
 assert.ok(pressure.initialEvents.p1.some(e=>e.message==='|-ability|p1a: Mew|Pressure'));
 const intimidate=find('ability-Intimidate');
 assert.equal(intimidate.initial.sides[1].pokemon[0].boosts.atk,-1);
 assert.ok(intimidate.initialEvents.p1.some(e=>e.message==='|-ability|p1a: Mew|Intimidate|boost'));
 const scarf=find('item-ChoiceScarf');
 assert.equal(scarf.initial.sides[0].pokemon[0].speed,180);
 assert.ok(scarf.steps[0].after.sides[0].pokemon[0].volatiles.choicelock);
 assert.ok(find('item-LifeOrb').steps[1].events.p1.some(e=>e.message.includes('[from] item: Life Orb')));
 for(const name of ['Drizzle','Drought','SandStream','SnowWarning']){
  const c=find(`weather-${name}`);assert.equal(c.initial.field.weatherState.duration,0);
  assert.equal(c.steps.at(-1).after.field.weatherState.duration,0);
 }
 const prankster=find('ability-Prankster').steps[0].events.p1.filter(e=>e.message.startsWith('|move|'));
 assert.match(prankster[0].message,/p1a: Mew\|Swords Dance/);
});
