'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const {assertReady}=require('../reference/neutral-turn-oracle.cjs');
const {cases,runCase}=require('../reference/shared-engine-cases.cjs');

test('native hit family fixes counts, interruptions, healing, recoil and Gen5 faint order',()=>{
 assertReady();const ps=require('../reference/vendor/pokemon-showdown/dist/sim');
 const selected=cases.filter(c=>c.hitFamily);assert.equal(selected.length,97);
 const records=selected.map(c=>runCase(ps,c));
 assert.equal(records.reduce((n,c)=>n+c.steps.length,0),253);
 assert.equal(records.filter(c=>c.setup.fullBattle).length,4);
 for(const c of records){
  assert.ok(c.steps.every(s=>s.accepted.every(a=>a.accepted)),c.id);
  if(c.setup.fullBattle)assert.ok(c.steps.at(-1).after.terminal,c.id);
 }
 const find=id=>records.find(c=>c.id===id);
 const messages=id=>find(id).steps[0].events.p1.map(e=>e.message);
 for(const count of [2,3,4,5]){
  const c=find(`G4-phase-count-${count}`),draw=c.steps[0].draws.find(d=>d.semanticContext==='multihit-count');
  assert.equal(draw.bound,20);assert.ok(messages(c.id).includes(`|-hitcount|p2a: Mew|${count}`));
 }
 assert.ok(messages('G4-phase-skilllink').includes('|-hitcount|p2a: Mew|5'));
 assert.ok(!find('G4-phase-skilllink').steps[0].draws.some(d=>d.semanticContext==='multihit-count'));
 assert.equal(messages('G4-phase-multihit-lifeorb').filter(s=>s.includes('item: Life Orb')).length,1);
 assert.ok(messages('G4-hit-contact-stop-33').includes('|-hitcount|p2a: Mew|1'));
 assert.equal(find('G4-hit-recoil-double-ko').steps[0].after.winner,1);
 for(const id of ['G4-hit-struggle-contact-double-ko','G4-hit-contact-double-ko','G4-hit-contact-terminal-multihit-1','G4-hit-explosion-double-ko'])assert.equal(find(id).steps[0].after.winner,2,id);
 assert.ok(messages('G4-hit-contact-terminal-multihit-1').at(-1).startsWith('|-hitcount|p2: Mew|1'));
 assert.ok(!messages('G4-hit-rockhead-recoil').some(s=>s.includes('[from] Recoil')));
 assert.ok(messages('G4-hit-rockhead-struggle').some(s=>s.includes('[from] recoil')));
 assert.ok(messages('G4-hit-LiquidOoze-GigaDrain').some(s=>s.includes('[from] ability: Liquid Ooze')));
 assert.equal(find('G4-phase-drain-target-cap').steps[0].after.sides[0].pokemon[0].hp,101);
 for(const id of ['G4-hit-explosion-immune','G4-hit-explosion-protect'])assert.ok(!find(id).steps[0].draws.some(d=>d.semanticContext==='accuracy'));
 for(const move of ['FlareBlitz','Scald','FlameWheel','SacredFire']){
  const c=find(`G4-hit-defrost-${move}`);assert.ok(!c.steps[0].draws.some(d=>d.semanticContext==='freeze'));
  assert.ok(messages(c.id).some(s=>s.startsWith('|-curestatus|p1a: Mew|frz|[from] move:')));
 }
 assert.equal(find('G4-hit-scald-target-frozen').steps[0].after.sides[1].pokemon[0].status,'frz');
 const blocked=find('G4-hit-scald-frozen-confusion-blocked');assert.equal(blocked.steps[1].after.sides[0].pokemon[0].status,'frz');
 assert.ok(blocked.steps[1].events.p1.some(e=>e.message.includes('[from] confusion')));
 assert.ok(!blocked.steps[1].events.p1.some(e=>e.message.startsWith('|-curestatus|')));
 assert.ok(messages('G4-hit-defrost-miss').some(s=>s.endsWith('|[miss]')));
 for(const id of ['G4-hit-frozen-target-fire-Synchronize','G4-hit-frozen-target-fire-contact'])assert.ok(!messages(id).some(s=>s.startsWith('|-status|')));
 const contact=messages('G4-hit-frozen-target-fire-contact');assert.ok(contact.findIndex(s=>s.startsWith('|-curestatus|'))>contact.findIndex(s=>s.includes('item: Rocky Helmet')));
 assert.ok(find('G4-hit-frozen-target-fire-contact').steps[0].draws.some(d=>d.semanticContext==='secondary'&&d.value===3));
 const terminal=messages('G4-hit-frozen-target-fire-terminal');assert.ok(terminal.findIndex(s=>s.startsWith('|-curestatus|'))>terminal.findIndex(s=>s.startsWith('|win|')));
 assert.equal(find('G4-hit-spin-contact-1').steps[1].after.sides[0].hazards[1],1);
 assert.equal(find('G4-hit-spin-contact-100').steps[1].after.sides[0].hazards[1],0);
 const spin=find('G4-hit-spin-contact-100').steps[1].events.p1.map(e=>e.message);assert.ok(spin.findIndex(s=>s.startsWith('|-sideend|'))>spin.findIndex(s=>s.includes('ability: Rough Skin')));
 for(const c of records.filter(c=>c.id.startsWith('G4-hit-protect-')))assert.ok(!c.steps[0].draws.some(d=>d.semanticContext==='multihit-count'));
});
