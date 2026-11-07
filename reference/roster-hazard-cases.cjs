'use strict';
// Independent native inputs: no simulator imports or derived expectations.
module.exports = function rosterHazardCases(mon) {
 const m=(moves,o={})=>mon('Mew',moves,o), l=(moves,o={})=>mon('Snorlax',moves,o), g=(moves,o={})=>mon('Gengar',moves,o);
 const step=(p1,p2)=>Object.fromEntries(Object.entries({p1,p2}).filter(([,v])=>v));
 const both=(a,b)=>step(`move ${a}`,`move ${b}`);
 const fast={evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:252}};
 const cases=[];
 for(const move of ['Stealth Rock','Spikes'])cases.push({id:`G4-place-${move}`,seed:1,
  p1:[m([move,'Tackle'],fast)],p2:[m(['Protect','Recover']),l(['Tackle']),g(['Tackle']),m(['Tackle']),l(['Tackle']),g(['Tackle'])],
  steps:[both(1,1),both(1,2),both(1,2),both(1,2),step('move 2','switch 6'),step('move 2','switch 2'),step('move 2','switch 3')]});
 for(const layers of [1,2])cases.push({id:`G4-spikes-layer-${layers}`,seed:0,p1:[m(['Spikes','Recover'],fast)],p2:[m(['Recover']),m(['Tackle'])],steps:[...Array.from({length:layers},()=>both(1,1)),step('move 2','switch 2')]});
 for(const reverse of [false,true])cases.push({id:`G4-order-${reverse}`,seed:9,
  p1:[m(reverse?['Spikes','Stealth Rock','Tackle']:['Stealth Rock','Spikes','Tackle'],fast)],p2:[m(['Recover']),l(['Tackle'],{item:'Leftovers'})],
  steps:[both(1,1),both(2,1),step('move 3','switch 2')]});
 cases.push({id:'G4-entry-ko-chain',seed:21,p1:[m(['Stealth Rock','Tackle'],fast)],p2:[m(['Recover']),m(['Tackle']),l(['Tackle']),g(['Tackle'])],
  setup:[{side:'p2',mon:1,hp:1},{side:'p2',mon:2,hp:1}],steps:[both(1,1),step('move 2','switch 2'),step(undefined,'switch 3'),step(undefined,'switch 4')]},
 {id:'G4-entry-ko-terminal',seed:1,p1:[m(['Stealth Rock','Tackle'],fast)],p2:[m(['Protect','Tackle']),l(['Tackle'])],
  setup:[{side:'p2',hp:1},{side:'p2',mon:1,hp:1}],steps:[both(1,1),both(2,2),step(undefined,'switch 2')]},
 {id:'G4-pivot-entry-ko',seed:1,p1:[m(['Spikes','Tackle'],fast)],p2:[m(['U-turn']),l(['Tackle']),m(['Tackle'])],
  setup:[{side:'p2',mon:1,hp:1}],steps:[both(1,1),step(undefined,'switch 2'),step(undefined,'switch 3')]});
 for(const target of ['Mew','Gengar'])for(const protect of [false,true])cases.push({id:`G4-spin-${target}-${protect}`,seed:1,
  p1:[m(['Stealth Rock','Spikes','Protect','Recover'])],p2:[mon(target,['Stealth Rock','Spikes',protect?'Protect':'Recover']),l(['Rapid Spin','Tackle'])],
  steps:[both(1,1),both(2,2),step('move 4','switch 2'),both(protect?3:4,1)]});
 // Rapid Spin against Ghost: setters switch after placing hazards, then spin.
 cases.push({id:'G4-spin-ghost',seed:0,p1:[m(['Stealth Rock','Spikes']),g(['Protect'])],p2:[l(['Rapid Spin','Protect'])],
  steps:[both(1,2),both(2,2),step('switch 2','move 1')]},
 {id:'G4-spin-ko',seed:1,p1:[m(['Stealth Rock','Recover'])],p2:[l(['Rapid Spin'])],
  setup:[{side:'p1',hp:1}],steps:[both(1,1)]},
 {id:'G4-double-entry',seed:9,p1:[m(['Stealth Rock']),l(['Tackle'])],p2:[m(['Spikes']),m(['Tackle'])],
  steps:[both(1,1),step('switch 2','switch 2')]});
 cases.push({id:'G4-side-no-target',seed:1,p1:[l(['Stealth Rock'])],p2:[m(['Tackle']),m(['Tackle'])],setup:[{side:'p2',hp:1,pp:0}],steps:[both(1,1)]},
 {id:'G4-minimum-entry',seed:0,p1:[m(['Spikes','Recover'])],p2:[l(['Protect']),m(['Tackle'],{level:1})],steps:[both(1,1),step('move 2','switch 2')]});
 for(const seed of [0,21])for(const swap of [false,true]) {
  const a=[m(['Stealth Rock','Spikes','U-turn','Earthquake']),l(['Rapid Spin','U-turn','Earthquake']),g(['U-turn','Earthquake']),m(['U-turn','Earthquake']),l(['U-turn','Earthquake']),g(['U-turn','Earthquake'])];
  const b=[m(['Stealth Rock','Spikes','U-turn','Earthquake'],fast),l(['Rapid Spin','U-turn','Earthquake']),m(['U-turn','Earthquake']),g(['U-turn','Earthquake']),m(['U-turn','Earthquake']),l(['U-turn','Earthquake'])];
  cases.push({id:`G4-full-${seed}-${swap}`,seed,p1:swap?b:a,p2:swap?a:b,fullBattle:true,family:'hazards'});
 }
 return cases.map(c=>({...c,g4:true}));
};
