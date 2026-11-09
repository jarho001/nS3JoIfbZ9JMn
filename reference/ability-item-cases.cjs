'use strict';
// Independent expectations for the integrated hook family. No capability
// is enabled in JAX merely by adding a case or exporting an ability/item ID.
module.exports=function abilityItemCases(mon) {
 const m=(moves,o={})=>mon('Mew',moves,{ability:'Synchronize',...o});
 const both=(a,b)=>({p1:`move ${a}`,p2:`move ${b}`});
 const fast={evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:252}};
 const cases=[];
 for(const ability of ['Pressure','Intimidate','Prankster','Huge Power','Pure Power','Adaptability',
   'Technician','Hustle','Compound Eyes','Super Luck','Sniper','Guts','Quick Feet','Marvel Scale',
   'Battle Armor','Shell Armor','Thick Fat','Filter','Solid Rock','Tinted Lens']) {
  cases.push({id:`G4-hooks-ability-${ability.replaceAll(' ','')}`,seed:21,
    p1:[m(['Swords Dance','Tackle','Surf','Thunder Wave'],{ability})],
    p2:[m(['Tackle','Ice Beam','Recover'],fast)],steps:[both(1,1),both(2,2),both(3,3),both(4,1)]});
 }
 for(const item of ['Choice Band','Choice Specs','Choice Scarf','Life Orb','Expert Belt',
   'Muscle Band','Wise Glasses','Silk Scarf','Mystic Water','Scope Lens']) {
  cases.push({id:`G4-hooks-item-${item.replaceAll(' ','')}`,seed:21,
    p1:[m(['Tackle','Surf'],{item})],p2:[m(['Tackle','Recover'],fast)],
    steps:[both(1,1),both(1,2),both(1,1)]});
 }
 for(const ability of ['Drizzle','Drought','Sand Stream','Snow Warning']) {
  cases.push({id:`G4-hooks-weather-${ability.replaceAll(' ','')}`,seed:0,
    p1:[m(['Surf','Recover'],{ability}),m(['Tackle'])],p2:[m(['Tackle','Recover'])],
    steps:[both(1,2),{p1:'switch 2',p2:'move 1'},{p1:'switch 2',p2:'move 2'},both(2,2)]});
 }
 for(const [weather,ability] of [['Rain Dance','Swift Swim'],['Sunny Day','Chlorophyll'],
   ['Sandstorm','Sand Rush'],['Hail','Ice Body']]) {
  cases.push({id:`G4-hooks-weather-speed-${ability.replaceAll(' ','')}`,seed:1,
    p1:[m([weather,'Tackle','Recover'],{ability})],p2:[m(['Tackle','Recover'],fast)],
    steps:[both(1,2),both(2,1),both(3,2)]});
 }
 // Status-dependent hooks are installed through the same trusted fixture path
 // as the existing status corpus; they are not generator-membership witnesses.
 for(const ability of ['Guts','Quick Feet','Marvel Scale'])cases.push({
   id:`G4-hooks-status-${ability.replaceAll(' ','')}`,seed:21,
   p1:[m(['Tackle','Surf'],{ability})],p2:[m(['Tackle','Recover'],fast)],
   setup:[{side:'p1',status:'par'}],steps:[both(1,1),both(2,2)]});
 cases.push({id:'G4-hooks-pressure-targets',seed:0,p1:[m(['Recover'],{ability:'Pressure'})],
   p2:[m(['Stealth Rock','Spikes','Rain Dance','Recover'])],steps:[both(1,1),both(1,2),both(1,3),both(1,4)]},
  {id:'G4-hooks-pressure-last-pp',seed:0,p1:[m(['Recover'],{ability:'Pressure'})],
   p2:[m(['Tackle'])],setup:[{side:'p2',pp:1}],steps:[both(1,1),both(1,1)]},
  {id:'G4-hooks-intimidate-tied',seed:1,p1:[m(['Tackle'],{ability:'Intimidate'})],
   p2:[m(['Tackle'],{ability:'Intimidate'})],steps:[both(1,1)]},
  {id:'G4-hooks-weather-dual-tied',seed:1,p1:[m(['Tackle','Recover'],{ability:'Drizzle'})],
   p2:[m(['Tackle','Recover'],{ability:'Drought'})],steps:[both(1,1),both(2,2)]},
  {id:'G4-hooks-weather-initial-speed',seed:1,p1:[m(['Tackle','Recover'],{ability:'Drizzle'})],
   p2:[m(['Tackle','Recover'],{ability:'Swift Swim'})],steps:[both(1,1),both(2,2)]},
  {id:'G4-hooks-choice-last-pp-protect',seed:0,p1:[m(['Tackle','Surf'],{...fast,item:'Choice Band'})],
   p2:[m(['Protect','Recover'])],setup:[{side:'p1',ppByMove:{tackle:1}}],steps:[both(1,1),both(1,2)]},
  {id:'G4-hooks-choice-sleep-status',seed:0,p1:[m(['Swords Dance','Tackle'],{item:'Choice Band'})],
   p2:[m(['Spore','Recover'],fast)],steps:[both(1,1),both(1,2),both(1,2)]});
 for(const [id,sourceBench,targetBench] of [['terminal',false,false],['target-bench',false,true],['source-bench',true,false]])cases.push({
   id:`G4-hooks-life-orb-${id}`,seed:0,
   p1:[m(['Tackle'],{...fast,item:'Life Orb'}),...(sourceBench?[m(['Tackle'])]:[])],
   p2:[m(['Tackle']),...(targetBench?[m(['Tackle'])]:[])],setup:[{side:'p1',hp:1},{side:'p2',hp:1}],steps:[both(1,1)]});
 cases.push({id:'G4-hooks-life-orb-struggle',seed:0,p1:[m(['Tackle'],{...fast,item:'Life Orb'})],
   p2:[m(['Recover'])],setup:[{side:'p1',pp:0}],steps:[both(1,1)]},
  {id:'G4-hooks-adaptability-stab',seed:0,p1:[mon('Vaporeon',['Water Gun'],{ability:'Adaptability'})],
   p2:[m(['Recover'])],steps:[both(1,1)]},
  {id:'G4-hooks-tinted-lens-resist',seed:0,p1:[m(['Surf'],{ability:'Tinted Lens'})],
   p2:[mon('Vaporeon',['Recover'],{ability:'Synchronize'})],steps:[both(1,1)]});
 for(const ability of ['Filter','Solid Rock'])cases.push({id:`G4-hooks-super-effective-${ability.replaceAll(' ','')}`,seed:0,
   p1:[m(['Water Gun'])],p2:[mon('Rhydon',['Recover'],{ability})],steps:[both(1,1)]});
 cases.push({id:'G4-hooks-expert-belt-super-effective',seed:0,p1:[m(['Water Gun'],{item:'Expert Belt'})],
   p2:[mon('Rhydon',['Recover'],{ability:'Synchronize'})],steps:[both(1,1)]},
  {id:'G4-hooks-special-item-effects',seed:0,p1:[m(['Surf'],{item:'Choice Specs'})],p2:[m(['Recover'])],steps:[both(1,1)]},
  {id:'G4-hooks-wise-glasses-special',seed:0,p1:[m(['Surf'],{item:'Wise Glasses'})],p2:[m(['Recover'])],steps:[both(1,1)]},
  {id:'G4-hooks-mystic-water-special',seed:0,p1:[m(['Water Gun'],{item:'Mystic Water'})],p2:[m(['Recover'])],steps:[both(1,1)]},
  {id:'G4-hooks-eviolite-sand',seed:0,p1:[mon('Rhydon',['Sandstorm','Recover'],{ability:'Synchronize',item:'Eviolite',...fast})],
   p2:[m(['Ice Beam'])],steps:[both(1,1),both(2,1)]},
  {id:'G4-hooks-damage-handler-interaction',seed:21,p1:[m(['Swords Dance','Tackle'],{ability:'Sniper',item:'Life Orb'})],
   p2:[m(['Reflect','Tackle'],{ability:'Filter'})],steps:[both(1,1),both(2,2)]},
  {id:'G4-hooks-critical-stage-stacking',seed:0,p1:[m(['Tackle'],{ability:'Super Luck',item:'Scope Lens'})],
   p2:[m(['Recover'])],steps:[both(1,1)]});
 cases.push({id:'G4-hooks-choice-all-pp-empty',seed:0,p1:[m(['Tackle'],{item:'Choice Band',...fast})],
   p2:[m(['Recover'])],setup:[{side:'p1',pp:0}],steps:[both(1,1)]},
  {id:'G4-hooks-life-orb-pivot-ko',seed:0,p1:[m(['U-turn'],{item:'Life Orb',...fast}),m(['Tackle'])],
   p2:[m(['Tackle'])],setup:[{side:'p1',hp:1}],steps:[both(1,1)]},
  {id:'G4-hooks-entry-hazard-ko-skips-start',seed:0,p1:[m(['Recover']),m(['Tackle'],{ability:'Pressure'})],
   p2:[m(['Stealth Rock','Recover'])],setup:[{side:'p1',mon:1,hp:1}],
   steps:[both(1,1),{p1:'switch 2',p2:'move 2'}]});
 for(const seed of [0,21])for(const swap of [false,true]) {
   const a=[m(['Tackle'],{ability:'Huge Power',item:'Life Orb',...fast}),
     mon('Vaporeon',['Water Gun'],{ability:'Adaptability',item:'Mystic Water'}),
     m(['Pound'],{ability:'Pressure',item:'Choice Band'}),
     mon('Rhydon',['Earthquake'],{ability:'Solid Rock',item:'Eviolite'}),
     m(['Reflect','Surf'],{ability:'Prankster'}),m(['Surf'],{ability:'Swift Swim',item:'Choice Specs'})];
   const b=[m(['Surf'],{ability:'Drizzle',item:'Damp Rock'}),
     m(['Ice Beam'],{ability:'Battle Armor',item:'Life Orb'}),m(['Tackle'],{ability:'Intimidate',item:'Choice Scarf'}),
     m(['Tackle'],{ability:'Sniper',item:'Scope Lens'}),
     mon('Rhydon',['Tackle'],{ability:'Filter',item:'Eviolite'}),m(['Ice Beam'],{ability:'Snow Warning',item:'Leftovers'})];
   cases.push({id:`G4-hooks-full-${seed}-${swap}`,seed,p1:swap?b:a,p2:swap?a:b,fullBattle:true,family:'ability-item'});
 }
 return cases.map(c=>({...c,generic:true,hookFamily:true}));
};
