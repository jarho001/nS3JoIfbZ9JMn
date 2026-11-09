'use strict';
// Native-only expectations for the next integrated hook family. No capability
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
 return cases.map(c=>({...c,generic:true,hookFamily:true}));
};
