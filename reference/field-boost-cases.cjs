'use strict';
// Independent field/stat inputs. Implementation and coverage are separate.
module.exports=function fieldBoostCases(mon) {
 const m=(moves,o={})=>mon('Mew',moves,{ability:'Synchronize',...o});
 const set=(species,moves,o={})=>mon(species,moves,{ability:'Synchronize',...o});
 const both=(a,b)=>({p1:`move ${a}`,p2:`move ${b}`});
 const fast={evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:252}};
 const cases=[];
 for(const weather of ['Rain Dance','Sunny Day','Sandstorm','Hail'])cases.push({
  id:`G4-field-weather-${weather.replaceAll(' ','')}`,seed:1,
  p1:[m([weather,'Surf','Recover'],fast)],p2:[m(['Flamethrower','Recover'])],
  steps:[both(1,2),both(1,1),both(2,2),both(3,2),both(3,2),both(3,2)]});
 for(const screen of ['Reflect','Light Screen'])cases.push({
  id:`G4-field-screen-${screen.replaceAll(' ','')}`,seed:0,
  p1:[m([screen,'Recover'],fast)],p2:[m(['Tackle','Surf'])],
  steps:[both(1,1),both(1,2),both(2,1),both(2,2),both(2,1),both(2,2)]});
 for(const move of ['Swords Dance','Calm Mind','Iron Defense','Agility','Growl','Screech','Double Team'])cases.push({
  id:`G4-field-boost-${move.replaceAll(' ','')}`,seed:9,
  p1:[m([move,'Tackle','Surf'],fast),m(['Tackle'])],p2:[m(['Tackle','Surf'])],
  steps:[both(1,1),both(2,1),both(3,2),{p1:'switch 2',p2:'move 1'}]});
 cases.push({id:'G4-field-boost-cap',seed:1,p1:[m(['Swords Dance'],fast)],p2:[m(['Recover'])],
   steps:Array.from({length:4},()=>both(1,1))},
  {id:'G4-field-weather-replace',seed:9,p1:[m(['Rain Dance','Sunny Day','Surf'],fast)],p2:[m(['Hail','Recover'])],
   steps:[both(1,1),both(2,2),both(3,2)]},
  {id:'G4-field-weather-immunity',seed:21,p1:[set('Golem',['Sandstorm','Hail'])],p2:[set('Articuno',['Recover'])],
   steps:[both(1,1),both(2,1)]},
  {id:'G4-field-screen-critical',seed:21,p1:[m(['Reflect'],fast)],p2:[m(['Tackle'])],steps:[both(1,1)]});
 cases.push({id:'G4-field-weather-tied',seed:1,p1:[m(['Rain Dance','Recover'])],p2:[m(['Recover'])],steps:[both(1,1),both(2,1)]},
  {id:'G4-field-weather-double-ko',seed:9,p1:[m(['Sandstorm'],fast)],p2:[m(['Protect'])],setup:[{side:'p1',hp:1},{side:'p2',hp:1}],steps:[both(1,1)]},
  {id:'G4-field-screen-minimum',seed:1,p1:[m(['Reflect'],{level:100})],p2:[m(['Tackle'],{level:1})],steps:[both(1,1)]});
 for(const move of ['Swords Dance','Light Screen'])cases.push({id:`G4-field-tied-${move.replaceAll(' ','')}`,seed:1,
  p1:[m([move,'Recover'])],p2:[m(['Recover'])],steps:[both(1,1),both(2,1)]});
 cases.push({id:'G4-field-screens-both',seed:1,p1:[m(['Reflect','Recover'])],p2:[m(['Reflect','Recover'])],
  steps:[both(1,1),...Array.from({length:5},()=>both(2,2))]},
  {id:'G4-field-agility-tied',seed:1,p1:[m(['Agility','Recover'])],p2:[m(['Recover'])],steps:[both(1,1),both(2,1)]});
 for(const [id,move,item] of [['rain-rock','Rain Dance','Damp Rock'],['screen-clay','Reflect','Light Clay']])cases.push({
  id:`G4-field-${id}`,seed:9,p1:[m([move,'Recover'],{...fast,item})],p2:[m(['Tackle','Recover'])],
  steps:[both(1,2),...Array.from({length:8},()=>both(2,1))]});
 for(const move of ['Shell Smash','Shift Gear'])cases.push({id:`G4-field-native-order-${move.replaceAll(' ','')}`,seed:0,
  p1:[m([move,'Recover'])],p2:[m(['Recover'])],steps:[both(1,1),both(2,1)]});
 cases.push({id:'G4-field-hazard-source-order',seed:1,p1:[m(['Stealth Rock','Spikes','Rain Dance','Recover'],fast)],
   p2:[m(['Recover']),m(['Tackle'])],steps:[both(1,1),both(2,1),both(3,1),{p1:'move 4',p2:'switch 2'}]});
 cases.push({id:'G4-field-status-residuals',seed:1,
   p1:[m(['Sandstorm','Reflect','Recover'],{...fast,item:'Leftovers'})],
   p2:[m(['Protect','Toxic','Recover'],{item:'Leftovers'})],
   setup:[{side:'p1',status:'tox'},{side:'p2',status:'brn'}],
   steps:[both(1,1),both(2,2),both(3,1),both(3,3),both(3,3)]},
  {id:'G4-field-stages-critical',seed:21,p1:[m(['Iron Defense','Recover'],fast)],
   p2:[m(['Tackle','Growl'])],steps:[both(1,1),both(2,2),both(1,1)]},
  {id:'G4-field-negative-speed-accuracy',seed:9,p1:[m(['Scary Face','Sand Attack','Tackle'],fast)],
   p2:[m(['Tackle'])],steps:[both(1,1),both(2,1),both(3,1)]});
 cases.push({id:'G4-field-confusion-boost',seed:0,p1:[m(['Swords Dance','Tackle'],fast)],
   p2:[m(['Confuse Ray','Recover'])],steps:[both(1,1),both(2,2)]},
  // Invert three native LCG steps (two accuracy draws, then critical).
  {id:'G4-field-critical-negative-attack',seed:21,seedWords:[35775,7112,3144,14623],p1:[m(['Growl'],fast)],
   p2:[m(['Tackle'])],steps:[both(1,1)]},
  {id:'G4-field-four-screen-handlers',seed:1,p1:[m(['Reflect','Light Screen','Surf'])],
   p2:[m(['Reflect','Light Screen','Tackle'])],steps:[both(1,1),both(2,2),both(3,3)]});
 cases.push({id:'G4-field-sun-status-immunity-metadata',seed:1,
   p1:[m(['Sunny Day','Recover'],fast)],p2:[m(['Spore','Recover'])],steps:[both(1,1),both(2,2)]});
 for(const seed of [0,21])for(const swap of [false,true]) {
   const a=[m(['Rain Dance','Light Screen','Calm Mind','Surf'],{...fast,item:'Damp Rock'}),
     m(['Sandstorm','Reflect','Swords Dance','Tackle'],{item:'Light Clay'}),m(['Spore','Ice Beam']),
     m(['Spikes','Flamethrower']),m(['Confuse Ray','Sludge Bomb']),m(['Thunderbolt'])];
   const b=[m(['Sunny Day','Reflect','Iron Defense','Flamethrower'],{item:'Heat Rock'}),
     m(['Hail','Light Screen','Agility','Ice Beam'],fast),m(['Toxic','Tackle']),
     m(['Stealth Rock','Surf']),m(['Rest','Sludge Bomb']),m(['Thunderbolt'])];
   cases.push({id:`G4-field-full-${seed}-${swap}`,seed,p1:swap?b:a,p2:swap?a:b,
     fullBattle:true,family:'field-boost'});
 }
 return cases.map(c=>({...c,generic:true,fieldFamily:true}));
};
