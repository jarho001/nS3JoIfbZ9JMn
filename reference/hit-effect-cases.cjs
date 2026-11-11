'use strict';
// Native inputs for the shared hit loop and reactive damage integration.
module.exports=function hitEffectCases(mon){
 const m=(moves,o={})=>mon('Mew',moves,{ability:'Synchronize',...o});
 const fast={evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:252}};
 const tank={evs:{hp:252,atk:0,def:252,spa:0,spd:252,spe:0}};
 const both=(a=1,b=1)=>({p1:`move ${a}`,p2:`move ${b}`});
 const cases=require('./move-phase-cases.cjs')(mon).filter(c=>
   /selfdestruct|multihit|recoil-drain|count-[2345]$|recoil-target-cap|drain-target-cap|skilllink|roughskin/.test(c.id)&&!c.id.includes('TripleKick')).map(c=>/multihit-lifeorb|recoil-drain-(HeadSmash|TakeDown)$/.test(c.id)?{...c,seedWords:[35861,12493,8975,48894]}:c);
 for(const ability of ['Rough Skin','Iron Barbs'])for(const item of ['','Rocky Helmet']){
  for(const move of ['Double Slap','Bullet Seed','Double-Edge','Giga Drain'])cases.push({
   id:`G4-hit-contact-${ability.replaceAll(' ','')}-${item?'helmet':'bare'}-${move.replaceAll(' ','')}`,
   seed:0,seedWords:[35861,12493,8975,48894],p1:[m([move],fast),m(['Tackle'])],
   p2:[m(['Recover'],{ability,item,...tank})],steps:[both()]});
 }
 for(const hp of [1,33,100])cases.push({id:`G4-hit-contact-stop-${hp}`,seed:0,seedWords:[35861,12493,8975,48894],
  p1:[m(['Double Slap'],fast),m(['Tackle'])],p2:[m(['Recover'],{ability:'Rough Skin',item:'Rocky Helmet',...tank})],
  setup:[{side:'p1',hp}],steps:[both()]});
 for(const [ability,move] of [['Rock Head','Double-Edge'],['Rock Head','Tackle'],['Liquid Ooze','Giga Drain'],['Liquid Ooze','Drain Punch']]){
  const ooze=ability==='Liquid Ooze';
  cases.push({id:`G4-hit-${ability.replaceAll(' ','')}-${move.replaceAll(' ','')}`,seed:0,
    p1:[m([move],{...fast,...(ooze?{}:{ability})})],p2:[m(['Recover'],{...tank,...(ooze?{ability}:{})})],
    setup:ooze?[{side:'p1',hp:1}]:[],steps:[both()]});
 }
 for(const [id,move,sourceHp,targetHp] of [['recoil-source-ko','Double-Edge',1,100],['recoil-double-ko','Double-Edge',1,1],
  ['contact-double-ko','Tackle',1,1],['drain-full','Giga Drain',175,100]])cases.push({id:`G4-hit-${id}`,seed:0,
   p1:[m([move],fast)],p2:[m(['Recover'],{...(id.startsWith('contact')?{ability:'Iron Barbs',item:'Rocky Helmet'}:{}),...tank})],
   setup:[{side:'p1',hp:sourceHp},{side:'p2',hp:targetHp}],steps:[both()]});
 cases.push({id:'G4-hit-rockhead-struggle',seed:0,p1:[m(['Tackle'],{ability:'Rock Head',...fast})],p2:[m(['Recover'],tank)],setup:[{side:'p1',pp:0}],steps:[both()]});
 for(const hp of [1,100])cases.push({id:`G4-hit-contact-terminal-multihit-${hp}`,seed:0,seedWords:[35861,12493,8975,48894],p1:[m(['Double Slap'],fast)],p2:[m(['Recover'],{ability:'Rough Skin',item:'Rocky Helmet',...tank})],setup:[{side:'p1',hp:1},{side:'p2',hp}],steps:[both()]});
 for(const move of ['Bullet Seed','Double-Edge','Giga Drain']){
  cases.push({id:`G4-hit-protect-${move.replaceAll(' ','')}`,seed:0,p1:[m([move],fast)],p2:[m(['Protect'])],steps:[both()]});
  cases.push({id:`G4-hit-immunity-${move.replaceAll(' ','')}`,seed:0,p1:[m([move==='Bullet Seed'?'Double Slap':move],fast)],p2:[mon('Gengar',['Recover'],{ability:'Levitate'})],steps:[both()]});
 }
 cases.push({id:'G4-hit-boost-weather-screen',seed:21,p1:[m(['Swords Dance','Bullet Seed'],{ability:'Skill Link',item:'Life Orb'})],p2:[m(['Rain Dance','Reflect','Recover'],fast)],steps:[both(1,1),both(2,2),both(2,3)]},
  {id:'G4-hit-paralysis-tied-contact',seed:21,p1:[m(['Double Slap'])],p2:[m(['Tackle'],{ability:'Iron Barbs',item:'Rocky Helmet'})],setup:[{side:'p1',status:'par'},{side:'p2',status:'par'}],steps:[both()]},
  {id:'G4-hit-ooze-survives',seed:0,p1:[m(['Giga Drain'],fast)],p2:[m(['Recover'],{ability:'Liquid Ooze',...tank})],setup:[{side:'p1',hp:100}],steps:[both()]},
  {id:'G4-hit-rockhead-recoil',seed:0,p1:[m(['Brave Bird'],{ability:'Rock Head',...fast})],p2:[m(['Recover'],tank)],steps:[both()]});
 for(const [id,foe,setup,ability] of [
  ['terminal',m(['Tackle'],tank),[],'Synchronize'],
  ['double-ko',m(['Tackle'],tank),[{side:'p2',hp:1}],'Synchronize'],
  ['protect',m(['Protect'],tank),[],'Synchronize'],
  ['immune',mon('Gengar',['Tackle'],{ability:'Levitate'}),[],'Synchronize'],
  ['miss',m(['Tackle'],tank),[],'Hustle']])cases.push({id:`G4-hit-explosion-${id}`,seed:0,p1:[m(['Explosion'],{ability,...fast})],p2:[foe],setup,steps:[both()]});
 for(const [id,hp,targetHp,item] of [['double-ko',1,1,''],['orb',100,100,'Life Orb']])cases.push({id:`G4-hit-struggle-contact-${id}`,seed:0,p1:[m(['Tackle'],{...fast,item})],p2:[m(['Recover'],{ability:'Rough Skin',item:'Rocky Helmet',...tank})],setup:[{side:'p1',hp,pp:0},{side:'p2',hp:targetHp}],steps:[both()]});
 for(const ability of ['Synchronize','Immunity'])cases.push({id:`G4-hit-secondary-multihit-${ability}`,seed:0,p1:[m(['Twineedle'],fast)],p2:[m(['Recover'],{ability,...tank})],steps:[both(),both()]});
 cases.push({id:'G4-hit-secondary-recoil-rockhead',seed:0,p1:[m(['Volt Tackle'],{ability:'Rock Head',...fast})],p2:[m(['Recover'],tank)],steps:[both(),both()]});
 for(const move of ['Flare Blitz','Scald','Flame Wheel','Sacred Fire'])cases.push({id:`G4-hit-defrost-${move.replaceAll(' ','')}`,seed:0,p1:[m([move],fast)],p2:[m(['Recover'],tank)],setup:[{side:'p1',status:'frz'}],steps:[both()]});
 cases.push({id:'G4-hit-scald-target-frozen',seed:0,p1:[m(['Scald'],fast)],p2:[m(['Recover'],tank)],setup:[{side:'p2',status:'frz'}],steps:[both()]},
  {id:'G4-hit-scald-choice-frozen',seed:0,p1:[m(['Scald'],{...fast,item:'Choice Band'})],p2:[m(['Recover'],tank)],setup:[{side:'p1',status:'frz'}],steps:[both()]},
  {id:'G4-hit-scald-frozen-confusion',seed:0,p1:[m(['Tackle','Scald'])],p2:[m(['Confuse Ray','Recover'],fast)],setup:[{side:'p1',status:'frz'}],steps:[both(1,1),both(2,2)]});
 cases.push({id:'G4-hit-scald-frozen-confusion-blocked',seed:0,seedWords:[35861,12493,8975,48894],p1:[m(['Tackle','Scald'])],p2:[m(['Confuse Ray','Recover'],fast)],setup:[{side:'p1',status:'frz'}],steps:[both(1,1),both(2,2)]});
 cases.push({id:'G4-hit-defrost-miss',seed:0,p1:[m(['Flare Blitz'],{ability:'Hustle',...fast})],p2:[m(['Recover'],tank)],setup:[{side:'p1',status:'frz'}],steps:[both()]},
  {id:'G4-hit-defrost-protected',seed:0,p1:[m(['Scald'],fast)],p2:[m(['Protect'],tank)],setup:[{side:'p1',status:'frz'}],steps:[both()]});
 cases.push({id:'G4-hit-frozen-target-fire-Synchronize',seed:0,p1:[m(['Sacred Fire'],fast)],p2:[m(['Recover'])],setup:[{side:'p2',status:'frz'}],steps:[both()]});
 const fireContact={seed:0,seedWords:[39407,18577,41884,47404],p1:[m(['Flare Blitz'],fast)],p2:[m(['Recover'],{ability:'Rough Skin',item:'Rocky Helmet'})],steps:[both()]};
 cases.push({...fireContact,id:'G4-hit-frozen-target-fire-contact',setup:[{side:'p2',status:'frz'}]},
  {...fireContact,id:'G4-hit-frozen-target-fire-terminal',setup:[{side:'p1',hp:1},{side:'p2',status:'frz'}]});
 for(const hp of [1,100])cases.push({id:`G4-hit-spin-contact-${hp}`,seed:0,p1:[m(['Protect','Rapid Spin'],fast),m(['Tackle'])],p2:[m(['Spikes','Recover'],{ability:'Rough Skin'})],setup:[{side:'p1',hp}],steps:[both(1,1),both(2,2)]});
 for(const seed of [0,21])for(const swap of [false,true]){
  const a=[m(['Bullet Seed'],{ability:'Skill Link',item:'Life Orb',...fast}),m(['Double-Edge'],{ability:'Rock Head'}),
   m(['Drain Punch']),m(['Rock Blast'],{item:'Choice Band'}),m(['Brave Bird']),m(['Giga Drain'],{item:'Leftovers'})];
  const b=[m(['Tackle'],{ability:'Rough Skin',item:'Rocky Helmet'}),m(['Ice Beam'],{ability:'Iron Barbs'}),
   m(['Surf'],{ability:'Liquid Ooze'}),m(['Double Slap'],{ability:'Skill Link'}),m(['Drain Punch'],{item:'Life Orb'}),m(['Double-Edge'])];
  cases.push({id:`G4-hit-full-${seed}-${swap}`,seed,p1:swap?b:a,p2:swap?a:b,fullBattle:true,family:'hit-effects'});
 }
 return cases.map(c=>({...c,generic:true,phaseFamily:false,hookFamily:false,hitFamily:true}));
};
