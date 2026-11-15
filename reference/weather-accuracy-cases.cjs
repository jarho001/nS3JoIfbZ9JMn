'use strict';
module.exports=function(){
 const m=(x={})=>({species:'Mew',moves:['Tackle','Recover'],ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...x});
 const b=(a=1,c=2)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[];
 const add=(id,x)=>cases.push({id:'G4-weather-accuracy-'+id,seed:1,generic:true,weatherAccuracyFamily:true,p1:[m()],p2:[m()],steps:[b()],...x});
 for(const move of ['Thunder','Blizzard']){
  const key=move.toLowerCase(),source=(x={})=>m({moves:[move,'Recover'],evs:{spe:252},...x});
  for(const seed of [1,2,3])add(key+'-neutral-'+seed,{seed,p1:[source()]});
  for(const weather of ['Rain Dance','Sunny Day','Hail','Sandstorm']){
   const w=weather.toLowerCase().replace(' ','');
   add(key+'-'+w,{p1:[source({moves:[weather,move,'Recover']})],steps:[b(),b(2)]});
   add(key+'-'+w+'-midturn',{p1:[source({evs:{spe:0}})],p2:[m({moves:[weather,'Recover'],evs:{spe:252}})],steps:[b(1,1)]});
  }
  const weather=move==='Thunder'?'Rain Dance':'Hail';
  add(key+'-weather-stages',{p1:[source({moves:[weather,move,'Recover']})],setup:[{side:'p1',boosts:{accuracy:-6}},{side:'p2',boosts:{evasion:6}}],steps:[b(),b(2)]});
  add(key+'-weather-unaware',{unawareFamily:true,p1:[source({ability:'Unaware',moves:[weather,move,'Recover'],evs:{spe:0}})],p2:[m({ability:'Unaware'})],steps:[b(),b(2)]});
  for(const ability of ['No Guard','Compound Eyes','Hustle','Victory Star'])add(key+'-'+ability.toLowerCase().replace(' ',''),{p1:[source({ability,moves:[weather,move,'Recover']})],steps:[b(),b(2)]});
  add(key+'-weather-victory-tie',{accuracyAbilityFamily:true,p1:[source({ability:'Victory Star',moves:[weather,move,'Recover'],evs:{spe:0}})],p2:[m({ability:'Victory Star'})],steps:[b(),b(2)]});
  add(key+'-weather-substitute',{p1:[source({moves:[weather,move,'Recover']})],p2:[m({moves:['Substitute','Recover']})],steps:[b(1,1),b(2)]});
  add(key+'-weather-protect',{p1:[source({moves:[weather,move,'Recover']})],p2:[m({moves:['Protect','Recover']})],steps:[b(),b(2,1)]});
  for(const charge of ['Fly','Dig'])add(key+'-'+charge.toLowerCase(),{p1:[source({evs:{spe:0}})],p2:[m({moves:[charge,'Recover'],evs:{spe:252}})],steps:[b(1,1),b(1,1)]});
  add(key+'-weather-sleep-talk',{seed:1,p1:[source({moves:['Rest','Sleep Talk',move],evs:{spe:0}})],p2:[m({moves:[weather,'Recover'],evs:{spe:252}})],setup:[{side:'p1',hp:100}],steps:[b(1,1),b(2)]});
  add(key+'-terminal',{p1:[source()],p2:[m({level:1})]});
 }
 add('thunder-sun-unaware',{unawareFamily:true,p1:[m({ability:'Unaware',moves:['Sunny Day','Thunder','Recover']})],p2:[m({ability:'Unaware'})],steps:[b(),b(2)]});
 add('thunder-immune',{p1:[m({moves:['Thunder','Recover'],evs:{spe:252}})],p2:[m({species:'Donphan'})]});
 add('thunder-absorb',{p1:[m({moves:['Thunder','Recover'],evs:{spe:252}})],p2:[m({ability:'Volt Absorb'})],setup:[{side:'p2',hp:100}]});
 return cases;
};
module.exports.membershipWitnesses=function(){
 require('./neutral-turn-oracle.cjs').assertReady();const ps=require('./vendor/pokemon-showdown/dist/sim'),assert=require('node:assert/strict');return [['thunder',37,'Kyogre'],['blizzard',11,'Abomasnow']].map(([id,n,species])=>{const seed=[n,2,3,4],set=ps.Teams.getGenerator('gen5randombattle',seed).getTeam().find(s=>s.species===species&&s.moves.some(m=>ps.Dex.toID(m)===id));assert.ok(set,id);return {kind:'move',id,seed,set};});
};
