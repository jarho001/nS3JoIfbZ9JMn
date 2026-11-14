'use strict';
// Native-only continuation: expand callback sort/lifecycle audit before implementation.
module.exports=function(){
 const m=(x={})=>({species:'Mew',moves:['Mega Punch','Gastro Acid','Fly','Recover'],ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...x});
 const b=(a=1,c=4)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[];
 const add=(id,x={})=>cases.push({id:'G4-accuracy-ability-'+id,seed:1,generic:true,accuracyAbilityFamily:true,p1:[m({evs:{spe:252}})],p2:[m()],steps:[b()],...x});
 for(const ability of ['No Guard','Victory Star']){
  const key=ability.toLowerCase().replaceAll(' ','');
  for(const side of ['source','target'])for(const move of ['Mega Punch','Toxic','Screech']){
   add(key+'-'+side+'-'+move.toLowerCase().replaceAll(' ',''),{p1:[m({moves:[move,'Gastro Acid','Fly','Recover'],ability:side==='source'?ability:'Pressure',evs:{spe:252}})],p2:[m({ability:side==='target'?ability:'Pressure'})]});
  }
  add(key+'-stages',{p1:[m({ability,evs:{spe:252}})],setup:[{side:'p1',boosts:{accuracy:-2}},{side:'p2',boosts:{evasion:2}}]});
  add(key+'-suppressed',{p2:[m({ability})],steps:[b(2),b()]});
  add(key+'-protected',{p1:[m({ability,evs:{spe:252}})],p2:[m({moves:['Protect','Recover']})],steps:[b(1,1)]});
  add(key+'-fly',{p1:[m({ability,evs:{spe:252}})],steps:[b(1,3),b(1,1)]});
 }
 add('noguard-ghost',{p1:[m({ability:'No Guard',moves:['Tackle'],evs:{spe:252}})],p2:[m({species:'Gengar',moves:['Recover']})],steps:[b(1,1)]});
 return cases;
};
module.exports.membershipWitnesses=function(){
 require('./neutral-turn-oracle.cjs').assertReady();const ps=require('./vendor/pokemon-showdown/dist/sim'),assert=require('node:assert/strict');
 return [['No Guard',44,'Golurk'],['Victory Star',38,'Victini']].map(([ability,n,species])=>{
  const seed=[n,2,3,4],set=ps.Teams.getGenerator('gen5randombattle',seed).getTeam().find(s=>s.ability===ability&&s.species===species);assert.ok(set,ability);return {kind:'ability',id:ps.Dex.toID(ability),seed,set};
 });
};
