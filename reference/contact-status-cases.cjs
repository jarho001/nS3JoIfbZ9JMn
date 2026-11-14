'use strict';
module.exports=function(){
 const m=(x={})=>({species:'Mew',moves:['Tackle','Surf','Gastro Acid','Recover'],ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...x});
 const b=(a=1,c=4)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[];
 const add=(id,x={})=>cases.push({id:'G4-contact-status-'+id,seed:1,generic:true,contactStatusFamily:true,p1:[m({evs:{spe:252}})],p2:[m()],steps:[b()],...x});
 for(const [ability,immune]of [['Static','Limber'],['Flame Body','Water Veil']]){
  const key=ability.toLowerCase().replaceAll(' ','');
  for(const move of ['Tackle','Surf','Earthquake','Double Hit'])add(key+'-'+move.toLowerCase().replaceAll(' ',''),{p1:[m({moves:[move,'Recover'],evs:{spe:252}})],p2:[m({ability})],steps:[b(),b(2)]});
  add(key+'-suppressed',{p2:[m({ability})],steps:[b(3),b()]});
  add(key+'-target-ko',{p2:[m({ability,level:1})]});
  add(key+'-helmet-source-ko',{p2:[m({ability,item:'Rocky Helmet'})],setup:[{side:'p1',hp:1}]});
  add(key+'-already-statused',{p2:[m({ability})],setup:[{side:'p1',status:'psn'}]});
  for(const [id,extra]of [['immune',{ability:immune}],['lum',{item:'Lum Berry'}],['synchronize',{ability:'Synchronize'}],['magicguard',{ability:'Magic Guard'}],['electric',{species:'Pikachu'}],['fire',{species:'Flareon'}]])add(key+'-'+id,{p1:[m({evs:{spe:252},...extra})],p2:[m({ability})]});
  add(key+'-struggle',{p2:[m({ability})],setup:[{side:'p1',pp:0}]});
  add(key+'-substitute',{p1:[m({evs:{spe:252},moves:['Tackle','Recover']})],p2:[m({ability,moves:['Substitute','Recover']})],steps:[b(2,1),b(1,2)]});
  for(const seed of [2,3,4,5])add(key+'-seed-'+seed,{seed,p2:[m({ability})]});
 }
 return cases;
};
module.exports.membershipWitnesses=function(){
 require('./neutral-turn-oracle.cjs').assertReady();const ps=require('./vendor/pokemon-showdown/dist/sim'),assert=require('node:assert/strict');
 return [['Static',9,'Ampharos'],['Flame Body',78,'Magmortar']].map(([ability,n,species])=>{const seed=[n,2,3,4],set=ps.Teams.getGenerator('gen5randombattle',seed).getTeam().find(s=>s.ability===ability&&s.species===species);assert.ok(set,ability);return {kind:'ability',id:ps.Dex.toID(ability),seed,set};});
};
