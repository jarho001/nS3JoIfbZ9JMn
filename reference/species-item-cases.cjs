'use strict';
module.exports=function(){
 const m=(x={})=>({species:'Mew',moves:['Tackle','Surf','Trick','Transform'],ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...x});
 const b=(a=1,c=1)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[];
 const add=(id,item,species,x={})=>cases.push({id:'G4-species-item-'+id,seed:1,generic:true,speciesItemFamily:true,p1:[m({species,item,evs:{spe:252}})],p2:[m({species:'Snorlax',moves:['Growl','Recover'],ability:'Thick Fat'})],steps:[b(),b(2)],...x});
 for(const [k,item,species]of [['ball','Light Ball','Pikachu'],['club','Thick Club','Marowak'],['dew','Soul Dew','Latios']]){
  add(k+'-ordinary',item,species);
  add(k+'-critical',item,species,{seed:4});
  add(k+'-wrong-species',item,'Mew');
  add(k+'-no-item', '',species);
  add(k+'-burn',item,species,{setup:[{side:'p1',status:'brn'}]});
  add(k+'-staged',item,species,{p1:[m({species,item,moves:['Swords Dance','Tackle','Surf'],evs:{spe:252}})],steps:[b(),b(2),b(3)]});
  add(k+'-guts',item,species,{p1:[m({species,item,ability:'Guts',evs:{spe:252}})],setup:[{side:'p1',status:'brn'}]});
  add(k+'-technician',item,species,{p1:[m({species,item,ability:'Technician',evs:{spe:252}})]});
  add(k+'-exchange',item,species,{steps:[b(3),b(),b(2)]});
  add(k+'-knockoff',item,species,{p2:[m({species:'Snorlax',moves:['Knock Off','Growl'],evs:{spe:252},nature:'Timid'})],steps:[b(1,1),b(2,2)]});
  add(k+'-transform-owner',item,species,{p2:[m({moves:['Growl','Tackle','Surf']})],steps:[b(4),b(2),b(3)]});
  add(k+'-transform-nonowner',item,'Mew',{p2:[m({species,moves:['Growl','Tackle','Surf']})],steps:[b(4),b(2),b(3)]});
 }
 add('club-cubone','Thick Club','Cubone');
 add('dew-latias','Soul Dew','Latias',{p1:[m({species:'Latias',gender:'F',item:'Soul Dew',evs:{spe:252}})]});
 for(const species of ['Latios','Latias','Mew']){
  add('dew-defense-'+species,'', 'Mew',{p2:[m({species,gender:species==='Latias'?'F':'M',item:'Soul Dew',moves:['Growl','Recover']})]});
 }
 return cases;
};
module.exports.membershipWitnesses=function(){
 require('./neutral-turn-oracle.cjs').assertReady();const ps=require('./vendor/pokemon-showdown/dist/sim'),assert=require('node:assert/strict');
 return [['Light Ball',21,'Pikachu'],['Thick Club',57,'Marowak'],['Soul Dew',3,'Latios']].map(([item,n,species])=>{
  const seed=[n,2,3,4],set=ps.Teams.getGenerator('gen5randombattle',seed).getTeam().find(s=>s.item===item&&s.species===species);assert.ok(set,item);return {item:ps.Dex.toID(item),seed,set};
 });
};
