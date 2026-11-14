'use strict';
// Synthetic branch installations; generator witnesses below prove membership separately.
module.exports=function(){
 const m=(moves,x={})=>({species:'Mew',moves,ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...x});
 const b=(a=1,c=1)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[];
 const add=(id,ability,x={})=>cases.push({id:'G4-rain-recovery-'+id,seed:1,generic:true,rainRecoveryFamily:true,p1:[m(['Rain Dance','Recover','Gastro Acid'],{evs:{spe:252}})],p2:[m(['Growl'],{ability})],steps:[b(),b(2)],...x});
 for(const ability of ['Rain Dish','Hydration']){
  const key=ability==='Rain Dish'?'dish':'hydration';
  add(key+'-partial',ability,{setup:[{side:'p2',hp:80}]});
  add(key+'-full',ability);
  add(key+'-cap',ability,{setup:[{side:'p2',hp:174}]});
  add(key+'-tie',ability,{p1:[m(['Rain Dance','Recover'])],setup:[{side:'p2',hp:80,status:'brn'}]});
  add(key+'-gas',ability,{setup:[{side:'p2',hp:80,status:'brn'}],steps:[b(),b(3)]});
  add(key+'-expiry',ability,{setup:[{side:'p2',hp:80}],steps:[b(),...Array.from({length:4},()=>b(2)),b(2)]});
  add(key+'-wrong-weather',ability,{p1:[m(['Sunny Day','Recover'],{evs:{spe:252}})],setup:[{side:'p2',hp:80,status:'brn'}]});
  add(key+'-no-weather',ability,{setup:[{side:'p2',hp:80,status:'brn'}],steps:[b(2)]});
  add(key+'-minimum',ability,{p2:[m(['Growl'],{species:'Shedinja',ability})]});
 }
 for(const status of ['brn','psn','tox','par','frz'])add('hydration-'+status,'Hydration',{setup:[{side:'p2',hp:80,status}]});
 add('dish-leftovers-burn','Rain Dish',{p2:[m(['Growl'],{ability:'Rain Dish',item:'Leftovers'})],setup:[{side:'p2',hp:80,status:'brn'}]});
 add('hydration-leftovers-toxic','Hydration',{p2:[m(['Growl'],{ability:'Hydration',item:'Leftovers'})],setup:[{side:'p2',hp:80,status:'tox'}]});
 add('hydration-slp','Hydration',{p1:[m(['Rain Dance','Spore'],{evs:{spe:252}})],steps:[b(),b(2)]});
 add('hydration-rest','Hydration',{p2:[m(['Rest'],{ability:'Hydration'})],setup:[{side:'p2',hp:80}]});
 add('hydration-expiry-status','Hydration',{p1:[m(['Rain Dance','Recover','Thunder Wave'],{evs:{spe:252}})],steps:[b(),b(2),b(2),b(2),b(3),b(2)]});
 add('both-tie','Hydration',{p1:[m(['Rain Dance','Recover'],{ability:'Rain Dish'})],setup:[{side:'p1',hp:80,status:'brn'},{side:'p2',hp:80,status:'tox'}]});
 add('both-hydration-tie','Hydration',{p1:[m(['Rain Dance','Recover'],{ability:'Hydration'})],setup:[{side:'p1',hp:80,status:'brn'},{side:'p2',hp:80,status:'tox'}]});
 add('dish-mirrored','Pressure',{p1:[m(['Rain Dance','Recover'],{ability:'Rain Dish',evs:{spe:252}})],setup:[{side:'p1',hp:80}]});
 add('hydration-gas-before','Hydration',{p1:[m(['Gastro Acid','Rain Dance'],{evs:{spe:252}})],setup:[{side:'p2',hp:80,status:'tox'}],steps:[b(),b(2)]});
 add('dish-ko','Rain Dish',{p1:[m(['Rain Dance','Tackle'],{evs:{spe:252}})],p2:[m(['Growl'],{ability:'Rain Dish'}),m(['Growl'])],setup:[{side:'p2',hp:1}],steps:[b(),b(2)]});
 return cases;
};
module.exports.membershipWitnesses=function(){
 require('./neutral-turn-oracle.cjs').assertReady();const ps=require('./vendor/pokemon-showdown/dist/sim'),assert=require('node:assert/strict');
 return [['Rain Dish',59,'Pelipper'],['Hydration',1,'Swanna']].map(([ability,n,species])=>{
  const seed=[n,2,3,4],set=ps.Teams.getGenerator('gen5randombattle',seed).getTeam().find(s=>s.ability===ability&&s.species===species);assert.ok(set,ability);return {ability:ps.Dex.toID(ability),seed,set};
 });
};
