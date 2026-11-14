'use strict';
module.exports=function(){
 const m=(moves,x={})=>({species:'Mew',moves,ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...x});
 const b=(a=1,c=1)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[];
 const add=(id,ability,x={})=>cases.push({id:'G4-residual-recovery-'+id,seed:1,generic:true,residualRecoveryFamily:true,p1:[m(['Recover','Gastro Acid','Tackle'],{evs:{spe:252}})],p2:[m(['Growl','Rest'],{ability})],steps:[b(),b()],...x});
 for(const ability of ['Shed Skin','Poison Heal']){
  const key=ability==='Shed Skin'?'shed':'poison';
  add(key+'-healthy',ability);
  for(const status of ['brn','psn','tox','par','frz'])add(key+'-'+status,ability,{setup:[{side:'p2',hp:80,status}]});
  add(key+'-rest',ability,{setup:[{side:'p2',hp:80}],steps:[b(1,2),b(),b()]});
  add(key+'-gas',ability,{setup:[{side:'p2',hp:80,status:'tox'}],steps:[b(2),b()]});
  add(key+'-leftovers',ability,{p2:[m(['Growl'],{ability,item:'Leftovers'})],setup:[{side:'p2',hp:80,status:'tox'}]});
  add(key+'-tie',ability,{p1:[m(['Recover'])],setup:[{side:'p2',hp:80,status:'brn'}]});
 }
 for(const seed of [2,3,4,5,6,7])add('shed-seed-'+seed,'Shed Skin',{seed,setup:[{side:'p2',hp:80,status:'tox'}],steps:[b(),b(),b()]});
 add('shed-both-tie','Shed Skin',{p1:[m(['Recover'],{ability:'Shed Skin'})],setup:[{side:'p1',hp:80,status:'brn'},{side:'p2',hp:80,status:'tox'}]});
 add('cross-tie','Poison Heal',{p1:[m(['Recover'],{ability:'Shed Skin'})],setup:[{side:'p1',hp:80,status:'tox'},{side:'p2',hp:80,status:'tox'}]});
 add('poison-cap','Poison Heal',{setup:[{side:'p2',hp:174,status:'psn'}]});
 add('poison-full','Poison Heal',{setup:[{side:'p2',status:'tox'}]});
 add('poison-stage-cap','Poison Heal',{setup:[{side:'p2',hp:80,status:'tox',toxicStage:14}],steps:[b(),b(),b()]});
 add('poison-direct','Poison Heal',{setup:[{side:'p2',status:'psn'}],steps:[b(3),b(3)]});
 add('poison-mirrored','Pressure',{p1:[m(['Recover'],{ability:'Poison Heal',evs:{spe:252}})],setup:[{side:'p1',hp:80,status:'tox'}]});
 add('poison-minimum','Poison Heal',{p2:[m(['Growl'],{species:'Shedinja',ability:'Poison Heal'})],setup:[{side:'p2',status:'psn'}]});
 add('shed-low-hp','Shed Skin',{setup:[{side:'p2',hp:1,status:'brn'}],p2:[m(['Growl'],{ability:'Shed Skin'}),m(['Growl'])],steps:[b()]});
 return cases;
};
module.exports.membershipWitnesses=function(){
 require('./neutral-turn-oracle.cjs').assertReady();const ps=require('./vendor/pokemon-showdown/dist/sim'),assert=require('node:assert/strict');
 return [['Shed Skin',4,'Arbok'],['Poison Heal',38,'Breloom']].map(([ability,n,species])=>{
  const seed=[n,2,3,4],set=ps.Teams.getGenerator('gen5randombattle',seed).getTeam().find(s=>s.ability===ability&&s.species===species);assert.ok(set,ability);return {ability:ps.Dex.toID(ability),seed,set};
 });
};
