'use strict';
module.exports=function(){
 const m=(x={})=>({species:'Mew',moves:['Tackle','Recover'],ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...x});
 const b=(a=1,c=1)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[];
 const add=(id,x={})=>cases.push({id:'G4-damage-item-'+id,seed:1,generic:true,damageItemFamily:true,p1:[m({evs:{spe:252}})],p2:[m()],steps:[b()],...x});
 for(const [key,move]of [['physical','Tackle'],['special','Surf']]){
  for(const [species,tag]of [['Vigoroth','nfe'],['Mew','fully-evolved']])for(const item of ['Eviolite',''])add(`eviolite-${key}-${tag}-${item?'held':'empty'}`,{p1:[m({moves:[move,'Recover'],evs:{spe:252}})],p2:[m({species,item})]});
  add(`eviolite-${key}-boosted`,{p1:[m({moves:[move,'Recover'],evs:{spe:252}})],p2:[m({species:'Vigoroth',item:'Eviolite'})],setup:[{side:'p2',boosts:{def:2,spd:2}}]});
 }
 add('eviolite-critical',{p1:[m({moves:['Storm Throw','Recover'],evs:{spe:252}})],p2:[m({species:'Vigoroth',item:'Eviolite'})],setup:[{side:'p2',boosts:{def:2}}]});
 add('eviolite-foulplay',{p1:[m({moves:['Foul Play','Recover'],evs:{spe:252}})],p2:[m({species:'Vigoroth',item:'Eviolite'})]});
 add('eviolite-psyshock',{p1:[m({moves:['Psyshock','Recover'],evs:{spe:252}})],p2:[m({species:'Vigoroth',item:'Eviolite'})]});
 add('eviolite-knockoff',{itemLifecycleFamily:true,p1:[m({moves:['Knock Off','Tackle','Recover'],evs:{spe:252}})],p2:[m({species:'Vigoroth',item:'Eviolite'})],steps:[b(),b(2)]});
 for(const species of ['Vigoroth','Mew'])add('eviolite-transform-'+species.toLowerCase(),{identityFamily:true,p1:[m({species,item:'Eviolite',moves:['Transform','Recover'],evs:{spe:252}})],p2:[m({species:species==='Mew'?'Vigoroth':'Mew'})],steps:[b(1,2),b(2,1)]});
 add('eviolite-substitute',{p1:[m({moves:['Recover','Tackle'],evs:{spe:252}})],p2:[m({species:'Vigoroth',item:'Eviolite',moves:['Substitute','Recover']})],steps:[b(1,1),b(2,2)]});
 for(const [key,move,species]of [['super','Surf','Arcanine'],['neutral','Surf','Mew'],['resist','Surf','Blastoise'],['immune','Thunderbolt','Rhydon'],['quad','Surf','Rhydon'],['net-neutral','Ice Beam','Walrein'],['physical','Low Kick','Regice']])for(const item of ['Expert Belt',''])add(`expertbelt-${key}-${item?'held':'empty'}`,{p1:[m({item,moves:[move,'Recover'],evs:{spe:252}})],p2:[m({species,level:key==='quad'?100:50})]});
 add('expertbelt-hiddenpower',{p1:[m({item:'Expert Belt',moves:['Hidden Power','Recover'],evs:{spe:252},ivs:{hp:30,atk:30,def:30,spa:30,spd:30,spe:30}})],p2:[m({species:'Regice'})]});
 add('expertbelt-multihit',{p1:[m({item:'Expert Belt',moves:['Double Kick','Recover'],evs:{spe:252}})],p2:[m({species:'Regice'})]});
 add('expertbelt-critical',{p1:[m({item:'Expert Belt',moves:['Storm Throw','Recover'],evs:{spe:252}})],p2:[m({species:'Regice'})]});
 add('expertbelt-filter',{p1:[m({item:'Expert Belt',moves:['Surf','Recover'],evs:{spe:252}})],p2:[m({species:'Arcanine',ability:'Filter'})]});
 add('expertbelt-tintedlens',{p1:[m({item:'Expert Belt',ability:'Tinted Lens',moves:['Surf','Recover'],evs:{spe:252}})],p2:[m({species:'Blastoise'})]});
 add('expertbelt-substitute',{p1:[m({item:'Expert Belt',moves:['Surf','Recover'],evs:{spe:252}})],p2:[m({species:'Arcanine',moves:['Substitute','Recover']})],steps:[b(2,1),b()]});
 add('expertbelt-trick',{itemLifecycleFamily:true,p1:[m({item:'Expert Belt',moves:['Trick','Surf','Recover'],evs:{spe:252}})],p2:[m({species:'Arcanine'})],steps:[b(),b(2)]});
 add('expertbelt-eviolite',{p1:[m({item:'Expert Belt',moves:['Low Kick','Recover'],evs:{spe:252}})],p2:[m({species:'Vigoroth',item:'Eviolite'})],steps:[b(),b()]});
 return cases;
};
module.exports.membershipWitnesses=function(){
 require('./neutral-turn-oracle.cjs').assertReady();const ps=require('./vendor/pokemon-showdown/dist/sim'),assert=require('node:assert/strict');return [['expertbelt',2,'Seaking'],['eviolite',13,'Vigoroth']].map(([id,n,species])=>{const seed=[n,2,3,4],set=ps.Teams.getGenerator('gen5randombattle',seed).getTeam().find(s=>s.species===species&&ps.Dex.toID(s.item)===id);assert.ok(set,id);return {kind:'item',id,seed,set};});
};
