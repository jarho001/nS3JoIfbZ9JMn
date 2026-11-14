'use strict';
module.exports=function(){
 const m=(x={})=>({species:'Mew',moves:['Tackle','Surf','Gastro Acid','Recover'],ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...x});
 const b=(a=1,c=1)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[];
 const add=(id,x={})=>cases.push({id:'G4-hp-modifier-'+id,seed:1,generic:true,hpModifierFamily:true,p1:[m({evs:{spe:252}})],p2:[m()],steps:[b(),b()],...x});
 for(const move of ['Tackle','Surf']){
  const key=move.toLowerCase();
  for(const hp of [87,88,89,176])add('defeatist-'+key+'-'+hp,{p1:[m({ability:'Defeatist',evs:{hp:4,spe:252}})],setup:[{side:'p1',hp}],steps:[b(move==='Tackle'?1:2)]});
  add('defeatist-'+key+'-staged',{p1:[m({ability:'Defeatist',evs:{hp:4,spe:252}})],setup:[{side:'p1',hp:88,boosts:{atk:2,spa:2}}],steps:[b(move==='Tackle'?1:2)]});
  add('multiscale-'+key,{p2:[m({ability:'Multiscale'})],steps:[b(move==='Tackle'?1:2),b(move==='Tackle'?1:2)]});
  add('multiscale-'+key+'-chipped',{p2:[m({ability:'Multiscale'})],setup:[{side:'p2',hp:174}],steps:[b(move==='Tackle'?1:2)]});
  add('both-'+key,{p1:[m({ability:'Defeatist',evs:{hp:4,spe:252},item:'Life Orb'})],p2:[m({ability:'Multiscale'})],setup:[{side:'p1',hp:96}],steps:[b(move==='Tackle'?1:2),b(move==='Tackle'?1:2)]});
 }
 for(const ability of ['Sniper','Tinted Lens'])add('multiscale-'+ability.toLowerCase().replaceAll(' ','')+'-tie',{p1:[m({ability})],p2:[m({ability:'Multiscale'})]});
 add('multiscale-suppressed-tie',{p1:[m({ability:'Sniper'})],p2:[m({ability:'Multiscale'})],steps:[b(3),b()]});
 add('multiscale-multi',{p1:[m({moves:['Double Hit'],evs:{spe:252}})],p2:[m({ability:'Multiscale'})],steps:[b()]});
 add('multiscale-critical',{seed:21,p2:[m({ability:'Multiscale'})],steps:[b()]});
 add('defeatist-thickfat-tie',{p1:[m({ability:'Defeatist'})],p2:[m({ability:'Thick Fat'})],steps:[b(2)]});
 add('defeatist-suppressed-tie',{p1:[m({ability:'Defeatist'})],p2:[m({ability:'Thick Fat'})],steps:[b(4,3),b(2)]});
 for(const ability of ['Rough Skin','Mummy'])add('defeatist-multi-'+ability.toLowerCase().replaceAll(' ',''),{p1:[m({ability:'Defeatist',moves:['Double Hit'],evs:{hp:4,spe:252}})],p2:[m({ability})],setup:[{side:'p1',hp:ability==='Mummy'?88:89}],steps:[b()]});
 add('defeatist-choice',{p1:[m({ability:'Defeatist',item:'Choice Band',evs:{hp:4,spe:252}})],setup:[{side:'p1',hp:88}],steps:[b()]});
 add('sniper-suppressed-multiscale-tie',{p1:[m({ability:'Sniper'})],p2:[m({ability:'Multiscale'})],steps:[b(4,3),b()]});
 add('overgrow-suppressed-thickfat-tie',{p1:[m({ability:'Overgrow'})],p2:[m({ability:'Thick Fat'})],steps:[b(4,3),b(2)]});
 add('defeatist-target-suppressed-tie',{p1:[m({ability:'Defeatist'})],p2:[m({ability:'Thick Fat'})],steps:[b(3),b(2)]});
 return cases;
};
module.exports.membershipWitnesses=function(){
 require('./neutral-turn-oracle.cjs').assertReady();const ps=require('./vendor/pokemon-showdown/dist/sim'),assert=require('node:assert/strict');
 return [['Multiscale',10,'Dragonite'],['Defeatist',91,'Archeops']].map(([ability,n,species])=>{
  const seed=[n,2,3,4],set=ps.Teams.getGenerator('gen5randombattle',seed).getTeam().find(s=>s.ability===ability&&s.species===species);assert.ok(set,ability);return {kind:'ability',id:ps.Dex.toID(ability),seed,set};
 });
};
