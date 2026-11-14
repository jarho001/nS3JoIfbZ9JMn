'use strict';
module.exports=function(){
 const m=(x={})=>({species:'Mew',moves:['Tackle','Surf','Toxic','Recover'],ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...x});
 const b=(a=1,c=4)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[];
 const add=(id,x={})=>cases.push({id:'G4-faint-ability-'+id,seed:1,generic:true,faintAbilityFamily:true,p1:[m({ability:'Moxie',evs:{spe:252}})],p2:[m({level:1,ability:'Aftermath'}),m({level:1})],steps:[b()],...x});
 for(const ability of ['Pressure','Moxie','Magic Guard','Mummy'])for(const move of [1,2])add(ability.toLowerCase().replaceAll(' ','')+'-'+move,{p1:[m({ability,evs:{spe:252}})],steps:[b(move)]});
 add('moxie-last',{p2:[m({level:1})]});
 add('moxie-reserve',{p2:[m({level:1}),m({level:1})],steps:[b(),{p2:'switch 2'},b()]});
 for(const hp of [1,43,44])add('aftermath-source-hp-'+hp,{setup:[{side:'p1',hp}]});
 add('aftermath-terminal-double',{p2:[m({level:1,ability:'Aftermath'})],setup:[{side:'p1',hp:1}]});
 add('aftermath-helmet',{p2:[m({level:1,ability:'Aftermath',item:'Rocky Helmet'}),m({level:1})]});
 for(const stage of [5,6])add('moxie-cap-'+stage,{p2:[m({level:1}),m({level:1})],setup:[{side:'p1',boosts:{atk:stage}}]});
 add('moxie-poison',{p1:[m({ability:'Moxie',moves:['Recover'],evs:{spe:252}})],p2:[m({level:1,ability:'Aftermath',moves:['Tackle']}),m({level:1})],setup:[{side:'p2',hp:1,status:'psn'}],steps:[b(1,1)]});
 add('moxie-mummy',{p2:[m({level:1,ability:'Mummy'}),m({level:1})]});
 add('aftermath-suppressed',{p1:[m({ability:'Moxie',moves:['Gastro Acid','Tackle'],evs:{spe:252}})],steps:[b(),b(2)]});
 add('moxie-recoil-ko',{p1:[m({ability:'Moxie',moves:['Double-Edge'],evs:{spe:252}})],p2:[m({level:1}),m({level:1})],setup:[{side:'p1',hp:1}]});
 add('moxie-explosion',{p1:[m({ability:'Moxie',moves:['Explosion'],evs:{spe:252}})],p2:[m({level:1}),m({level:1})]});
 add('aftermath-multi',{p1:[m({ability:'Moxie',moves:['Double Hit'],evs:{spe:252}})]});
 add('moxie-suppressed',{p2:[m({level:1,ability:'Aftermath',moves:['Gastro Acid']}),m({level:1})],steps:[b(4,1),b(1,1)]});
 add('moxie-aftermath-orb',{p1:[m({ability:'Moxie',item:'Life Orb',evs:{spe:252}})],setup:[{side:'p1',hp:60}]});
 return cases;
};
module.exports.membershipWitnesses=function(){
 require('./neutral-turn-oracle.cjs').assertReady();const ps=require('./vendor/pokemon-showdown/dist/sim'),assert=require('node:assert/strict');
 return [['Aftermath',6,'Skuntank'],['Moxie',7,'Salamence']].map(([ability,n,species])=>{
  const seed=[n,2,3,4],set=ps.Teams.getGenerator('gen5randombattle',seed).getTeam().find(s=>s.ability===ability&&s.species===species);assert.ok(set,ability);return {kind:'ability',id:ps.Dex.toID(ability),seed,set};
 });
};
