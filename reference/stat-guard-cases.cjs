'use strict';
module.exports=function(){
 const m=(x={})=>({species:'Mew',moves:['Growl','Recover'],ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...x});
 const b=(a=1,c=2)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[];
 const add=(id,x)=>cases.push({id:'G4-stat-guard-'+id,seed:1,generic:true,statGuardFamily:true,p1:[m({evs:{spe:252}})],p2:[m()],steps:[b()],...x});
 for(const [key,ability]of [['pecks','Big Pecks'],['cutter','Hyper Cutter']]){
  for(const move of ['Growl','Charm','Screech','Tickle','Metal Sound'])add(key+'-'+move.toLowerCase().replace(' ',''),{p1:[m({moves:[move,'Recover'],evs:{spe:252}})],p2:[m({ability})]});
  add(key+'-capped',{p1:[m({moves:['Tickle','Recover'],evs:{spe:252}})],p2:[m({ability})],setup:[{side:'p2',boosts:{atk:-6,def:-6}}]});
  add(key+'-self',{p2:[m({ability,moves:['Shell Smash','Recover']})],steps:[b(2,1)]});
  add(key+'-suppressed',{abilityChangeFamily:true,p1:[m({moves:['Gastro Acid','Tickle','Recover'],evs:{spe:252}})],p2:[m({ability})],steps:[b(),b(2)]});
  add(key+'-intimidate',{p1:[m({ability:'Intimidate',evs:{spe:252}})],p2:[m({ability})]});
  add(key+'-switch-intimidate',{p1:[m(),m({ability:'Intimidate'})],p2:[m({ability})],steps:[{p1:'switch 2',p2:'move 2'}]});
  add(key+'-herb',{p1:[m({moves:['Tickle','Recover'],evs:{spe:252}})],p2:[m({ability,item:'White Herb'})]});
  add(key+'-reflected',{p1:[m({ability,moves:['Tickle','Recover'],evs:{spe:252}})],p2:[m({ability:'Magic Bounce'})]});
  add(key+'-roleplay',{abilityChangeFamily:true,p1:[m({moves:['Role Play','Recover'],evs:{spe:252}})],p2:[m({ability,moves:['Tickle','Recover']})],steps:[b(1,1)]});
  add(key+'-swap',{abilityChangeFamily:true,p1:[m({moves:['Skill Swap','Recover'],evs:{spe:252}})],p2:[m({ability,moves:['Tickle','Recover']})],steps:[b(1,1)]});
  add(key+'-trace',{p1:[m({ability:'Trace',moves:['Recover'],evs:{spe:252}})],p2:[m({ability,moves:['Tickle','Recover']})],steps:[b(1,1)]});
 }
 add('clear-growl-control',{p2:[m({ability:'Clear Body'})],steps:[b(),b()]});
 add('clear-intimidate-control',{p1:[m({ability:'Intimidate',evs:{spe:252}})],p2:[m({ability:'Clear Body'})],steps:[b(),b()]});
 add('clear-capped-control',{p1:[m({moves:['Tickle','Recover'],evs:{spe:252}})],p2:[m({ability:'Clear Body'})],setup:[{side:'p2',boosts:{atk:-6,def:-6}}]});
 for(const [key,ability]of [['cutter','Hyper Cutter'],['clear','Clear Body']])add(key+'-capped-intimidate',{p1:[m(),m({ability:'Intimidate'})],p2:[m({ability})],setup:[{side:'p2',boosts:{atk:-6}}],steps:[{p1:'switch 2',p2:'move 2'}]});
 return cases;
};
module.exports.membershipWitnesses=function(){
 require('./neutral-turn-oracle.cjs').assertReady();const ps=require('./vendor/pokemon-showdown/dist/sim'),assert=require('node:assert/strict');return [['Big Pecks',16,'Pidgeot'],['Hyper Cutter',207,'Kingler']].map(([name,n,species])=>{const seed=[n,2,3,4],set=ps.Teams.getGenerator('gen5randombattle',seed).getTeam().find(s=>s.species===species&&s.ability===name);assert.ok(set,name);return {kind:'ability',id:ps.Dex.toID(name),seed,set};});
};
