'use strict';
// Native-only next group: direct-move survival and item consumption.
module.exports=function(){
 const m=(x={})=>({species:'Mew',moves:['Tackle','Recover','Gastro Acid','Double Hit'],ability:'Pressure',item:'',level:100,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...x});
 const b=(a=1,c=2)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[];
 const add=(id,x={})=>cases.push({id:'G4-survival-'+id,seed:1,generic:true,survivalFamily:true,p1:[m({species:'Snorlax',evs:{spe:252}})],p2:[m({level:1})],steps:[b(),b()],...x});
 for(const [key,ability,item]of [['sturdy','Sturdy',''],['sash','Pressure','Focus Sash'],['both','Sturdy','Focus Sash'],['control','Pressure','']]){
  const target=m({level:1,ability,item});
  add(key+'-full',{p2:[target],steps:key==='control'?[b()]:[b(),b()]});
  add(key+'-chipped',{p2:[target],setup:[{side:'p2',hp:1}],steps:[b()]});
  add(key+'-nonlethal',{p2:[m({ability,item})],steps:[b()]});
  add(key+'-multi',{p2:[target],steps:[b(4)]});
  add(key+'-suppressed',{p2:[target],steps:[b(3),b()]});
  add(key+'-burn',{p1:[m({moves:['Recover']})],p2:[target],setup:[{side:'p2',status:'brn',hp:1}],steps:[b(1,1)]});
 }
 for(const [key,ability,item]of [['sturdy','Sturdy',''],['sash','Pressure','Focus Sash'],['both','Sturdy','Focus Sash']])for(const move of ['Drain Punch','Double-Edge'])add(key+'-'+move.toLowerCase().replaceAll(/[^a-z]/g,''),{p1:[m({species:'Snorlax',moves:[move],evs:{spe:252}})],p2:[m({level:1,ability,item})],setup:[{side:'p1',hp:100}],steps:[b()]});
 for(const [key,ability,item]of [['sturdy','Sturdy',''],['sash','Pressure','Focus Sash']])add(key+'-one-hp',{p1:[m({species:'Snorlax',moves:['Aerial Ace'],evs:{spe:252}})],p2:[m({species:'Shedinja',moves:['Recover'],ability,item})],steps:[b(1,1)]});
 return cases;
};
module.exports.membershipWitnesses=function(){
 require('./neutral-turn-oracle.cjs').assertReady();const ps=require('./vendor/pokemon-showdown/dist/sim'),assert=require('node:assert/strict');
 return [['ability','Sturdy',14,'Crustle'],['item','Focus Sash',57,'Qwilfish']].map(([kind,name,n,species])=>{
  const seed=[n,2,3,4],set=ps.Teams.getGenerator('gen5randombattle',seed).getTeam().find(s=>s[kind]===name&&s.species===species);assert.ok(set,name);return {kind,id:ps.Dex.toID(name),seed,set};
 });
};
