'use strict';
module.exports=function(){
 const m=(x={})=>({species:'Mew',moves:['Tackle','Recover'],ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...x});
 const b=(a=1,c=1)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[];
 const add=(id,x={})=>cases.push({id:'G4-conditional-attack-'+id,seed:1,generic:true,conditionalAttackFamily:true,p1:[m({evs:{spe:252}})],p2:[m()],steps:[b()],...x});
 for(const move of ['Sucker Punch','Payback']){
  const key=move.toLowerCase().replaceAll(' ','');const source=(x={})=>m({moves:[move,'Recover'],evs:{spe:252},...x});
  for(const other of ['Tackle','Recover','Protect','Mach Punch','Hyper Beam'])add(key+'-vs-'+other.toLowerCase().replaceAll(' ',''),{p1:[source()],p2:[m({moves:[other,other==='Recover'?'Tackle':'Recover']})]});
  add(key+'-slow',{p1:[source({evs:{spe:0}})],p2:[m({evs:{spe:252}})]});
  add(key+'-tie',{p1:[source({evs:{spe:0}})]});
  add(key+'-priority-first',{p1:[source({evs:{spe:0}})],p2:[m({moves:['Mach Punch'],evs:{spe:252}})]});
  add(key+'-switch',{p1:[source()],p2:[m(),m({species:'Snorlax'})],steps:[{p1:'move 1',p2:'switch 2'}]});
  add(key+'-pivot',{p1:[source({evs:{spe:0}})],p2:[m({moves:['U-turn'],evs:{spe:252}}),m({species:'Snorlax'})],steps:[b(),{p2:'switch 2'}]});
  add(key+'-recharge',{p1:[source()],p2:[m({moves:['Hyper Beam']})],steps:[b(2),b()]});
  add(key+'-paralyzed-target',{p1:[source()],setup:[{side:'p2',status:'par'}]});
  add(key+'-ghost',{p1:[source()],p2:[m({species:'Gengar'})]});
  add(key+'-choice',{p1:[source({item:'Choice Band'})]});
  add(key+'-slow-technician',{p1:[source({ability:'Technician',evs:{spe:0}})],p2:[m({evs:{spe:252}})]});
  add(key+'-slow-sleep-talk',{seed:4,p1:[source({moves:['Rest','Sleep Talk',move],evs:{spe:0}})],p2:[m({moves:['Tackle','Recover'],evs:{spe:252}})],setup:[{side:'p1',hp:100}],steps:[b(1,2),b(2)]});
  add(key+'-technician',{p1:[source({ability:'Technician'})]});
  add(key+'-substitute',{p1:[source()],p2:[m({moves:['Substitute','Tackle'],evs:{spe:0}})],steps:[b(2),b(1,2)]});
  add(key+'-sleep-talk',{seed:4,p1:[source({moves:['Rest','Sleep Talk',move]})],p2:[m({moves:['Tackle','Recover']})],setup:[{side:'p1',hp:100}],steps:[b(1,2),b(2)]});
 }
 return cases;
};
module.exports.membershipWitnesses=function(){
 require('./neutral-turn-oracle.cjs').assertReady();const ps=require('./vendor/pokemon-showdown/dist/sim'),assert=require('node:assert/strict');
 return [['suckerpunch',3,'Arbok'],['payback',45,'Ambipom']].map(([id,n,species])=>{const seed=[n,2,3,4],set=ps.Teams.getGenerator('gen5randombattle',seed).getTeam().find(s=>s.species===species&&s.moves.some(m=>ps.Dex.toID(m)===id));assert.ok(set,id);return {kind:'move',id,seed,set};});
};
