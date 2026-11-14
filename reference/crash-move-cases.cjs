'use strict';
module.exports=function(){
 const m=(x={})=>({species:'Mew',moves:['Tackle','Recover'],ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...x});
 const b=(a=1,c=2)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[];
 const add=(id,x={})=>cases.push({id:'G4-crash-'+id,seed:1,generic:true,crashMoveFamily:true,p1:[m({evs:{spe:252}})],p2:[m()],steps:[b()],...x});
 for(const move of ['Jump Kick','High Jump Kick']){
  const key=move.toLowerCase().replaceAll(' ','');const source=(x={})=>m({moves:[move,'Recover'],evs:{spe:252},...x});
  add(key+'-hit',{p1:[source()]});
  add(key+'-protect',{p1:[source()],p2:[m({moves:['Protect']})],steps:[b(1,1)]});
  add(key+'-ghost',{p1:[source()],p2:[m({species:'Gengar'})]});
  for(const ability of ['Magic Guard','Rock Head','Reckless','No Guard'])add(key+'-'+ability.toLowerCase().replaceAll(' ',''),{p1:[source({ability})],p2:[m({moves:['Protect']})],steps:[b(1,1)]});
  add(key+'-low-hp',{p1:[source()],p2:[m({moves:['Protect']})],setup:[{side:'p1',hp:50}],steps:[b(1,1)]});
  add(key+'-reserve',{p1:[source(),m()],p2:[m({moves:['Protect']})],setup:[{side:'p1',hp:50}],steps:[b(1,1),{p1:'switch 2'}]});
  add(key+'-reckless-hit',{p1:[source({ability:'Reckless'})]});
  add(key+'-lifeorb',{p1:[source({item:'Life Orb'})],p2:[m({moves:['Protect']})],steps:[b(1,1)]});
  add(key+'-helmet',{p1:[source()],p2:[m({item:'Rocky Helmet'})]});
  add(key+'-one-hp-sash',{p1:[source({species:'Shedinja',item:'Focus Sash'})],p2:[m({moves:['Protect']})],steps:[b(1,1)]});
  add(key+'-even-hp',{p1:[source({evs:{spe:252,hp:4}})],p2:[m({moves:['Protect']})],steps:[b(1,1)]});
  add(key+'-fly',{p1:[source({evs:{spe:0}})],p2:[m({moves:['Fly'],evs:{spe:252}})],steps:[b(1,1),b(1,1)]});
  add(key+'-wonderguard',{p1:[source()],p2:[m({ability:'Wonder Guard'})]});
  add(key+'-noguard-hit',{seed:15,p1:[source({ability:'No Guard'})]});
  add(key+'-sleep-talk',{seed:4,p1:[source({moves:['Rest','Sleep Talk',move]})],p2:[m({moves:['Protect','Recover']})],setup:[{side:'p1',hp:100}],steps:[b(1,2),b(2,1)]});
  add(key+'-no-target',{p1:[source({species:'Registeel',level:100,evs:{spe:0}})],p2:[m({moves:['Explosion'],level:100,evs:{spe:252}}),m()],steps:[b(1,1)]});
  add(key+'-paralysis',{seed:21,p1:[source()],setup:[{side:'p1',status:'par'}]});
  for(const seed of [13,15])add(key+'-seed-'+seed,{seed,p1:[source()]});
 }
 return cases;
};
module.exports.membershipWitnesses=function(){
 require('./neutral-turn-oracle.cjs').assertReady();const ps=require('./vendor/pokemon-showdown/dist/sim'),assert=require('node:assert/strict');
 return [['jumpkick',4,'Lopunny'],['highjumpkick',20,'Mienshao']].map(([id,n,species])=>{const seed=[n,2,3,4],set=ps.Teams.getGenerator('gen5randombattle',seed).getTeam().find(s=>s.species===species&&s.moves.some(m=>ps.Dex.toID(m)===id));assert.ok(set,id);return {kind:'move',id,seed,set};});
};
