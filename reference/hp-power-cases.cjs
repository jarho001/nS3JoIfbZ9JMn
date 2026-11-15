'use strict';
module.exports=function(){
 const m=(x={})=>({species:'Mew',moves:['Tackle','Recover'],ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...x});
 const b=(a=1,c=2)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[];
 const add=(id,x={})=>cases.push({id:'G4-hp-power-'+id,seed:1,generic:true,hpPowerFamily:true,p1:[m({evs:{spe:252}})],p2:[m()],steps:[b()],...x});
 for(const [key,move,ability,absorb]of [['eruption','Eruption','Blaze','Flash Fire'],['waterspout','Water Spout','Torrent','Water Absorb']]){
  const user=(x={})=>m({moves:[move,'Recover'],evs:{spe:252},...x});
  for(const hp of [175,174,100,87,71,2,1])add(key+'-hp'+hp,{p1:[user()],setup:[{side:'p1',hp}]});
  for(const hp of [69,70,71,72])add(key+'-technician-hp'+hp,{p1:[user({ability:'Technician'})],setup:[{side:'p1',hp}]});
  for(const hp of [58,59])add(key+'-ability-hp'+hp,{p1:[user({ability})],setup:[{side:'p1',hp}]});
  add(key+'-slower',{p1:[user({evs:{spe:0}})],p2:[m({evs:{spe:252}})],setup:[{side:'p1',hp:100}],steps:[b(1,1)]});
  add(key+'-sitrus',{p1:[user({item:'Sitrus Berry',evs:{spe:0}})],p2:[m({evs:{spe:252}})],setup:[{side:'p1',hp:100}],steps:[b(1,1)]});
  add(key+'-burn',{p1:[user()],setup:[{side:'p1',status:'brn'}],steps:[b(),b()]});
  add(key+'-specs',{p1:[user({item:'Choice Specs'})],setup:[{side:'p1',hp:87}]});
  add(key+'-technician-glasses',{p1:[user({ability:'Technician',item:'Wise Glasses'})],setup:[{side:'p1',hp:71}]});
  add(key+'-custap',{p1:[user({ability:'Unburden',item:'Custap Berry',evs:{spe:0}})],p2:[m({evs:{spe:252}})],setup:[{side:'p1',hp:43}],steps:[b(1,1)]});
  for(const weather of ['Sunny Day','Rain Dance'])add(key+'-'+weather.toLowerCase().replaceAll(' ',''),{p1:[user({moves:[weather,move,'Recover']})],setup:[{side:'p1',hp:87}],steps:[b(),b(2)]});
  add(key+'-sleep-talk',{seed:2,p1:[user({moves:['Rest','Sleep Talk',move],evs:{spe:0}})],p2:[m({evs:{spe:252}})],setup:[{side:'p1',hp:100}],steps:[b(1,1),b(2,1)]});
  add(key+'-substitute',{p1:[user({moves:[move,'Protect']})],p2:[m({moves:['Substitute','Recover']})],setup:[{side:'p1',hp:87}],steps:[b(2,1),b()]});
  add(key+'-absorbed',{p1:[user()],p2:[m({ability:absorb})],setup:[{side:'p1',hp:87},{side:'p2',hp:100}]});
  add(key+'-protect',{p1:[user()],p2:[m({moves:['Protect','Recover']})],setup:[{side:'p1',hp:87}],steps:[b(1,1)]});
  add(key+'-target-hp',{p1:[user()],setup:[{side:'p1',hp:87},{side:'p2',hp:75}]});
  add(key+'-terminal',{p1:[user()],p2:[m({level:1})],setup:[{side:'p1',hp:87}]});
 }
 return cases;
};
module.exports.membershipWitnesses=function(){
 require('./neutral-turn-oracle.cjs').assertReady();const ps=require('./vendor/pokemon-showdown/dist/sim'),assert=require('node:assert/strict');return [['eruption',20,'Typhlosion'],['waterspout',36,'Wailord']].map(([id,n,species])=>{const seed=[n,2,3,4],set=ps.Teams.getGenerator('gen5randombattle',seed).getTeam().find(s=>s.species===species&&s.moves.some(m=>ps.Dex.toID(m)===id));assert.ok(set,id);return {kind:'move',id,seed,set};});
};
