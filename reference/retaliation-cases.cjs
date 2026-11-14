'use strict';
module.exports=function(){
 const m=(x={})=>({species:'Mew',moves:['Tackle','Recover'],ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...x});
 const b=(a=1,c=1)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[];
 const add=(id,x={})=>cases.push({id:'G4-retaliation-'+id,seed:1,generic:true,retaliationFamily:true,p1:[m({evs:{spe:252}})],p2:[m()],steps:[b()],...x});
 for(const [key,move,attack,wrong,immune]of [['counter','Counter','Tackle','Surf','Gengar'],['mirrorcoat','Mirror Coat','Surf','Tackle','Umbreon']]){
  const user=(x={})=>m({moves:[move,'Recover'],evs:{spe:252},...x}),foe=(x={})=>m({moves:[attack,'Recover'],...x});
  add(key,{p1:[user()],p2:[foe()],steps:[b(),b()]});
  add(key+'-struggle-slot4',{p1:[user({moves:['Tackle','Recover','Protect',move]})],p2:[foe()],setup:[{side:'p1',pp:0}]});
  add(key+'-no-damage',{p1:[user()],p2:[foe()],steps:[b(1,2)]});
  add(key+'-wrong-category',{p1:[user()],p2:[foe({moves:[wrong,'Recover']})]});
  add(key+'-immune-target',{p1:[user()],p2:[foe({species:immune})]});
  add(key+'-one-hp',{p1:[user()],p2:[foe()],setup:[{side:'p1',hp:1}]});
  add(key+'-burned-attacker',{p1:[user()],p2:[foe()],setup:[{side:'p2',status:'brn'}]});
  add(key+'-boosted-attacker',{p1:[user()],p2:[foe()],setup:[{side:'p2',boosts:{atk:1,spa:1}}]});
  add(key+'-sash-unburden',{p1:[user({item:'Focus Sash',ability:'Unburden',level:1})],p2:[foe({level:100})]});
  add(key+'-target-substitute',{p1:[user()],p2:[foe({moves:['Substitute',attack,'Recover']})],steps:[b(2,1),b(1,2)]});
  add(key+'-own-substitute',{p1:[user({moves:['Substitute',move,'Recover']})],p2:[foe()],steps:[b(),b(2),b(2)]});
  add(key+'-helmet',{p1:[user()],p2:[foe({item:'Rocky Helmet'})]});
  add(key+'-lifeorb',{p1:[user({item:'Life Orb'})],p2:[foe()]});
  add(key+'-mummy',{abilityChangeFamily:true,p1:[user({ability:'Technician'})],p2:[foe({ability:'Mummy'})]});
  add(key+'-reserve',{p1:[user()],p2:[foe(),foe()],setup:[{side:'p2',hp:1}],steps:[b(),{p2:'switch 2'},b()]});
  add(key+'-helmet-terminal',{p1:[user({item:'Rocky Helmet'})],p2:[foe({moves:[key==='counter'?attack:'Grass Knot','Recover']})],setup:[{side:'p2',hp:1}]});
  add(key+'-blank-terminal',{p1:[user()],p2:[foe({item:'Life Orb',moves:[wrong,'Recover']})],setup:[{side:'p2',hp:1}]});
  add(key+'-terminal',{p1:[user()],p2:[foe()],setup:[{side:'p2',hp:1}]});
  add(key+'-queued-fail',{p1:[user()],p2:[foe()],steps:[b(),b(1,2)]});
   for(const [tag,incoming]of [['frozen',attack],['fire-thaw',key==='counter'?'Fire Punch':'Flamethrower']])add(key+'-'+tag,{p1:[user()],p2:[foe({moves:[incoming,'Recover']})],setup:[{side:'p1',status:'frz'}]});
  add(key+'-sleep-talk',{seed:2,p1:[user({moves:['Rest','Sleep Talk',move],evs:{spe:0}})],p2:[foe({evs:{spe:252}})],setup:[{side:'p1',hp:100}],steps:[b(),b(2)]});
 }
 add('counter-doublekick',{p1:[m({moves:['Counter','Recover'],evs:{spe:252}})],p2:[m({moves:['Double Kick','Recover']})]});
 add('counter-both',{p1:[m({moves:['Counter','Recover']})],p2:[m({moves:['Counter','Recover']})]});
 add('mirrorcoat-both',{p1:[m({moves:['Mirror Coat','Recover']})],p2:[m({moves:['Mirror Coat','Recover']})]});
 add('counter-mirrorcoat-tie',{p1:[m({moves:['Counter','Recover']})],p2:[m({moves:['Mirror Coat','Recover']})]});
 for(const species of ['Mew','Gengar'])add('counter-pivot-'+species.toLowerCase(),{p1:[m({moves:['Counter','Recover'],evs:{spe:252}})],p2:[m({moves:['U-turn','Recover']}),m({species})],steps:[b(),{p2:'switch 2'}]});
 add('counter-pursuit-tie',{p1:[m({moves:['Counter','Recover']})],p2:[m({moves:['Pursuit','Recover']})]});
 add('legacy-pressure',{p1:[m({evs:{spe:252}})],p2:[m()]});
 return cases;
};
module.exports.membershipWitnesses=function(){
 require('./neutral-turn-oracle.cjs').assertReady();const ps=require('./vendor/pokemon-showdown/dist/sim'),assert=require('node:assert/strict');return ['counter','mirrorcoat'].map(id=>{const seed=[42,2,3,4],set=ps.Teams.getGenerator('gen5randombattle',seed).getTeam().find(s=>s.species==='Wobbuffet'&&s.moves.some(m=>ps.Dex.toID(m)===id));assert.ok(set,id);return {kind:'move',id,seed,set};});
};
