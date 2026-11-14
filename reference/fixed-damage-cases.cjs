'use strict';
module.exports=function(){
 const m=(x={})=>({species:'Mew',moves:['Tackle','Recover'],ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...x});
 const b=(a=1,c=1)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[];
 const add=(id,x={})=>cases.push({id:'G4-fixed-damage-'+id,seed:1,generic:true,fixedDamageFamily:true,p1:[m({evs:{spe:252}})],p2:[m()],steps:[b()],...x});
 for(const [key,move]of [['superfang','Super Fang'],['seismictoss','Seismic Toss']]){
  const user=(x={})=>m({moves:[move,'Recover'],evs:{spe:252},...x});
  add(key,{p1:[user()],steps:[b(),b()]});
  add(key+'-partial',{p1:[user()],setup:[{side:'p2',hp:101}]});
  add(key+'-onehp',{p1:[user()],setup:[{side:'p2',hp:1}]});
  add(key+'-ghost',{p1:[user()],p2:[m({species:'Gengar'})]});
  add(key+'-protect',{p1:[user()],p2:[m({moves:['Protect','Tackle']})]});
  add(key+'-burn-boosts',{p1:[user()],p2:[m()],setup:[{side:'p1',status:'brn',boosts:{atk:6,spa:6}},{side:'p2',boosts:{def:6,spd:6}}]});
  add(key+'-multiscale',{p1:[user()],p2:[m({ability:'Multiscale'})]});
  add(key+'-filter-eviolite',{p1:[user({item:'Expert Belt'})],p2:[m({species:'Vigoroth',ability:'Filter',item:'Eviolite'})]});
  add(key+'-reflect',{p1:[user()],p2:[m({moves:['Reflect','Recover']})],steps:[b(2,1),b()]});
  add(key+'-substitute',{p1:[user()],p2:[m({moves:['Substitute','Tackle']})],steps:[b(2,1),b(1,2),b(1,2)]});
  add(key+'-helmet',{p1:[user()],p2:[m({item:'Rocky Helmet'})]});
  add(key+'-helmet-source-ko',{p1:[user()],p2:[m({item:'Rocky Helmet'})],setup:[{side:'p1',hp:1}]});
  add(key+'-lifeorb',{p1:[user({item:'Life Orb'})]});
  add(key+'-magicguard-lifeorb',{p1:[user({ability:'Magic Guard',item:'Life Orb'})]});
  add(key+'-mummy',{abilityChangeFamily:true,p1:[user({ability:'Technician'})],p2:[m({ability:'Mummy'})],steps:[b(),b()]});
  add(key+'-dig',{p1:[user()],p2:[m({moves:['Dig','Tackle']})],steps:[b(2,1),b()]});
  add(key+'-reserve',{p1:[user()],p2:[m(),m()],setup:[{side:'p2',hp:1}],steps:[b(),{p2:'switch 2'},b()]});
 }
 add('superfang-miss',{seed:13,p1:[m({moves:['Super Fang','Recover'],evs:{spe:252}})]});
 for(const level of [1,100])add('seismictoss-level'+level,{p1:[m({moves:['Seismic Toss','Recover'],level,evs:{spe:252}})],p2:[m({moves:['Growl','Tackle']})]});
 add('seismictoss-sash',{p1:[m({moves:['Seismic Toss','Recover'],level:100,evs:{spe:252}})],p2:[m({item:'Focus Sash',level:1})]});
 add('seismictoss-sturdy',{p1:[m({moves:['Seismic Toss','Recover'],level:100,evs:{spe:252}})],p2:[m({ability:'Sturdy',level:1})]});
 add('seismictoss-gem-unburden',{p1:[m({moves:['Seismic Toss','Recover'],item:'Fighting Gem',ability:'Unburden',evs:{spe:252}})],steps:[b(),b()]});
 add('seismictoss-gem-ghost',{p1:[m({moves:['Seismic Toss','Recover'],item:'Fighting Gem',ability:'Unburden',evs:{spe:252}})],p2:[m({species:'Gengar'})]});
 add('seismictoss-gem-terminal',{p1:[m({moves:['Seismic Toss','Recover'],item:'Fighting Gem',ability:'Unburden',level:100,evs:{spe:252}})],p2:[m({level:1})]});
 add('seismictoss-moxie-terminal',{p1:[m({moves:['Seismic Toss','Recover'],ability:'Moxie',level:100,evs:{spe:252}})],p2:[m({level:1})]});
 add('superfang-wonderguard',{p1:[m({moves:['Super Fang','Recover'],evs:{spe:252}})],p2:[m({ability:'Wonder Guard'})]});
 add('seismictoss-wonderguard-super',{p1:[m({moves:['Seismic Toss','Recover'],evs:{spe:252}})],p2:[m({species:'Vigoroth',ability:'Wonder Guard'})]});
 add('superfang-static',{p1:[m({moves:['Super Fang','Recover'],evs:{spe:252}})],p2:[m({ability:'Static'})]});
 add('seismictoss-aftermath-terminal',{p1:[m({moves:['Seismic Toss','Recover'],evs:{spe:252}})],p2:[m({ability:'Aftermath'})],setup:[{side:'p1',hp:1},{side:'p2',hp:1}]});
 add('superfang-sleep-talk',{seed:4,p1:[m({moves:['Rest','Sleep Talk','Super Fang'],evs:{spe:252}})],p2:[m({moves:['Recover','Tackle']})],setup:[{side:'p1',hp:100}],steps:[b(1,1),b(2,1)]});
 return cases;
};
module.exports.membershipWitnesses=function(){
 require('./neutral-turn-oracle.cjs').assertReady();const ps=require('./vendor/pokemon-showdown/dist/sim'),assert=require('node:assert/strict');return [['superfang',14,'Crobat'],['seismictoss',36,'Blissey']].map(([id,n,species])=>{const seed=[n,2,3,4],set=ps.Teams.getGenerator('gen5randombattle',seed).getTeam().find(s=>s.species===species&&s.moves.some(m=>ps.Dex.toID(m)===id));assert.ok(set,id);return {kind:'move',id,seed,set};});
};
