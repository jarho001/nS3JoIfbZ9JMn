'use strict';
module.exports=function(){
 const m=(x={})=>({species:'Mew',moves:['Tackle','Recover'],ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...x});
 const b=(a=1,c=2)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[];
 const add=(id,x)=>cases.push({id:'G4-confusion-secondary-'+id,seed:1,generic:true,confusionSecondaryFamily:true,p1:[m()],p2:[m()],steps:[b()],...x});
 for(const [key,move,seed]of [['signal','Signal Beam',5],['hurricane','Hurricane',5],['punch','Dynamic Punch',1]]){
  const source=(x={})=>m({moves:[move,'Recover'],ability:'No Guard',evs:{spe:252},...x});
  add(key+'-positive',{seed,p1:[source()]});
  add(key+'-followup',{seed,p1:[source()],steps:[b(),b(2,1),b(2,1)]});
  if(key!=='punch')add(key+'-negative',{p1:[source()]});
  for(const ability of ['Own Tempo','Shield Dust','Magic Guard'])add(key+'-'+ability.toLowerCase().replace(' ',''),{seed,p1:[source()],p2:[m({ability})]});
  add(key+'-grace',{seed,p1:[source({ability:'Serene Grace'})]});
  add(key+'-lum',{seed,p1:[source()],p2:[m({item:'Lum Berry'})]});
  add(key+'-substitute',{seed,p1:[source()],p2:[m({moves:['Substitute','Recover']})],steps:[b(2,1),b()]});
  add(key+'-protect',{seed,p1:[source()],p2:[m({moves:['Protect','Recover']})],steps:[b(1,1)]});
  add(key+'-existing',{seed,p1:[source({moves:['Confuse Ray',move,'Recover']})],steps:[b(),b(2)]});
  add(key+'-terminal',{seed,p1:[source()],p2:[m({level:1})]});
  add(key+'-sleep-talk',{seed:1,p1:[source({moves:['Rest','Sleep Talk',move],evs:{spe:0}})],p2:[m({evs:{spe:252}})],setup:[{side:'p1',hp:100}],steps:[b(1,1),b(2,1)]});
 }
 for(const weather of ['Rain Dance','Sunny Day','Hail'])add('hurricane-'+weather.toLowerCase().replace(' ',''),{seed:5,p1:[m({moves:[weather,'Hurricane','Recover'],evs:{spe:252}})],steps:[b(),b(2)]});
 add('punch-ghost',{p1:[m({moves:['Dynamic Punch','Recover'],ability:'No Guard',evs:{spe:252}})],p2:[m({species:'Dusknoir'})]});
 add('punch-helmet',{p1:[m({moves:['Dynamic Punch','Recover'],ability:'No Guard',evs:{spe:252}})],p2:[m({item:'Rocky Helmet'})]});
 add('punch-mummy',{abilityChangeFamily:true,p1:[m({moves:['Dynamic Punch','Recover'],ability:'No Guard',evs:{spe:252}})],p2:[m({ability:'Mummy'})]});
 return cases;
};
module.exports.membershipWitnesses=function(){
 require('./neutral-turn-oracle.cjs').assertReady();const ps=require('./vendor/pokemon-showdown/dist/sim'),assert=require('node:assert/strict');return [['signalbeam',2,'Beheeyem'],['hurricane',13,'Moltres'],['dynamicpunch',44,'Golurk']].map(([id,n,species])=>{const seed=[n,2,3,4],set=ps.Teams.getGenerator('gen5randombattle',seed).getTeam().find(s=>s.species===species&&s.moves.some(m=>ps.Dex.toID(m)===id));assert.ok(set,id);return {kind:'move',id,seed,set};});
};
