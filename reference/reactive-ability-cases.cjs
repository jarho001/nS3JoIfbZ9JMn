'use strict';
module.exports=function(){
 const m=(x={})=>({species:'Mew',moves:['Tackle','Surf','Gastro Acid','Recover'],ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...x});
 const b=(a=1,c=4)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[];
 const add=(id,x={})=>cases.push({id:'G4-reactive-ability-'+id,seed:1,generic:true,reactiveAbilityFamily:true,p1:[m({evs:{spe:252}})],p2:[m()],steps:[b()],...x});
 for(const ability of ['Weak Armor','Cursed Body']){
  const key=ability.toLowerCase().replaceAll(' ','');
  for(const move of ['Tackle','Surf','Double Hit'])add(key+'-'+move.toLowerCase().replaceAll(' ',''),{p1:[m({moves:[move,'Recover'],evs:{spe:252}})],p2:[m({ability})],steps:[b(),b(ability==='Cursed Body'?2:1)]});
  add(key+'-suppressed',{p2:[m({ability})],steps:[b(3),b()]});
  add(key+'-ko',{p2:[m({ability,level:1})]});
  add(key+'-helmet-source-ko',{p2:[m({ability,item:'Rocky Helmet'})],setup:[{side:'p1',hp:1}]});
  add(key+'-struggle',{p2:[m({ability})],setup:[{side:'p1',pp:0}]});
 }
 for(const seed of [2,3,4,5,6,7,8,9])add('cursedbody-seed-'+seed,{seed,p2:[m({ability:'Cursed Body'})]});
 add('cursedbody-expiry',{p2:[m({ability:'Cursed Body'})],steps:[b(),b(4),b(4),b(4),b(4),b()]});
 add('weakarmor-caps',{p2:[m({ability:'Weak Armor'})],setup:[{side:'p2',boosts:{def:-6,spe:6}}]});
 add('cursedbody-one-hp-sash',{p1:[m({moves:['Aerial Ace'],evs:{spe:252}})],p2:[m({ability:'Cursed Body',species:'Shedinja',item:'Focus Sash'})]});
 add('cursedbody-existing-disable',{p2:[m({ability:'Cursed Body',moves:['Disable','Recover']})],steps:[b(1,1),b(2,2)]});
 add('cursedbody-last-pp',{p2:[m({ability:'Cursed Body'})],setup:[{side:'p1',ppByMove:{tackle:1}}]});
 add('weakarmor-defense-cap',{p2:[m({ability:'Weak Armor'})],setup:[{side:'p2',boosts:{def:-6}}]});
 add('weakarmor-speed-cap',{p2:[m({ability:'Weak Armor'})],setup:[{side:'p2',boosts:{spe:6}}]});
 add('cursedbody-sleep-talk',{seed:4,p1:[m({moves:['Rest','Sleep Talk','Tackle'],evs:{spe:252}})],p2:[m({ability:'Cursed Body'})],setup:[{side:'p1',hp:100}],steps:[b(1),b(2)]});
 add('cursedbody-substitute',{p1:[m({moves:['Substitute','Tackle','Recover'],evs:{spe:252}})],p2:[m({ability:'Cursed Body',moves:['Substitute','Recover']})],steps:[b(3,1),b(2,2)]});
 return cases;
};
module.exports.membershipWitnesses=function(){
 require('./neutral-turn-oracle.cjs').assertReady();const ps=require('./vendor/pokemon-showdown/dist/sim'),assert=require('node:assert/strict');
 return [['Weak Armor',11,'Vanilluxe'],['Cursed Body',38,'Froslass']].map(([ability,n,species])=>{
  const seed=[n,2,3,4],set=ps.Teams.getGenerator('gen5randombattle',seed).getTeam().find(s=>s.ability===ability&&s.species===species);assert.ok(set,ability);return {kind:'ability',id:ps.Dex.toID(ability),seed,set};
 });
};
