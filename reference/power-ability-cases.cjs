'use strict';
module.exports=function(){
 const m=(x={})=>({species:'Mew',moves:['Tackle','Recover'],ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...x});
 const b=(a=1,c=2)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[];
 const add=(id,x={})=>cases.push({id:'G4-power-ability-'+id,seed:1,generic:true,powerAbilityFamily:true,p1:[m({evs:{spe:252}})],p2:[m()],steps:[b()],...x});
 for(const [ability,moves]of [['Iron Fist',['Mach Punch','Drain Punch','Fire Punch','Shadow Punch','Tackle']],['Reckless',['Double-Edge','Take Down','Brave Bird','Flare Blitz','Wild Charge','Tackle']]]){
  const key=ability.toLowerCase().replaceAll(' ','');
  for(const move of moves)for(const control of [false,true])add(key+'-'+move.toLowerCase().replaceAll(/[^a-z]/g,'')+(control?'-control':''),{p1:[m({ability:control?'Pressure':ability,moves:[move,'Recover'],evs:{spe:252}})]});
  const move=ability==='Iron Fist'?'Drain Punch':'Double-Edge';
  for(const item of ['Muscle Band','Life Orb','Choice Band'])add(key+'-'+item.toLowerCase().replaceAll(' ',''),{p1:[m({ability,item,moves:[move,'Recover'],evs:{spe:252}})]});
  add(key+'-suppressed',{p1:[m({ability,moves:[move,'Recover'],evs:{spe:252}})],p2:[m({moves:['Gastro Acid','Recover']})],steps:[b(2,1),b()]});
  add(key+'-mummy',{abilityChangeFamily:true,p1:[m({ability,moves:[move,'Recover'],evs:{spe:252}})],p2:[m({ability:'Mummy'})],steps:[b(),b()]});
  add(key+'-burn',{p1:[m({ability,moves:[move,'Recover'],evs:{spe:252}})],setup:[{side:'p1',status:'brn'}]});
  add(key+'-low-level',{p1:[m({ability,level:1,moves:[move,'Recover'],evs:{spe:252}})]});
  add(key+'-dryskin',{p1:[m({ability,moves:[ability==='Iron Fist'?'Fire Punch':'Flare Blitz','Recover'],evs:{spe:252},item:'Muscle Band'})],p2:[m({ability:'Dry Skin'})]});
  add(key+'-struggle',{p1:[m({ability,evs:{spe:252}})],setup:[{side:'p1',pp:0}]});
 }
 return cases;
};
module.exports.membershipWitnesses=function(){
 require('./neutral-turn-oracle.cjs').assertReady();const ps=require('./vendor/pokemon-showdown/dist/sim'),assert=require('node:assert/strict');
 return [['Reckless',20,'Mienshao'],['Iron Fist',64,'Hitmonchan']].map(([ability,n,species])=>{const seed=[n,2,3,4],set=ps.Teams.getGenerator('gen5randombattle',seed).getTeam().find(s=>s.ability===ability&&s.species===species);assert.ok(set,ability);return {kind:'ability',id:ps.Dex.toID(ability),seed,set};});
};
