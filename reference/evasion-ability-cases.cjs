'use strict';
module.exports=function(){
 const m=(x={})=>({species:'Mew',moves:['Recover','Protect','Tackle'],ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...x});
 const b=(a=1,c=1)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[];
 const add=(id,x={})=>cases.push({id:'G4-evasion-ability-'+id,seed:1,generic:true,evasionAbilityFamily:true,p1:[m({moves:['Mega Punch','Confuse Ray','Gastro Acid'],evs:{spe:252}})],p2:[m()],steps:[b()],...x});
 for(const ability of ['Tangled Feet','Wonder Skin']){
  const key=ability.toLowerCase().replaceAll(' ','');
  for(const move of ['Mega Punch','Thunder Wave','Toxic','Confuse Ray','Charm','Gastro Acid'])add(key+'-'+move.toLowerCase().replaceAll(' ',''),{p1:[m({moves:[move],evs:{spe:252}})],p2:[m({ability})]});
  add(key+'-confused',{p2:[m({ability})],steps:[b(2),b(),b()]});
  add(key+'-suppressed',{p2:[m({ability})],steps:[b(2),b(3),b()]});
  add(key+'-switch',{p2:[m({ability}),m({ability})],steps:[b(2),{p1:'move 1',p2:'switch 2'},b()]});
  add(key+'-always',{p1:[m({moves:['Aerial Ace'],evs:{spe:252}})],p2:[m({ability})]});
  for(const source of ['Compound Eyes','Hustle','Victory Star'])add(key+'-'+source.toLowerCase().replaceAll(' ','')+'-tie',{p1:[m({ability:source,moves:['Confuse Ray','Rock Slide','Toxic']})],p2:[m({ability})],steps:[b(),b(2),b(3)]});
 }
 add('wonderskin-no-guard',{p1:[m({ability:'No Guard',moves:['Toxic'],evs:{spe:252}})],p2:[m({ability:'Wonder Skin'})]});
 add('wonderskin-stage',{p1:[m({moves:['Toxic'],evs:{spe:252}})],p2:[m({ability:'Wonder Skin'})],setup:[{side:'p1',boosts:{accuracy:2}},{side:'p2',boosts:{evasion:-1}}]});
 add('wonderskin-threshold',{seed:11,p1:[m({moves:['Toxic'],evs:{spe:252}})],p2:[m({ability:'Wonder Skin'})]});
 add('control-threshold',{seed:11,p1:[m({moves:['Toxic'],evs:{spe:252}})]});
 add('tangledfeet-rounding',{seed:135,p1:[m({ability:'Compound Eyes',moves:['Confuse Ray','Rock Slide'],evs:{spe:252}})],p2:[m({ability:'Tangled Feet'})],steps:[b(),b(2)]});
 add('tangledfeet-suppressed-tie',{p1:[m({ability:'Victory Star',moves:['Confuse Ray','Gastro Acid','Rock Slide']})],p2:[m({ability:'Tangled Feet'})],steps:[b(),b(2),b(3)]});
 return cases;
};
module.exports.membershipWitnesses=function(){
 require('./neutral-turn-oracle.cjs').assertReady();const ps=require('./vendor/pokemon-showdown/dist/sim'),assert=require('node:assert/strict');
 return [['Tangled Feet',29,'Chatot'],['Wonder Skin',50,'Delcatty']].map(([ability,n,species])=>{
  const seed=[n,2,3,4],set=ps.Teams.getGenerator('gen5randombattle',seed).getTeam().find(s=>s.ability===ability&&s.species===species);assert.ok(set,ability);return {kind:'ability',id:ps.Dex.toID(ability),seed,set};
 });
};
