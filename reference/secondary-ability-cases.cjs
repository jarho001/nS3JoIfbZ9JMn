'use strict';
module.exports=function(){
 const m=(moves,x={})=>({species:'Mew',moves,ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...x});
 const b=(a=1,c=1)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[];
 const add=(id,move='Flamethrower',x={})=>cases.push({id:'G4-secondary-ability-'+id,seed:1,generic:true,secondaryAbilityFamily:true,p1:[m([move,'Tackle'],{evs:{spe:252}})],p2:[m(['Tackle'])],steps:[b()],...x});
 for(const [key,ability,side]of [['dust','Shield Dust','p2'],['grace','Serene Grace','p1']]){
  for(const move of ['Flamethrower','Ice Beam','Thunderbolt','Body Slam','Rock Slide'])for(const seed of [0,1]){
   const x={seed};if(side==='p2')x.p2=[m(['Tackle'],{ability})];else x.p1=[m([move,'Tackle'],{ability,evs:{spe:252}})];add(key+'-'+move.toLowerCase().replace(/[^a-z]/g,'')+'-'+seed,move,x);
  }
 }
 add('grace-mummy','Body Slam',{p1:[m(['Body Slam'],{ability:'Serene Grace',evs:{spe:252}})],p2:[m(['Tackle'],{ability:'Mummy'})]});
 add('both','Body Slam',{p1:[m(['Body Slam'],{ability:'Serene Grace',evs:{spe:252}})],p2:[m(['Tackle'],{ability:'Shield Dust'})]});
 add('dust-primary','Toxic',{p2:[m(['Tackle'],{ability:'Shield Dust'})]});
 add('dust-thaw','Flamethrower',{p2:[m(['Tackle'],{ability:'Shield Dust'})],setup:[{side:'p2',status:'frz'}]});
 add('dust-gas','Flamethrower',{p1:[m(['Gastro Acid','Flamethrower'],{evs:{spe:252}})],p2:[m(['Tackle'],{ability:'Shield Dust'})],steps:[b(),b(2)]});
 add('grace-gas','Body Slam',{p1:[m(['Tackle','Body Slam'],{ability:'Serene Grace'})],p2:[m(['Gastro Acid','Tackle'],{evs:{spe:252}})],steps:[b(),b(2,2)]});
 add('dust-sub','Body Slam',{p1:[m(['Recover','Body Slam'])],p2:[m(['Substitute','Tackle'],{ability:'Shield Dust',evs:{spe:252}})],steps:[b(),b(2,2)]});
 add('grace-sub','Body Slam',{p1:[m(['Recover','Body Slam'],{ability:'Serene Grace'})],p2:[m(['Substitute','Tackle'],{evs:{spe:252}})],steps:[b(),b(2,2)]});
 add('dust-mirrored','Flamethrower',{p1:[m(['Tackle'],{ability:'Shield Dust'})],p2:[m(['Flamethrower'],{evs:{spe:252}})]});
 return cases;
};
module.exports.membershipWitnesses=function(){
 require('./neutral-turn-oracle.cjs').assertReady();const ps=require('./vendor/pokemon-showdown/dist/sim'),assert=require('node:assert/strict');
 return [['Serene Grace',12],['Shield Dust',65]].map(([ability,i])=>{const seed=[i,2,3,4],set=ps.Teams.getGenerator('gen5randombattle',seed).getTeam().find(s=>s.ability===ability);assert.ok(set);return {ability:ps.Dex.toID(ability),seed,set};});
};
