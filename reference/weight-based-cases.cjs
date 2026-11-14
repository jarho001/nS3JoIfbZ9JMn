'use strict';
module.exports=function(){
 const m=(x={})=>({species:'Mew',moves:['Tackle','Recover'],ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...x});
 const b=(a=1,c=2)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[];
 const add=(id,x={})=>cases.push({id:'G4-weight-based-'+id,seed:1,generic:true,weightBasedFamily:true,p1:[m({evs:{spe:252}})],p2:[m()],steps:[b()],...x});
 for(const move of ['Low Kick','Grass Knot']){
  const key=move.toLowerCase().replaceAll(' ','');const source=(x={})=>m({moves:[move,'Recover'],evs:{spe:252},...x});
  for(const species of ['Vulpix','Magikarp','Flareon','Zweilous','Venusaur','Crustle'])add(key+'-'+species.toLowerCase(),{p1:[source()],p2:[m({species})]});
  for(const species of ['Vulpix','Flareon','Zweilous'])add(key+'-technician-'+species.toLowerCase(),{p1:[source({ability:'Technician'})],p2:[m({species})]});
  add(key+'-item',{p1:[source({item:move==='Low Kick'?'Muscle Band':'Wise Glasses'})],p2:[m({species:'Crustle'})]});
  add(key+'-roughskin',{p1:[source()],p2:[m({species:'Flareon',ability:'Rough Skin'})]});
  add(key+'-target-transform',{identityFamily:true,p1:[source({species:'Snorlax'})],p2:[m({moves:['Transform','Recover']})],steps:[b(2,1),b()]});
  add(key+'-source-transform',{identityFamily:true,p1:[m({moves:['Transform','Recover'],evs:{spe:252}})],p2:[m({species:'Pikachu',moves:[move,'Recover']})],steps:[b(),b()]});
  add(key+'-imposter',{identityFamily:true,p1:[source({species:'Snorlax'})],p2:[m({species:'Ditto',ability:'Imposter'})]});
  add(key+'-substitute',{p1:[source()],p2:[m({moves:['Substitute','Recover']})],steps:[b(2,1),b()]});
 }
 const source=(x={})=>m({species:'Steelix',moves:['Heavy Slam','Recover'],evs:{spe:252},...x});
 for(const species of ['Kangaskhan','Venusaur','Crustle','Snorlax'])add('heavyslam-'+species.toLowerCase(),{p1:[source()],p2:[m({species})]});
 add('heavyslam-ratio-three',{p1:[source({species:'Aggron'})],p2:[m({species:'Rhydon'})]});
 add('heavyslam-technician',{p1:[source({ability:'Technician'})],p2:[m({species:'Snorlax'})]});
 add('heavyslam-muscleband',{p1:[source({item:'Muscle Band'})],p2:[m({species:'Snorlax'})]});
 add('heavyslam-source-transform',{identityFamily:true,p1:[m({species:'Aggron',moves:['Transform','Recover'],evs:{spe:252}})],p2:[m({species:'Pikachu',moves:['Heavy Slam','Recover']})],steps:[b(),b()]});
 add('heavyslam-target-transform',{identityFamily:true,p1:[source()],p2:[m({moves:['Transform','Recover']})],steps:[b(2,1),b()]});
 add('heavyslam-imposter',{identityFamily:true,p1:[source()],p2:[m({species:'Ditto',ability:'Imposter'})]});
 return cases;
};
module.exports.membershipWitnesses=function(){
 require('./neutral-turn-oracle.cjs').assertReady();const ps=require('./vendor/pokemon-showdown/dist/sim'),assert=require('node:assert/strict');
 return [['heavyslam',4,'Aggron'],['grassknot',7,'Emboar'],['lowkick',21,'Bisharp']].map(([id,n,species])=>{const seed=[n,2,3,4],set=ps.Teams.getGenerator('gen5randombattle',seed).getTeam().find(s=>s.species===species&&s.moves.some(m=>ps.Dex.toID(m)===id));assert.ok(set,id);return {kind:'move',id,seed,set};});
};
