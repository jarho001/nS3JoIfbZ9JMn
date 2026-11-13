'use strict';
module.exports=function(){
 const m=(moves,extra={})=>({species:'Mew',moves,ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...extra});
 const b=(a=1,c=1)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[],add=(id,move,x={})=>cases.push({id:'G4-reflection-'+id,seed:1,generic:true,identityFamily:true,abilityChangeFamily:true,reflectionFamily:true,hitFamily:true,p1:[m([move,'Tackle'],{evs:{spe:252}})],p2:[m(['Tackle'],{ability:'Magic Bounce'})],steps:[b()],...x});
 for(const move of ['Toxic','Will-O-Wisp','Thunder Wave','Spore','Confuse Ray','Growl','Taunt','Stealth Rock','Spikes','Gastro Acid','Worry Seed','Torment'])add(move.toLowerCase().replace(/[^a-z]/g,''),move);
 add('ground-immunity','Thunder Wave',{p2:[m(['Tackle'],{species:'Sandslash',ability:'Magic Bounce'})]});
 add('both-bounce','Toxic',{p1:[m(['Toxic'],{ability:'Magic Bounce',evs:{spe:252}})]});
 add('protected','Toxic',{p2:[m(['Protect'],{ability:'Magic Bounce'})]});
 add('substitute','Taunt',{p2:[m(['Substitute','Tackle'],{ability:'Magic Bounce',evs:{spe:252}})],p1:[m(['Tackle','Taunt'])],steps:[b(),b(2,2)]});
 add('semi-invulnerable','Toxic',{p1:[m(['Tackle','Toxic'])],p2:[m(['Fly'],{ability:'Magic Bounce',evs:{spe:252}})],steps:[b(2),b(2)]});
 add('hazard-in-air','Spikes',{p1:[m(['Tackle','Spikes'])],p2:[m(['Fly'],{ability:'Magic Bounce',evs:{spe:252}})],steps:[b(2),b(2)]});
 add('suppressed','Toxic',{p1:[m(['Tackle','Toxic'],{ability:'Magic Bounce'})],p2:[m(['Gastro Acid','Tackle'],{ability:'Magic Bounce',evs:{spe:252}})],steps:[b(),b(2,2)]});
 add('called','Sleep Talk',{p1:[m(['Sleep Talk','Taunt'])],p2:[m(['Spore'],{ability:'Magic Bounce',evs:{spe:252}})]});
 add('phazing','Roar',{p1:[m(['Roar','Tackle']),m(['Tackle'])],p2:[m(['Tackle'],{ability:'Magic Bounce',evs:{spe:252}})]});
 add('mirrored','Tackle',{p1:[m(['Tackle'],{ability:'Magic Bounce'})],p2:[m(['Toxic'],{evs:{spe:252}})]});
 return cases;
};
