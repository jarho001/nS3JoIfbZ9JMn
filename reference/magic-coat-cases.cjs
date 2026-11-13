'use strict';
module.exports=function(){
 const m=(moves,extra={})=>({species:'Mew',moves,ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...extra});
 const b=(a=1,c=1)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[],add=(id,move='Toxic',x={})=>cases.push({id:'G4-magic-coat-'+id,seed:1,generic:true,identityFamily:true,abilityChangeFamily:true,magicCoatFamily:true,hitFamily:true,p1:[m([move,'Tackle'])],p2:[m(['Magic Coat','Tackle'])],steps:[b()],...x});
 for(const move of ['Toxic','Spore','Growl','Stealth Rock','Gastro Acid','Worry Seed','Taunt'])add(move.toLowerCase().replace(/[^a-z]/g,''),move);
 add('repeat','Toxic',{steps:[b(),b()]});
 add('expiration','Toxic',{steps:[b(2),b(1,2)]});
 add('both-bounce','Toxic',{p1:[m(['Toxic'],{ability:'Magic Bounce'})],p2:[m(['Magic Coat'],{ability:'Magic Bounce'})]});
 add('prankster-bounce','Toxic',{p1:[m(['Toxic'],{ability:'Prankster'})],p2:[m(['Tackle'],{ability:'Magic Bounce'})]});
 add('prankster-pivot','U-turn',{p1:[m(['U-turn']),m(['Tackle'])],p2:[m(['Magic Coat'],{ability:'Prankster'})],steps:[b(),{p1:'switch 2'}]});
 add('pivot','U-turn',{p1:[m(['U-turn']),m(['Tackle'])],steps:[b(),{p1:'switch 2'}]});
 add('taunt','Taunt',{p1:[m(['Taunt','Tackle'],{ability:'Prankster',evs:{spe:252}})],steps:[b()]});
 add('called','Spore',{p1:[m(['Spore','Toxic'],{ability:'Prankster'})],p2:[m(['Sleep Talk','Magic Coat'],{ability:'Prankster',evs:{spe:252}})],steps:[b(),b(2)]});
 add('phazing','Roar',{p1:[m(['Roar']),m(['Tackle'])]});
 add('ground-immunity','Thunder Wave',{p2:[m(['Magic Coat'],{species:'Sandslash'})]});
 add('mirrored','Magic Coat',{p1:[m(['Magic Coat'])],p2:[m(['Toxic'])]});
 return cases;
};
