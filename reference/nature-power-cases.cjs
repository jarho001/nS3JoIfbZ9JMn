'use strict';
module.exports=function(){
 const m=(species,moves,extra={})=>({species,moves,ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...extra});
 const b=(a=1,c=1)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[],add=(id,x={})=>cases.push({id:'G4-nature-power-'+id,seed:1,generic:true,identityFamily:true,abilityChangeFamily:true,naturePowerFamily:true,hitFamily:true,p1:[m('Mew',['Nature Power','Tackle'],{evs:{spe:252}})],p2:[m('Mew',['Tackle','Recover'])],steps:[b(),b(1,2)],...x});
 add('ordinary');add('no-pressure',{p2:[m('Mew',['Tackle','Recover'],{ability:'Synchronize'})]});
 add('levitate',{p2:[m('Gengar',['Tackle'],{ability:'Levitate'})],steps:[b()]});
 add('flying',{p2:[m('Charizard',['Tackle'])],steps:[b()]});
 add('protect',{p2:[m('Mew',['Protect'])],steps:[b()]});
 add('substitute',{p2:[m('Mew',['Substitute','Tackle'],{evs:{spe:252},nature:'Timid'})]});
 add('prankster',{p1:[m('Mew',['Nature Power'],{ability:'Prankster'})],p2:[m('Mew',['Tackle'],{evs:{spe:252}})],steps:[b()]});
 add('choice',{p1:[m('Mew',['Nature Power','Tackle'],{item:'Choice Band',evs:{spe:252}})]});
 add('lifeorb',{p1:[m('Mew',['Nature Power'],{item:'Life Orb',evs:{spe:252}})],steps:[b()]});
 add('taunt',{p2:[m('Mew',['Taunt','Tackle'],{evs:{spe:252},nature:'Timid'})],steps:[b()]});
 add('zero-earthquake-pp',{p1:[m('Mew',['Nature Power','Earthquake'],{evs:{spe:252}})],setup:[{side:'p1',ppByMove:{earthquake:0}}],steps:[b()]});
 add('sleep-talk-filter',{p1:[m('Mew',['Rest','Sleep Talk','Nature Power'],{evs:{spe:252}})],setup:[{side:'p1',hp:100}],steps:[b(),b(2)]});
 add('tie',{p1:[m('Mew',['Nature Power'])],p2:[m('Mew',['Recover'])],steps:[b()]});
 add('mirrored',{p1:[m('Mew',['Tackle'])],p2:[m('Mew',['Nature Power'],{evs:{spe:252}})],steps:[b()]});
 return cases;
};
