'use strict';
module.exports=function(){
 const m=(moves,x={})=>({species:'Mew',moves,ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...x});
 const b=(a=1,c=1)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[];
 const add=(id,move='Giga Drain',x={})=>cases.push({id:'G4-sap-sipper-'+id,seed:1,generic:true,sapSipperFamily:true,p1:[m([move,'Tackle'],{evs:{spe:252}})],p2:[m(['Growl'],{ability:'Sap Sipper'})],steps:[b()],...x});
 add('ordinary');add('physical','Vine Whip');add('spore','Spore');add('inaccurate','Power Whip');
 add('cap','Spore',{steps:Array.from({length:7},()=>b())});
 add('protect','Spore',{p2:[m(['Protect'],{ability:'Sap Sipper'})]});
 add('substitute','Spore',{p1:[m(['Recover','Spore'])],p2:[m(['Substitute','Growl'],{ability:'Sap Sipper',evs:{spe:252}})],steps:[b(),b(2,2)]});
 add('gas','Giga Drain',{p1:[m(['Gastro Acid','Giga Drain'],{evs:{spe:252}})],steps:[b(),b(2)]});
 add('replace','Spore',{p1:[m(['Simple Beam','Spore'],{evs:{spe:252}})],steps:[b(),b(2)]});
 add('wrong','Flamethrower');add('hidden','Hidden Power Grass');
 add('drain-no-heal','Giga Drain',{setup:[{side:'p1',hp:50}]});
 add('reflected','Spore',{p1:[m(['Spore'],{ability:'Sap Sipper',evs:{spe:252}})],p2:[m(['Growl'],{ability:'Magic Bounce'})]});
 add('sleep-talk','Giga Drain',{p1:[m(['Sleep Talk','Giga Drain'])],p2:[m(['Spore','Growl'],{ability:'Sap Sipper',evs:{spe:252}})],steps:[b(),b(1,2)]});
 add('entry-trace','Spore',{p1:[m(['Spore'],{ability:'Trace',evs:{spe:252}})],p2:[m(['Growl'],{ability:'Forecast'}),m(['Growl'],{ability:'Sap Sipper'})],steps:[{p1:'move 1',p2:'switch 2'}]});
 add('switch-reset','Spore',{p2:[m(['Growl'],{ability:'Sap Sipper'}),m(['Growl'])],steps:[b(),{p1:'move 2',p2:'switch 2'},{p1:'move 1',p2:'switch 2'}]});
 add('mirrored','Spore',{p1:[m(['Growl'],{ability:'Sap Sipper'})],p2:[m(['Spore'],{evs:{spe:252}})]});
 return cases;
};
module.exports.membershipWitnesses=function(){
 require('./neutral-turn-oracle.cjs').assertReady();const ps=require('./vendor/pokemon-showdown/dist/sim'),assert=require('node:assert/strict'),seed=[10,2,3,4];
 const set=ps.Teams.getGenerator('gen5randombattle',seed).getTeam().find(s=>s.ability==='Sap Sipper');assert.ok(set);return [{ability:'sapsipper',seed,set}];
};
