'use strict';
module.exports=function(){
 const m=(moves,x={})=>({species:'Mew',moves,ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...x});
 const b=(a=1,c=1)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[];
 const add=(id,ability,move,x={})=>cases.push({id:'G4-type-boost-'+id,seed:1,generic:true,typeBoostFamily:true,p1:[m([move,'Tackle'],{evs:{spe:252}})],p2:[m(['Growl'],{ability})],steps:[b()],...x});
 for(const [key,ability,move,type]of [['storm','Storm Drain','Surf','Water'],['rod','Lightning Rod','Thunderbolt','Electric'],['motor','Motor Drive','Thunderbolt','Electric']]){
  add(key+'-physical',ability,type==='Water'?'Waterfall':'Thunder Punch');
  add(key+'-inaccurate',ability,type==='Water'?'Hydro Pump':'Zap Cannon');
  add(key+'-choice',ability,move,{p1:[m([move,'Tackle'],{item:'Choice Specs',evs:{spe:252}})]});
  add(key+'-entry-trace',ability,move,{p1:[m([move],{ability:'Trace',evs:{spe:252}})],p2:[m(['Growl'],{ability:'Forecast'}),m(['Growl'],{ability})],steps:[{p1:'move 1',p2:'switch 2'}]});
  add(key+'-replace',ability,move,{p1:[m(['Worry Seed',move],{evs:{spe:252}})],steps:[b(),b(2)]});
  add(key+'-sleep-talk',ability,move,{p1:[m(['Sleep Talk',move])],p2:[m(['Spore','Growl'],{ability,evs:{spe:252}})],steps:[b(),b(1,2)]});
  add(key+'-ordinary',ability,move);add(key+'-cap',ability,move,{steps:Array.from({length:7},()=>b())});
  add(key+'-protect',ability,move,{p2:[m(['Protect'],{ability})]});
  add(key+'-sub',ability,move,{p1:[m(['Recover',move])],p2:[m(['Substitute','Growl'],{ability,evs:{spe:252}})],steps:[b(),b(2,2)]});
  add(key+'-hidden',ability,'Hidden Power '+type);add(key+'-wrong',ability,'Flamethrower');
  add(key+'-gas',ability,move,{p1:[m(['Gastro Acid',move],{evs:{spe:252}})],steps:[b(),b(2)]});
  add(key+'-fly',ability,move,{p1:[m([move])],p2:[m(['Fly'],{ability,evs:{spe:252}})]});
  add(key+'-pivot',ability,move,{p1:[m(['U-turn',move],{evs:{spe:252}}),m(['Tackle'])],steps:[b(),{p1:'switch 2'}]});
  add(key+'-mirrored',ability,move,{p1:[m(['Growl'],{ability})],p2:[m([move],{evs:{spe:252}})]});
 }
 for(const ability of ['Lightning Rod','Motor Drive']){
  const key=ability==='Lightning Rod'?'rod':'motor';add(key+'-status',ability,'Thunder Wave');
  add(key+'-ground',ability,'Thunder Wave',{p2:[m(['Growl'],{species:'Rhydon',ability})]});
  add(key+'-reflected',ability,'Thunder Wave',{p1:[m(['Thunder Wave'],{ability,evs:{spe:252}})],p2:[m(['Growl'],{ability:'Magic Bounce'})]});
 }
 return cases;
};
module.exports.membershipWitnesses=function(){
 require('./neutral-turn-oracle.cjs').assertReady();const ps=require('./vendor/pokemon-showdown/dist/sim'),assert=require('node:assert/strict');
 return [['Lightning Rod',2],['Motor Drive',32],['Storm Drain',33]].map(([ability,i])=>{const seed=[i,2,3,4],set=ps.Teams.getGenerator('gen5randombattle',seed).getTeam().find(s=>s.ability===ability);assert.ok(set,ability);return {ability:ps.Dex.toID(ability),seed,set};});
};
