'use strict';
// Native synthetic behavior cases; generator membership is separate below.
module.exports=function(){
 const m=(moves,extra={})=>({species:'Mew',moves,ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...extra});
 const b=(a=1,c=1)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[];
 const add=(id,ability='Water Absorb',move='Surf',x={})=>cases.push({id:'G4-absorption-'+id,seed:1,generic:true,absorptionFamily:true,p1:[m([move,'Recover'],{evs:{spe:252}})],p2:[m(['Growl'],{ability})],setup:[{side:'p2',hp:60}],steps:[b()],...x});
 for(const [ability,move,physical,inaccurate,type]of [['Water Absorb','Surf','Waterfall','Hydro Pump','Water'],['Volt Absorb','Thunderbolt','Thunder Punch','Zap Cannon','Electric']]){
  const key=type.toLowerCase();add(key+'-partial',ability,move);add(key+'-physical',ability,physical);
  add(key+'-full',ability,move,{setup:[]});add(key+'-cap',ability,move,{setup:[{side:'p2',hp:174}]});
  add(key+'-rounding',ability,move,{p2:[m(['Growl'],{ability,evs:{hp:16}})]});
  add(key+'-minimum',ability,move,{p2:[m(['Growl'],{species:'Shedinja',ability})],setup:[]});
  add(key+'-wrong-type',ability,'Tackle');add(key+'-inaccurate',ability,inaccurate);
  add(key+'-hidden-power',ability,'Hidden Power '+type);
  add(key+'-protect',ability,move,{p2:[m(['Protect'],{ability})]});
  for(const hp of [60,175])add(key+'-tied-'+hp,ability,move,{p1:[m([move])],setup:[{side:'p2',hp}]});
  add(key+'-substitute',ability,move,{p1:[m(['Recover',move])],p2:[m(['Substitute','Growl'],{ability,evs:{spe:252}})],setup:[{side:'p2',hp:120}],steps:[b(),b(2,2)]});
  add(key+'-life-orb',ability,move,{p1:[m([move],{item:'Life Orb',evs:{spe:252}})],setup:[{side:'p1',hp:1},{side:'p2',hp:60}]});
  add(key+'-choice',ability,move,{p1:[m([move,'Tackle'],{item:'Choice Specs',evs:{spe:252}})],steps:[b(),b()]});
  add(key+'-suppressed',ability,move,{p1:[m(['Gastro Acid',move],{evs:{spe:252}})],setup:[{side:'p2',hp:120}],steps:[b(),b(2)]});
  add(key+'-acquired',ability,move,{p1:[m(['Recover',move],{ability,evs:{spe:252}})],p2:[m(['Role Play','Growl'])],steps:[b(),b(2,2)]});
  add(key+'-trace',ability,move,{p1:[m([move],{ability,evs:{spe:252}})],p2:[m(['Growl'],{ability:'Trace'})]});
  add(key+'-burn',ability,move,{setup:[{side:'p2',hp:60,status:'brn'}]});
 }
 add('water-toxic-cycle','Water Absorb','Surf',{setup:[{side:'p2',status:'tox'}],steps:[b(),b()]});
 add('water-fly','Water Absorb','Surf',{p1:[m(['Surf'])],p2:[m(['Fly'],{ability:'Water Absorb',evs:{spe:252}})],steps:[b(),b()]});
 add('water-dive','Water Absorb','Surf',{p1:[m(['Surf'])],p2:[m(['Dive'],{ability:'Water Absorb',evs:{spe:252}})],steps:[b(),b()]});
 add('water-rain-self','Water Absorb','Rain Dance',{p1:[m(['Rain Dance'],{ability:'Water Absorb',evs:{spe:252}})]});
 add('water-transform-base-hp','Water Absorb','Surf',{p1:[m(['Recover','Surf'],{ability:'Water Absorb',evs:{spe:252}})],p2:[m(['Transform'],{species:'Ditto'})],setup:[{side:'p1',hp:60},{side:'p2',hp:60}],steps:[b(),b(2,2)]});
 add('water-sleep-talk','Water Absorb','Surf',{p1:[m(['Rest','Sleep Talk','Surf'],{evs:{spe:252}})],setup:[{side:'p1',hp:70},{side:'p2',hp:60}],steps:[b(),b(2),b(2)]});
 add('water-sleep-talk-tied','Water Absorb','Surf',{p1:[m(['Rest','Sleep Talk','Surf'])],setup:[{side:'p1',hp:70},{side:'p2',hp:60}],steps:[b(),b(2),b(2)]});
 add('water-source-ooze','Water Absorb','Surf',{p1:[m(['Surf'],{ability:'Liquid Ooze',evs:{spe:252}})]});
 add('electric-ground-protect','Volt Absorb','Thunderbolt',{p2:[m(['Protect'],{species:'Rhydon',ability:'Volt Absorb'})]});
 add('water-mummy-loss','Water Absorb','Surf',{p1:[m(['Tackle','Growl'],{ability:'Water Absorb',evs:{spe:252}})],p2:[m(['Growl','Surf'],{ability:'Mummy'})],setup:[{side:'p1',hp:120}],steps:[b(),b(2,2)]});
 for(const hp of [60,175])add('electric-status-'+hp,'Volt Absorb','Thunder Wave',{setup:[{side:'p2',hp}]});
 for(const move of ['Thunderbolt','Thunder Wave'])add('electric-ground-'+move.replace(' ',''),'Volt Absorb',move,{p2:[m(['Growl'],{species:'Rhydon',ability:'Volt Absorb'})]});
 add('electric-reflected','Volt Absorb','Thunder Wave',{p1:[m(['Thunder Wave'],{ability:'Volt Absorb',evs:{spe:252}})],p2:[m(['Growl'],{ability:'Magic Bounce'})],setup:[{side:'p1',hp:60}]});
 add('mirrored-water','Water Absorb','Surf',{p1:[m(['Growl'],{ability:'Water Absorb',evs:{spe:252}})],p2:[m(['Surf'])],setup:[{side:'p1',hp:60}]});
 return cases;
};
module.exports.membershipWitnesses=function(){
 require('./neutral-turn-oracle.cjs').assertReady();const ps=require('./vendor/pokemon-showdown/dist/sim'),assert=require('node:assert/strict');
 return [['Water Absorb',1],['Volt Absorb',2]].map(([ability,word])=>{const seed=[word,2,3,4],set=ps.Teams.getGenerator('gen5randombattle',seed).getTeam().find(s=>s.ability===ability);assert.ok(set,ability+' membership');return{ability:ps.Dex.toID(ability),seed,set};});
};
