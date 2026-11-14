'use strict';
// Directed synthetic behavior cases; membership is independently witnessed below.
module.exports=function(){
 const m=(moves,extra={})=>({species:'Mew',moves,ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...extra});
 const b=(a=1,c=1)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[];
 const add=(id,ability='Overgrow',move='Seed Bomb',x={})=>cases.push({id:'G4-low-hp-'+id,seed:1,generic:true,lowHpFamily:true,p1:[m([move,'Recover'],{ability,evs:{spe:252}})],p2:[m(['Recover'])],setup:[{side:'p1',hp:58}],steps:[b()],...x});
 for(const [ability,physical,special]of [['Overgrow','Seed Bomb','Giga Drain'],['Blaze','Fire Punch','Flamethrower'],['Torrent','Waterfall','Surf'],['Swarm','U-turn','Hidden Power Bug']]){
  const key=ability.toLowerCase();add(key+'-physical',ability,physical);add(key+'-special',ability,special);
  add(key+'-above',ability,special,{setup:[{side:'p1',hp:59}]});
  add(key+'-wrong-type',ability,'Tackle');
 }
 add('exact-third','Overgrow','Giga Drain',{p1:[m(['Giga Drain'],{ability:'Overgrow',evs:{hp:16,spe:252}})],setup:[{side:'p1',hp:59}]});
 add('choice-band','Blaze','Fire Punch',{p1:[m(['Fire Punch'],{ability:'Blaze',item:'Choice Band',evs:{spe:252}})]});
 add('choice-specs','Torrent','Surf',{p1:[m(['Surf'],{ability:'Torrent',item:'Choice Specs',evs:{spe:252}})]});
 add('thick-fat','Blaze','Flamethrower',{p1:[m(['Flamethrower'],{ability:'Blaze',item:'Choice Specs',evs:{spe:252}})],p2:[m(['Recover'],{ability:'Thick Fat'})]});
 add('flower-gift','Blaze','Fire Punch',{p2:[m(['Sunny Day','Recover'],{species:'Cherrim',ability:'Flower Gift'})],steps:[b(),b(1,2)]});
 add('odd-stat-rounding','Blaze','Flamethrower',{p1:[m(['Flamethrower'],{ability:'Blaze',evs:{spa:4,spe:252},item:'Choice Specs'})],p2:[m(['Recover'],{ability:'Thick Fat'})]});
 add('stat-control','Pressure','Fire Punch');
 for(const [id,move,hp]of [['tied-special','Flamethrower',58],['tied-above','Flamethrower',59],['tied-wrong-type','Surf',58]])add(id,'Blaze',move,{p1:[m([move],{ability:'Blaze'})],p2:[m(['Recover'],{ability:'Thick Fat'})],setup:[{side:'p1',hp}]});
 add('critical','Blaze','Fire Punch',{seed:21});
 add('hidden-power','Torrent','Hidden Power Water');
 add('drain-crossing','Overgrow','Giga Drain',{steps:[b(),b()]});
 add('damaged-crossing','Torrent','Surf',{p1:[m(['Surf'],{ability:'Torrent'})],p2:[m(['Tackle'],{evs:{spe:252}})],setup:[{side:'p1',hp:65}],steps:[b(),b()]});
 add('heal-crossing','Blaze','Flamethrower',{steps:[b(2),b()]});
 add('suppressed','Blaze','Flamethrower',{p2:[m(['Gastro Acid'],{evs:{spe:252}})]});
 add('acquired','Overgrow','Role Play',{p1:[m(['Role Play','Giga Drain'],{evs:{spe:252}})],p2:[m(['Recover'],{ability:'Overgrow'})],steps:[b(),b(2)]});
 add('trace','Overgrow','Giga Drain',{p1:[m(['Giga Drain'],{ability:'Trace',evs:{spe:252}})],p2:[m(['Recover'],{ability:'Overgrow'})]});
 add('mummy-loss','Blaze','Fire Punch',{p2:[m(['Recover'],{ability:'Mummy'})],steps:[b(),b()]});
 add('burn-boost','Blaze','Fire Punch',{setup:[{side:'p1',hp:58,status:'brn'}]});
 add('boost-stage','Torrent','Waterfall',{p1:[m(['Swords Dance','Waterfall'],{ability:'Torrent',evs:{spe:252}})],steps:[b(),b(2)]});
 add('multihit','Overgrow','Bullet Seed');
 add('substitute','Blaze','Flamethrower',{p1:[m(['Growl','Flamethrower'],{ability:'Blaze'})],p2:[m(['Substitute','Recover'],{evs:{spe:252}})],setup:[{side:'p1',hp:58}],steps:[b(),b(2,2)]});
 add('mirrored','Swarm','Hidden Power Bug',{p1:[m(['Recover'],{evs:{spe:252}})],p2:[m(['Hidden Power Bug'],{ability:'Swarm'})],setup:[{side:'p2',hp:58}]});
 return cases;
};
module.exports.membershipWitnesses=function(){
 require('./neutral-turn-oracle.cjs').assertReady();
 const ps=require('./vendor/pokemon-showdown/dist/sim'),assert=require('node:assert/strict');
 return [['Overgrow',1],['Blaze',7],['Torrent',8],['Swarm',31]].map(([ability,word])=>{
  const seed=[word,2,3,4],set=ps.Teams.getGenerator('gen5randombattle',seed).getTeam().find(s=>s.ability===ability);
  assert.ok(set,ability+' positive generator witness');return {ability:ps.Dex.toID(ability),seed,set};
 });
};
