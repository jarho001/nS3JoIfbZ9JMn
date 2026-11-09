'use strict';
// Next-family native inputs. This file is deliberately not imported by the
// frozen schema6 reader until its active CPU regression has finished.
module.exports=function movePhaseCases(mon) {
 const m=(moves,o={})=>mon('Mew',moves,{ability:'Synchronize',...o});
 const fast={evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:252}};
 const tank={evs:{hp:252,atk:0,def:252,spa:0,spd:252,spe:0}};
 const both=(a,b)=>({p1:`move ${a}`,p2:`move ${b}`});
 const cases=[];
 for(const move of ['Double Slap','Bullet Seed','Rock Blast','Dual Chop','Triple Kick','Bonemerang'])cases.push({
  id:`G4-phase-multihit-${move.replaceAll(' ','')}`,seed:0,
  p1:[m([move],fast)],p2:[m(['Recover'],tank)],steps:[both(1,1),both(1,1)]});
 for(const move of ['Take Down','Double-Edge','Flare Blitz','Head Smash','Wild Charge','Brave Bird',
   'Absorb','Mega Drain','Giga Drain','Drain Punch','Leech Life'])cases.push({
  id:`G4-phase-recoil-drain-${move.replaceAll(' ','')}`,seed:0,
  p1:[m([move],fast)],p2:[m(['Recover'],tank)],setup:move.includes('Drain')||['Absorb','Leech Life'].includes(move)?[{side:'p1',hp:100}]:[],steps:[both(1,1),both(1,1)]});
 for(const move of ['Self-Destruct','Explosion'])cases.push({
  id:`G4-phase-selfdestruct-${move}`,seed:0,p1:[m([move],fast),m(['Tackle'])],
  p2:[m(['Tackle'],tank),m(['Tackle'])],steps:[both(1,1)]});
 for(const move of ['Hyper Beam','Giga Impact','Rock Wrecker','Blast Burn','Frenzy Plant','Hydro Cannon'])cases.push({
  id:`G4-phase-recharge-${move.replaceAll(' ','')}`,seed:0,
  p1:[m([move,'Recover'],fast),m(['Tackle'])],p2:[m(['Recover'],tank)],steps:[both(1,1),both(1,1),both(1,1),both(1,1)]});
 for(const move of ['Outrage','Thrash','Petal Dance'])cases.push({
  id:`G4-phase-rampage-${move.replaceAll(' ','')}`,seed:0,
  p1:[m(['Pound',move],fast),m(['Tackle'])],p2:[m(['Recover'],tank)],
  steps:[both(2,1),both(1,1),both(1,1),both(1,1)]});
 for(const move of ['Solar Beam','Sky Attack','Skull Bash','Razor Wind','Fly','Dig','Dive','Bounce'])cases.push({
  id:`G4-phase-charge-${move.replaceAll(' ','')}`,seed:0,
  p1:[m(['Pound',move],fast),m(['Tackle'])],p2:[m(['Tackle'],tank)],steps:[both(2,1),both(1,1)]});
 cases.push({id:'G4-phase-solar-sun',seed:0,p1:[m(['Solar Beam'],{ability:'Drought',...fast})],p2:[m(['Recover'],tank)],steps:[both(1,1)]},
  {id:'G4-phase-solar-rain',seed:0,p1:[m(['Solar Beam'],{ability:'Drizzle',...fast})],p2:[m(['Recover'],tank)],steps:[both(1,1),both(1,1)]},
  {id:'G4-phase-power-herb',seed:0,p1:[m(['Solar Beam'],{item:'Power Herb',...fast})],p2:[m(['Recover'],tank)],steps:[both(1,1)]},
  {id:'G4-phase-taunt-interrupts',seed:0,p1:[m(['Taunt','Tackle'],fast)],p2:[m(['Recover','Tackle'],tank)],steps:[both(1,1),both(2,2)]},
  {id:'G4-phase-encore',seed:0,p1:[m(['Recover','Encore','Tackle'])],p2:[m(['Tackle','Recover'],fast)],steps:[both(1,1),both(2,1),both(3,1)]},
  {id:'G4-phase-disable',seed:0,p1:[m(['Recover','Disable'])],p2:[m(['Tackle','Recover'],fast)],steps:[both(1,1),both(2,1),both(1,2)]},
  {id:'G4-phase-torment',seed:0,p1:[m(['Torment','Tackle'],fast)],p2:[m(['Tackle','Recover'])],steps:[both(1,1),both(2,2),both(2,1)]},
  {id:'G4-phase-multihit-lifeorb',seed:0,p1:[m(['Double Slap'],{item:'Life Orb',...fast})],p2:[m(['Recover'],tank)],steps:[both(1,1)]},
  {id:'G4-phase-skilllink',seed:0,p1:[m(['Bullet Seed'],{ability:'Skill Link',...fast})],p2:[m(['Recover'],tank)],steps:[both(1,1)]},
  {id:'G4-phase-roughskin-stops-hits',seed:0,seedWords:[35861,12493,8975,48894],p1:[m(['Double Slap'],fast),m(['Tackle'])],p2:[m(['Recover'],{ability:'Rough Skin',...tank})],setup:[{side:'p1',hp:1}],steps:[both(1,1)]});
 // Invert two real LCG steps: accuracy100 then the native20-entry count.
 const modulus=1n<<64n,A=0x5D588B656C078965n,C=0x269EC3n;
 let inverse=1n;for(let i=0;i<6;i++)inverse=inverse*(2n-A*inverse)%modulus;
 for(const [count,last] of [[2,0x1000000000000000n],[3,0x8000000000000000n],[4,0xC000000000000000n],[5,0xF000000000000000n]]){
  let word=last;for(let i=0;i<2;i++)word=((word-C)*inverse%modulus+modulus)%modulus;
  const seedWords=[48n,32n,16n,0n].map(shift=>Number((word>>shift)&65535n));
  cases.push({id:`G4-phase-count-${count}`,seed:0,seedWords,p1:[m(['Bullet Seed'],fast)],p2:[m(['Recover'],tank)],steps:[both(1,1)]});
 }
 cases.push({id:'G4-phase-multihit-target-ko',seed:0,p1:[m(['Bullet Seed'],fast)],p2:[m(['Recover']),m(['Tackle'])],setup:[{side:'p2',hp:1}],steps:[both(1,1)]},
  {id:'G4-phase-recoil-target-cap',seed:0,p1:[m(['Double-Edge'],fast)],p2:[m(['Recover']),m(['Tackle'])],setup:[{side:'p2',hp:1}],steps:[both(1,1)]},
  {id:'G4-phase-drain-target-cap',seed:0,p1:[m(['Giga Drain'],fast)],p2:[m(['Recover']),m(['Tackle'])],setup:[{side:'p1',hp:100},{side:'p2',hp:1}],steps:[both(1,1)]});
 return cases.map(c=>({...c,generic:true,phaseFamily:true}));
};
