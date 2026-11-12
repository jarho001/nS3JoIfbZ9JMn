#!/usr/bin/env node
'use strict';
// Independent native expectations. No JAX implementation is imported here.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {assertReady, withObservedBattleDraws} = require('./neutral-turn-oracle.cjs');
const {projectPackets} = require('../environment/protocol.cjs');
const {actionSpace} = require('../environment/action-space.cjs');
const pin = require('../reference-lock.json').commit;
const ivs = {hp:31,atk:31,def:31,spa:31,spd:31,spe:31};
function mon(species, moves, options={}) {
  const ability = {Mew:'Synchronize', Snorlax:'Immunity', Gengar:'Levitate'}[species];
  return {species, ability, moves, item:'', level:50, nature:'Hardy', ivs,
    evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0}, ...options};
}
const mew = (moves, o) => mon('Mew', moves, o);
const lax = (moves, o) => mon('Snorlax', moves, o);
const ghost = (moves, o) => mon('Gengar', moves, o);
const attack = ['Tackle','Mega Punch'];
const cases = [
 ...['Tackle','Earthquake','Toxic'].map(move=>({id:`protect-immunity-${move}`,seed:1,
  p1:[lax([move])],p2:[ghost(['Protect'])],steps:[{p1:'move 1',p2:'move 1'}]})),
 ...['Gengar','Snorlax'].map(species=>({id:`trusted-toxic-immune-${species}`,seed:1,
  p1:[mon(species,['Protect'])],p2:[mew(['Tackle'])],setup:[{side:'p1',status:'tox'}],steps:[{p1:'move 1',p2:'move 1'}]})),
 {id:'paralyzed-forced-return',seed:1,p1:[ghost(['Protect'],{level:1}),mew(['Tackle'],{level:1})],
  p2:[lax(['Mega Punch'],{level:100})],setup:[{side:'p1',status:'par'}],steps:[{p1:'switch 2',p2:'move 1'},{p1:'switch 2'}]},
 {id:'fainted-leftovers-handler-sort',seed:1,p1:[lax(['Tackle'],{item:'Leftovers'})],
  p2:[lax(['Tackle'],{item:'Leftovers'}),lax(['Tackle'],{item:'Leftovers'})],
  setup:[{side:'p1',hp:100},{side:'p2',hp:1}],steps:[{p1:'move 1',p2:'move 1'}]},
 ...['brn','tox'].map(status=>({id:`fainted-${status}-handler-sort`,seed:1,
  p1:[mew(['Tackle'])],p2:[mew(['Tackle']),mew(['Tackle'])],
  setup:[{side:'p1',hp:100,status},{side:'p2',hp:1,status}],
  campaign:true,policy:'tactical',policySeeds:[0,0],steps:[]})),
 {id:'wisp-hit',seed:0,p1:[mew(['Will-O-Wisp'])],p2:[lax(attack)],steps:[{p1:'move 1',p2:'move 1'}]},
 {id:'wisp-miss',seed:1,p1:[mew(['Will-O-Wisp'])],p2:[lax(attack)],steps:[{p1:'move 1',p2:'move 1'}]},
 {id:'toxic-immunity',seed:1,p1:[mew(['Toxic'])],p2:[lax(attack)],steps:[{p1:'move 1',p2:'move 1'}]},
 {id:'toxic-type-immunity',seed:1,p1:[mew(['Toxic'])],p2:[ghost(['Toxic'])],steps:[{p1:'move 1',p2:'move 1'}]},
 {id:'normal-ghost',seed:1,p1:[lax(['Tackle'])],p2:[ghost(['Toxic'])],steps:[{p1:'move 1',p2:'move 1'}]},
 {id:'levitate',seed:1,p1:[lax(['Earthquake'])],p2:[ghost(['Toxic'])],steps:[{p1:'move 1',p2:'move 1'}]},
 {id:'lum-synchronize',seed:1,p1:[mew(['Will-O-Wisp'],{evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:252}})],p2:[mew(['Tackle'],{item:'Lum Berry'})],steps:[{p1:'move 1',p2:'move 1'}]},
 {id:'two-burn-ko-order',seed:1,p1:[mew(['Protect'],{evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:252}})],p2:[mew(['Protect'])],setup:[{side:'p1',status:'brn',hp:1},{side:'p2',status:'brn',hp:1}],steps:[{p1:'move 1',p2:'move 1'}]},
 {id:'leftovers-before-burn',seed:1,p1:[mew(['Protect'],{item:'Leftovers'})],p2:[lax(['Protect'])],setup:[{side:'p1',status:'brn',hp:15}],steps:[{p1:'move 1',p2:'move 1'}]},
 {id:'thunder-wave',seed:1,p1:[mew(['Thunder Wave'])],p2:[lax(attack)],steps:[{p1:'move 1',p2:'move 1'}]},
 {id:'recover-partial',seed:1,p1:[mew(['Recover'])],p2:[lax(attack)],setup:[{side:'p1',hp:10}],steps:[{p1:'move 1',p2:'move 1'}]},
 ...[0,1,21].map(seed=>({id:`paralysis-${seed}`,seed,p1:[mew(attack)],p2:[lax(attack)],setup:[{side:'p1',status:'par'}],steps:[{p1:'move 2',p2:'move 2'}]})),
 {id:'toxic-switch-reset',seed:1,p1:[mew(['Recover']),ghost(['Protect'])],p2:[lax(['Protect'])],setup:[{side:'p1',status:'tox'}],steps:[{p1:'move 1',p2:'move 1'},{p1:'move 1',p2:'move 1'},{p1:'switch 2',p2:'move 1'},{p1:'switch 2',p2:'move 1'}]},
 {id:'voluntary-retarget',seed:1,p1:[mew(['Tackle']),ghost(['Protect'])],p2:[lax(['Tackle'])],steps:[{p1:'switch 2',p2:'move 1'}]},
 {id:'pivot-resume',seed:1,p1:[mew(['U-turn']),lax(attack)],p2:[lax(attack)],steps:[{p1:'move 1',p2:'move 2'},{p1:'switch 2'}]},
 {id:'forced-replacement',seed:21,p1:[mew(['Mega Punch'],{level:100})],p2:[lax(attack,{level:1}),ghost(['Protect'])],steps:[{p1:'move 1',p2:'move 2'},{p2:'switch 2'}]},
 {id:'struggle-double-ko',seed:1,p1:[mew(['Tackle'])],p2:[ghost(['Toxic'])],setup:[{side:'p1',hp:1,pp:0},{side:'p2',hp:1,pp:0}],steps:[{p1:'move 1',p2:'move 1'}]},
 {id:'protect-repeat',seed:1,p1:[mew(['Protect'],{evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:252}})],p2:[mew(['Tackle'])],steps:[{p1:'move 1',p2:'move 1'},{p1:'move 1',p2:'move 1'},{p1:'move 1',p2:'move 1'}]},
 {id:'pivot-protected',seed:1,p1:[mew(['U-turn']),lax(attack)],p2:[lax(['Protect'])],steps:[{p1:'move 1',p2:'move 1'}]},
 {id:'commit-replace',seed:1,p1:[mew(attack,{evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:252}})],p2:[mew(attack)],steps:[{p1:'move 1'},{p1:'move 2'},{p2:'move 1'}]},
];

// One case catalog, grouped by family. Boundary mutations are always labeled.
cases.push(
 {id:'legacy-pound-tied',seed:1,p1:[mew(['Tackle','Pound'])],p2:[mew(['Pound','Tackle'])],steps:[{p1:'move 2',p2:'move 1'},{p2:'move 2',p1:'move 1'}]},
 {id:'legacy-accuracy',seed:9,p1:[mew(attack,{evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:252}})],p2:[mew(attack)],steps:[{p1:'move 2',p2:'move 1'},{p1:'move 1',p2:'move 2'}]},
 ...[1,50,100].map(level=>({id:`damage-level-${level}`,seed:0,p1:[lax(['Earthquake'],{level})],p2:[mew(['Tackle'],{level})],steps:[{p1:'move 1',p2:'move 1'}]})),
 {id:'burn-control',seed:0,p1:[mew(attack)],p2:[lax(attack)],setup:[{side:'p1',status:'brn'}],steps:[{p1:'move 2',p2:'move 1'}]},
 {id:'protect-counter-cap',seed:1,p1:[mew(['Protect'])],p2:[lax(['Tackle'])],setup:[{side:'p1',stall:256}],steps:[{p1:'move 1',p2:'move 1'}]},
 {id:'toxic-counter-cap',seed:1,p1:[mew(['Recover'])],p2:[lax(['Protect'])],setup:[{side:'p1',status:'tox',toxicStage:15}],steps:[{p1:'move 1',p2:'move 1'}]},
 ...['brn','par','tox'].map(status=>({id:`lum-${status}`,seed:1,p1:[mew(['Protect'],{item:'Lum Berry'})],p2:[lax(['Protect'])],setup:[{side:'p1',status}],steps:[{p1:'move 1',p2:'move 1'}]})),
 {id:'already-statused',seed:0,p1:[mew(['Will-O-Wisp'])],p2:[lax(['Tackle'])],setup:[{side:'p2',status:'par'}],steps:[{p1:'move 1',p2:'move 1'}]},
 {id:'two-burn-reserves',seed:1,p1:[mew(['Protect']),lax(['Tackle'])],p2:[mew(['Protect']),ghost(['Toxic'])],setup:[{side:'p1',status:'brn',hp:1},{side:'p2',status:'brn',hp:1}],steps:[{p1:'move 1',p2:'move 1'},{p1:'switch 2',p2:'switch 2'}]},
 {id:'double-switch',seed:1,p1:[mew(attack),lax(attack)],p2:[mew(attack),ghost(['Toxic'])],steps:[{p1:'switch 2',p2:'switch 2'},{p1:'switch 2',p2:'switch 2'}]},
 ...[false,true].map(slow=>({id:`pivot-${slow?'slow':'fast'}`,seed:1,p1:[mew(['U-turn'],{level:slow?1:50}),lax(attack)],p2:[lax(['Tackle'],{level:slow?50:1}),ghost(['Protect'])],steps:[{p1:'move 1',p2:'move 1'}]})),
 {id:'pivot-no-reserve',seed:0,p1:[mew(['U-turn'])],p2:[lax(['Tackle'])],steps:[{p1:'move 1',p2:'move 1'}]},
 {id:'commit-cancel',seed:1,p1:[mew(attack)],p2:[lax(attack)],steps:[{p1:'move 1'},{p1:'undo'},{p2:'move 2'},{p1:'move 2'}]},
 {id:'last-pp',seed:1,p1:[mew(attack)],p2:[lax(attack)],setup:[{side:'p1',pp:1}],steps:[{p1:'move 2',p2:'move 1'}]},
 ...[0,1,9,21].map(seed=>({id:`tackle-tie-${seed}`,seed,p1:[mew(['Tackle'])],p2:[mew(['Tackle'])],steps:[{p1:'move 1',p2:'move 1'}]})),
);

// Migrate the accepted v6/v7 boundary inputs without modifying their old APIs.
const legacyAccuracy=[
 ['tackle-tackle',0,252,0,50,50,1,1],['tackle-mega',1,252,0,50,50,1,2],
 ['mega-tackle',1,0,252,50,50,2,1],['mega-mega',1,0,252,50,50,2,2],
 ['critical',21,0,252,50,50,1,2],['direct-ko',21,252,0,100,1,2,1],
];
for(const [id,seed,a,b,l1,l2,m1,m2] of legacyAccuracy)cases.push({id:`M03-${id}`,seed,
 p1:[mew(attack,{level:l1,evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:a}})],
 p2:[mew(attack,{level:l2,evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:b}})],steps:[{p1:`move ${m1}`,p2:`move ${m2}`} ]});
for(const [id,seed,a,b,l1,l2,status,m1,m2,hp] of [
 ['control',1,252,0,50,50,null,1,1],['critical',21,252,0,50,50,'p1',2,1],
 ['p2-faster',1,0,252,50,50,'p2',2,1],['miss',0,252,0,50,50,'p1',2,1],
 ['direct-ko',21,252,0,100,1,'p1',2,1],['residual-ko',0,0,252,50,50,'p1',1,2,1],
])cases.push({id:`M04-${id}`,seed,
 p1:[mew(attack,{level:l1,evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:a}})],
 p2:[mew(attack,{level:l2,evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:b}})],
 ...(status?{setup:[{side:status,status:'brn',...(hp?{hp}:{})}]}:{}),steps:[{p1:`move ${m1}`,p2:`move ${m2}`} ]});
for(const move of ['tackle','megapunch'])cases.push({id:`M03-last-${move}`,seed:move==='tackle'?1:0,
 p1:[mew(attack,{evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:252}})],p2:[mew(attack)],
 setup:[{side:'p1',ppByMove:{[move]:1}}],steps:[{p1:`move ${move==='tackle'?1:2}`,p2:'move 2'},{p1:`move ${move==='tackle'?2:1}`,p2:'move 1'}]});
cases.push(
 ...[false,true].map(alt=>({id:`privacy-hp-${alt?'b':'a'}`,seed:1,p1:[lax(['Protect'])],p2:[mew(['Recover'])],setup:[{side:'p2',hp:alt?85:86}],steps:[{p1:'move 1',p2:'move 1'}]})),
 {id:'near-full-recover',seed:1,p1:[mew(['Recover'])],p2:[lax(['Protect'])],setup:[{side:'p1',hp:86}],steps:[{p1:'move 1',p2:'move 1'}]},
 {id:'near-full-switch',seed:1,p1:[lax(['Tackle']),mew(['Tackle'])],p2:[lax(['Protect'])],setup:[{side:'p1',mon:1,hp:174}],steps:[{p1:'switch 2',p2:'move 1'}]},
 {id:'struggle-protected',seed:1,p1:[mew(['Protect'],{evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:252}})],p2:[mew(['Tackle'])],setup:[{side:'p2',pp:0}],steps:[{p1:'move 1',p2:'move 1'}]},
 {id:'burned-struggle',seed:1,p1:[mew(['Tackle'])],p2:[lax(['Tackle'])],setup:[{side:'p1',status:'brn',pp:0}],steps:[{p1:'move 1',p2:'move 1'}]},
 {id:'paralysis-pivot-refresh',seed:3,p1:[mew(['Thunder Wave'])],p2:[lax(['U-turn']),ghost(['Toxic'])],steps:[{p1:'move 1',p2:'move 1'},{p2:'switch 2'}]},
 {id:'two-burn-swapped',seed:1,p1:[mew(['Protect'])],p2:[mew(['Protect'],{evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:252}})],setup:[{side:'p1',status:'brn',hp:1},{side:'p2',status:'brn',hp:1}],steps:[{p1:'move 1',p2:'move 1'}]},
);

// Companion boundaries for counters, pivot order, Lum cures and privacy.
const allzero={hp:0,atk:0,def:0,spa:0,spd:0,spe:0};
const fast={...allzero,spe:252};
for(const stall of [2,4,8,16,32,64,128,256]) {
 // Synthetic counter with a real PRNG seed whose next upper word is zero.
 // This observes success at the otherwise extremely rare 2**32 boundary;
 // no sampled value or native callback is overridden.
 const mod=1n<<64n,A=0x5D588B656C078965n,C=0x269EC3n;
 let inv=1n;for(let i=0;i<6;i++)inv=inv*(2n-A*inv)%mod;
 const seedValue=((-C*inv)%mod+mod)%mod;
 const seedWords=[48n,32n,16n,0n].map(shift=>Number((seedValue>>shift)&65535n));
 cases.push({id:`protect-success-${stall}`,seed:1,seedWords,p1:[mew(['Protect'],{evs:fast})],p2:[mew(['Tackle'])],
  setup:[{side:'p1',stall}],steps:[{p1:'move 1',p2:'move 1'}]});
}
for(const seed of [0,9,21])cases.push({id:`protect-chain-${seed}`,seed,p1:[mew(['Protect'],{evs:fast})],p2:[mew(['Tackle'])],
 steps:Array.from({length:5},()=>({p1:'move 1',p2:'move 1'}))});
for(const [move,status] of [['Will-O-Wisp','brn'],['Toxic','tox'],['Thunder Wave','par']])cases.push({
 id:`lum-inflicted-${status}`,seed:0,p1:[ghost([move])],p2:[status==='tox'?mew(['Tackle'],{item:'Lum Berry'}):lax(['Tackle'],{item:'Lum Berry'})],steps:[{p1:'move 1',p2:'move 1'}]});
cases.push(
 {id:'already-same-status',seed:0,p1:[mew(['Will-O-Wisp'])],p2:[lax(['Tackle'])],setup:[{side:'p2',status:'brn'}],steps:[{p1:'move 1',p2:'move 1'}]},
 {id:'pivot-slower-resume',seed:1,p1:[lax(['U-turn']),ghost(['Toxic'])],p2:[mew(['Tackle'])],steps:[{p1:'move 1',p2:'move 1'},{p1:'switch 2'}]},
 {id:'pivot-ko-resume',seed:1,p1:[mew(['U-turn'],{level:100}),lax(['Tackle'])],p2:[lax(['Tackle'],{level:1}),ghost(['Toxic'])],steps:[{p1:'move 1',p2:'move 1'},{p1:'switch 2'},{p2:'switch 2'}]},
 {id:'pivot-paralyzed',seed:21,p1:[mew(['U-turn']),ghost(['Toxic'])],p2:[lax(['Tackle'])],setup:[{side:'p1',status:'par'}],steps:[{p1:'move 1',p2:'move 1'}]},
 {id:'pivot-burn-last-pp',seed:0,p1:[mew(['U-turn']),ghost(['Toxic'])],p2:[lax(['Tackle'])],setup:[{side:'p1',status:'brn',pp:1}],steps:[{p1:'move 1',p2:'move 1'},{p1:'switch 2'}]},
 {id:'protect-paralyzed',seed:21,p1:[mew(['Protect'])],p2:[lax(['Tackle'])],setup:[{side:'p1',status:'par',stall:4}],steps:[{p1:'move 1',p2:'move 1'}]},
 {id:'toxic-pivot',seed:1,p1:[mew(['U-turn']),ghost(['Toxic'])],p2:[lax(['Tackle'])],setup:[{side:'p1',status:'tox',toxicStage:3}],steps:[{p1:'move 1',p2:'move 1'},{p1:'switch 2'}]},
 ...[false,true].map(alt=>({id:`privacy-${alt?'b':'a'}`,seed:1,p1:[mew(attack)],p2:[lax(attack),ghost(alt?['Recover']:['Protect'],{item:alt?'Lum Berry':'Leftovers'})],steps:[{p2:alt?'move 2':'move 1'}]})),
);

for(const [id,moves,tied,seed] of [
 ['M01-duel-tied',['Tackle'],true,1],['M01-duel-unequal',['Tackle'],false,21],
 ['M02-duel-two-move',['Tackle','Pound'],true,9],['M03-duel-accuracy',attack,false,0],
])cases.push({id,seed,legacyDuel:true,family:'legacy',p1:[mew(moves,{evs:tied?allzero:fast})],p2:[mew(moves)],steps:[]});

// Predeclared deterministic scripts. They use native requests only, never JAX.
for (const [index,family] of ['direct','status','pivot','exhaustion'].entries()) {
 const physical=['Earthquake','U-turn','Mega Punch','Tackle'];
 const teams=family==='status' ? [
  [mew(['Will-O-Wisp','Recover','U-turn','Mega Punch'],{item:'Lum Berry'}),lax(['Toxic','Earthquake','U-turn','Protect'],{item:'Leftovers'}),ghost(['Thunder Wave','U-turn','Mega Punch','Protect'])],
  [mew(['Toxic','Protect','U-turn','Earthquake'],{item:'Leftovers'}),lax(['Will-O-Wisp','Earthquake','U-turn','Protect'],{item:'Lum Berry'}),ghost(['Toxic','Earthquake','U-turn','Recover'])]
 ] : [[mew(physical,{evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:252}}),lax(physical),ghost(physical)],
       [lax(physical),ghost(physical),mew(physical)]];
 for(const mirrored of [false,true]) {
  const [p1,p2]=mirrored?[teams[1],teams[0]]:teams;
  cases.push({id:`F0${index+1}-${mirrored?'p2':'p1'}`,family,seed:mirrored?21:1,p1,p2,
   ...(family==='exhaustion'?{setup:['p1','p2'].flatMap(side=>[0,1,2].map(mon=>({side,mon,pp:0})))}:{}),
   fullBattle:true,steps:[]});
 }
}
cases.push(...require('./roster-hazard-cases.cjs')(mon));
cases.push(...require('./generic-data-cases.cjs')(mon));
cases.push(...require('./status-duration-cases.cjs')(mon));
cases.push(...require('./field-boost-cases.cjs')(mon));
cases.push(...require('./ability-item-cases.cjs')(mon));
cases.push(...require('./hit-effect-cases.cjs')(mon));
cases.push(...require('./recharge-cases.cjs')(mon));
cases.push(...require('./lock-phase-cases.cjs')(mon));

function scriptedChoices(battle,scenario,boundary) {
 const choices={};
 for(const side of battle.sides) {
  const req=side.activeRequest;if(!req||req.wait)continue;
  const space=actionSpace(req);const commands=space.commands.filter(Boolean);
  if(req.forceSwitch){choices[side.id]=commands[0];continue;}
  if(commands.includes('move 1')&&req.active[0].moves[0].id==='struggle'){choices[side.id]='move 1';continue;}
  if(scenario.legacyDuel){const available=commands.filter(c=>c.startsWith('move '));choices[side.id]=available[(battle.turn+side.n)%available.length];continue;}
  const pokemon=side.active[0],foe=side.foe.active[0];
  const moveCommand=id=>{const n=req.active[0].moves.findIndex(m=>m.id===id&&!m.disabled);return n<0?null:`move ${n+1}`;};
  if(scenario.family==='direct'&&battle.turn===1&&side.n===0&&commands.includes('switch 2')){choices[side.id]='switch 2';continue;}
  if(scenario.family==='status'&&battle.turn<=3) {
   const status=moveCommand('willowisp')??moveCommand('toxic')??moveCommand('thunderwave');
   if(status&&battle.turn===1){choices[side.id]=status;continue;}
   if(battle.turn===2&&moveCommand('protect')){choices[side.id]=moveCommand('protect');continue;}
   if(battle.turn===3&&moveCommand('recover')){choices[side.id]=moveCommand('recover');continue;}
  }
  if(scenario.family==='hazards') {
   const wanted=battle.turn===1?'stealthrock':battle.turn===2?'spikes':(side.sideConditions.spikes||side.sideConditions.stealthrock)?'rapidspin':null;
   if(wanted&&moveCommand(wanted)){choices[side.id]=moveCommand(wanted);continue;}
  }
  if(scenario.family==='move-phases') {
   const wanted=battle.turn%5===1?['taunt','encore','disable','torment','sunnyday','reflect']:[];
   const status=wanted.map(moveCommand).find(Boolean);if(status){choices[side.id]=status;continue;}
   const attacks=req.active[0].moves.map((m,i)=>!m.disabled&&battle.dex.moves.get(m.id).category!=='Status'?`move ${i+1}`:null).filter(Boolean);
   choices[side.id]=attacks[0]??commands.find(c=>c.startsWith('move '))??commands[0];continue;
  }
  if(scenario.family==='ability-item') {
   if(!side.sideConditions.reflect&&moveCommand('reflect')){choices[side.id]=moveCommand('reflect');continue;}
   const attacks=req.active[0].moves.map((m,i)=>!m.disabled&&battle.dex.moves.get(m.id).category!=='Status'?`move ${i+1}`:null).filter(Boolean);
   choices[side.id]=attacks[0]??commands.find(c=>c.startsWith('move '))??commands[0];continue;
  }
  if(scenario.family==='field-boost') {
   const wanted=battle.turn===1?['raindance','sunnyday','sandstorm','hail']:
     battle.turn===2?['reflect','lightscreen']:battle.turn===3?['calmmind','swordsdance','irondefense','agility']:[];
   const early=wanted.map(moveCommand).find(Boolean);
   if(early){choices[side.id]=early;continue;}
   const attacks=req.active[0].moves.map((m,i)=>!m.disabled&&battle.dex.moves.get(m.id).category!=='Status'?`move ${i+1}`:null).filter(Boolean);
   choices[side.id]=attacks[0]??commands.find(c=>c.startsWith('move '))??commands[0];continue;
  }
  if(scenario.family==='status-duration') {
   const early=battle.turn===1?moveCommand('spore'):battle.turn===2?moveCommand('confuseray'):null;
   if(early){choices[side.id]=early;continue;}
   if(battle.turn%7===0&&pokemon.hp<pokemon.maxhp/2&&moveCommand('rest')){choices[side.id]=moveCommand('rest');continue;}
   const attacks=req.active[0].moves.map((m,i)=>!m.disabled&&battle.dex.moves.get(m.id).category!=='Status'?`move ${i+1}`:null).filter(Boolean);
   choices[side.id]=attacks[0]??commands.find(c=>c.startsWith('move '))??commands[0];continue;
  }
  const preferred=scenario.family==='pivot'?['uturn','earthquake','megapunch','tackle']:
   foe.species.name==='Gengar'?['uturn','earthquake','megapunch','tackle']:['earthquake','megapunch','uturn','tackle'];
  choices[side.id]=preferred.map(moveCommand).find(Boolean)??commands.find(c=>c.startsWith('move '))??commands[0];
 }
 return choices;
}

// Campaign decisions depend only on public native requests and a separate,
// predeclared policy seed. They never consume or replace the battle PRNG.
function campaignChoices(battle,scenario,boundary) {
 const choices={};
 for(const side of battle.sides) {
  const req=side.activeRequest;if(!req||req.wait)continue;
  const commands=actionSpace(req).commands.filter(Boolean);
  let word=(scenario.policySeeds[side.n] + Math.imul(boundary+1,0x9e3779b9))>>>0;
  word=Math.imul(word^(word>>>16),0x21f0aaad)>>>0;
  word=Math.imul(word^(word>>>15),0x735a2d97)>>>0;word=(word^(word>>>15))>>>0;
  const random=()=>commands[word%commands.length];
  if(req.forceSwitch||scenario.policy==='random'){choices[side.id]=random();continue;}
  const moves=commands.filter(c=>c.startsWith('move '));
  const byId=id=>moves.find(c=>req.active[0].moves[Number(c.split(' ')[1])-1].id===id);
  const edge=scenario.policy==='edge'&&battle.turn<=8;
  const priority=edge?['protect','thunderwave','willowisp','toxic','uturn','recover']:
   battle.turn<=2?['thunderwave','willowisp','toxic']:['uturn','megapunch','earthquake','tackle','pound'];
  choices[side.id]=priority.map(byId).find(Boolean)??moves[word%moves.length]??random();
 }
 return choices;
}

function observeInitialization(ps, action) {
 const original=ps.PRNG.prototype.random;const draws=[];
 ps.PRNG.prototype.random=function(...args) {
  const before=this.getSeed().split(',').map(Number); const value=original.apply(this,args);
  draws.push({args,value,before,after:this.getSeed().split(',').map(Number),context:'initialization'}); return value;
 };
 try {return {battle:action(),draws};} finally {ps.PRNG.prototype.random=original;}
}
function capture(battle, ids, options={}) {
 const ref=p=>p?{side:p.side.n,mon:ids.get(p)}:null;
 // Side.emitRequest(...,true) adds a resend notification to the mutable request.
 // Freeze its decision data now; raw notifications remain in privatePackets.
 const requestSnapshot=request=>{
  if(!request)return request;
  const snapshot=structuredClone(request);delete snapshot.update;
  // Empty means no disabled-source information. Preserve every nonempty source.
  for(const active of snapshot.active??[])for(const move of active.moves??[])if(move.disabledSource==='')delete move.disabledSource;
  return snapshot;
 };
 function effectState(value,depth=0) {
  if(depth>32)throw Error('Effect-state capture depth exceeded');
  if(value===null||typeof value!=='object') {
   if(typeof value==='function'||typeof value==='symbol')throw Error('Uncaptured effect-state value');
   return value;
  }
  if(ids.has(value))return {$pokemon:ref(value)};
  if(battle.sides.includes(value))return {$side:value.n};
  if(value===battle)return {$battle:true};
  if(value===battle.field)return {$field:true};
  if(value.effectType&&value.id)return {$effect:value.id,type:value.effectType};
  if(Array.isArray(value))return value.map(v=>effectState(v,depth+1));
  if(Object.getPrototypeOf(value)!==Object.prototype&&Object.getPrototypeOf(value)!==null)throw Error('Uncaptured effect-state object '+value.constructor?.name);
  return Object.fromEntries(Object.keys(value).sort().map(k=>[k,effectState(value[k],depth+1)]));
 }
 const typed=x=>x===undefined?{$undefined:true}:x;
 const moveState=x=>x?{id:x.id,type:x.type,category:x.category,basePower:x.basePower}:null;
 const tracking=p=>({lastMove:moveState(p.lastMove),lastMoveEncore:moveState(p.lastMoveEncore),lastMoveUsed:moveState(p.lastMoveUsed),
  lastMoveTargetLoc:typed(p.lastMoveTargetLoc),lastDamage:p.lastDamage,activeTurns:p.activeTurns,activeMoveActions:p.activeMoveActions,
  moveThisTurn:p.moveThisTurn,moveThisTurnResult:typed(p.moveThisTurnResult),moveLastTurnResult:typed(p.moveLastTurnResult),
  hurtThisTurn:p.hurtThisTurn,timesAttacked:p.timesAttacked,trapped:p.trapped,maybeTrapped:p.maybeTrapped,maybeDisabled:p.maybeDisabled,
  newlySwitched:p.newlySwitched,usedItemThisTurn:p.usedItemThisTurn,lastItem:p.lastItem,statsRaisedThisTurn:p.statsRaisedThisTurn,
  statsLoweredThisTurn:p.statsLoweredThisTurn,moveSlotsUsed:p.moveSlots.map(m=>m.used),ateBerry:p.ateBerry,previouslySwitchedIn:p.previouslySwitchedIn,
  attackedBy:p.attackedBy.map(a=>({source:ref(a.source),damage:a.damage,move:a.move,thisTurn:a.thisTurn,slot:a.slot,damageValue:typed(a.damageValue)}))});
 const queue=battle.queue.list.map(a=>({choice:a.choice,order:a.order,priority:a.priority??0,speed:a.speed??0,
  actor:ref(a.pokemon),target:ref(a.target),move:a.moveid??null,source:a.sourceEffect?.id??null}));
 return {
  rng:battle.prng.getSeed().split(',').map(Number),turn:battle.turn,terminal:battle.ended,
  winner:battle.ended?(battle.winner==='p1'?1:battle.winner==='p2'?2:0):-1,
  midTurn:battle.midTurn,requestState:battle.requestState,queue,
  ...(options.tracking?{phaseState:{lastMove:moveState(battle.lastMove),lastSuccessfulMoveThisTurn:battle.lastSuccessfulMoveThisTurn,lastDamage:battle.lastDamage}}:{}),
  ...(options.effects?{effectOrder:battle.effectOrder}:{}),
  ...(options.field?{field:{weather:battle.field.weather,weatherState:effectState(battle.field.weatherState),
    pseudoWeather:effectState(battle.field.pseudoWeather),terrain:battle.field.terrain,terrainState:effectState(battle.field.terrainState)}}:{}),
  sides:battle.sides.map(side=>({
   ...(options.tracking?{phaseState:{faintedThisTurn:ref(side.faintedThisTurn),faintedLastTurn:ref(side.faintedLastTurn)}}:{}),
   ...(options.field?{sideConditions:effectState(side.sideConditions)}:{}),
   hazards:[Number(!!side.sideConditions.stealthrock),side.sideConditions.spikes?.layers??0,({stealthrock:1,spikes:2})[Object.keys(side.sideConditions).find(k=>['stealthrock','spikes'].includes(k))]??0],
   active:ids.get(side.active[0]),order:side.pokemon.map(p=>ids.get(p)),
   request:requestSnapshot(side.activeRequest),actions:side.activeRequest?actionSpace(side.activeRequest):null,
   pending:side.choice.actions.map(a=>({choice:a.choice,actor:ref(a.pokemon),move:a.moveid??null,target:ref(a.target)})),
   choiceDone:side.isChoiceDone(),
   pokemon:[...side.pokemon].sort((a,b)=>ids.get(a)-ids.get(b)).map(p=>({
    ...(options.tracking?{phaseState:tracking(p)}:{}),
    id:ids.get(p),species:p.species.name,level:p.level,gender:p.gender,hp:p.hp,maxhp:p.maxhp,
    name:p.name,types:p.getTypes(),ivs:p.set.ivs,evs:p.set.evs,nature:p.set.nature??'Hardy',
    hpType:p.hpType,hpPower:p.hpPower,shiny:!!p.set.shiny,baseAbility:p.baseAbility,
    ...(options.effects?{statusState:effectState(p.statusState),volatiles:effectState(p.volatiles),
      abilityState:effectState(p.abilityState),itemState:effectState(p.itemState),speciesState:effectState(p.speciesState)}:{}),
    status:p.status,toxicStage:p.statusState.stage??0,item:p.item,ability:p.ability,
    active:p.isActive,fainted:p.fainted,speed:p.speed,stats:p.storedStats,
    moves:p.moveSlots.map(m=>({id:m.id,pp:m.pp,maxpp:m.maxpp,disabled:m.disabled})),
    boosts:{...p.boosts},stall:p.volatiles.stall?{counter:p.volatiles.stall.counter,duration:p.volatiles.stall.duration}:null,
    protect:!!p.volatiles.protect,switchFlag:p.switchFlag,forceSwitchFlag:p.forceSwitchFlag,
   }))
  }))
 };
}
function events(battle,start=0) {
 return Object.fromEntries(['p1','p2'].map(side=>[side,projectPackets(['update\n'+battle.log.slice(start).join('\n')],side).records]));
}
function semanticContext(draw) {
 const stack=draw.stack.join('\n');
 if(stack.includes('BattleActions.modifyDamage')&&draw.context==='speed-sort')return 'damage-modifiers';
 if(draw.eventId==='WeatherChange')return 'weather-change';
 if(draw.eventId==='Weather')return 'weather-order';
 if(draw.effectId==='lockedmove')return 'rampage-duration';
 if(draw.effectId==='sleeptalk')return 'sleep-talk-selection';
 if(draw.effectId==='slp')return 'sleep-duration';
 if(draw.effectId==='frz')return 'freeze';
 if(draw.effectId==='confusion') {
  if(draw.context==='damage')return 'confusion-damage';
  if(draw.to!==undefined)return 'confusion-duration';
  return 'confusion';
 }
 if(stack.includes('BattleActions.secondaries'))return 'secondary';
 if(draw.context==='accuracy'||draw.context==='critical'||draw.context==='damage')return draw.context;
 if(stack.includes('onBeforeMove'))return 'paralysis';
 if(stack.includes('onStallMove'))return 'stall';
 if(stack.includes('BattleQueue.sort')||stack.includes('BattleQueue.insertChoice'))return 'queue-sort';
 if(stack.includes('Battle.fieldEvent'))return 'residual-handlers';
 if(stack.includes('BattleActions.runSwitch'))return 'switch-order';
 if(draw.eventId==='BeforeTurn')return 'before-turn';
 if(stack.includes('hitStepMoveHitLoop')&&draw.context==='unsupported')return 'multihit-count';
 if(stack.includes('hitStepMoveHitLoop'))return 'hit-update';
 if(draw.eventId==='Update'||stack.includes('runPursuitActivation'))return 'update';
 throw Error('Unclassified native draw context: '+JSON.stringify(draw));
}
function runCase(ps, scenario, {tracking=true}={}) {
 tracking=tracking||!!scenario.lockFamily;
 const {battle,draws:initializationDraws}=observeInitialization(ps,()=>new ps.Battle({formatid:'gen5randombattle',seed:scenario.seedWords??[scenario.seed,2,3,4],
  p1:{name:'p1',team:scenario.p1},p2:{name:'p2',team:scenario.p2}}));
 try {
  const ids=new Map(battle.sides.flatMap(side=>side.pokemon.map((p,i)=>[p,i])));
  const rng=battle.prng.getSeed();
  for(const setup of scenario.setup??[]) {
   const p=battle[setup.side].pokemon[setup.mon??0];
   if(setup.status)p.setStatus(setup.status,p);
   if(setup.hp!==undefined)p.sethp(setup.hp);
   if(setup.pp!==undefined)for(const m of p.moveSlots)m.pp=setup.pp;
   if(setup.ppByMove)for(const m of p.moveSlots)if(Object.hasOwn(setup.ppByMove,m.id))m.pp=setup.ppByMove[m.id];
   if(setup.toxicStage!==undefined)p.statusState.stage=setup.toxicStage;
   if(setup.stall!==undefined){p.addVolatile('stall');p.volatiles.stall.counter=setup.stall;}
  }
  if(scenario.setup)battle.makeRequest('move');
  assert.equal(battle.prng.getSeed(),rng,'trusted boundary setup must not consume RNG');
  const snapshot=()=>capture(battle,ids,{tracking,effects:tracking||scenario.statusFamily||scenario.fieldFamily||scenario.hookFamily||scenario.hitFamily||scenario.rechargeFamily||scenario.lockFamily,field:tracking||scenario.fieldFamily||scenario.statusFamily||scenario.hookFamily||scenario.hitFamily||scenario.rechargeFamily||scenario.lockFamily});
  const record={id:scenario.id,classification:scenario.setup?'synthetic':'native-constructed',setup:scenario,initializationDraws,
   initial:snapshot(),initialEvents:events(battle),steps:[]};
  let boundary=0;
  while((scenario.fullBattle||scenario.legacyDuel||scenario.campaign)?!battle.ended:boundary<scenario.steps.length) {
   if(boundary>=512) {
    if(scenario.campaign){record.truncated=true;break;}
    throw Error(`${scenario.id}: truncation at 512 request boundaries`);
   }
   const choices=scenario.campaign?campaignChoices(battle,scenario,boundary):
    (scenario.fullBattle||scenario.legacyDuel)?scriptedChoices(battle,scenario,boundary):scenario.steps[boundary];
   boundary++;
   const before=snapshot(), start=battle.log.length,acks=[];
   const oldSend=battle.send; battle.send=function(type,data){if(type==='sideupdate')acks.push(data);return oldSend.call(this,type,data);};
   let observed;
   try {observed=withObservedBattleDraws(ps,battle,()=>Object.entries(choices).map(([side,choice])=>{
    if(choice==='undo'){battle.undoChoice(side);return {side,choice,accepted:!battle[side].choice.actions.length};}
    return {side,choice,accepted:battle.choose(side,choice)};
   }),{includeEffects:tracking||scenario.statusFamily||scenario.fieldFamily||scenario.hookFamily||scenario.hitFamily||scenario.rechargeFamily||scenario.lockFamily});} finally {battle.send=oldSend;}
   record.steps.push({choices,accepted:observed.result,before,after:snapshot(),
    draws:observed.draws.map(draw=>({...draw,semanticContext:semanticContext(draw)})),events:events(battle,start),privatePackets:acks});
  }
  if(scenario.fullBattle||scenario.legacyDuel)assert.ok(battle.ended,scenario.id+': must reach real native terminal');
  return record;
 } finally {battle.destroy();}
}
function main(argv) {
 let batch='all',output,campaign,campaignStart=0,campaignIndices,coverageOutput;
 for(let i=0;i<argv.length;i++) {
  if(argv[i]==='--batch')batch=argv[++i];else if(argv[i]==='--output')output=argv[++i];else if(argv[i]==='--campaign')campaign=argv[++i];else if(argv[i]==='--campaign-start')campaignStart=Number(argv[++i]);else if(argv[i]==='--campaign-indices')campaignIndices=argv[++i].split(',').map(Number);else if(argv[i]==='--coverage')coverageOutput=argv[++i];else throw Error('Unknown argument '+argv[i]);
 }
 assertReady(); const ps=require('./vendor/pokemon-showdown/dist/sim');
 if(campaign) {
  assert.equal(output,undefined,'campaign streams NDJSON on stdout');
  const config=JSON.parse(fs.readFileSync(campaign,'utf8'));
  assert.equal(config.schemaVersion,1);
  assert.ok(Number.isInteger(campaignStart)&&campaignStart>=0&&campaignStart<config.cases.length);
  if(campaignIndices)assert.ok(campaignIndices.every(i=>Number.isInteger(i)&&i>=0&&i<config.cases.length));
  const selected=campaignIndices?campaignIndices.map(i=>config.cases[i]):config.cases.slice(campaignStart);
  const coverage={schemaVersion:1,pin,config:campaign,starts:0,completed:0,truncated:0,boundaries:0,decisions:0,ordinary:0,synthetic:0,effects:{},interactions:{},drawContexts:{},perStart:[]};
  const increment=(table,key)=>{table[key]=(table[key]??0)+1;};
  for(const scenario of selected) {
   const record=runCase(ps,scenario);
   if(!coverageOutput){fs.writeSync(1,JSON.stringify(record)+'\n');continue;}
   coverage.starts++;coverage[record.classification==='synthetic'?'synthetic':'ordinary']++;
   const last=record.steps.at(-1)?.after??record.initial;
   coverage[last.terminal?'completed':'truncated']++;
   coverage.perStart.push({id:record.id,boundaries:record.steps.length,winner:last.winner,truncated:!!record.truncated});
   for(const step of record.steps) {
    coverage.boundaries++;coverage.decisions+=Object.keys(step.choices).length;
    const names=new Set();
    for(const event of step.events.p1) {
     const parts=event.message.split('|'),kind=parts[1];if(!kind)continue;
     if(kind==='move')names.add('move:'+parts[3]);
     else if(kind==='-status'||kind==='-curestatus')names.add(kind+':'+parts[3]);
     else if(kind==='-enditem')names.add('item:'+parts[3]);
     else if(kind==='switch')names.add(event.message.includes('[from] U-turn')?'pivot-switch':'switch');
     else if(kind==='-damage'&&parts[4])names.add(parts[4]);
     else if(kind==='-heal'&&parts[4])names.add(parts[4]);
     else if(kind==='-activate')names.add('activate:'+parts[3]);
     else names.add(kind);
    }
    const sorted=[...names].sort();
    for(const name of sorted)increment(coverage.effects,name);
    for(let i=0;i<sorted.length;i++)for(let j=i+1;j<sorted.length;j++)increment(coverage.interactions,sorted[i]+' + '+sorted[j]);
    for(const draw of step.draws)increment(coverage.drawContexts,draw.semanticContext);
   }
  }
  if(coverageOutput)fs.writeFileSync(coverageOutput,JSON.stringify(coverage,null,2)+'\n',{flag:'wx'});
  return;
 }
 if(batch==='tracking') {
  if(!output)throw Error('Tracking requires a fresh NDJSON output path');
  const fd=fs.openSync(output,'wx');let boundaries=0;
  const write=x=>fs.writeSync(fd,JSON.stringify(x)+'\n');
  try {
   write({kind:'header',schemaVersion:1,pin,format:'gen5randombattle',caseSchema:'gen5-causal-v1'});
   for(const c of cases){const record=runCase(ps,c,{tracking:true});boundaries+=record.steps.length;write({kind:'case',...record});}
   write({kind:'summary',cases:cases.length,boundaries});
  } finally {fs.closeSync(fd);}
  return;
 }
 const selected=batch==='locks'?cases.filter(c=>c.lockFamily):batch==='recharge'?cases.filter(c=>c.rechargeFamily):batch==='hits'?cases.filter(c=>c.hitFamily):batch==='hooks'?cases.filter(c=>c.hookFamily&&!c.hitFamily&&!c.rechargeFamily&&!c.lockFamily):batch==='field'?cases.filter(c=>c.fieldFamily):batch==='status'?cases.filter(c=>c.statusFamily):batch==='generic'?cases.filter(c=>c.generic&&!c.statusFamily&&!c.fieldFamily&&!c.hookFamily&&!c.hitFamily&&!c.rechargeFamily&&!c.lockFamily):batch==='g4'?cases.filter(c=>c.g4):batch==='all'?cases.filter(c=>!c.g4&&!c.generic):batch==='boundaries'?cases.filter(c=>!c.g4&&!c.generic&&!c.fullBattle):batch==='battles'?cases.filter(c=>!c.g4&&!c.generic&&c.fullBattle):cases.filter(c=>c.id===batch);
 if(!selected.length)throw Error('Empty native case selection: '+batch);
 const result={schemaVersion:1,pin,format:'gen5randombattle',caseSchema:selected.some(c=>c.lockFamily)?'gen5-lock-phase-v1':selected.some(c=>c.rechargeFamily)?'gen5-recharge-v1':selected.some(c=>c.hitFamily)?'gen5-hit-effects-v1':selected.some(c=>c.hookFamily)?'gen5-ability-item-v1':selected.some(c=>c.fieldFamily)?'gen5-field-boost-v1':selected.some(c=>c.statusFamily)?'gen5-status-duration-v1':selected.some(c=>c.generic)?'shared-engine-v3':selected.some(c=>c.g4)?'shared-engine-v2':'shared-engine-v1',cases:selected.map(c=>runCase(ps,c))};
 const json=JSON.stringify(result,null,2)+'\n';
 if(output)fs.writeFileSync(output,json,{flag:'wx'});else process.stdout.write(json);
}
if(require.main===module)main(process.argv.slice(2));
module.exports={cases,runCase,capture,observeInitialization};
