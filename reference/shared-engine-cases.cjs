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
function scriptedChoices(battle,scenario,boundary) {
 const choices={};
 for(const side of battle.sides) {
  const req=side.activeRequest;if(!req||req.wait)continue;
  const space=actionSpace(req);const commands=space.commands.filter(Boolean);
  if(req.forceSwitch){choices[side.id]=commands[0];continue;}
  if(commands.includes('move 1')&&req.active[0].moves[0].id==='struggle'){choices[side.id]='move 1';continue;}
  const pokemon=side.active[0],foe=side.foe.active[0];
  const moveCommand=id=>{const n=req.active[0].moves.findIndex(m=>m.id===id&&!m.disabled);return n<0?null:`move ${n+1}`;};
  if(scenario.family==='direct'&&battle.turn===1&&side.n===0&&commands.includes('switch 2')){choices[side.id]='switch 2';continue;}
  if(scenario.family==='status'&&battle.turn<=3) {
   const status=moveCommand('willowisp')??moveCommand('toxic')??moveCommand('thunderwave');
   if(status&&battle.turn===1){choices[side.id]=status;continue;}
   if(battle.turn===2&&moveCommand('protect')){choices[side.id]=moveCommand('protect');continue;}
   if(battle.turn===3&&moveCommand('recover')){choices[side.id]=moveCommand('recover');continue;}
  }
  const preferred=scenario.family==='pivot'?['uturn','earthquake','megapunch','tackle']:
   foe.species.name==='Gengar'?['uturn','earthquake','megapunch','tackle']:['earthquake','megapunch','uturn','tackle'];
  choices[side.id]=preferred.map(moveCommand).find(Boolean)??commands.find(c=>c.startsWith('move '))??commands[0];
 }
 return choices;
}

function observeInitialization(ps, action) {
 const original=ps.PRNG.prototype.random;const draws=[];
 ps.PRNG.prototype.random=function(...args) {
  const before=this.getSeed().split(',').map(Number); const value=original.apply(this,args);
  draws.push({args,value,before,after:this.getSeed().split(',').map(Number)}); return value;
 };
 try {return {battle:action(),draws};} finally {ps.PRNG.prototype.random=original;}
}
function capture(battle, ids) {
 const ref=p=>p?{side:p.side.n,mon:ids.get(p)}:null;
 const queue=battle.queue.list.map(a=>({choice:a.choice,order:a.order,priority:a.priority??0,speed:a.speed??0,
  actor:ref(a.pokemon),target:ref(a.target),move:a.moveid??null,source:a.sourceEffect?.id??null}));
 return {
  rng:battle.prng.getSeed().split(',').map(Number),turn:battle.turn,terminal:battle.ended,
  winner:battle.ended?(battle.winner==='p1'?1:battle.winner==='p2'?2:0):-1,
  midTurn:battle.midTurn,requestState:battle.requestState,queue,
  sides:battle.sides.map(side=>({
   active:ids.get(side.active[0]),order:side.pokemon.map(p=>ids.get(p)),
   request:side.activeRequest,actions:side.activeRequest?actionSpace(side.activeRequest):null,
   pending:side.choice.actions.map(a=>({choice:a.choice,actor:ref(a.pokemon),move:a.moveid??null,target:ref(a.target)})),
   choiceDone:side.isChoiceDone(),
   pokemon:[...side.pokemon].sort((a,b)=>ids.get(a)-ids.get(b)).map(p=>({
    id:ids.get(p),species:p.species.name,level:p.level,gender:p.gender,hp:p.hp,maxhp:p.maxhp,
    status:p.status,toxicStage:p.statusState.stage??0,item:p.item,ability:p.ability,
    active:p.isActive,fainted:p.fainted,speed:p.speed,stats:p.storedStats,
    moves:p.moveSlots.map(m=>({id:m.id,pp:m.pp,maxpp:m.maxpp,disabled:m.disabled})),
    boosts:p.boosts,stall:p.volatiles.stall?{counter:p.volatiles.stall.counter,duration:p.volatiles.stall.duration}:null,
    protect:!!p.volatiles.protect,switchFlag:p.switchFlag,forceSwitchFlag:p.forceSwitchFlag,
   }))
  }))
 };
}
function events(battle,start=0) {
 return Object.fromEntries(['p1','p2'].map(side=>[side,projectPackets(['update\n'+battle.log.slice(start).join('\n')],side).records]));
}
function runCase(ps, scenario) {
 const {battle,draws:initializationDraws}=observeInitialization(ps,()=>new ps.Battle({formatid:'gen5randombattle',seed:[scenario.seed,2,3,4],
  p1:{name:'p1',team:scenario.p1},p2:{name:'p2',team:scenario.p2}}));
 try {
  const ids=new Map(battle.sides.flatMap(side=>side.pokemon.map((p,i)=>[p,i])));
  const rng=battle.prng.getSeed();
  for(const setup of scenario.setup??[]) {
   const p=battle[setup.side].pokemon[setup.mon??0];
   if(setup.status)p.setStatus(setup.status,p);
   if(setup.hp!==undefined)p.sethp(setup.hp);
   if(setup.pp!==undefined)for(const m of p.moveSlots)m.pp=setup.pp;
   if(setup.toxicStage!==undefined)p.statusState.stage=setup.toxicStage;
   if(setup.stall!==undefined){p.addVolatile('stall');p.volatiles.stall.counter=setup.stall;}
  }
  if(scenario.setup)battle.makeRequest('move');
  assert.equal(battle.prng.getSeed(),rng,'trusted boundary setup must not consume RNG');
  const record={id:scenario.id,classification:scenario.setup?'synthetic':'native-constructed',setup:scenario,initializationDraws,
   initial:capture(battle,ids),initialEvents:events(battle),steps:[]};
  let boundary=0;
  while(scenario.fullBattle?!battle.ended:boundary<scenario.steps.length) {
   if(boundary>=512)throw Error(`${scenario.id}: truncation at 512 request boundaries`);
   const choices=scenario.fullBattle?scriptedChoices(battle,scenario,boundary):scenario.steps[boundary];
   boundary++;
   const before=capture(battle,ids), start=battle.log.length,acks=[];
   const oldSend=battle.send; battle.send=function(type,data){if(type==='sideupdate')acks.push(data);return oldSend.call(this,type,data);};
   let observed;
   try {observed=withObservedBattleDraws(ps,battle,()=>Object.entries(choices).map(([side,choice])=>{
    if(choice==='undo'){battle.undoChoice(side);return {side,choice,accepted:!battle[side].choice.actions.length};}
    return {side,choice,accepted:battle.choose(side,choice)};
   }));} finally {battle.send=oldSend;}
   record.steps.push({choices,accepted:observed.result,before,after:capture(battle,ids),
    draws:observed.draws,events:events(battle,start),privatePackets:acks});
  }
  if(scenario.fullBattle)assert.ok(battle.ended,scenario.id+': must reach real native terminal');
  return record;
 } finally {battle.destroy();}
}
function main(argv) {
 let batch='all',output;
 for(let i=0;i<argv.length;i++) {
  if(argv[i]==='--batch')batch=argv[++i];else if(argv[i]==='--output')output=argv[++i];else throw Error('Unknown argument '+argv[i]);
 }
 assertReady(); const ps=require('./vendor/pokemon-showdown/dist/sim');
 const selected=batch==='all'?cases:batch==='boundaries'?cases.filter(c=>!c.fullBattle):batch==='battles'?cases.filter(c=>c.fullBattle):cases.filter(c=>c.id===batch);
 if(!selected.length)throw Error('Empty native case selection: '+batch);
 const result={schemaVersion:1,pin,format:'gen5randombattle',caseSchema:'shared-engine-v1',cases:selected.map(c=>runCase(ps,c))};
 const json=JSON.stringify(result,null,2)+'\n';
 if(output)fs.writeFileSync(output,json,{flag:'wx'});else process.stdout.write(json);
}
if(require.main===module)main(process.argv.slice(2));
module.exports={cases,runCase,capture,observeInitialization};
