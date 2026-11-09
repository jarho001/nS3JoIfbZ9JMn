'use strict';
// Inputs only. Shared runner supplies independent state/draw/view expectations.
module.exports=function statusDurationCases(mon) {
  const set=(moves,o={})=>mon('Mew',moves,{ability:'Synchronize',...o});
  const lax=(moves,o={})=>mon('Snorlax',moves,{ability:'Immunity',...o});
  const both=(a,b)=>({p1:`move ${a}`,p2:`move ${b}`});
  const fast={evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:252}};
  const cases=[];
  for(const seed of [0,1,9,21])for(const slower of [false,true])cases.push({
    id:`G4-status-sleep-${seed}-${slower}`,seed,
    p1:[set(['Spore','Recover'],slower?{}:fast)],p2:[set(['Recover'],slower?fast:{})],
    steps:[both(1,1),...Array.from({length:4},()=>both(2,1))]});
  cases.push({id:'G4-status-sleep-switch-reset',seed:1,
    p1:[set(['Spore','Recover'],fast)],p2:[set(['Recover']),lax(['Protect'])],
    steps:[both(1,1),{p1:'move 2',p2:'switch 2'},both(2,1),{p1:'move 2',p2:'switch 2'},both(2,1),both(2,1),both(2,1)]},
  {id:'G4-status-sleep-clause',seed:0,p1:[set(['Spore','Recover'],fast)],p2:[set(['Recover']),set(['Recover'])],
    steps:[both(1,1),{p1:'move 1',p2:'switch 2'},both(1,1)]},
  {id:'G4-status-sleep-lum',seed:1,p1:[set(['Spore','Recover'],fast)],p2:[set(['Tackle'],{item:'Lum Berry'})],steps:[both(1,1),both(2,1)]},
  {id:'G4-status-rest-full-failure',seed:1,p1:[set(['Rest'])],p2:[lax(['Protect'])],steps:[both(1,1)]},
  {id:'G4-status-rest',seed:1,p1:[set(['Rest','Recover'],fast)],p2:[lax(['Tackle'],{level:1})],
    setup:[{side:'p1',hp:50,status:'par'}],steps:[both(1,1),both(1,1),both(2,1),both(2,1),both(2,1)]},
  {id:'G4-status-rest-lum',seed:0,p1:[set(['Rest','Recover'],{...fast,item:'Lum Berry'})],p2:[lax(['Tackle'])],
    setup:[{side:'p1',hp:1}],steps:[both(1,1),both(2,1)]},
  {id:'G4-status-rest-exempt-clause',seed:9,p1:[set(['Recover','Spore'])],p2:[set(['Rest'],fast),set(['Recover'])],
    setup:[{side:'p2',hp:1}],steps:[both(1,1),{p1:'move 2',p2:'switch 2'}]});
  for(const seed of [0,1,9,21])cases.push({id:`G4-status-confusion-${seed}`,seed,
    p1:[set(['Confuse Ray','Recover'],fast)],p2:[lax(['Recover','Protect'])],
    steps:[both(1,1),both(2,1),both(2,1),both(2,1),both(2,1),both(2,1)]});
  cases.push({id:'G4-status-confusion-reapply',seed:1,p1:[set(['Confuse Ray'],fast)],p2:[lax(['Recover'])],steps:[both(1,1),both(1,1)]},
  {id:'G4-status-confusion-lum',seed:1,p1:[set(['Confuse Ray'],fast)],p2:[lax(['Tackle'],{item:'Lum Berry'})],steps:[both(1,1)]},
  {id:'G4-status-confusion-paralysis',seed:21,p1:[set(['Confuse Ray','Recover'],fast)],p2:[set(['Recover'])],
    setup:[{side:'p2',status:'par'}],steps:[both(1,1),both(2,1),both(2,1),both(2,1)]});
  for(const seed of [0,1,9,21])cases.push({id:`G4-status-freeze-${seed}`,seed,
    p1:[set(['Recover'])],p2:[lax(['Protect'])],setup:[{side:'p1',status:'frz'}],steps:Array.from({length:6},()=>both(1,1))});
  cases.push({id:'G4-status-poison',seed:1,p1:[set(['Poison Powder','Recover'],fast)],p2:[set(['Recover'])],steps:[both(1,1),both(2,1),both(2,1)]},
  {id:'G4-status-poison-immunity',seed:1,p1:[set(['Poison Powder'],fast)],p2:[lax(['Recover'])],steps:[both(1,1)]},
  {id:'G4-status-poison-lum',seed:1,p1:[set(['Poison Powder'],fast)],p2:[set(['Tackle'],{item:'Lum Berry'})],steps:[both(1,1)]});
  // Native secondary branches are observed across a bounded, fixed seed grid.
  // No critical, roll, secondary result or duration is replaced.
  for(const move of ['Ice Beam','Flamethrower','Sludge Bomb','Thunderbolt'])for(const seed of [0,1,9,21])cases.push({
    id:`G4-status-secondary-${move.replaceAll(' ','')}-${seed}`,seed,
    p1:[set([move,'Recover'],fast)],p2:[set(['Recover'])],steps:[both(1,1),both(1,1),both(1,1),both(2,1)]});
  cases.push({id:'G4-status-secondary-ko',seed:1,p1:[set(['Ice Beam'],fast)],p2:[set(['Recover'])],
    setup:[{side:'p2',hp:1}],steps:[both(1,1)]},
  {id:'G4-status-secondary-protected',seed:1,p1:[set(['Ice Beam'],fast)],p2:[set(['Protect'])],steps:[both(1,1)]},
  {id:'G4-status-secondary-existing',seed:1,p1:[set(['Ice Beam'],fast)],p2:[set(['Recover'])],
    setup:[{side:'p2',status:'par'}],steps:[both(1,1)]});
  // Invert four real Gen 5 LCG steps (gender, accuracy, duration, self-hit)
  // to exercise confusion KO. No draw result or native callback is replaced.
  const modulus=1n<<64n,A=0x5D588B656C078965n,C=0x269EC3n;
  let inverse=1n;for(let i=0;i<6;i++)inverse=inverse*(2n-A*inverse)%modulus;
  let word=0xC000000000000000n;
  for(let i=0;i<4;i++)word=((word-C)*inverse%modulus+modulus)%modulus;
  const seedWords=[48n,32n,16n,0n].map(shift=>Number((word>>shift)&65535n));
  cases.push({id:'G4-status-confusion-self-ko',seed:1,seedWords,
    p1:[set(['Confuse Ray','Recover'],fast)],p2:[lax(['Tackle'])],setup:[{side:'p2',hp:1}],
    steps:[both(1,1)]});
  for(const seed of [0,21])cases.push({id:`G4-status-fire-thaw-${seed}`,seed,
    p1:[set(['Flamethrower'],fast)],p2:[set(['Recover'])],setup:[{side:'p2',status:'frz'}],steps:[both(1,1)]});
  for(const seed of [0,21])for(const swap of [false,true]) {
    const a=[set(['Spore','Confuse Ray','Flamethrower','Recover'],{...fast,item:'Lum Berry'}),
      lax(['Ice Beam','Thunderbolt']),set(['Sludge Bomb','Rest']),lax(['Flamethrower']),
      set(['Ice Beam','Recover']),lax(['Thunderbolt'])];
    const b=[set(['Rest','Ice Beam','Thunderbolt','Tackle']),lax(['Confuse Ray','Flamethrower']),
      set(['Spore','Sludge Bomb']),lax(['Ice Beam']),set(['Flamethrower','Recover']),lax(['Thunderbolt'])];
    cases.push({id:`G4-status-full-${seed}-${swap}`,seed,p1:swap?b:a,p2:swap?a:b,
      fullBattle:true,family:'status-duration'});
  }
  return cases.map(c=>({...c,generic:true,statusFamily:true}));
};
