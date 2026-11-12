'use strict';
// Native-only witness on currently implemented neutral sets; no JAX execution.
const fs=require('node:fs'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const root=require('node:path').resolve(__dirname,'..');
const {capture,observeInitialization,semanticContext}=require(root+'/reference/shared-engine-cases.cjs');
const {assertReady,withObservedBattleDraws}=require(root+'/reference/neutral-turn-oracle.cjs');
const {projectPackets}=require(root+'/environment/protocol.cjs');
assertReady();const ps=require(root+'/reference/vendor/pokemon-showdown/dist/sim');
const team=()=>Array.from({length:6},()=>({species:'Mew',moves:['Tackle'],ability:'Synchronize',level:50,nature:'Hardy',gender:'N',evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0}}));
const config={formatid:'gen5randombattle',seed:[1,2,3,4],p1:{name:'p1',team:team()},p2:{name:'p2',team:team()}};
const setup={generic:true,p1:structuredClone(config.p1.team),p2:structuredClone(config.p2.team),seed:1,seedWords:[...config.seed]};
const {battle:b,draws:initializationDraws}=observeInitialization(ps,()=>new ps.Battle(config));
const ids=new Map(b.sides.flatMap(s=>s.pokemon.map((p,i)=>[p,i])));
const snap=()=>capture(b,ids,{tracking:true,effects:true,field:true});
const events=start=>Object.fromEntries(['p1','p2'].map(side=>[side,projectPackets(['update\n'+b.log.slice(start).join('\n')],side).records]));
const steps=[],samples=[],chain=crypto.createHash('sha256');let boundaries=0,rows=events(0).p1.length;
const initial=snap(),initialEvents=events(0);
try{
 while(!b.ended){
  assert.ok(boundaries<1002);const before=snap(),start=b.log.length;
  const observation=withObservedBattleDraws(ps,b,()=>['p1','p2'].map(side=>({side,choice:'switch 2',accepted:b.choose(side,'switch 2')})),{includeEffects:true});
  assert.ok(observation.result.every(a=>a.accepted));const visible=events(start),after=snap();
  const record={boundary:++boundaries,before,after,draws:observation.draws,events:visible};
  steps.push({choices:{p1:'switch 2',p2:'switch 2'},accepted:observation.result,before,after,events:visible,draws:observation.draws.map(d=>({...d,semanticContext:semanticContext(d)}))});
  rows+=visible.p1.length;chain.update(JSON.stringify(record)+'\n');
  if(boundaries<=2||b.ended||visible.p1.some(e=>/^\|(bigerror|message)\|/.test(e.message)))samples.push(record);
 }
 const result={id:'G4-turn-limit-neutral',setup,initialEvents,steps,status:'native_only_not_JAX_acceptance',pin:require(root+'/reference-lock.json').commit,config,initial,initializationDraws,boundaries,rows,chainSha256:chain.digest('hex'),samples};
 fs.writeFileSync(process.argv[2],JSON.stringify(result)+'\n',{flag:'wx'});
 console.log(JSON.stringify({boundaries,rows,warnings:samples.length-2,terminal:b.ended}));
}finally{b.destroy();}
