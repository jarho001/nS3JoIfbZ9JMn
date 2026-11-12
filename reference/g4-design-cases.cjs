'use strict';
// Native design witnesses only; these mechanics are not enabled in JAX.
const fs=require('node:fs'),path=require('node:path'),root=path.resolve(__dirname,'..');
const {runCase}=require(root+'/reference/shared-engine-cases.cjs');
const {assertReady}=require(root+'/reference/neutral-turn-oracle.cjs');
assertReady();const ps=require(root+'/reference/vendor/pokemon-showdown/dist/sim');
const m=(species,ability,moves=['Tackle'])=>({species,ability,moves,level:50,nature:'Hardy',item:'',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0}});
const cases=[
 {id:'called-reflected-frame-chain',generic:true,p1:[m('Mew','Synchronize',['Sleep Talk','Taunt'])],p2:[{...m('Mew','Magic Bounce',['Spore','Recover']),evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:252}}],steps:[{p1:'move 1',p2:'move 1'}]},
 {id:'query-shadowtag',generic:true,p1:[m('Mew','Synchronize'),m('Mew','Synchronize')],p2:[m('Gothitelle','Frisk')],steps:[]},
 {id:'hidden-trap-retry',generic:true,p1:[m('Mew','Synchronize'),m('Mew','Synchronize')],p2:[m('Gothitelle','Shadow Tag')],steps:[{p1:'switch 2'}]},
 {id:'gen5-ghost-trap',generic:true,p1:[m('Gengar','Levitate'),m('Mew','Synchronize')],p2:[m('Gothitelle','Shadow Tag')],steps:[{p1:'switch 2'}]},
 {id:'transform-base-and-effective',generic:true,p1:[m('Mew','Synchronize',['Transform','Recover'])],p2:[m('Snorlax','Immunity',['Hidden Power Ice','Protect'])],steps:[{p1:'move 1',p2:'move 2'}]},
].map(c=>({seed:1,...c}));
cases.push(cases.shift());
const records=cases.map(c=>runCase(ps,c));
fs.writeFileSync(process.argv[2],JSON.stringify({status:'native_only_design_evidence',pin:require(root+'/reference-lock.json').commit,cases:records},null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify({cases:records.length,boundaries:records.reduce((n,c)=>n+c.steps.length,0)}));
