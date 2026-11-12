'use strict';
module.exports=function(){
const mon=(moves,o={})=>({species:'Mew',moves,ability:'Synchronize',item:'',level:50,nature:'Hardy',gender:'N',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:252},...o});
const both=(a=1,b=1)=>({p1:`move ${a}`,p2:`move ${b}`});
const slow={evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0}};
const cases=[];
function add(id,foe,extra={}){cases.push({id:'G4-substitute-'+id,generic:true,hitFamily:true,substituteFamily:true,seed:1,p1:[mon(['Substitute','Recover','Tackle']),mon(['Tackle'])],p2:[mon(foe==='Recover'?['Recover','Tackle']:[foe,'Recover'],slow)],steps:[both(),both(2),both(3)],...extra});}
for(const move of ['Pound','Double Slap','Bullet Seed','Giga Drain','Double-Edge','Hyper Beam','Outrage','Fly','Sky Attack','Thunder Wave','Toxic','Spore','Confuse Ray','Taunt','Encore','Disable','Torment','Air Slash','Rapid Spin'])add(move.replaceAll(' ',''),move);
add('weak','Recover',{setup:[{side:'p1',hp:43}],steps:[both()]});
add('already-present','Recover',{steps:[both(),both()]});
add('status-before-sub','Recover',{setup:[{side:'p1',status:'tox'}],steps:[both(),both(2),both(2)]});
add('rough-skin','Recover',{p1:[mon(['Substitute','Recover'],{ability:'Rough Skin',item:'Rocky Helmet'}),mon(['Tackle'])],p2:[mon(['Pound'],slow)],steps:[both(),both(2),both(2)]});
add('life-orb','Pound',{p2:[mon(['Pound'],{...slow,item:'Life Orb'})]});
add('switch-clear','Recover',{steps:[both(),{p1:'switch 2',p2:'move 1'},{p1:'switch 2',p2:'move 1'}]});
for(const c of cases){if(c.id.endsWith('-Taunt'))c.steps=[both(),both(3),both(3)];if(c.id.endsWith('-Encore'))c.steps=[both(),both(),both()];}

add('break-multihit','Bullet Seed',{p1:[mon(['Substitute'],{level:100})],p2:[mon(['Bullet Seed'],{...slow,ability:'Skill Link',level:100,nature:'Adamant',evs:{hp:0,atk:252,def:0,spa:0,spd:0,spe:0}})],steps:[both()]});
add('sound','Hyper Voice');
add('struggle','Tackle',{setup:[{side:'p2',pp:0}],steps:[both(),both(2)]});
add('drain-low','Giga Drain',{setup:[{side:'p2',hp:1}],steps:[both()]});
add('drain-ooze','Giga Drain',{p1:[mon(['Substitute','Recover'],{ability:'Liquid Ooze'})],setup:[{side:'p2',hp:1}],steps:[both()]});
add('recoil-ko','Double-Edge',{setup:[{side:'p2',hp:1}],steps:[both()]});
add('rock-head','Double-Edge',{p2:[mon(['Double-Edge'],{...slow,ability:'Rock Head'})],steps:[both()]});
add('protect','Pound',{p1:[mon(['Substitute','Protect','Tackle']),mon(['Tackle'])],steps:[both(),both(2),both(3)]});
add('pivot','U-turn',{p2:[mon(['U-turn'],slow),mon(['Tackle'])],steps:[both(),{p2:'switch 2'},both(2)]});
add('spin-hazards','Rapid Spin',{p1:[mon(['Stealth Rock','Substitute','Recover'])],steps:[both(1,2),both(2),both(3)]});
add('sleep-blocked','Spore',{steps:[both(),both(2)]});
add('weak-boundary','Recover',{setup:[{side:'p1',hp:43}],steps:[both()]});
add('minimal-success','Recover',{setup:[{side:'p1',hp:44}],steps:[both()]});
add('shedinja','Recover',{p1:[mon(['Substitute'],{species:'Shedinja',ability:'Pressure'})],steps:[both()]});
add('tied-status','Thunder Wave',{p1:[mon(['Substitute','Recover'],slow)],steps:[both(1,2),both(2)]});
add('tied-secondary','Air Slash',{p1:[mon(['Substitute','Recover'],slow)],steps:[both(1,2),both(2)]});
add('selfdestruct','Self-Destruct',{steps:[both()]});
for(const c of [...cases].filter(c=>['break-multihit','recoil-ko','drain-ooze','pivot','weak','Taunt','HyperBeam'].some(k=>c.id==='G4-substitute-'+k))){const m=structuredClone(c);m.id+='-mirror';[m.p1,m.p2]=[m.p2,m.p1];m.steps=m.steps.map(s=>Object.fromEntries(Object.entries(s).map(([k,v])=>[k==='p1'?'p2':'p1',v])));if(m.setup)m.setup=m.setup.map(s=>({...s,side:s.side==='p1'?'p2':'p1'}));cases.push(m);}
for(const mirror of [false,true]){const a=Array.from({length:6},(_,i)=>mon(['Substitute','Tackle'],i%2?slow:{}));const b=Array.from({length:6},(_,i)=>mon([i%2?'Bullet Seed':'Double-Edge','Tackle'],{...slow,ability:i%2?'Skill Link':'Rock Head'}));cases.push({id:'G4-substitute-complete-'+(mirror?2:1),generic:true,hitFamily:true,substituteFamily:true,seed:mirror?21:1,p1:mirror?b:a,p2:mirror?a:b,steps:[],fullBattle:true,family:'substitute'});}
return cases;
};
