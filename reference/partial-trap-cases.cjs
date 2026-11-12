'use strict';
module.exports=function(){
const mon=(moves,extra={})=>({species:'Mew',ability:'Synchronize',moves,item:'',level:50,nature:'Hardy',gender:'N',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...extra});
const fast={evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:252}};
const both=(a=1,b=1)=>({p1:`move ${a}`,p2:`move ${b}`});
const cases=[],add=(id,steps,extra={})=>cases.push({id:'G4-partial-trap-'+id,seed:1,generic:true,hitFamily:true,partialTrapFamily:true,p1:[mon(['Bind','Recover'],fast),mon(['Tackle'])],p2:[mon(['Recover','Substitute','Rapid Spin','U-turn']),mon(['Tackle'])],steps,...extra});
for(const item of ['','Binding Band','Grip Claw'])add('expiry-'+(item||'ordinary'),[both(),...Array.from({length:8},()=>both(2))],{p1:[mon(['Bind','Recover'],{...fast,item}),mon(['Tackle'])]});
add('repeat',[both(),both(),both(),both(2)]);
add('source-switch',[both(),{p1:'switch 2',p2:'move 1'},both()]);
add('substitute-clears',[both(),both(2,2),both()]);
add('substitute-blocks',[both(2,2),both(),both()]);
add('spin-clears',[both(),both(2,3),{p1:'move 2',p2:'switch 2'}]);
add('pivot-escapes',[both(),both(2,4),{p2:'switch 2'}]);
add('known-trap-retry',[both(),{p2:'switch 2'},{p1:'move 2'},{p2:'move 1'}]);
add('no-bench',[both(),both(2)],{p2:[mon(['Recover'])]});
add('source-ko',[both()],{p1:[mon(['Bind'],fast),mon(['Tackle'])],p2:[mon(['Tackle'])],setup:[{side:'p1',hp:1}]});

for(const move of ['Clamp','Fire Spin','Magma Storm','Sand Tomb','Whirlpool','Wrap'])add(move.replaceAll(' ',''),[both(),both(2),both(2)],{p1:[mon([move,'Recover'],fast),mon(['Tackle'])]});
add('expiry-other-duration',[both(),...Array.from({length:7},()=>both(2))],{seed:21});
add('tied-both',[both(),...Array.from({length:6},()=>both(2))],{p1:[mon(['Bind','Recover']),mon(['Tackle'])],p2:[mon(['Bind','Recover']),mon(['Tackle'])]});
add('burn-leftovers',[both(),both(2),both(2)],{p2:[mon(['Recover'],{item:'Leftovers'}),mon(['Tackle'])],setup:[{side:'p2',status:'brn'}]});
add('source-pivot',[both(),both(2),{p1:'switch 2'},both()],{p1:[mon(['Bind','U-turn'],fast),mon(['Tackle'])]});
add('spin-after-subdamage',[both(),both(2,3)],{p1:[mon(['Bind','Substitute'],fast),mon(['Tackle'])]});
add('residual-ko',[both()],{p2:[mon(['Tackle']),mon(['Tackle'])],setup:[{side:'p2',hp:12}]});
for(const id of ['known-trap-retry','substitute-clears','spin-after-subdamage']){const m=structuredClone(cases.find(c=>c.id==='G4-partial-trap-'+id));m.id+='-mirror';[m.p1,m.p2]=[m.p2,m.p1];m.steps=m.steps.map(s=>Object.fromEntries(Object.entries(s).map(([k,v])=>[k==='p1'?'p2':'p1',v])));if(m.setup)m.setup=m.setup.map(s=>({...s,side:s.side==='p1'?'p2':'p1'}));cases.push(m);}
return cases;
};
