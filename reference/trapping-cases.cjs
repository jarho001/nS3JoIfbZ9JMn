'use strict';
module.exports=function(mon){
 const m=(species='Mew',ability='Synchronize',moves=['Tackle','Recover'],options={})=>mon(species,moves,{ability,...options});
 const both=(a=1,b=1)=>({p1:`move ${a}`,p2:`move ${b}`});
 const cases=[];
 const add=(id,p1,p2,steps,extra={})=>cases.push({id:'G4-trap-'+id,seed:1,generic:true,hookFamily:true,trappingFamily:true,p1,p2,steps,...extra});
 const bench=()=>m();
 for(const [id,source,ability]of [['shadow','Gothitelle','Shadow Tag'],['arena','Dugtrio','Arena Trap'],['magnet','Magnezone','Magnet Pull']]){
  const target=id==='magnet'?m('Magnezone','Magnet Pull'):m();
  add(id+'-disclose',[target,bench()],[m(source,ability)],[{p1:'switch 2'},{p1:'switch 2'},both(),{p1:'switch 2'}]);
  add(id+'-opponent-pending',[target,bench()],[m(source,ability)],[{p2:'move 1'},{p1:'switch 2'},{p1:'move 1'}]);
  add(id+'-no-bench',[target],[m(source,ability)],[both()]);
 }
 add('ghost-shadow',[m('Gengar','Levitate'),bench()],[m('Gothitelle','Shadow Tag')],[{p1:'switch 2'},both()]);
 add('shadow-self-immune',[m('Gothitelle','Shadow Tag'),bench()],[m('Gothitelle','Shadow Tag')],[{p1:'switch 2'},{p2:'move 1'}]);
 add('arena-levitate',[m('Gengar','Levitate'),bench()],[m('Dugtrio','Arena Trap')],[{p1:'switch 2'},{p2:'move 1'}]);
 add('arena-flying',[m('Charizard','Pressure'),bench()],[m('Dugtrio','Arena Trap')],[{p1:'switch 2'},{p2:'move 1'}]);
 add('magnet-nonsteel',[m(),bench()],[m('Magnezone','Magnet Pull')],[{p1:'switch 2'},{p2:'move 1'}]);
 add('query-frisk-undo',[m(),bench()],[m('Gothitelle','Frisk')],[{p1:'switch 2'},{p1:'undo'},{p1:'move 1'},{p2:'move 1'}]);
 add('query-frisk-item',[m('Mew','Synchronize',['Tackle'],{item:'Leftovers'}),bench()],[m('Gothitelle','Frisk')],[both()]);
 add('query-frisk-entry',[m(),bench()],[m(),m('Gothitelle','Frisk')],[{p1:'move 1',p2:'switch 2'},{p1:'switch 2'},{p2:'move 1'}]);
 add('query-frisk-replace',[m(),bench()],[m('Gothitelle','Frisk')],[{p1:'switch 2'},{p1:'switch 2'},{p2:'move 1'}]);
 add('query-custom-arena',[m(),bench()],[m('Dugtrio','Synchronize')],[{p1:'switch 2'},{p1:'undo'},{p2:'move 1'}]);
 add('query-custom-magnet',[m('Magnezone','Magnet Pull'),bench()],[m('Magnezone','Synchronize')],[{p1:'switch 2'},{p1:'undo'},{p2:'move 1'}]);
 add('pivot-escapes',[m('Mew','Synchronize',['U-turn']),bench()],[m('Gothitelle','Shadow Tag')],[both(),{p1:'switch 2'}]);
 add('forced-escapes',[m(),bench()],[m('Gothitelle','Shadow Tag')],[both(),{p1:'switch 2'}],{setup:[{side:'p1',hp:1}]});
 add('trap-source-switch',[m(),bench()],[m('Gothitelle','Shadow Tag'),bench()],[{p1:'move 1',p2:'switch 2'},{p1:'switch 2'},{p2:'move 1'}]);
 add('recharge-hidden',[m('Mew','Synchronize',['Hyper Beam']),bench()],[m('Gothitelle','Shadow Tag')],[both(),{p1:'switch 2'},both()]);
 const a=[m('Mew','Synchronize',['U-turn','Tackle']),m('Magnezone','Magnet Pull',['Tackle','Recover']),m('Gothitelle','Shadow Tag'),m('Dugtrio','Arena Trap'),m('Charizard','Pressure'),m()];
 const b=[m('Gothitelle','Frisk',['Tackle'],{item:'Leftovers'}),m('Dugtrio','Arena Trap'),m('Magnezone','Magnet Pull'),m(),m('Gothitelle','Shadow Tag'),m('Mew','Pressure')];
 for(const mirror of [false,true])add('complete-'+(mirror?2:1),mirror?b:a,mirror?a:b,[],{fullBattle:true,family:'trapping',seed:mirror?21:1});
 for(const c of [...cases]){
  if(!['shadow-opponent-pending','arena-levitate','query-frisk-undo','query-frisk-item','magnet-disclose'].some(id=>c.id==='G4-trap-'+id))continue;
  const mirror=structuredClone(c);mirror.id+='-mirror';[mirror.p1,mirror.p2]=[mirror.p2,mirror.p1];
  mirror.steps=mirror.steps.map(step=>Object.fromEntries(Object.entries(step).map(([side,choice])=>[side==='p1'?'p2':'p1',choice])));
  cases.push(mirror);
 }
 return cases;
};
