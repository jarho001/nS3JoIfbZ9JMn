'use strict';
module.exports=function(mon){
 const m=(moves,extra={})=>mon('Mew',moves,{gender:'N',...extra}),fast={evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:252}};
 const both=(a=1,b=1)=>({p1:`move ${a}`,p2:`move ${b}`});
 const cases=[],add=(id,steps,extra={})=>cases.push({id:'G4-pursuit-'+id,generic:true,pursuitFamily:true,hitFamily:true,seed:1,p1:[m(['Tackle','U-turn','Substitute'],fast),m(['Tackle'])],p2:[m(['Pursuit','Recover']),m(['Tackle'])],steps,...extra});
 add('ordinary',[both(),both()]);add('switch',[{p1:'switch 2',p2:'move 1'},both()]);
 add('switch-tied',[{p1:'switch 2',p2:'move 1'}],{p1:[m(['Tackle']),m(['Tackle'])]});
 add('victim-ko',[{p1:'switch 2',p2:'move 1'},{p1:'switch 2'}],{setup:[{side:'p1',hp:1}]});
 add('both-switch',[{p1:'switch 2',p2:'switch 2'}]);
 add('pivot-slower',[both(2),{p1:'switch 2'},both()]);
 add('pivot-faster',[both(),{p1:'switch 2'},both()],{p1:[m(['U-turn','Tackle']),m(['Tackle'])],p2:[m(['Pursuit'],fast),m(['Tackle'])]});
 add('pivot-ko',[both(2),{p1:'switch 2'}],{setup:[{side:'p1',hp:1}]});
 add('substitute',[both(3),{p1:'switch 2',p2:'move 1'}]);
 add('protect',[both()],{p1:[m(['Protect'],fast),m(['Tackle'])]});
 for(const status of ['par','frz'])add('status-'+status,[{p1:'switch 2',p2:'move 1'}],{setup:[{side:'p2',status}]});
 add('status-slp',[both(),{p1:'switch 2',p2:'move 1'}],{p1:[m(['Spore'],fast),m(['Tackle'])]});
 add('pressure-last-pp',[{p1:'switch 2',p2:'move 1'},both(1,2)],{p1:[m(['Tackle'],{...fast,ability:'Pressure'}),m(['Tackle'])],setup:[{side:'p2',pp:1}]});
 add('contact-source-ko',[{p1:'switch 2',p2:'move 1'}],{p1:[m(['Tackle'],{...fast,ability:'Rough Skin'}),m(['Tackle'])],setup:[{side:'p2',hp:1}]});
 add('contact-double-ko',[{p1:'switch 2',p2:'move 1'},{p1:'switch 2',p2:'switch 2'}],{p1:[m(['Tackle'],{...fast,ability:'Rough Skin'}),m(['Tackle'])],setup:[{side:'p1',hp:1},{side:'p2',hp:1}]});
 add('phazing',[both()],{p1:[m(['Roar'],fast),m(['Tackle'])]});
 add('choice',[{p1:'switch 2',p2:'move 1'},both()],{p2:[m(['Pursuit','Recover'],{item:'Choice Band'}),m(['Tackle'])]});
 add('technician',[{p1:'switch 2',p2:'move 1'},both()],{p2:[m(['Pursuit'],{ability:'Technician'}),m(['Tackle'])]});
 add('both-pursuit',[both(),both()],{p1:[m(['Pursuit']),m(['Tackle'])]});
 for(const id of ['switch-tied','victim-ko','pivot-slower']){const c=structuredClone(cases.find(c=>c.id==='G4-pursuit-'+id));c.id+='-mirror';[c.p1,c.p2]=[c.p2,c.p1];c.steps=c.steps.map(s=>Object.fromEntries(Object.entries(s).map(([k,v])=>[k==='p1'?'p2':'p1',v])));if(c.setup)c.setup=c.setup.map(s=>({...s,side:s.side==='p1'?'p2':'p1'}));cases.push(c);}
 return cases;
};
