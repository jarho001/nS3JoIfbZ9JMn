'use strict';
module.exports=function(mon){
 const m=(moves,extra={})=>mon('Mew',moves,{gender:'N',...extra});
 const fast={evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:252}};
 const both=(a=1,b=1)=>({p1:`move ${a}`,p2:`move ${b}`});
 const cases=[],add=(id,move,extra={})=>cases.push({id:'G4-phaze-'+id,generic:true,phazingFamily:true,hitFamily:true,seed:1,p1:[m([move,'Recover'],fast),m(['Tackle'])],p2:[m(['Recover','Tackle']),m(['Tackle']),m(['Pound'])],steps:[both(),both(2)],...extra});
 for(const move of ['Roar','Whirlwind','Dragon Tail','Circle Throw']){
  const key=move.replaceAll(' ','');add(key,move);
  add(key+'-no-bench',move,{p2:[m(['Recover'])],steps:[both()]});
  add(key+'-substitute',move,{p2:[m(['Substitute']),m(['Tackle'])],steps:[both()]});
  add(key+'-protect',move,{p2:[m(['Protect']),m(['Tackle'])],steps:[both()]});
  add(key+'-suction',move,{p2:[m(['Recover'],{ability:'Suction Cups'}),m(['Tackle'])],steps:[both()]});
 }
 add('roar-soundproof','Roar',{p2:[m(['Recover'],{ability:'Soundproof'}),m(['Tackle'])],steps:[both()]});
 add('whirlwind-soundproof','Whirlwind',{p2:[m(['Recover'],{ability:'Soundproof'}),m(['Tackle'])],steps:[both()]});
 add('tied-cancel','Roar',{p1:[m(['Roar']),m(['Tackle'])],p2:[m(['Whirlwind']),m(['Tackle'])],steps:[both()]});
 add('hazards','Roar',{p1:[m(['Roar','Stealth Rock','Spikes'],fast)],steps:[both(2),both(3),both()]});
 add('hazard-ko','Roar',{p1:[m(['Roar','Stealth Rock'],fast)],setup:[{side:'p2',mon:1,hp:1}],p2:[m(['Recover']),m(['Tackle'])],steps:[both(2),both()]});
 add('target-ko','Dragon Tail',{p2:[m(['Tackle']),m(['Tackle'])],setup:[{side:'p2',hp:1}],steps:[both()]});
 add('source-contact-ko','Dragon Tail',{p2:[m(['Recover'],{ability:'Rough Skin'}),m(['Tackle'])],setup:[{side:'p1',hp:1}],steps:[both()]});
 add('source-orb-ko','Dragon Tail',{p1:[m(['Dragon Tail'],{...fast,item:'Life Orb'}),m(['Tackle'])],setup:[{side:'p1',hp:1}],steps:[both()]});
 add('clear-bound-recharge','Roar',{p1:[m(['Roar','Bind'],fast)],p2:[m(['Hyper Beam']),m(['Tackle'])],steps:[both(2),both()]});
 add('soundproof-damage','Hyper Voice',{p2:[m(['Recover'],{ability:'Soundproof'}),m(['Tackle'])],steps:[both()]});
 add('source-orb-terminal','Dragon Tail',{p1:[m(['Dragon Tail'],{...fast,item:'Life Orb'})],setup:[{side:'p1',hp:1}],steps:[both()]});
 add('pivot-interrupted','Roar',{p2:[m(['U-turn']),m(['Tackle'])],steps:[both(),{p2:'switch 2'}]});
 add('soundproof-ghost','Hyper Voice',{p2:[mon('Gengar',['Recover'],{ability:'Soundproof',gender:'M'}),m(['Tackle'])],steps:[both()]});
 for(const id of ['tied-cancel','Roar','source-orb-ko']){const c=structuredClone(cases.find(c=>c.id==='G4-phaze-'+id));c.id+='-mirror';[c.p1,c.p2]=[c.p2,c.p1];c.steps=c.steps.map(s=>Object.fromEntries(Object.entries(s).map(([k,v])=>[k==='p1'?'p2':'p1',v])));if(c.setup)c.setup=c.setup.map(s=>({...s,side:s.side==='p1'?'p2':'p1'}));cases.push(c);}
 return cases;
};
