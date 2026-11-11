'use strict';
// Native-constructed inputs; boundary installations remain explicitly synthetic.
module.exports=function rechargeCases(mon) {
 const m=(moves,o={})=>mon('Mew',moves,{ability:'Synchronize',...o});
 const fast={evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:252}};
 const tank={evs:{hp:252,atk:0,def:252,spa:0,spd:252,spe:0}};
 const both=(a=1,b=1)=>({p1:`move ${a}`,p2:`move ${b}`});
 const cases=require('./move-phase-cases.cjs')(mon).filter(c=>c.id.startsWith('G4-phase-recharge-'));
 const add=(id,extra={})=>cases.push({id:`G4-recharge-${id}`,seed:1,p1:[m(['Hyper Beam','Recover'],fast),m(['Tackle'])],p2:[m(['Recover'],tank)],steps:[both(),both()],...extra});
 add('slot2-pending-undo',{p1:[m(['Pound','Hyper Beam'],fast),m(['Tackle'])],steps:[both(2),{p1:'move 1'},{p1:'undo'},both()]});
 add('reject-switch',{steps:[both(),{p1:'switch 2'},both()]});
 add('reject-slot2',{steps:[both(),{p1:'move 2'},both()]});
 add('protect',{p2:[m(['Protect','Recover'],tank)],steps:[both(),both(1,2)]});
 add('immune',{p2:[mon('Gengar',['Recover'])],steps:[both()]});
 add('pressure',{p2:[m(['Recover'],{ability:'Pressure',...tank})]});
 for(const item of ['Choice Band','Choice Specs','Choice Scarf','Life Orb'])add(item.replaceAll(' ',''),{p1:[m(['Pound','Giga Impact'],{item,...fast}),m(['Tackle'])],steps:[both(2),both()]});
 for(const move of ['Spore','Thunder Wave','Confuse Ray'])add(move.replaceAll(' ',''),{p2:[m([move,'Recover'],tank)],steps:[both(),both(1,2),both(1,2)]});
 add('last-pp',{setup:[{side:'p1',pp:0,ppByMove:{hyperbeam:1}}],steps:[both(),both(),both()]});
 add('target-ko',{p2:[m(['Recover'],tank),m(['Recover'],tank)],setup:[{side:'p2',hp:1}],steps:[both(),{p2:'switch 2'},both()]});
 add('terminal',{setup:[{side:'p2',hp:1}],steps:[both()]});
 for(const hp of [1,100])add(`contact-${hp}`,{p1:[m(['Giga Impact'],fast),m(['Tackle'])],p2:[m(['Recover'],{ability:'Rough Skin',item:'Rocky Helmet',...tank})],setup:[{side:'p1',hp}],steps:[both()]});
 add('orb-ko',{p1:[m(['Hyper Beam'],{item:'Life Orb',...fast}),m(['Tackle'])],setup:[{side:'p1',hp:1}],steps:[both()]});
 add('residual-ko',{setup:[{side:'p1',hp:1,status:'brn'}],steps:[both()]});
 add('both-tied',{p1:[m(['Hyper Beam'])],p2:[m(['Giga Impact'])],steps:[both(),both(),both()]});
 add('both-recharge',{seed:0,p1:[m(['Hyper Beam'])],p2:[m(['Giga Impact'])],steps:[both(),both(),both()]});
 add('frozen-recharge',{seed:9,p2:[m(['Ice Beam','Recover'],tank)],steps:[both(),both(1,2),both(1,2)]});
 add('ko-before-recharge',{p1:[m(['Hyper Beam']),m(['Tackle'])],p2:[m(['Tackle'],{...tank,evs:{...tank.evs,spe:252}})],setup:[{side:'p1',hp:35}],steps:[both(),both()]});
 add('duration-sort',{p1:[m(['Hyper Beam'])],p2:[m(['Protect','Recover'])],steps:[both(1,2),both(),both(1,2)]});
 add('forced-before-recharge',{p1:[m(['Hyper Beam'],fast),m(['Tackle'])],p2:[m(['Tackle']),m(['Tackle'])],setup:[{side:'p1',hp:30},{side:'p2',hp:1}],steps:[both(),{p2:'switch 2'},both()]});
 for(const mirror of [false,true]){
  const teams=[Array.from({length:6},(_,i)=>m([i%2?'Giga Impact':'Hyper Beam'],{...(i%2?{}:fast),item:i%3?'':'Life Orb'})),Array.from({length:6},(_,i)=>m([i%2?'Hyper Beam':'Giga Impact'],{item:i%3?'':'Leftovers'}))];
  cases.push({id:`G4-recharge-complete-${mirror?2:1}`,seed:mirror?21:1,p1:teams[mirror?1:0],p2:teams[mirror?0:1],fullBattle:true,family:'ability-item',steps:[]});
 }
 return cases.map(c=>({...c,generic:true,phaseFamily:false,rechargeFamily:true}));
};
