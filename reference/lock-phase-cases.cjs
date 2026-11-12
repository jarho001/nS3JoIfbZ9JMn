'use strict';
module.exports=function lockPhaseCases(mon){
 const m=(moves,o={})=>mon('Mew',moves,{ability:'Synchronize',...o});
 const fast={evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:252}};
 const tank={evs:{hp:252,atk:0,def:252,spa:0,spd:252,spe:0}};
 const both=(a=1,b=1)=>({p1:`move ${a}`,p2:`move ${b}`});
 const cases=require('./move-phase-cases.cjs')(mon).filter(c=>/G4-phase-(rampage|charge|solar-|power-herb|taunt|encore|disable|torment)/.test(c.id));
 const add=(id,extra={})=>cases.push({id:`G4-lock-${id}`,seed:1,p1:[m(['Pound','Outrage'],fast),m(['Tackle'])],p2:[m(['Recover'],tank)],steps:[both(2),both(),both(),both()],...extra});
 for(const seed of [1,2,9,21])add(`duration-${seed}`,{seed});
 add('immune',{p2:[mon('Gengar',['Recover'],{ability:'Levitate',...tank})],p1:[m(['Pound','Thrash'],fast),m(['Tackle'])],steps:[both(2)]});
 add('protect-first',{p2:[m(['Protect','Recover'],tank)],steps:[both(2),both(2,2),both(1,2),both(1,2)]});
 add('protect-locked',{p2:[m(['Recover','Protect'],tank)],steps:[both(2),both(1,2),both(1,1),both(1,1)]});
 for(const move of ['Spore','Thunder Wave','Confuse Ray'])add(move.replaceAll(' ',''),{p2:[m([move,'Recover'],tank)],steps:[both(2),both(1,2),both(1,2),both(1,2)]});
 add('lum-fatigue',{p1:[m(['Pound','Outrage'],{item:'Lum Berry',...fast}),m(['Tackle'])]});
 add('choice-fatigue',{p1:[m(['Pound','Outrage'],{item:'Choice Band',...fast}),m(['Tackle'])],steps:[both(2),both(),{p1:'move 1'},{p2:'move 1'},{p1:'move 1'}]});
 add('last-pp',{setup:[{side:'p1',ppByMove:{outrage:1}}]});
 add('reject-switch',{steps:[both(2),{p1:'switch 2'},both()]});
 add('terminal',{p2:[m(['Recover'],tank)],setup:[{side:'p2',hp:1}],steps:[both(2)]});
 add('contact-ko',{p1:[m(['Outrage'],fast),m(['Tackle'])],p2:[m(['Recover'],{ability:'Rough Skin',item:'Rocky Helmet',...tank})],setup:[{side:'p1',hp:1}],steps:[both()]});
 for(const move of ['Solar Beam','Fly','Skull Bash']){
  const id=move.replaceAll(' ','');
  cases.push({id:`G4-lock-charge-${id}-sleep`,seed:1,p1:[m(['Pound',move],fast),m(['Tackle'])],p2:[m(['Spore','Recover'],tank)],steps:[both(2),both(1,2),both(1,2)]});
  cases.push({id:`G4-lock-charge-${id}-herb`,seed:1,p1:[m([move],{item:'Power Herb',...fast})],p2:[m(['Recover'],tank)],steps:[both()]});
 }

 const restrict=(id,p1,p2,steps,extra={})=>cases.push({id:`G4-lock-${id}`,seed:1,p1,p2,steps,...extra});
 restrict('taunt-expiry',[m(['Taunt','Tackle'],fast)],[m(['Recover','Tackle'],tank)],[both(),both(2,2),both(2,2),both(2,1)]);
 restrict('taunt-late',[m(['Taunt','Tackle'],tank)],[m(['Recover','Tackle'],fast)],[both(),both(2,2),both(2,2),both(2,2),both(2,1)]);
 restrict('taunt-switch',[m(['Taunt','Tackle'],fast)],[m(['Recover']),m(['Recover','Tackle'])],[{p1:'move 1',p2:'switch 2'},both(2,2)]);
 restrict('taunt-repeated',[m(['Taunt','Tackle'],fast)],[m(['Recover','Tackle'],tank)],[both(),both(1,2)]);
 restrict('disable-queued',[m(['Recover','Disable'],fast)],[m(['Tackle','Recover'],tank)],[both(),both(2),both(1,2)]);
 restrict('disable-no-history',[m(['Disable'],fast)],[m(['Recover'],tank)],[both()]);
 restrict('disable-zero-pp',[m(['Recover','Disable'],fast)],[m(['Tackle','Recover'],tank)],[both(),both(2,2)],{setup:[{side:'p2',ppByMove:{tackle:1}}]});
 restrict('encore-no-history',[m(['Encore'],fast)],[m(['Recover'],tank)],[both()]);
 restrict('encore-override',[m(['Recover','Encore','Tackle'],fast)],[m(['Tackle','Recover'],tank)],[both(),both(2,2),both(3)]);
 restrict('encore-pp-expiry',[m(['Recover','Encore','Tackle'],fast)],[m(['Tackle','Recover'],tank)],[both(),both(2),both(3,2)],{setup:[{side:'p2',ppByMove:{tackle:2}}]});
 restrict('torment-miss',[m(['Torment','Tackle'],fast)],[m(['Tackle','Recover'],tank)],[both(),both(2,2)],{seed:0});
 restrict('torment-rampage',[m(['Torment','Tackle'],fast)],[m(['Outrage'],tank)],[both(),both(2),both(2)]);
 for(const [move,counter]of [['Fly','Gust'],['Bounce','Gust'],['Dig','Earthquake'],['Dive','Surf']]){
  restrict(`counter-${move}`,[m([move],fast)],[m([counter],tank)],[both(),both()]);
 }
 restrict('solar-technician-rain',[m(['Solar Beam'],{ability:'Technician',item:'Wise Glasses',...fast})],[m(['Recover'],{ability:'Drizzle',...tank})],[both(),both()]);
 restrict('dig-sand-immunity',[m(['Dig'],fast)],[m(['Recover'],{ability:'Sand Stream',...tank})],[both(),both()]);
 restrict('fly-pressure',[m(['Pound','Fly'],fast)],[m(['Recover'],{ability:'Pressure',...tank})],[both(2),both()]);
 restrict('sky-flinch',[m(['Sky Attack'],fast)],[m(['Recover'],tank)],[both(),both()]);
 restrict('sky-flinch-after-action',[m(['Sky Attack'],tank)],[m(['Recover'],fast)],[both(),both()]);
 restrict('sky-tied',[m(['Sky Attack'])],[m(['Sky Attack'])],[both(),both()]);
 restrict('rampage-residual-ko',[m(['Outrage'],fast),m(['Tackle'])],[m(['Recover'],tank)],[both()],{setup:[{side:'p1',status:'brn',hp:1}]});
 for(const move of ['Air Slash','Iron Head','Dark Pulse','Rock Slide','Waterfall','Bite','Twister']){
  restrict(`flinch-${move.replaceAll(' ','')}`,[m([move],fast)],[m(['Recover'],tank)],[both(),both()]);
 }
 restrict('pending-charge',[m(['Pound','Solar Beam'],fast),m(['Tackle'])],[m(['Recover'],tank)],[both(2),{p1:'move 1'},{p1:'undo'},both()]);
 for(const mirror of [false,true]){
  const a=[m(['Outrage','Taunt'],fast),m(['Solar Beam','Sunny Day'],{item:'Power Herb',...fast}),m(['Fly','Recover']),m(['Skull Bash','Disable'],fast),m(['Petal Dance','Encore']),m(['Thrash','Torment'])];
  const b=[m(['Giga Impact','Reflect']),m(['Sky Attack','Protect']),m(['Bounce','Recover']),m(['Razor Wind','Recover']),m(['Dig','Protect']),m(['Dive','Recover'])];
  cases.push({id:`G4-lock-complete-${mirror?2:1}`,seed:mirror?21:1,p1:mirror?b:a,p2:mirror?a:b,fullBattle:true,family:'move-phases',steps:[]});
 }
 return cases.map(c=>({...c,generic:true,phaseFamily:false,lockFamily:true}));
};
