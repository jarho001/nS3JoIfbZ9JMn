'use strict';
module.exports=function(){
 const m=(moves,extra={})=>({species:'Mew',moves,ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...extra});
 const b=(a=1,c=1)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[],add=(id,ability='Pressure',x={})=>cases.push({id:'G4-trace-'+id,seed:1,generic:true,identityFamily:true,abilityChangeFamily:true,traceFamily:true,hitFamily:true,p1:[m(['Tackle'],{ability:'Trace'})],p2:[m(['Tackle'],{ability,evs:{spe:252}})],steps:[b()],...x});
 for(const ability of ['Pressure','Intimidate','Drizzle','Drought','Prankster','Levitate','Immunity','Magic Bounce','Simple','Wonder Guard'])add(ability.toLowerCase().replace(/[^a-z]/g,''),ability);
 for(const ability of ['Trace','Forecast','Imposter','Illusion','Zen Mode'])add('blocked-'+ability.toLowerCase().replace(/[^a-z]/g,''),ability);
 add('delayed','Trace',{p1:[m(['Worry Seed','Tackle'],{ability:'Trace'})],p2:[m(['Tackle'],{ability:'Trace',evs:{spe:252}})],steps:[b(),b(2)]});
 add('switch-reset','Pressure',{p1:[m(['Tackle'],{ability:'Trace'}),m(['Tackle'])],p2:[m(['Tackle'],{ability:'Intimidate',evs:{spe:252}})],steps:[{p1:'switch 2',p2:'move 1'},{p1:'switch 2',p2:'move 1'}]});
 add('acquire','Trace',{p1:[m(['Skill Swap','Worry Seed'],{ability:'Forecast'})],p2:[m(['Tackle'],{ability:'Trace',evs:{spe:252}})],steps:[b(),b(2)]});
 add('suppressed-acquire','Trace',{p1:[m(['Tackle','Skill Swap'],{ability:'Forecast'})],p2:[m(['Gastro Acid','Tackle'],{ability:'Trace',evs:{spe:252}})],steps:[b(),b(2,2)]});
 add('faint-seek','Forecast',{p1:[m(['Tackle'],{ability:'Trace'}),m(['Tackle'])],setup:[{side:'p1',hp:1}],steps:[b(),{p1:'switch 2'}]});
 add('mirrored','Trace',{p1:[m(['Tackle'],{ability:'Pressure',evs:{spe:252}})],p2:[m(['Tackle'],{ability:'Trace'})]});
 return cases;
};
