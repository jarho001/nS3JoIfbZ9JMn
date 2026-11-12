'use strict';
module.exports=function(){
const m=(species,moves,extra={})=>({species,moves,ability:species==='Gengar'?'Levitate':'Synchronize',item:'',level:50,nature:'Hardy',gender:species==='Gengar'?'M':'N',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...extra});
const fast={evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:252}},both=(a=1,b=1)=>({p1:`move ${a}`,p2:`move ${b}`});
const cases=[],add=(id,extra={})=>cases.push({id:'G4-identity-'+id,seed:1,generic:true,identityFamily:true,hitFamily:true,p1:[m('Mew',['Transform','Tackle'],fast),m('Mew',['Tackle'])],p2:[m('Gengar',['Mega Punch','Thunder Wave'],{level:60}),m('Mew',['Tackle'])],steps:[both(),both(),{p1:'switch 2',p2:'move 1'},{p1:'switch 2',p2:'move 1'}],...extra});
add('ordinary');add('both-transform',{p2:[m('Mew',['Transform','Tackle'])],steps:[both(),both()]});
add('target-substitute',{p1:[m('Mew',['Transform'])],p2:[m('Mew',['Substitute','Tackle'],fast)],steps:[both()]});
add('copy-boosts',{p1:[m('Mew',['Transform','Tackle'])],p2:[m('Mew',['Calm Mind','Tackle'],fast)],steps:[both(),both(2,2)]});
add('copy-intimidate',{p2:[m('Mew',['Recover','Tackle'],{ability:'Intimidate'})],steps:[both(),both(2,2)]});
add('copy-frisk',{p1:[m('Mew',['Transform','Tackle'],{...fast,item:'Leftovers'})],p2:[m('Mew',['Recover','Tackle'],{ability:'Frisk',item:'Life Orb'})],steps:[both(),both(2,2)]});
add('copy-pressure',{p2:[m('Mew',['Recover','Tackle'],{ability:'Pressure'})],steps:[both(),both(2,2)]});
add('copy-own-hiddenpower',{p1:[m('Mew',['Transform'],{...fast,ivs:{hp:30,atk:30,def:30,spa:30,spd:30,spe:30}})],p2:[m('Mew',['Hidden Power','Recover'])],steps:[both(1,2),both(1,2)]});
add('imposter-entry',{p1:[m('Ditto',['Transform'],{ability:'Imposter',gender:'N'}),m('Mew',['Tackle'])],steps:[both(),{p1:'switch 2',p2:'move 1'},{p1:'switch 2',p2:'move 1'}]});
add('imposter-substitute',{p1:[m('Mew',['Tackle']),m('Ditto',['Transform'],{ability:'Imposter'})],p2:[m('Mew',['Substitute','Tackle'],fast)],steps:[both(),{p1:'switch 2',p2:'move 2'}]});
add('copied-trap',{p2:[m('Gothitelle',['Recover','Tackle'],{ability:'Shadow Tag',gender:'F'})],steps:[both(),{p1:'switch 2'},{p1:'move 1',p2:'move 2'}]});

add('imposter-intimidate',{p1:[m('Ditto',['Transform'],{ability:'Imposter'})],p2:[m('Mew',['Recover','Tackle'],{ability:'Intimidate'})],steps:[both()]});
add('imposter-hazards',{p1:[m('Mew',['Tackle']),m('Ditto',['Transform'],{ability:'Imposter'})],p2:[m('Charizard',['Stealth Rock','Tackle'],{ability:'Pressure'})],steps:[both(),{p1:'switch 2',p2:'move 2'}]});
add('choice-transform',{p1:[m('Mew',['Transform','Tackle'],{...fast,item:'Choice Band'})],p2:[m('Mew',['Recover','Tackle'])],steps:[both(),both(2,2)]});
add('faint-restore',{p1:[m('Mew',['Transform'],fast),m('Mew',['Tackle'])],p2:[m('Gengar',['Mega Punch','Earthquake'],{ability:'Pressure'})],steps:[both(1,2)],setup:[{side:'p1',hp:1}]});
add('unknown-flying-query',{p2:[m('Charizard',['Recover','Tackle'],{ability:'Pressure'}),m('Dugtrio',['Tackle'],{ability:'Synchronize'})],steps:[both(),{p1:'move 2',p2:'switch 2'},{p1:'switch 2'},{p1:'undo'},{p2:'move 1'}]});
add('same-frisk',{p1:[m('Mew',['Transform'],{...fast,ability:'Frisk'})],p2:[m('Mew',['Recover'],{ability:'Frisk',item:'Life Orb'})],steps:[both()]});
for(const c of cases)if(c.id==='G4-identity-copied-trap')c.steps=[both(),{p1:'switch 2',p2:'move 2'}];
return cases;
};
