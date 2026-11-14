'use strict';
module.exports=function(){
 const m=(moves,x={})=>({species:'Mew',moves,ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...x});
 const b=(a=1,c=1)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[];
 const add=(id,ability='Natural Cure',x={})=>cases.push({id:'G4-switch-recovery-'+id,seed:1,generic:true,switchRecoveryFamily:true,p1:[m(['Tackle','U-turn'],{ability,evs:{spe:252}}),m(['Tackle'])],p2:[m(['Growl'])],steps:[{p1:'switch 2',p2:'move 1'}],...x});
 for(const status of ['brn','psn','tox','par','frz'])add('cure-'+status,'Natural Cure',{setup:[{side:'p1',status}]});
 add('cure-clear');add('cure-sleep','Natural Cure',{p2:[m(['Spore','Growl'],{evs:{spe:252},nature:'Timid'})],steps:[b(),{p1:'switch 2',p2:'move 2'}]});
 add('cure-pivot','Natural Cure',{setup:[{side:'p1',status:'brn'}],steps:[b(2),{p1:'switch 2'}]});
 add('cure-gas','Natural Cure',{p2:[m(['Gastro Acid','Growl'],{evs:{spe:252},nature:'Timid'})],setup:[{side:'p1',status:'brn'}],steps:[b(),{p1:'switch 2',p2:'move 2'}]});
 add('cure-mummy','Natural Cure',{p2:[m(['Growl'],{ability:'Mummy'})],setup:[{side:'p1',status:'brn'}],steps:[b(),{p1:'switch 2',p2:'move 1'}]});
 for(const hp of [60,150,175])add('regen-'+hp,'Regenerator',{setup:[{side:'p1',hp}]});
 add('regen-round','Regenerator',{p1:[m(['Tackle'],{ability:'Regenerator',evs:{hp:16,spe:252}}),m(['Tackle'])],setup:[{side:'p1',hp:60}]});
 add('regen-pivot','Regenerator',{setup:[{side:'p1',hp:60}],steps:[b(2),{p1:'switch 2'}]});
 add('regen-pursuit','Regenerator',{p2:[m(['Pursuit'])],setup:[{side:'p1',hp:150}]});
 add('regen-pursuit-ko','Regenerator',{p2:[m(['Pursuit'])],setup:[{side:'p1',hp:1}],steps:[{p1:'switch 2',p2:'move 1'},{p1:'switch 2'}]});
 add('regen-gas','Regenerator',{p2:[m(['Gastro Acid','Growl'],{evs:{spe:252},nature:'Timid'})],setup:[{side:'p1',hp:60}],steps:[b(),{p1:'switch 2',p2:'move 2'}]});
 add('regen-mummy','Regenerator',{p2:[m(['Growl'],{ability:'Mummy'})],setup:[{side:'p1',hp:60}],steps:[b(),{p1:'switch 2',p2:'move 1'}]});
 add('cure-phaze','Natural Cure',{p2:[m(['Roar'])],setup:[{side:'p1',status:'brn'}],steps:[b()]});
 add('regen-phaze','Regenerator',{p2:[m(['Roar'])],setup:[{side:'p1',hp:60}],steps:[b()]});
 add('mirrored','Natural Cure',{p1:[m(['Growl'])],p2:[m(['Tackle'],{ability:'Natural Cure'}),m(['Tackle'])],setup:[{side:'p2',status:'tox'}],steps:[{p1:'move 1',p2:'switch 2'}]});
 return cases;
};
module.exports.membershipWitnesses=function(){
 require('./neutral-turn-oracle.cjs').assertReady();const ps=require('./vendor/pokemon-showdown/dist/sim'),assert=require('node:assert/strict');
 return [['Regenerator',3],['Natural Cure',24]].map(([ability,i])=>{const seed=[i,2,3,4],set=ps.Teams.getGenerator('gen5randombattle',seed).getTeam().find(s=>s.ability===ability);assert.ok(set);return {ability:ps.Dex.toID(ability),seed,set};});
};
