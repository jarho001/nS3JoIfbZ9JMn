'use strict';
module.exports=function(){
 const m=(species,moves,extra={})=>({species,moves,ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...extra});
 const b=(a=1,c=1)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[],add=(id,extra={})=>cases.push({id:'G4-called-'+id,seed:1,generic:true,identityFamily:true,abilityChangeFamily:true,calledMoveFamily:true,hitFamily:true,p1:[m('Mew',['Rest','Sleep Talk','Tackle'],{evs:{spe:252}}),m('Mew',['Tackle'])],p2:[m('Mew',['Tackle','Recover'])],steps:[b(),b(2),b(2),b(2)],...extra});
 add('awake',{steps:[b(2)]});
 add('ordinary');add('one-target',{p1:[m('Mew',['Rest','Sleep Talk'],{evs:{spe:252}})],steps:[b(),b(2),b(2)]});
 add('empty',{p1:[m('Mew',['Sleep Talk'])],p2:[m('Mew',['Spore','Recover'],{evs:{spe:252}})],steps:[b(),b(1,2)]});
 add('disabled',{p1:[m('Mew',['Tackle','Rest','Sleep Talk'],{evs:{spe:252}})],p2:[m('Mew',['Tackle','Disable'],{evs:{spe:252},nature:'Timid'})],steps:[b(),b(2,2),b(3),b(3)]});
 add('zero-called-pp',{setup:[{side:'p1',ppByMove:{tackle:0}}]});
 add('choice',{p1:[m('Mew',['Rest','Sleep Talk','Tackle'],{item:'Choice Band',evs:{spe:252}})],p2:[m('Mew',['Spore','Recover'],{evs:{spe:252},nature:'Timid'})],steps:[b(2),b(2,2)]});
 add('charge-filter',{p1:[m('Mew',['Rest','Sleep Talk','Solar Beam'],{evs:{spe:252}})],steps:[b(),b(2)]});
 add('pivot',{p1:[m('Mew',['Rest','Sleep Talk','U-turn'],{evs:{spe:252}}),m('Mew',['Tackle'])],steps:[b(),b(2),{p1:'switch 2'},b()]});
 add('recharge',{p1:[m('Mew',['Rest','Sleep Talk','Hyper Beam'],{evs:{spe:252}})],steps:[b(),b(2),b(1)]});
 add('transform',{p1:[m('Mew',['Rest','Sleep Talk','Transform'],{evs:{spe:252}})],p2:[m('Mew',['Tackle'])],steps:[b(),b(2)]});
 add('hazard',{p1:[m('Mew',['Rest','Sleep Talk','Stealth Rock'],{evs:{spe:252}})],steps:[b(),b(2)]});
 add('tie',{p1:[m('Mew',['Rest','Sleep Talk','Tackle'])],p2:[m('Mew',['Spore','Recover'])],steps:[b(2),b(2,2),b(2,2)]});
 add('lifeorb',{p1:[m('Mew',['Rest','Sleep Talk','Tackle'],{item:'Life Orb',evs:{spe:252}})],steps:[b(),b(2)]});
 add('explosion',{p1:[m('Mew',['Rest','Sleep Talk','Explosion'],{evs:{spe:252}})],steps:[b(),b(2)]});
 add('rampage',{p1:[m('Mew',['Rest','Sleep Talk','Outrage'],{evs:{spe:252}})],steps:[b(),b(2)]});
 add('substitute',{p1:[m('Mew',['Rest','Sleep Talk','Substitute'],{evs:{spe:252}})],steps:[b(),b(2)]});
 add('taunt',{p1:[m('Mew',['Rest','Sleep Talk','Tackle'])],p2:[m('Mew',['Spore','Taunt','Tackle'],{evs:{spe:252}})],steps:[b(2),b(2,2)]});
 add('phazing',{p1:[m('Mew',['Rest','Sleep Talk','Roar'],{evs:{spe:252}})],p2:[m('Mew',['Tackle']),m('Mew',['Tackle'])],steps:[b(),b(2)]});
 add('mirrored',{p1:[m('Mew',['Tackle','Recover'])],p2:[m('Mew',['Rest','Sleep Talk','Tackle'],{evs:{spe:252}})],setup:[{side:'p2',hp:100}],steps:[b(),b(1,2),b(2,2)]});
 // Prefix Rest must be legal from full HP only after taking a prior hit.
 for(const c of cases)if(!['awake','empty','choice','tie','taunt','mirrored'].some(id=>c.id==='G4-called-'+id))c.setup=[...(c.setup??[]),{side:'p1',hp:100}];
 return cases;
};
