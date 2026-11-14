'use strict';
// Standalone next family: global ModifyBoost callbacks cross stat/speed query paths.
module.exports=function(){
 const m=(x={})=>({species:'Mew',moves:['Tackle','Surf','Gastro Acid','Recover'],ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...x});
 const b=(a=1,c=4)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[];
 const add=(id,x={})=>cases.push({id:'G4-unaware-'+id,seed:1,generic:true,unawareFamily:true,p1:[m({evs:{spe:252}})],p2:[m()],steps:[b()],...x});
 for(const position of ['source','target'])for(const stage of [-3,3])for(const move of [1,2]){
  add(position+'-'+stage+'-'+move,{p1:[m({ability:position==='source'?'Unaware':'Pressure',evs:{spe:252}})],p2:[m({ability:position==='target'?'Unaware':'Pressure'})],setup:[{side:'p1',boosts:{atk:stage,spa:stage}},{side:'p2',boosts:{def:stage,spd:stage}}],steps:[b(move)]});
 }
 for(const position of ['source','target'])for(const stage of [-6,6])add(position+'-accuracy-'+stage,{p1:[m({ability:position==='source'?'Unaware':'Pressure',evs:{spe:252}})],p2:[m({ability:position==='target'?'Unaware':'Pressure'})],setup:[{side:'p1',boosts:{accuracy:stage}},{side:'p2',boosts:{evasion:-stage}}]});
 add('dual-tie',{p1:[m({ability:'Unaware'})],p2:[m({ability:'Unaware'})],steps:[b(),b(2),b(4)]});
 add('dual-always',{p1:[m({ability:'Unaware',moves:['Aerial Ace']})],p2:[m({ability:'Unaware'})]});
 add('dual-gas',{p1:[m({ability:'Unaware'})],p2:[m({ability:'Unaware'})],setup:[{side:'p1',boosts:{atk:3,spa:3}}],steps:[b(3,3),b(2,2)]});
 add('dual-switch',{p1:[m({ability:'Unaware'}),m({ability:'Unaware'})],p2:[m({ability:'Unaware'}),m({ability:'Unaware'})],steps:[{p1:'switch 2',p2:'switch 2'},b()]});
 add('critical',{seed:21,p2:[m({ability:'Unaware'})],setup:[{side:'p1',boosts:{atk:3}},{side:'p2',boosts:{def:3}}]});
 add('suppressed',{p2:[m({ability:'Unaware'})],setup:[{side:'p1',boosts:{atk:3}}],steps:[b(3),b()]});
 add('no-guard',{p1:[m({ability:'No Guard',evs:{spe:252}})],p2:[m({ability:'Unaware'})],setup:[{side:'p1',boosts:{accuracy:-6}}]});
 add('dual-forced',{p1:[m({ability:'Unaware',moves:['Quick Attack','Recover']})],p2:[m({ability:'Unaware'}),m({ability:'Unaware'})],setup:[{side:'p2',hp:1}],steps:[b(1,4),{p2:'switch 2'},b(1,4)]});
 add('dual-rain',{p1:[m({ability:'Unaware',moves:['Rain Dance','Surf']})],p2:[m({ability:'Unaware'})],steps:[b(),b(2)]});
 add('dual-speed-stage',{p1:[m({ability:'Unaware'})],p2:[m({ability:'Unaware'})],setup:[{side:'p2',boosts:{spe:1}}]});
 add('dual-source-speed-stage',{p1:[m({ability:'Unaware'})],p2:[m({ability:'Unaware'})],setup:[{side:'p1',boosts:{spe:1}}]});
 add('dual-source-paralysis',{p1:[m({ability:'Unaware'})],p2:[m({ability:'Unaware'})],setup:[{side:'p1',status:'par'}]});
 add('dual-fast-source',{p1:[m({ability:'Unaware',evs:{spe:252}})],p2:[m({ability:'Unaware'})]});
 add('dual-fast-target',{p1:[m({ability:'Unaware'})],p2:[m({ability:'Unaware',evs:{spe:252}})]});
 add('dual-phaze',{p1:[m({ability:'Unaware',moves:['Roar']})],p2:[m({ability:'Unaware'}),m({ability:'Unaware'})],steps:[b(1,1)]});
 add('dual-pivot',{p1:[m({ability:'Unaware',moves:['U-turn']}),m({ability:'Unaware'})],p2:[m({ability:'Unaware'})],steps:[b(1,4),{p1:'switch 2'}]});
 add('dual-pursuit',{p1:[m({ability:'Unaware',moves:['Pursuit']})],p2:[m({ability:'Unaware'}),m({ability:'Unaware'})],steps:[{p1:'move 1',p2:'switch 2'}]});
 add('transform-pair',{p1:[m({moves:['Transform'],evs:{spe:252}})],p2:[m({ability:'Unaware'})],steps:[b(),b()]});
 for(const targetMove of ['Tackle','Recover'])add('dual-confusion-'+targetMove.toLowerCase(),{p1:[m({ability:'Unaware',moves:['Confuse Ray','Recover']})],p2:[m({ability:'Unaware',moves:[targetMove]})],setup:[{side:'p2',boosts:{atk:3}}],steps:[b(1,1),b(2,1),b(2,1)]});
 return cases;
};
module.exports.membershipWitnesses=function(){
 require('./neutral-turn-oracle.cjs').assertReady();const ps=require('./vendor/pokemon-showdown/dist/sim'),assert=require('node:assert/strict'),seed=[18,2,3,4];
 const set=ps.Teams.getGenerator('gen5randombattle',seed).getTeam().find(s=>s.ability==='Unaware'&&s.species==='Quagsire');assert.ok(set);return [{kind:'ability',id:'unaware',seed,set}];
};
