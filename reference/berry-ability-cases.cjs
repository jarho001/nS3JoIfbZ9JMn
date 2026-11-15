'use strict';
module.exports=function(){
 const m=(x={})=>({species:'Mew',moves:['Tackle','Recover'],ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...x});
 const b=(a=1,c=2)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[];
 const add=(id,x={})=>cases.push({id:'G4-berry-ability-'+id,seed:1,generic:true,itemLifecycleFamily:true,berryAbilityFamily:true,p1:[m({evs:{spe:252}})],p2:[m()],steps:[b()],...x});
 const harvest=(x={})=>m({ability:'Harvest',item:'Sitrus Berry',evs:{spe:252},...x});
 add('harvest-empty',{p1:[harvest({item:''})],steps:[b(),b()]});
 add('harvest-held',{p1:[harvest()],steps:[b(),b()]});
 for(const seed of [1,2,3,4])add('harvest-consumed-'+seed,{seed,p1:[harvest()],setup:[{side:'p1',hp:80}],steps:[b(),b()]});
 add('harvest-sun',{p1:[harvest({moves:['Sunny Day','Tackle','Recover']})],setup:[{side:'p1',hp:80}],steps:[b(),b(2)]});
 add('harvest-sun-cycle',{p1:[harvest({evs:{spe:0}})],p2:[m({moves:['Sunny Day','Recover'],evs:{spe:252}})],setup:[{side:'p1',hp:1}],steps:[b(1,1),b(),b()]});
 add('harvest-rain',{p1:[harvest({moves:['Rain Dance','Tackle','Recover']})],setup:[{side:'p1',hp:80}],steps:[b(),b(2)]});
 add('harvest-lum',{p1:[harvest({item:'Lum Berry'})],setup:[{side:'p1',status:'par'}],steps:[b(),b()]});
 add('harvest-chesto',{p1:[harvest({item:'Chesto Berry',moves:['Rest','Tackle']})],setup:[{side:'p1',hp:100}],steps:[b(),b(2)]});
 add('harvest-herb',{p1:[harvest({item:'White Herb',moves:['Shell Smash','Recover']})],steps:[b(),b(2)]});
 add('harvest-knockoff',{p1:[harvest({evs:{spe:0},moves:['Recover']})],p2:[m({moves:['Knock Off','Recover'],evs:{spe:252}})],steps:[b(1,1),b()]});
 add('harvest-mummy',{abilityChangeFamily:true,p1:[harvest()],p2:[m({ability:'Mummy'})],setup:[{side:'p1',hp:80}]});
 add('harvest-suppressed',{abilityChangeFamily:true,p1:[harvest({moves:['Recover'],evs:{spe:0}})],p2:[m({moves:['Gastro Acid','Recover'],evs:{spe:252}})],setup:[{side:'p1',hp:80}],steps:[b(1,1),b()]});
 add('harvest-both-empty',{p1:[harvest({item:'',moves:['Recover'],evs:{spe:0}})],p2:[harvest({item:'',moves:['Recover'],evs:{spe:0}})],steps:[b(1,1),b(1,1)]});
 add('harvest-speed-tie',{p1:[harvest({item:'',moves:['Recover'],evs:{spe:0}})],p2:[m({ability:'Speed Boost',moves:['Recover']})],steps:[b(1,1),b(1,1)]});
 for(const item of ['Sitrus Berry','Lum Berry','Chesto Berry','Oran Berry','Pecha Berry','Custap Berry']){
  const key=item.toLowerCase().replace(' ','');const status=item==='Lum Berry'?'par':item==='Pecha Berry'?'psn':undefined;
  add('unnerve-'+key,{p1:[m({item,moves:item==='Chesto Berry'?['Rest','Recover']:['Tackle','Recover'],evs:{spe:252}})],p2:[m({ability:'Unnerve'})],setup:[{side:'p1',hp:30,...status?{status}:{}}],steps:[b(),b()]});
 }
 add('unnerve-herb',{p1:[m({item:'White Herb',moves:['Shell Smash','Recover'],evs:{spe:252}})],p2:[m({ability:'Unnerve'})],steps:[b(),b(2)]});
 add('unnerve-switch-out',{p1:[m({item:'Sitrus Berry',evs:{spe:252}})],p2:[m({ability:'Unnerve'}),m()],setup:[{side:'p1',hp:30}],steps:[{p1:'move 1',p2:'switch 2'}]});
 add('unnerve-switch-in',{p1:[m({item:'Custap Berry',moves:['Recover']})],p2:[m(),m({ability:'Unnerve',evs:{spe:252}})],setup:[{side:'p1',hp:30}],steps:[{p1:'move 1',p2:'switch 2'},b()]});
 add('unnerve-suppressed',{abilityChangeFamily:true,p1:[m({item:'Sitrus Berry',moves:['Gastro Acid','Recover'],evs:{spe:252}})],p2:[m({ability:'Unnerve'})],setup:[{side:'p1',hp:30}],steps:[b(),b(2)]});
 add('unnerve-mummy',{abilityChangeFamily:true,p1:[m({ability:'Mummy',item:'Sitrus Berry',moves:['Recover']})],p2:[m({ability:'Unnerve',evs:{spe:252}})],setup:[{side:'p1',hp:50}],steps:[b(1,1)]});
 add('unnerve-ko',{p1:[m({item:'Sitrus Berry',evs:{spe:252}})],p2:[m({ability:'Unnerve',level:1})],setup:[{side:'p1',hp:30}]});
 add('unnerve-reentry',{p1:[m({moves:['Recover']})],p2:[m({ability:'Unnerve'}),m()],steps:[b(),{p1:'move 1',p2:'switch 2'},{p1:'move 1',p2:'switch 2'},b()]});
 add('unnerve-harvest',{p1:[harvest()],p2:[m({ability:'Unnerve'})],setup:[{side:'p1',hp:80}],steps:[b(),b()]});
 return cases;
};
module.exports.membershipWitnesses=function(){
 require('./neutral-turn-oracle.cjs').assertReady();const ps=require('./vendor/pokemon-showdown/dist/sim'),assert=require('node:assert/strict');return [['Harvest',13,'Tropius'],['Unnerve',5,'Aerodactyl']].map(([name,n,species])=>{const seed=[n,2,3,4],set=ps.Teams.getGenerator('gen5randombattle',seed).getTeam().find(s=>s.species===species&&s.ability===name);assert.ok(set,name);return {kind:'ability',id:ps.Dex.toID(name),seed,set};});
};
