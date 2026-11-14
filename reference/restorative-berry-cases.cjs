'use strict';
module.exports=function(){
 const m=(x={})=>({species:'Mew',moves:['Growl','Rest','Trick','Tackle'],ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...x});
 const b=(a=1,c=1)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[];
 const add=(id,item,x={})=>cases.push({id:'G4-restorative-berry-'+id,seed:1,generic:true,restorativeBerryFamily:true,p1:[m({evs:{spe:252}})],p2:[m({item})],steps:[b(),b()],...x});
 for(const item of ['Chesto Berry','Sitrus Berry']){
  const k=item==='Chesto Berry'?'chesto':'sitrus';
  add(k+'-healthy',item);
  for(const status of ['brn','psn','tox','par','frz'])add(k+'-'+status,item,{setup:[{side:'p2',hp:80,status}]});
  add(k+'-rest',item,{setup:[{side:'p2',hp:80}],steps:[b(1,2),b(),b()]});
  add(k+'-rest-tie',item,{p1:[m({item})],setup:[{side:'p1',hp:80},{side:'p2',hp:80}],steps:[b(2,2),b(),b()]});
  add(k+'-spore',item,{p1:[m({moves:['Spore','Growl'],evs:{spe:252}})],steps:[b(),b(2)]});
  add(k+'-exchange',item,{setup:[{side:'p1',hp:80}],steps:[b(3),b()]});
  add(k+'-switch',item,{p2:[m({item}),m()],setup:[{side:'p2',hp:80}],steps:[{p1:'move 1',p2:'switch 2'},{p1:'move 1',p2:'switch 2'},b()]});
  add(k+'-recycle',item,{p2:[m({item,moves:['Rest','Recycle','Growl']})],setup:[{side:'p2',hp:80}],steps:[b(),b(1,2),b(1,3)]});
 }
 for(const hp of [87,88,89,1])add('sitrus-hp-'+hp,'Sitrus Berry',{setup:[{side:'p2',hp}]});
 add('sitrus-cross-threshold','Sitrus Berry',{setup:[{side:'p2',hp:100}],steps:[b(4),b()]});
 add('sitrus-knockoff','Sitrus Berry',{p1:[m({moves:['Knock Off','Growl'],evs:{spe:252}})],setup:[{side:'p2',hp:90}],steps:[b(),b(2)]});
 add('sitrus-substitute','Sitrus Berry',{p2:[m({item:'Sitrus Berry',moves:['Substitute','Growl']})],setup:[{side:'p2',hp:110}],steps:[b(),b(1,2)]});
 add('sitrus-faint','Sitrus Berry',{p1:[m({moves:['Explosion'],level:100})],steps:[b()]});
 add('sitrus-residual-crossing','Sitrus Berry',{setup:[{side:'p2',hp:100,status:'brn'}],steps:[b(),b()]});
 add('sitrus-rounding','Sitrus Berry',{p2:[m({item:'Sitrus Berry',evs:{hp:16}})],setup:[{side:'p2',hp:88}]});
 add('chesto-confusion','Chesto Berry',{p1:[m({moves:['Confuse Ray','Growl'],evs:{spe:252}})],steps:[b(),b(2)]});
 return cases;
};
module.exports.membershipWitnesses=function(){
 require('./neutral-turn-oracle.cjs').assertReady();const ps=require('./vendor/pokemon-showdown/dist/sim'),assert=require('node:assert/strict');
 return [['Chesto Berry',58,'Regirock'],['Sitrus Berry',13,'Tropius']].map(([item,n,species])=>{
  const seed=[n,2,3,4],set=ps.Teams.getGenerator('gen5randombattle',seed).getTeam().find(s=>s.item===item&&s.species===species);assert.ok(set,item);return {item:ps.Dex.toID(item),seed,set};
 });
};
