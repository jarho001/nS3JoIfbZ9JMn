'use strict';
// Native precheck only: complete lifecycle capture/callback audit before implementation.
module.exports=function(){
 const m=(x={})=>({species:'Mew',moves:['Growl','Shell Smash','Gastro Acid','Tackle'],ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...x});
 const b=(a=1,c=1)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[];
 const add=(id,x={})=>cases.push({id:'G4-stat-drop-'+id,seed:1,generic:true,statDropFamily:true,p1:[m({evs:{spe:252}})],p2:[m()],steps:[b(),b()],...x});
 for(const [key,ability,item]of [['clear','Clear Body',''],['herb','Pressure','White Herb'],['both','Clear Body','White Herb']]){
  add(key+'-growl',{p2:[m({ability,item})]});
  add(key+'-self',{p1:[m({moves:['Recover'],evs:{spe:252}})],p2:[m({ability,item})],steps:[b(1,2),b()]});
  add(key+'-gas',{p2:[m({ability,item})],steps:[b(3),b(),b()]});
  add(key+'-intimidate',{p1:[m({ability:'Intimidate',evs:{spe:252}})],p2:[m({ability,item})]});
  add(key+'-tie',{p1:[m()],p2:[m({ability,item})]});
  for(const move of ['Charm','Screech','Metal Sound'])add(key+'-'+move.toLowerCase().replaceAll(' ',''),{p1:[m({moves:[move,'Growl'],evs:{spe:252}})],p2:[m({ability,item})]});
 }
 for(const ability of ['Simple','Contrary'])add('herb-'+ability.toLowerCase(),{p1:[m({moves:['Recover'],evs:{spe:252}})],p2:[m({ability,item:'White Herb'})],steps:[b(1,2),b(),b()]});
 add('herb-recycle',{p2:[m({item:'White Herb',moves:['Shell Smash','Recycle','Growl']})],steps:[b(),b(1,2),b(1,1)]});
 add('herb-switch-intimidate',{p1:[m({ability:'Intimidate',evs:{spe:252}}),m()],p2:[m({item:'White Herb'}),m({item:'White Herb'})],steps:[{p1:'switch 2',p2:'switch 2'},{p1:'switch 2',p2:'move 1'},b()]});
 add('herb-exchange',{p1:[m({moves:['Growl','Trick'],evs:{spe:252}})],p2:[m({item:'White Herb'})],steps:[b(2),b()]});
 return cases;
};
module.exports.membershipWitnesses=function(){
 require('./neutral-turn-oracle.cjs').assertReady();const ps=require('./vendor/pokemon-showdown/dist/sim'),assert=require('node:assert/strict');
 return [['ability','Clear Body',5,'Klinklang'],['item','White Herb',14,'Crustle']].map(([kind,name,n,species])=>{
  const seed=[n,2,3,4],set=ps.Teams.getGenerator('gen5randombattle',seed).getTeam().find(s=>s[kind]===name&&s.species===species);assert.ok(set,name);return {kind,id:ps.Dex.toID(name),seed,set};
 });
};
