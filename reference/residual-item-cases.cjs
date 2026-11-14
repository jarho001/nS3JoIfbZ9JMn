'use strict';
module.exports=function(){
 const m=(x={})=>({species:'Mew',moves:['Growl','Recover','Gastro Acid','Trick'],ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...x});
 const b=(a=1,c=1)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[];
 const add=(id,item,x={})=>cases.push({id:'G4-residual-item-'+id,seed:1,generic:true,residualItemFamily:true,p1:[m({evs:{spe:252}})],p2:[m({item})],steps:[b(),b(),b()],...x});
 for(const item of ['Black Sludge','Toxic Orb','Flame Orb']){
  const key=item==='Black Sludge'?'sludge':item==='Toxic Orb'?'toxic':'flame';
  add(key+'-ordinary',item);
  add(key+'-tie',item,{p1:[m({item})]});
  add(key+'-mirror',item,{p1:[m({item,evs:{spe:252}})],p2:[m()]});
  for(const status of ['brn','psn','tox','par'])add(key+'-existing-'+status,item,{setup:[{side:'p2',hp:140,status}]});
  for(const ability of ['Magic Guard','Shed Skin','Poison Heal','Immunity','Water Veil'])add(key+'-'+ability.toLowerCase().replaceAll(' ',''),item,{p2:[m({item,ability})],setup:[{side:'p2',hp:80}]});
  add(key+'-fire',item,{p2:[m({item,species:'Charizard'})]});
  add(key+'-poison',item,{p2:[m({item,species:'Arbok'})],setup:[{side:'p2',hp:80}]});
  add(key+'-steel',item,{p2:[m({item,species:'Registeel'})]});
  add(key+'-gas',item,{p2:[m({item,ability:item==='Toxic Orb'?'Immunity':item==='Flame Orb'?'Water Veil':'Magic Guard'})],steps:[b(3),b(),b()]});
  add(key+'-switch',item,{p2:[m({item}),m()],steps:[b(),{p1:'move 1',p2:'switch 2'},{p1:'move 1',p2:'switch 2'},b()]});
  add(key+'-trick',item,{steps:[b(4),b(),b()]});
 }
 add('sludge-full-poison','Black Sludge',{p2:[m({item:'Black Sludge',species:'Arbok'})]});
 add('sludge-cap','Black Sludge',{p2:[m({item:'Black Sludge',species:'Arbok'})],setup:[{side:'p2',hp:134}]});
 add('sludge-faint','Black Sludge',{p2:[m({item:'Black Sludge'}),m()],setup:[{side:'p2',hp:1}],steps:[b()]});
 add('sludge-minimum','Black Sludge',{p2:[m({item:'Black Sludge',species:'Shedinja'})],steps:[b()]});
 add('toxic-poison-heal-tie','Toxic Orb',{p1:[m({ability:'Shed Skin',item:'Flame Orb'})],p2:[m({item:'Toxic Orb',ability:'Poison Heal'})],setup:[{side:'p1',hp:80},{side:'p2',hp:80}],steps:[b(),b(),b(),b()]});
 return cases;
};
module.exports.membershipWitnesses=function(){
 require('./neutral-turn-oracle.cjs').assertReady();const ps=require('./vendor/pokemon-showdown/dist/sim'),assert=require('node:assert/strict');
 return [['Black Sludge',3,'Arbok'],['Toxic Orb',2,'Raticate'],['Flame Orb',278,'Sigilyph']].map(([item,n,species])=>{
  const seed=[n,2,3,4],set=ps.Teams.getGenerator('gen5randombattle',seed).getTeam().find(s=>s.item===item&&s.species===species);assert.ok(set,item);return {item:ps.Dex.toID(item),seed,set};
 });
};
