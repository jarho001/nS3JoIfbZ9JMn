'use strict';
module.exports=function(){
 const m=(x={})=>({species:'Mew',moves:['Tackle','Recover'],ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...x});
 const b=(a=1,c=2)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[];
 const add=(id,x={})=>cases.push({id:'G4-stat-selection-'+id,seed:1,generic:true,statSelectionFamily:true,p1:[m({moves:['Foul Play','Recover'],evs:{spe:252}})],p2:[m({evs:{atk:252}})],steps:[b()],...x});
 add('foulplay-base');
 add('foulplay-target-level',{p2:[m({level:100,evs:{atk:252}})]});
 add('foulplay-source-level',{p1:[m({level:25,moves:['Foul Play','Recover'],evs:{spe:252}})]});
 for(const [id,setup]of [['actor-attack',[{side:'p1',boosts:{atk:3}}]],['target-attack',[{side:'p2',boosts:{atk:3}}]],['target-defense',[{side:'p2',boosts:{def:2}}]],['burn-source',[{side:'p1',status:'brn'}]],['burn-target',[{side:'p2',status:'brn'}]]])add('foulplay-'+id,{setup});
 for(const [id,source,target,setup]of [
  ['source-hugepower',{ability:'Huge Power'},{},[]],['target-hugepower',{}, {ability:'Huge Power'},[]],
  ['source-band',{item:'Choice Band'},{},[]],['target-band',{}, {item:'Choice Band'},[]],
  ['source-guts',{ability:'Guts'},{},[{side:'p1',status:'brn'}]],['target-guts',{}, {ability:'Guts'},[{side:'p2',status:'brn'}]],
  ['source-unaware',{ability:'Unaware'},{},[{side:'p2',boosts:{atk:3,def:2}}]],
  ['target-unaware',{}, {ability:'Unaware'},[{side:'p2',boosts:{atk:3,def:2}}]],
 ])add('foulplay-'+id,{p1:[m({moves:['Foul Play','Recover'],evs:{spe:252},...source})],p2:[m({evs:{atk:252},...target})],setup});
 add('foulplay-critical',{seed:21,p2:[m({species:'Chansey',evs:{atk:252,def:252}})],setup:[{side:'p2',boosts:{atk:-6,def:6}}]});
 add('foulplay-suppressed',{p1:[m({moves:['Foul Play','Recover'],evs:{spe:252},ability:'Huge Power'})],p2:[m({moves:['Gastro Acid','Recover'],evs:{atk:252}})],steps:[b(2,1),b()]});
 for(const move of ['Psyshock','Psystrike','Secret Sword']){
  const key=move.toLowerCase().replaceAll(' ','');const source=(x={})=>m({moves:[move,'Recover'],evs:{spe:252},...x});const target=(x={})=>m({evs:{def:252},...x});
  add(key+'-critical',{seed:21,p1:[source()],p2:[target()],setup:[{side:'p1',boosts:{spa:-6}},{side:'p2',boosts:{def:6,spd:-6}}]});
  add(key+'-base',{p1:[source()],p2:[target()]});
  for(const [id,setup]of [['defense',[{side:'p2',boosts:{def:3}}]],['special-defense',[{side:'p2',boosts:{spd:3}}]],['offense',[{side:'p1',boosts:{atk:3,spa:2}}]],['burn',[{side:'p1',status:'brn'}]]])add(key+'-'+id,{p1:[source()],p2:[target()],setup});
  add(key+'-marvelscale',{p1:[source()],p2:[target({ability:'Marvel Scale'})],setup:[{side:'p2',status:'brn'}]});
  add(key+'-souldew',{p1:[source()],p2:[target({species:'Latios',item:'Soul Dew'})]});
  add(key+'-eviolite',{p1:[source()],p2:[target({species:'Chansey',item:'Eviolite'})]});
  for(const screen of ['Reflect','Light Screen'])add(key+'-'+screen.toLowerCase().replaceAll(' ',''),{p1:[source()],p2:[target({moves:[screen,'Recover']})],steps:[b(2,1),b()]});
  for(const item of ['Choice Band','Choice Specs'])add(key+'-'+item.toLowerCase().replaceAll(' ',''),{p1:[source({item})],p2:[target()]});
  add(key+'-sandstorm',{p1:[source()],p2:[target({species:'Gigalith',moves:['Sandstorm','Recover']})],steps:[b(2,1),b()]});
  add(key+'-source-unaware',{p1:[source({ability:'Unaware'})],p2:[target()],setup:[{side:'p2',boosts:{def:3,spd:3}}]});
  add(key+'-target-unaware',{p1:[source()],p2:[target({ability:'Unaware'})],setup:[{side:'p1',boosts:{spa:3}}]});
 }
 return cases;
};
module.exports.membershipWitnesses=function(){
 require('./neutral-turn-oracle.cjs').assertReady();const ps=require('./vendor/pokemon-showdown/dist/sim'),assert=require('node:assert/strict');
 return [['psyshock',3,'Latios'],['foulplay',13,'Amoonguss'],['psystrike',19,'Mewtwo'],['secretsword',81,'Keldeo']].map(([id,n,species])=>{const seed=[n,2,3,4],set=ps.Teams.getGenerator('gen5randombattle',seed).getTeam().find(s=>s.species===species&&s.moves.some(m=>ps.Dex.toID(m)===id));assert.ok(set,id);return {kind:'move',id,seed,set};});
};
