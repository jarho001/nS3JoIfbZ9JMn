'use strict';
module.exports=function(){
 const m=(x={})=>({species:'Mew',moves:['Tackle','Recover'],ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...x});
 const b=(a=1,c=2)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[];
 const mixed={atk:2,def:-2,spa:3,spd:-1,spe:2,accuracy:2,evasion:-2};
 const add=(id,x={})=>cases.push({id:'G4-boost-reset-'+id,seed:1,generic:true,boostResetFamily:true,p1:[m({moves:['Haze','Recover'],evs:{spe:252}})],p2:[m()],steps:[b()],...x});
 for(const move of ['Haze','Clear Smog']){
  const key=move==='Haze'?'haze':'smog',source=(x={})=>m({moves:[move,'Recover'],evs:{spe:252},...x});
  add(key+'-zero',{p1:[source()]});
  for(const side of ['p1','p2'])add(key+'-'+side,{p1:[source()],setup:[{side,boosts:mixed}]});
  add(key+'-both',{p1:[source()],setup:[{side:'p1',boosts:mixed},{side:'p2',boosts:mixed}]});
  for(const ability of ['Simple','Contrary','Defiant','Clear Body','Unaware'])add(key+'-'+ability.toLowerCase().replace(' ',''),{p1:[source()],p2:[m({ability})],setup:[{side:'p2',boosts:mixed}]});
  add(key+'-unaware-tie',{unawareFamily:true,p1:[source({ability:'Unaware',evs:{spe:0}})],p2:[m({ability:'Unaware'})],setup:[{side:'p1',boosts:{atk:2}},{side:'p2',boosts:{atk:2}}]});
  add(key+'-reserves',{p1:[source(),m()],p2:[m(),m()],setup:[{side:'p1',mon:1,boosts:mixed},{side:'p2',mon:1,boosts:mixed},{side:'p2',boosts:mixed}]});
  add(key+'-protect',{p1:[source()],p2:[m({moves:['Protect','Recover']})],setup:[{side:'p2',boosts:mixed}],steps:[b(1,1)]});
  add(key+'-substitute',{p1:[source()],p2:[m({moves:['Substitute','Recover']})],setup:[{side:'p2',boosts:{def:2,spd:2}}],steps:[b(2,1),b()]});
  add(key+'-midturn',{p1:[source({evs:{spe:0}})],p2:[m({moves:['Calm Mind','Recover'],evs:{spe:252}})],steps:[b(1,1),b(1,1)]});
  add(key+'-storedpower',{p1:[source()],p2:[m({moves:['Stored Power','Recover']})],setup:[{side:'p2',boosts:{spa:2,def:2,accuracy:2}}],steps:[b(1,1)]});
  add(key+'-steel',{p1:[source()],p2:[m({species:'Registeel'})],setup:[{side:'p2',boosts:mixed}]});
  add(key+'-sleep-talk',{seed:2,p1:[source({moves:['Rest','Sleep Talk',move],evs:{spe:0}})],p2:[m({evs:{spe:252}})],setup:[{side:'p1',hp:100},{side:'p2',boosts:{atk:2,def:2}}],steps:[b(1,1),b(2,1)]});
 }
 add('smog-terminal',{p1:[m({moves:['Clear Smog','Recover'],evs:{spe:252}})],p2:[m({level:1})],setup:[{side:'p2',boosts:{atk:2}}]});
 return cases;
};
module.exports.membershipWitnesses=function(){
 require('./neutral-turn-oracle.cjs').assertReady();const ps=require('./vendor/pokemon-showdown/dist/sim'),assert=require('node:assert/strict');return [['clearsmog',13,'Amoonguss'],['haze',17,'Tentacruel']].map(([id,n,species])=>{const seed=[n,2,3,4],set=ps.Teams.getGenerator('gen5randombattle',seed).getTeam().find(s=>s.species===species&&s.moves.some(m=>ps.Dex.toID(m)===id));assert.ok(set,id);return {kind:'move',id,seed,set};});
};
