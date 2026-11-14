'use strict';
module.exports=function(){
 const m=(x={})=>({species:'Mew',moves:['Growl','Shell Smash','Gastro Acid','Recover'],ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...x});
 const b=(a=1,c=4)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[];
 const add=(id,x={})=>cases.push({id:'G4-boost-ability-'+id,seed:1,generic:true,boostAbilityFamily:true,p1:[m({evs:{spe:252}})],p2:[m()],steps:[b()],...x});
 for(const move of ['Growl','Charm','Screech','Metal Sound','Tickle'])add('defiant-'+move.toLowerCase().replaceAll(' ',''),{p1:[m({moves:[move],evs:{spe:252}})],p2:[m({ability:'Defiant'})],steps:[b(),b()]});
 for(const stage of [-6,5,6])add('defiant-cap-'+stage,{p2:[m({ability:'Defiant'})],setup:[{side:'p2',boosts:{atk:stage}}]});
 add('defiant-self',{p2:[m({ability:'Defiant'})],steps:[b(4,2)]});
 add('defiant-gas',{p2:[m({ability:'Defiant'})],steps:[b(3),b()]});
 add('defiant-intimidate',{p1:[m({ability:'Intimidate',evs:{spe:252}})],p2:[m({ability:'Defiant'})]});
 add('defiant-herb',{p2:[m({ability:'Defiant',item:'White Herb'})],steps:[b(),b()]});
 for(const ability of ['Pressure','Marvel Scale'])for(const nature of ['Bold','Calm','Hardy'])add('download-'+ability.toLowerCase().replaceAll(' ','')+'-'+nature.toLowerCase(),{p1:[m({ability:'Download',evs:{spe:252}})],p2:[m({ability,nature})]});
 add('download-entry-boosts',{p1:[m({moves:['Tackle']}),m({ability:'Download'})],p2:[m()],setup:[{side:'p2',boosts:{def:2,spd:-2}}],steps:[{p1:'switch 2',p2:'move 4'}]});
 add('download-roleplay',{p1:[m({moves:['Role Play'],evs:{spe:252}})],p2:[m({ability:'Download'})]});
 add('download-skillswap',{p1:[m({moves:['Skill Swap'],evs:{spe:252}})],p2:[m({ability:'Download'})]});
 add('download-unaware',{p1:[m({ability:'Download',evs:{spe:252}})],p2:[m({ability:'Unaware'})],setup:[{side:'p1',boosts:{spa:3}}]});
 add('both-start',{p1:[m({ability:'Download',evs:{spe:252}})],p2:[m({ability:'Defiant'})]});
 add('defiant-defense-at-atk-cap',{p1:[m({moves:['Screech'],evs:{spe:252}})],p2:[m({ability:'Defiant'})],setup:[{side:'p2',boosts:{atk:6}}]});
 add('download-roleplay-cap',{p1:[m({moves:['Role Play'],evs:{spe:252}})],p2:[m({ability:'Download'})],setup:[{side:'p1',boosts:{spa:6}}]});
 add('defiant-switch-herb',{p1:[m(),m({ability:'Intimidate'})],p2:[m({evs:{spe:252}}),m({ability:'Defiant',item:'White Herb',evs:{spe:252}})],steps:[{p1:'switch 2',p2:'switch 2'}]});
 add('download-dual-tie',{p1:[m({ability:'Download'})],p2:[m({ability:'Download'})]});
 add('download-ignore-marvelscale',{p1:[m(),m({ability:'Download'})],p2:[m({ability:'Marvel Scale',nature:'Calm'})],setup:[{side:'p2',status:'brn'}],steps:[{p1:'switch 2',p2:'move 4'}]});
 add('download-trace',{p1:[m({ability:'Trace',evs:{spe:252}})],p2:[m({ability:'Download'})]});
 return cases;
};
module.exports.membershipWitnesses=function(){
 require('./neutral-turn-oracle.cjs').assertReady();const ps=require('./vendor/pokemon-showdown/dist/sim'),assert=require('node:assert/strict');
 return [['Defiant',5,'Farfetch’d'],['Download',9,'Genesect']].map(([ability,n,species])=>{
  const seed=[n,2,3,4],set=ps.Teams.getGenerator('gen5randombattle',seed).getTeam().find(s=>s.ability===ability&&s.species===species);assert.ok(set,ability);return {kind:'ability',id:ps.Dex.toID(ability),seed,set};
 });
};
