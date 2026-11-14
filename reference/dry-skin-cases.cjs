'use strict';
module.exports=function(){
 const m=(moves,x={})=>({species:'Mew',moves,ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...x});
 const b=(a=1,c=1)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[];
 const add=(id,move='Surf',x={})=>cases.push({id:'G4-dry-skin-'+id,seed:1,generic:true,drySkinFamily:true,p1:[m([move,'Tackle'],{evs:{spe:252}})],p2:[m(['Growl'],{ability:'Dry Skin'})],steps:[b()],...x});
 add('water-partial','Surf',{setup:[{side:'p2',hp:60}]});add('water-full');add('water-cap','Waterfall',{setup:[{side:'p2',hp:174}]});
 add('fire-special','Flamethrower');add('fire-physical','Fire Punch');add('status','Will-O-Wisp');add('wrong','Thunderbolt');
 add('technician-item','Ember',{p1:[m(['Ember'],{ability:'Technician',item:'Wise Glasses',evs:{spe:252}})]});
 add('physical-item','Fire Punch',{p1:[m(['Fire Punch'],{item:'Muscle Band',evs:{spe:252}})]});
 add('protect','Surf',{p2:[m(['Protect'],{ability:'Dry Skin'})]});
 add('substitute','Surf',{p1:[m(['Recover','Surf'])],p2:[m(['Substitute','Growl'],{ability:'Dry Skin',evs:{spe:252}})],steps:[b(),b(2,2)]});
 add('gas','Surf',{p1:[m(['Gastro Acid','Surf'],{evs:{spe:252}})],setup:[{side:'p2',hp:60}],steps:[b(),b(2)]});
 for(const [key,weather]of [['rain','Rain Dance'],['sun','Sunny Day'],['sand','Sandstorm'],['hail','Hail']]){
  add(key,weather,{p1:[m([weather,'Recover'],{evs:{spe:252}})],setup:[{side:'p2',hp:80}],steps:[b(),b(2)]});
  add(key+'-tie',weather,{p1:[m([weather,'Recover'])],setup:[{side:'p2',hp:80}],steps:[b(),b(2)]});
 }
 add('rain-cap','Rain Dance',{setup:[{side:'p2',hp:174}]});add('rain-full','Rain Dance');
 add('sun-ko','Sunny Day',{setup:[{side:'p2',hp:1}]});
 add('sun-minimum','Sunny Day',{p2:[m(['Growl'],{species:'Shedinja',ability:'Dry Skin'})]});
 add('rain-gas','Rain Dance',{p1:[m(['Rain Dance','Gastro Acid'],{evs:{spe:252}})],setup:[{side:'p2',hp:60}],steps:[b(),b(2)]});
 add('sun-gas','Sunny Day',{p1:[m(['Sunny Day','Gastro Acid'],{evs:{spe:252}})],steps:[b(),b(2)]});
 add('rain-expiry','Rain Dance',{p1:[m(['Rain Dance','Recover'],{evs:{spe:252}})],setup:[{side:'p2',hp:50}],steps:[b(),...Array.from({length:5},()=>b(2))]});
 add('sun-expiry','Sunny Day',{p1:[m(['Sunny Day','Recover'],{evs:{spe:252}})],steps:[b(),...Array.from({length:5},()=>b(2))]});
 add('rain-burn','Rain Dance',{p1:[m(['Rain Dance','Recover'],{evs:{spe:252}})],setup:[{side:'p2',hp:80,status:'brn'}],steps:[b(),b(2)]});
 add('sun-burn','Sunny Day',{p1:[m(['Sunny Day','Recover'],{evs:{spe:252}})],setup:[{side:'p2',hp:80,status:'brn'}],steps:[b(),b(2)]});
 add('entry-trace','Surf',{p1:[m(['Surf'],{ability:'Trace',evs:{spe:252}})],p2:[m(['Growl'],{ability:'Forecast'}),m(['Growl'],{ability:'Dry Skin'})],setup:[{side:'p2',mon:1,hp:60}],steps:[{p1:'move 1',p2:'switch 2'}]});
 add('mirrored','Surf',{p1:[m(['Growl'],{ability:'Dry Skin'})],p2:[m(['Surf'],{evs:{spe:252}})],setup:[{side:'p1',hp:60}]});
 // Retained historical inputs protect the opposite metadata branch: damage
 // modifiers have already visited the field and installed weatherState.target.
 const fs=require('node:fs'),zlib=require('node:zlib'),path=require('node:path');
 const old=JSON.parse(zlib.gunzipSync(fs.readFileSync(path.join(__dirname,'../test-logs/g4-cpu-conformance/field.json.gz'))));
 for(const suffix of ['RainDance','SunnyDay']){
  const original=old.cases.find(c=>c.id==='G4-field-weather-'+suffix).setup;
  cases.push({...original,id:'G4-dry-skin-legacy-'+suffix,drySkinFamily:true,fieldFamily:false});
 }
 return cases;
};
module.exports.membershipWitnesses=function(){
 require('./neutral-turn-oracle.cjs').assertReady();const ps=require('./vendor/pokemon-showdown/dist/sim'),assert=require('node:assert/strict'),seed=[23,2,3,4];
 const set=ps.Teams.getGenerator('gen5randombattle',seed).getTeam().find(s=>s.ability==='Dry Skin');assert.ok(set);return [{ability:'dryskin',seed,set}];
};
