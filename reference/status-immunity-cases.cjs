'use strict';
module.exports=function(){
 const m=(species,moves,extra={})=>({species,moves,ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...extra});
 const b=(a=1,c=1)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[],add=(id,x={})=>cases.push({id:'G4-status-immunity-'+id,seed:1,generic:true,identityFamily:true,abilityChangeFamily:true,statusImmunityFamily:true,hitFamily:true,p1:[m('Mew',['Entrainment','Tackle'],{evs:{spe:252}})],p2:[m('Mew',['Tackle','Recover'])],steps:[b()],...x});
 for(const [ability,status]of [['Immunity','psn'],['Immunity','tox'],['Water Veil','brn'],['Limber','par'],['Magma Armor','frz'],['Insomnia','slp'],['Vital Spirit','slp']])add('acquire-'+ability.replaceAll(' ','')+'-'+status,{p1:[m('Mew',['Entrainment','Tackle'],{ability,evs:{spe:252}})],setup:[{side:'p2',status}],steps:[b(),b(2,2)]});
 for(const [id,ability,move]of [['poison','Immunity','Toxic'],['par','Limber','Thunder Wave'],['burn','Water Veil','Will-O-Wisp'],['sleep','Insomnia','Spore'],['vital','Vital Spirit','Spore']])add('block-'+id,{p1:[m('Mew',[move,'Tackle'],{evs:{spe:252}})],p2:[m('Mew',['Tackle'],{ability})],steps:[b()]});
 for(const ability of ['Insomnia','Vital Spirit'])add('rest-'+ability.replaceAll(' ',''),{p1:[m('Mew',['Rest'],{ability})],p2:[m('Mew',['Recover'])],setup:[{side:'p1',hp:100}],steps:[b()]});
 add('both-cure',{p1:[m('Mew',['Skill Swap','Tackle'],{ability:'Water Veil'})],p2:[m('Mew',['Recover','Tackle'],{ability:'Immunity'})],setup:[{side:'p1',status:'psn'},{side:'p2',status:'brn'}],steps:[b(),b(2,2)]});
 add('role-immunity',{p1:[m('Mew',['Role Play','Tackle'],{evs:{spe:252}})],p2:[m('Mew',['Tackle'],{ability:'Immunity'})],setup:[{side:'p1',status:'psn'}],steps:[b()]});
 add('freeze-secondary',{p1:[m('Mew',['Ice Beam'],{level:1})],p2:[m('Mew',['Recover'],{ability:'Magma Armor'})],steps:Array.from({length:8},()=>b())});
 add('burn-secondary',{p1:[m('Mew',['Scald'],{level:1})],p2:[m('Mew',['Recover'],{ability:'Water Veil'})],steps:Array.from({length:5},()=>b())});
 for(const c of cases)if(c.setup?.some(s=>s.status==='slp')){
  c.p1[0].moves=['Spore','Entrainment','Tackle'];delete c.setup;c.steps=[b(),b(2),b(3,2)];
 }
 return cases;
};
