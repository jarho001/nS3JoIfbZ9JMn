'use strict';
module.exports=function(){
 const m=(species,moves,extra={})=>({species,moves,ability:'Pressure',item:'',level:50,nature:'Hardy',gender:'M',ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},...extra});
 const b=(a=1,c=1)=>({p1:`move ${a}`,p2:`move ${c}`}),cases=[],add=(id,x={})=>cases.push({id:'G4-ability-create-'+id,seed:1,generic:true,identityFamily:true,abilityChangeFamily:true,abilityCreateFamily:true,hitFamily:true,p1:[m('Mew',['Simple Beam','Tackle'],{evs:{spe:252}})],p2:[m('Mew',['Calm Mind','Tackle'])],steps:[b(),b(2,2)],...x});
 add('simple-beam');add('simple-same',{p2:[m('Mew',['Calm Mind','Tackle'],{ability:'Simple'})],steps:[b()]});
 add('simple-zen',{p2:[m('Mew',['Recover'],{ability:'Zen Mode'})],steps:[b()]});
 add('simple-sub',{p2:[m('Mew',['Substitute'],{evs:{spe:252},nature:'Timid'})],steps:[b()]});
 add('simple-protect',{p2:[m('Mew',['Protect'])],steps:[b()]});
 add('simple-illusion',{p2:[m('Zoroark',['Calm Mind'],{ability:'Illusion'}),m('Snorlax',['Tackle'])],steps:[b()]});
 add('worry',{p1:[m('Mew',['Worry Seed','Tackle'],{evs:{spe:252}})],p2:[m('Mew',['Rest','Tackle'])],setup:[{side:'p2',hp:100}],steps:[b(),b(2,2)]});
 add('worry-sleep',{p1:[m('Mew',['Spore','Worry Seed'],{evs:{spe:252}})],p2:[m('Mew',['Tackle'])],steps:[b(),b(2)]});
 add('worry-same',{p1:[m('Mew',['Worry Seed'])],p2:[m('Mew',['Tackle'],{ability:'Insomnia'})],steps:[b()]});
 add('worry-zen',{p1:[m('Mew',['Worry Seed'])],p2:[m('Mew',['Recover'],{ability:'Zen Mode'})],steps:[b()]});
 for(const ability of ['Simple','Contrary']){
  add(ability+'-boosts',{p1:[m('Mew',['Calm Mind','Swords Dance'],{ability})],p2:[m('Mew',['Recover'])],steps:[b(),b(2),b(2),b(2),b(2)]});
  add(ability+'-growl',{p1:[m('Mew',['Growl'])],p2:[m('Mew',['Tackle'],{ability})],steps:[b(),b()]});
  add(ability+'-intimidate',{p1:[m('Mew',['Tackle'],{ability})],p2:[m('Mew',['Tackle'],{ability:'Intimidate'})],steps:[b()]});
  add(ability+'-skull',{p1:[m('Mew',['Skull Bash'],{ability})],p2:[m('Mew',['Recover'])],steps:[b(),b()]});
 }
 add('worry-illusion',{p1:[m('Mew',['Worry Seed'],{evs:{spe:252}})],p2:[m('Zoroark',['Tackle'],{ability:'Illusion'}),m('Snorlax',['Tackle'])],steps:[b()]});
 add('old-immunity-event',{p1:[m('Mew',['Toxic'],{evs:{spe:252}})],p2:[m('Mew',['Tackle'],{ability:'Immunity'})],steps:[b()]});
 add('worry-zen-end',{p1:[m('Mew',['Tackle','Worry Seed'],{evs:{spe:252}})],p2:[m('Darmanitan',['Tackle'],{ability:'Zen Mode'})],setup:[{side:'p2',hp:80}],steps:[b(),b(2)]});
 for(const ability of ['Simple','Contrary'])add(ability+'-foe-cap',{p1:[m('Mew',['Growl'])],p2:[m('Mew',['Recover'],{ability})],steps:Array.from({length:ability==='Simple'?4:7},()=>b())});
 return cases;
};
