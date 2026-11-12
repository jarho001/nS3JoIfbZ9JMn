'use strict';
module.exports=function(mon){
 const fast={evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:252}},m=(moves,item='',extra={})=>mon('Mew',moves,{item,gender:'N',...extra});
 const both=(a=1,b=1)=>({p1:`move ${a}`,p2:`move ${b}`});
 const cases=[],add=(id,move,a,b,extra={})=>cases.push({id:'G4-items-'+id,generic:true,itemLifecycleFamily:true,hitFamily:true,seed:1,p1:[m(move==='Recycle'?[move,'Tackle']:[move,'Tackle','Recycle'],a,fast),m(['Tackle'])],p2:[m(['Tackle','Recover'],b),m(['Tackle'])],steps:[both(),both(2)],...extra});
 for(const move of ['Trick','Switcheroo'])for(const [id,a,b]of [['both','Life Orb','Leftovers'],['take','','Choice Band'],['give','Choice Scarf',''],['empty','',''],['same','Leftovers','Leftovers']])add(move+'-'+id,move,a,b,{steps:[both()]});
 add('sticky-trick','Trick','Life Orb','Leftovers',{p2:[m(['Tackle'],'Leftovers',{ability:'Sticky Hold'})],steps:[both()]});
 add('substitute-trick','Trick','Life Orb','Leftovers',{p2:[m(['Substitute'],'Leftovers',fast)],p1:[m(['Trick'],'Life Orb')],steps:[both()]});
 add('choice-to-choice','Trick','Choice Band','Choice Scarf',{steps:[both(),both()]});
 add('lum-recycle','Recycle','Lum Berry','',{p1:[m(['Recycle','Tackle'],'Lum Berry',fast)],p2:[m(['Thunder Wave','Tackle'])],steps:[both(2),both(1,2),both(2,1),both(1,2)]});
 add('herb-recycle','Solar Beam','Power Herb','',{p1:[m(['Solar Beam','Recycle'],'Power Herb',fast)],p2:[m(['Recover'])],steps:[both(),both(2),both()]});
 add('recycle-fail','Recycle','','',{steps:[both()]});
 add('recycle-held','Recycle','Leftovers','',{steps:[both()]});
 for(const [id,item,extra]of [['plain','Leftovers',{}],['empty','',{}],['sticky','Leftovers',{ability:'Sticky Hold'}],['helmet','Rocky Helmet',{}]])add('knock-'+id,'Knock Off','',item,{p2:[m(['Recover','Recycle'],item,extra)],steps:[both(),both(2,2)]});
 add('knock-sub','Knock Off','','Leftovers',{p1:[m(['Knock Off'])],p2:[m(['Substitute'],'Leftovers',fast)],steps:[both()]});
 add('knock-contact-ko','Knock Off','','Rocky Helmet',{setup:[{side:'p1',hp:1}],steps:[both()]});
 add('knock-target-ko','Knock Off','','Leftovers',{setup:[{side:'p2',hp:1}],p2:[m(['Tackle'],'Leftovers'),m(['Tackle'])],steps:[both()]});
 add('swap-switch','Trick','Power Herb','Lum Berry',{steps:[both(),{p1:'switch 2',p2:'switch 2'},{p1:'switch 2',p2:'switch 2'}]});
 for(const id of ['Trick-take','knock-sticky','lum-recycle']){const c=structuredClone(cases.find(c=>c.id==='G4-items-'+id));c.id+='-mirror';[c.p1,c.p2]=[c.p2,c.p1];c.steps=c.steps.map(s=>Object.fromEntries(Object.entries(s).map(([k,v])=>[k==='p1'?'p2':'p1',v])));if(c.setup)c.setup=c.setup.map(s=>({...s,side:s.side==='p1'?'p2':'p1'}));cases.push(c);}
 return cases;
};
