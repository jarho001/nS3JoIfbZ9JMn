'use strict';
// G4-DATA family, observed before generic table integration. Supplied sets
// establish mechanics, not eligibility of arbitrary move/ability combinations.
module.exports=function genericDataCases(mon) {
  const fast={evs:{hp:85,atk:85,def:85,spa:85,spd:85,spe:85}};
  const set=(species,moves,o={})=>mon(species,moves,{ability:'Synchronize',...fast,...o});
  const both=(p1,p2)=>({p1:`move ${p1}`,p2:`move ${p2}`});
  const cases=[];
  for(const [id,a,b] of [
    ['special-stab',set('Espeon',['Power Gem','Dragon Pulse']),set('Flygon',['Surf','Aura Sphere'],{ability:'Levitate'})],
    ['dual-resistance',set('Bronzong',['Dragon Pulse'],{ability:'Levitate'}),set('Musharna',['X-Scissor'])],
    ['normal-ghost',set('Zangoose',['Strength'],{ability:'Immunity'}),set('Gengar',['Power Gem'],{ability:'Levitate'})],
    ['ground-flying',set('Zangoose',['Earthquake'],{ability:'Immunity'}),set('Zapdos',['Swift'])],
    ['ground-levitate',set('Zangoose',['Earthquake'],{ability:'Immunity'}),set('Eelektross',['Surf'],{ability:'Levitate'})],
    ['type-immunity',set('Espeon',['Dragon Pulse']),set('Bronzong',['Aura Sphere'],{ability:'Levitate'})],
    ['hidden-power',set('Mew',['Hidden Power Ice'],{ivs:{hp:31,atk:30,def:30,spa:31,spd:31,spe:31}}),set('Flygon',['Hidden Power Grass'],{ability:'Levitate',ivs:{hp:31,atk:30,def:31,spa:30,spd:31,spe:31}})],
    ['happiness',set('Zangoose',['Return','Frustration'],{ability:'Immunity',happiness:0}),set('Mew',['Return','Frustration'],{happiness:200})],
    ['nature',set('Espeon',['Surf'],{nature:'Modest',ivs:{hp:7,atk:0,def:20,spa:31,spd:30,spe:5}}),set('Bronzong',['Aqua Tail'],{nature:'Sassy',ability:'Levitate',evs:{hp:252,atk:80,def:0,spa:0,spd:176,spe:0}})],
    ['fixed-gender',set('Nidoking',['Mega Kick'],{gender:'M',shiny:true}),set('Nidoqueen',['Slam'],{gender:'F'})],
    ['ghost-dark-steel',set('Mew',['Feint Attack']),set('Bronzong',['Protect','Recover'],{ability:'Levitate'})],
    ['priority',set('Zangoose',['Quick Attack','Extreme Speed'],{ability:'Immunity'}),set('Mew',['Surf'])],
  ]) cases.push({id:`G4-data-${id}`,seed:1,p1:[a],p2:[b],steps:[both(1,1),both(a.moves.length>1?2:1,b.moves.length>1?2:1)]});
  for(const species of ['Charizard','Swampert','Flygon','Abomasnow']) cases.push({
    id:`G4-data-entry-${species}`,seed:9,
    p1:[set('Mew',['Stealth Rock','Spikes','Recover'])],
    p2:[set('Bronzong',['Protect'],{ability:'Levitate'}),set(species,['Tackle'])],
    steps:[both(1,1),both(2,1),{p1:'move 3',p2:'switch 2'}]});
  for(const species of ['Charizard','Bronzong','Swampert'])cases.push({
    id:`G4-data-status-${species}`,seed:0,
    p1:[set('Mew',['Toxic','Will-O-Wisp','Thunder Wave'])],
    p2:[set(species,['Recover'])],steps:[both(1,1),both(2,1),both(3,1)]});
  cases.push({id:'G4-data-initial-gender-order',seed:1,
    p1:[set('Musharna',['Surf'],{gender:'F'}),set('Zapdos',['Swift']),set('Flygon',['Power Gem'])],
    p2:[set('Bronzong',['Surf']),set('Espeon',['Surf'],{gender:'M'}),set('Zangoose',['Strength'])],steps:[both(1,1)]});
  const teamA=[set('Espeon',['Power Gem']),set('Flygon',['Dragon Pulse'],{ability:'Levitate'}),
    set('Mew',['Surf']),set('Zangoose',['X-Scissor'],{ability:'Immunity'}),
    set('Bronzong',['Surf'],{ability:'Levitate'}),set('Eelektross',['Hidden Power Ice'],{ability:'Levitate'})];
  const teamB=[set('Flygon',['Earthquake'],{ability:'Levitate'}),set('Musharna',['Power Gem']),
    set('Bronzong',['Surf'],{ability:'Levitate'}),set('Nidoking',['Mega Kick']),
    set('Nidoqueen',['Aqua Tail']),set('Mew',['Aura Sphere'])];
  for(const seed of [1,21])for(const swap of [false,true])cases.push({
    id:`G4-data-full-${seed}-${swap}`,seed,p1:swap?teamB:teamA,p2:swap?teamA:teamB,
    fullBattle:true,family:'generic'});
  return cases.map(c=>({...c,generic:true}));
};
