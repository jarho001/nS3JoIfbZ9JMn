#!/usr/bin/env node
'use strict';
// Independent pre-entry constructor and arithmetic witnesses. These are not
// full native initialization: no SwitchIn/Start/Update hook has run yet.
const fs=require('node:fs');
const assert=require('node:assert/strict');
const {assertReady}=require('./neutral-turn-oracle.cjs');
const {observeInitialization}=require('./shared-engine-cases.cjs');
const PIN=require('../reference-lock.json').commit;
const STATS=['hp','atk','def','spa','spd','spe'];
function cases() {
  assertReady();
  const ps=require('./vendor/pokemon-showdown/dist/sim');
  const dex=ps.Dex.mod('gen5');
  const sets=require('./vendor/pokemon-showdown/data/random-battles/gen5/sets.json');
  const generator=ps.Teams.getGenerator('gen5randombattle',[2026,11,8,3]);
  const records=[];
  function record(id,set,classification) {
    // Capture one side before adding p2 starts the battle and entry hooks.
    const seed=[2026,11,8,4];
    const {battle,draws}=observeInitialization(ps,()=>{
      const b=new ps.Battle({formatid:'gen5randombattle',seed});
      b.setPlayer('p1',{name:'p1',team:[structuredClone(set)]}); return b;
    });
    try {
      assert.equal(battle.started,false);
      const p=battle.p1.pokemon[0];
      records.push({id,classification,set,seed,draws,rng:battle.prng.getSeed().split(',').map(Number),
        expected:{species:p.species.id,level:p.level,gender:p.gender,ability:p.ability,item:p.item,
          stats:STATS.map(s=>p.baseStoredStats[s]),hpType:p.hpType,hpPower:p.hpPower,
          ivs:STATS.map(s=>p.set.ivs[s]),evs:STATS.map(s=>p.set.evs[s]),
          moves:p.moveSlots.map(m=>({id:m.id,pp:m.pp,maxpp:m.maxpp,target:m.target}))}});
    } finally {battle.destroy();}
  }
  for(const id of Object.keys(sets).sort()) record(`generated-set:${id}`,
    generator.randomSet(id,{},false),'native-generated-set; team-eligibility not established');
  for(const nature of dex.natures.all()) for(const level of [1,100]) record(`nature:${nature.id}:${level}`,
    {species:'Mew',moves:['Hidden Power Ice','Return'],nature:nature.name,level,
      ivs:{hp:0,atk:2,def:30,spa:31,spd:1,spe:3},evs:{hp:252,atk:0,def:85,spa:255,spd:4,spe:1}},
    'native-constructed; arithmetic boundary');
  // Every possible pair of low IV bitplanes, including their independent type
  // and power projections. Order differs from stored-stat order (Speed fourth).
  const hp=[];
  const hpOrder=['hp','atk','def','spe','spa','spd'];
  for(let a=0;a<64;a++)for(let b=0;b<64;b++) {
    const ivs=Object.fromEntries(hpOrder.map((s,i)=>[s,((a>>i)&1)+2*((b>>i)&1)]));
    hp.push({ivs:STATS.map(s=>ivs[s]),...dex.getHiddenPower(ivs)});
  }
  const types=dex.types.names().sort();
  const chart=types.map(attack=>types.map(defense=>({
    immune:!dex.getImmunity(attack,defense),exponent:dex.getEffectiveness(attack,defense)})));
  return {schemaVersion:1,pin:PIN,phase:'pre-entry-set-construction',
    records,hiddenPower:hp,types,chart};
}
if(require.main===module) {
  const args=process.argv.slice(2);
  const data=JSON.stringify(cases(),null,2)+'\n';
  if(!args.length)process.stdout.write(data);
  else {assert.equal(args.length,2);assert.equal(args[0],'--output');fs.writeFileSync(args[1],data,{flag:'wx'});}
}
module.exports={cases};
