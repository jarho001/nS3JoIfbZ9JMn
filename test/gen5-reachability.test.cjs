'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const {inventory,jsonHash}=require('../reference/gen5-reachability.cjs');

test('pinned resolved export and conservative inventory reproduce exactly',()=>{
  const {data,ledger}=inventory();
  assert.deepEqual(data,JSON.parse(fs.readFileSync('rules/gen5-data-v1.json','utf8')));
  assert.deepEqual(ledger,JSON.parse(fs.readFileSync('rules/gen5-reachable-v1.json','utf8')));
  assert.equal(ledger.dataSha256,jsonHash(data));
  assert.equal(ledger.roots.length,605);
  assert.equal(new Set(ledger.roots.map(r=>r.species)).size,388);
  assert.equal(data.types.length,17);
  assert.ok(!data.types.some(t=>['Fairy','Stellar'].includes(t.name)));
  const species=new Set(data.species.map(r=>r.id));
  const moves=new Set(data.moves.map(r=>r.id));
  const abilities=new Set(data.abilities.map(r=>r.id));
  for(const root of ledger.roots) {
    assert.ok(species.has(root.species),root.id);
    for(const move of root.moves)assert.ok(moves.has(move.id),root.id+':'+move.input);
    for(const ability of root.abilities)assert.ok(abilities.has(ability),root.id+':'+ability);
  }
  const slots=new Map(ledger.callbacks.map(row=>[row.id,row]));
  function inspect(value) {
    if(!value||typeof value!=='object')return;
    if(value.$callback) {
      const cb=slots.get(value.$callback);assert.ok(cb,value.$callback);
      assert.ok(cb.definitions.length);
      for(const def of cb.definitions)assert.ok(ledger.sources[def.file]);
      return;
    }
    for(const child of Object.values(value))inspect(child);
  }
  inspect(data);
  const physical=slots.get('abilities:synchronize:onAfterSetStatus');
  assert.ok(physical.definitions.some(d=>d.file==='data/abilities.ts'));
  assert.ok(slots.get('conditions:slp:onSwitchIn').definitions.some(d=>d.file==='data/mods/gen5/conditions.ts'));
  assert.equal(ledger.generatorMethods.randomSet.file,'data/random-battles/gen5/teams.ts');
  assert.equal(ledger.generatorMethods.getForme.file,'data/random-battles/gen9/teams.ts');
  assert.equal(ledger.generatorMethods.getPokemonPool.file,'data/random-battles/gen9/teams.ts');
  assert.equal(ledger.status,'incomplete');
  assert.ok(ledger.gaps.some(g=>g.id==='indirect-closure'&&g.status==='unresolved'));
  for(const id of ['conditions:slp:onBeforeMove','moves:rest:onHit',
    'moves:reflect:condition.onAnyModifyDamage','conditions:sandstorm:onWeather',
    'abilities:pressure:onDeductPP','abilities:prankster:onModifyPriority',
    'abilities:overgrow:onModifyAtk','abilities:overgrow:onModifySpA','abilities:blaze:onModifyAtk','abilities:blaze:onModifySpA','abilities:torrent:onModifyAtk','abilities:torrent:onModifySpA','abilities:swarm:onModifyAtk','abilities:swarm:onModifySpA','abilities:waterabsorb:onTryHit','abilities:voltabsorb:onTryHit','abilities:magicguard:onDamage','abilities:immunity:onUpdate','abilities:trace:onStart','abilities:trace:onUpdate','abilities:mummy:onDamagingHit',
    'abilities:icebody:onWeather','items:lifeorb:onAfterMoveSecondarySelf',
    'conditions:mustrecharge:onStart','conditions:mustrecharge:onBeforeMove','conditions:choicelock:onStart','conditions:frz:onModifyMove',
    'abilities:skilllink:onModifyMove','abilities:rockhead:onDamage',
    'abilities:roughskin:onDamagingHit','abilities:ironbarbs:onDamagingHit',
    'abilities:liquidooze:onSourceTryHeal','items:rockyhelmet:onDamagingHit']) {
    const cb=slots.get(id);assert.equal(cb.status,'partial');
    assert.ok(fs.existsSync(cb.implementation.file));
    assert.ok(cb.evidence.every(e=>fs.existsSync(e.native)&&fs.existsSync(e.test)));
    assert.match(cb.implementation.scope,/complete .*validation pending/);
  }
  for(const id of ['conditions:frz:onAfterMoveSecondary']) {
    const cb=slots.get(id);assert.equal(cb.status,'unsupported');assert.equal(cb.implementation,null);
  }
  assert.ok(ledger.callbacks.every(c=>['partial','unsupported'].includes(c.status)));
  assert.equal(ledger.callbacks.filter(c=>c.status==='partial').length,360);
  assert.equal(ledger.callbacks.filter(c=>c.status==='unsupported').length,786);
  const audit=ledger.reachabilityAudit;
  const entries=new Map(ledger.entries.map(e=>[e.id,e]));
  assert.equal(audit.proofs['move-identity-bound'].bound.length,224);
  assert.equal(audit.proofs['item-identity-bound'].bound.length,45);
  for(const id of ['moves:metronome','moves:copycat','moves:recycle','items:powerherb','items:iceberry']) {
    assert.equal(entries.get(id).reachability,'proven_unreachable',id);
    assert.ok(audit.proofs[entries.get(id).reachabilityAudit].references.length);
  }
  for(const id of ['abilities:magicguard','moves:doubleedge','species:clefable'])
    assert.equal(entries.get(id).reachability,'required',id);
  for(const id of ['overgrow','blaze','torrent','swarm']) {
    assert.equal(entries.get('abilities:'+id).reachability,'required');
    assert.equal(entries.get('abilities:'+id).reachabilityAudit,'low-hp-ability-positive-witness');
    assert.equal(slots.get('abilities:'+id+':onModifyAtk').reachability,'unresolved');
  }
  for(const id of ['waterabsorb','voltabsorb']) {
    assert.equal(entries.get('abilities:'+id).reachability,'required');
    assert.equal(entries.get('abilities:'+id).reachabilityAudit,'type-absorption-positive-witness');
    assert.equal(slots.get('abilities:'+id+':onTryHit').reachability,'unresolved');
  }
  for(const id of ['raindish','hydration']) {
    assert.equal(entries.get('abilities:'+id).reachability,'required');
    assert.equal(entries.get('abilities:'+id).reachabilityAudit,'rain-recovery-positive-witness');
  }
  for(const id of ['shedskin','poisonheal']) {
    assert.equal(entries.get('abilities:'+id).reachability,'required');
    assert.equal(entries.get('abilities:'+id).reachabilityAudit,'residual-recovery-positive-witness');
  }
  for(const id of ['blacksludge','toxicorb','flameorb']) {
    assert.equal(entries.get('items:'+id).reachability,'required');
    assert.equal(entries.get('items:'+id).reachabilityAudit,'residual-item-positive-witness');
    assert.ok(slots.get('items:'+id+':onResidual').implementation);
  }
  for(const id of ['chestoberry','sitrusberry']) {
    assert.equal(entries.get('items:'+id).reachability,'required');
    assert.equal(entries.get('items:'+id).reachabilityAudit,'restorative-berry-positive-witness');
    assert.ok(slots.get('items:'+id+':onUpdate').implementation);
  }
  for(const id of ['lightball','thickclub','souldew']) {
    assert.equal(entries.get('items:'+id).reachability,'required');
    assert.equal(entries.get('items:'+id).reachabilityAudit,'species-stat-item-positive-witness');
    assert.ok(slots.get('items:'+id+':'+(id==='souldew'?'onModifySpD':'onModifyAtk')).implementation);
  }
  for(const [owner,field]of [['abilities:clearbody','onTryBoost'],['items:whiteherb','onStart']]){
    assert.equal(entries.get(owner).reachability,'required');assert.equal(entries.get(owner).reachabilityAudit,'stat-drop-positive-witness');assert.ok(slots.get(owner+':'+field).implementation);
  }
  for(const owner of ['abilities:sturdy','items:focussash']){
    assert.equal(entries.get(owner).reachability,'required');assert.equal(entries.get(owner).reachabilityAudit,'survival-positive-witness');assert.ok(slots.get(owner+':onDamage').implementation);
  }
  assert.equal(slots.get('abilities:sturdy:onTryHit').implementation,null);
  for(const [owner,field]of [['abilities:noguard','onAnyAccuracy'],['abilities:victorystar','onAnyModifyAccuracy']]){
    assert.equal(entries.get(owner).reachability,'required');assert.equal(entries.get(owner).reachabilityAudit,'accuracy-ability-positive-witness');assert.ok(slots.get(owner+':'+field).implementation);
  }
  for(const owner of ['abilities:tangledfeet','abilities:wonderskin']){
    assert.equal(entries.get(owner).reachability,'required');assert.equal(entries.get(owner).reachabilityAudit,'evasion-ability-positive-witness');assert.ok(slots.get(owner+':onModifyAccuracy').implementation);
  }
  for(const [owner,field]of [['abilities:multiscale','onSourceModifyDamage'],['abilities:defeatist','onModifyAtk']]){
    assert.equal(entries.get(owner).reachability,'required');assert.equal(entries.get(owner).reachabilityAudit,'hp-modifier-positive-witness');assert.ok(slots.get(owner+':'+field).implementation);
  }
  assert.equal(entries.get('abilities:unaware').reachability,'required');
  assert.equal(entries.get('abilities:unaware').reachabilityAudit,'unaware-positive-witness');assert.ok(slots.get('abilities:unaware:onAnyModifyBoost').implementation);
  for(const [owner,field]of [['abilities:download','onStart'],['abilities:defiant','onAfterEachBoost']]){
    assert.equal(entries.get(owner).reachability,'required');assert.equal(entries.get(owner).reachabilityAudit,'boost-ability-positive-witness');assert.ok(slots.get(owner+':'+field).implementation);
  }
  for(const [owner,field]of [['abilities:moxie','onSourceAfterFaint'],['abilities:aftermath','onDamagingHit']]){
    assert.equal(entries.get(owner).reachability,'required');assert.equal(entries.get(owner).reachabilityAudit,'faint-ability-positive-witness');assert.ok(slots.get(owner+':'+field).implementation);
  }
  for(const owner of ['abilities:weakarmor','abilities:cursedbody']){
    assert.equal(entries.get(owner).reachability,'required');assert.equal(entries.get(owner).reachabilityAudit,'reactive-ability-positive-witness');assert.ok(slots.get(owner+':onDamagingHit').implementation);
  }
  for(const owner of ['abilities:static','abilities:flamebody']){
    assert.equal(entries.get(owner).reachability,'required');assert.equal(entries.get(owner).reachabilityAudit,'contact-status-positive-witness');assert.ok(slots.get(owner+':onDamagingHit').implementation);
  }
  for(const owner of ['abilities:ironfist','abilities:reckless']){
    assert.equal(entries.get(owner).reachability,'required');assert.equal(entries.get(owner).reachabilityAudit,'power-ability-positive-witness');assert.ok(slots.get(owner+':onBasePower').implementation);
  }
  for(const owner of ['moves:jumpkick','moves:highjumpkick']){
    assert.equal(entries.get(owner).reachability,'required');assert.equal(entries.get(owner).reachabilityAudit,'crash-move-positive-witness');assert.ok(slots.get(owner+':onMoveFail').implementation);
  }
  for(const [owner,field]of [['moves:suckerpunch','onTry'],['moves:payback','basePowerCallback']]){
    assert.equal(entries.get(owner).reachability,'required');assert.equal(entries.get(owner).reachabilityAudit,'conditional-attack-positive-witness');assert.ok(slots.get(owner+':'+field).implementation);
  }
  for(const owner of ['moves:foulplay','moves:psyshock','moves:psystrike','moves:secretsword']){
    const entry=entries.get(owner);assert.equal(entry.reachability,'required');assert.equal(entry.reachabilityAudit,'stat-selection-positive-witness');assert.equal(entry.implementation.behavior.family,'stat_selection');assert.ok(entry.evidence.some(e=>e.test==='test/test_stat_selection_engine.py'));
  }
  for(const owner of ['moves:lowkick','moves:grassknot','moves:heavyslam']){
    assert.equal(entries.get(owner).reachability,'required');assert.equal(entries.get(owner).reachabilityAudit,'weight-based-positive-witness');assert.ok(slots.get(owner+':basePowerCallback').implementation);
  }
  for(const [owner,field]of [['moves:facade','onBasePower'],['moves:acrobatics','basePowerCallback']]){
    assert.equal(entries.get(owner).reachability,'required');assert.equal(entries.get(owner).reachabilityAudit,'conditional-power-positive-witness');assert.ok(slots.get(owner+':'+field).implementation);
  }
  assert.equal(entries.get('abilities:zenmode').reachability,'unresolved');
  assert.equal(slots.get('abilities:magicguard:onDamage').reachability,'unresolved');
  for(const row of [...ledger.entries,...ledger.callbacks,...ledger.roots])
    assert.ok(['required','proven_unreachable','unresolved'].includes(row.reachability));
  // Exclusion narrows scope without discarding synthetic implementation/evidence.
  assert.ok(entries.get('moves:copycat').implementation);
  assert.ok(entries.get('moves:copycat').evidence.length);
  // Existing native users must not contaminate raw owner provenance.
  require('../reference/vendor/pokemon-showdown/dist/sim').Dex.mod('gen5').moves.all();
  assert.deepEqual(inventory(),{data,ledger});
});
