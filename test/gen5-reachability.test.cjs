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
  for(const id of ['conditions:frz:onAfterMoveSecondary',
    'abilities:immunity:onUpdate','abilities:magicguard:onDamage']) {
    const cb=slots.get(id);assert.equal(cb.status,'unsupported');assert.equal(cb.implementation,null);
  }
  assert.ok(ledger.callbacks.every(c=>['partial','unsupported'].includes(c.status)));
  assert.equal(ledger.callbacks.filter(c=>c.status==='partial').length,194);
  assert.equal(ledger.callbacks.filter(c=>c.status==='unsupported').length,952);
  // Existing native users must not contaminate raw owner provenance.
  require('../reference/vendor/pokemon-showdown/dist/sim').Dex.mod('gen5').moves.all();
  assert.deepEqual(inventory(),{data,ledger});
});
