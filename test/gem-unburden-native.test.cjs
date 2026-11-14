'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),zlib=require('node:zlib');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native gems and Unburden preserve item-use order, source metadata and persistent conditions',()=>{
 require('../reference/neutral-turn-oracle.cjs').assertReady();
 const ps=require('../reference/vendor/pokemon-showdown/dist/sim'),{runCase}=require('../reference/shared-engine-cases.cjs'),spec=require('../reference/gem-unburden-cases.cjs');
 const actual=JSON.parse(require('node:child_process').execFileSync('node',['reference/shared-engine-cases.cjs','--batch','gemunburden'],{maxBuffer:64*1024*1024}));
 assert.deepEqual(normalizeStacks(JSON.parse(JSON.stringify(actual))),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/gem-unburden.json.gz')))));
 assert.equal(actual.cases.length,48);assert.equal(actual.cases.reduce((n,c)=>n+c.steps.length,0),89);assert.ok(actual.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
 for(const c of actual.cases)for(const side of ['p1','p2'])for(const mon of c.setup[side])assert.equal(new Set(mon.moves.map(m=>ps.Dex.toID(m))).size,mon.moves.length,c.id+' unique moves');
 const c=id=>actual.cases.find(c=>c.id==='G4-gem-unburden-'+id),p=(id,i=0)=>c(id).steps[i].after.sides[0].pokemon[0];
 for(const key of ['flying','fighting']){
  assert.equal(p(key).item,'');assert.equal(p(key).volatiles.gem,undefined);assert.equal(p(key).speed,304);
  const terminal=p(key+'-terminal');assert.equal(terminal.volatiles.gem.duration,1);assert.ok(terminal.volatiles.unburden.effectOrder<terminal.volatiles.gem.effectOrder);
  assert.equal(terminal.volatiles.gem.sourceEffect.$effect,key+'gem');
  assert.equal(p(key+'-protect').item,key+'gem');
 }
 assert.equal(p('unburden-sash').hp,1);assert.equal(p('unburden-sash').speed,14);assert.ok(p('unburden-sash').volatiles.unburden);
 assert.equal(p('unburden-recycle',1).item,'lumberry');assert.equal(p('unburden-recycle',1).speed,120);assert.deepEqual(p('unburden-recycle',2).volatiles.unburden,p('unburden-recycle').volatiles.unburden);
 assert.equal(p('unburden-naked').speed,152);assert.equal(p('fighting-hiddenpower-par').speed,75);
 assert.equal(p('flying-status').item,'flyinggem');assert.equal(p('fighting-ghost').item,'fightinggem');assert.equal(p('fighting-miss').item,'fightinggem');
 assert.equal(p('unburden-trick-other').volatiles.unburden.source.$pokemon.side,1);
 assert.equal(p('unburden-knockoff').volatiles.unburden.source.$pokemon.side,0);
 const both=c('unburden-trick-both').steps[0].after.sides;assert.ok(both[1].pokemon[0].volatiles.unburden.effectOrder<both[0].pokemon[0].volatiles.unburden.effectOrder);
 const repeated=c('unburden-repeat-take').steps.map(s=>s.after.sides[0].pokemon[0].volatiles.unburden);assert.equal(repeated[0].source.$pokemon.side,1);assert.deepEqual(repeated[1],repeated[0]);
 assert.equal(p('unburden-suppress-after',1).volatiles.unburden,undefined);assert.equal(p('fighting-mummy').volatiles.unburden,undefined);
 for(const ability of ['pressure','unburden'])assert.equal(p('unburden-transform-'+ability,1).volatiles.unburden,undefined);
 assert.equal(p('unburden-switch-reset',2).speed,152);assert.equal(p('unburden-switch-reset',2).volatiles.unburden,undefined);
 assert.ok(c('gem-residual-tie').steps[0].draws.some(d=>d.semanticContext==='residual-handlers'));
 assert.deepEqual(spec.membershipWitnesses().map(w=>w.set.species),['Drifblim','Hitmonlee','Drifblim']);
});
