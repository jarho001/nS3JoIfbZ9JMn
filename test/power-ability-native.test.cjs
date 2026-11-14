'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),zlib=require('node:zlib');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native-only Iron Fist and Reckless preserve punch/recoil power, modifier chains and suppression',()=>{
 require('../reference/neutral-turn-oracle.cjs').assertReady();
 const ps=require('../reference/vendor/pokemon-showdown/dist/sim'),{runCase}=require('../reference/shared-engine-cases.cjs'),spec=require('../reference/power-ability-cases.cjs');
 const actual={pin:require('../reference-lock.json').commit,witnesses:spec.membershipWitnesses(),cases:spec().map(c=>runCase(ps,c))};
 assert.deepEqual(normalizeStacks(JSON.parse(JSON.stringify(actual))),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/power-ability-precheck.json.gz')))));
 assert.equal(actual.cases.length,40);assert.equal(actual.cases.reduce((n,c)=>n+c.steps.length,0),44);
 assert.ok(actual.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
 const c=id=>actual.cases.find(c=>c.id==='G4-power-ability-'+id),damage=id=>{const x=c(id),line=x.steps[0].events.p2.find(e=>e.message.startsWith('|-damage|p2a:'));return x.initial.sides[1].pokemon[0].maxhp-Number(line.message.split('|')[3].split('/')[0]);};
 for(const [key,moves]of [['ironfist',['machpunch','drainpunch','firepunch','shadowpunch']],['reckless',['doubleedge','takedown','bravebird','flareblitz','wildcharge']]]){
  for(const move of moves)assert.ok(damage(key+'-'+move)>damage(key+'-'+move+'-control'),key+' '+move);
  assert.equal(damage(key+'-tackle'),damage(key+'-tackle-control'));
  assert.equal(c(key+'-suppressed').steps[0].after.sides[0].pokemon[0].volatiles.gastroacid.id,'gastroacid');
 }
 assert.deepEqual(actual.witnesses.map(w=>w.set.species),['Mienshao','Hitmonchan']);
});
