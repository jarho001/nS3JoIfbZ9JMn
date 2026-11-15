'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),zlib=require('node:zlib');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native Eruption and Water Spout use live source HP with pre-Technician integer power',()=>{
 require('../reference/neutral-turn-oracle.cjs').assertReady();
 const ps=require('../reference/vendor/pokemon-showdown/dist/sim'),{runCase}=require('../reference/shared-engine-cases.cjs'),spec=require('../reference/hp-power-cases.cjs');
 const actual=JSON.parse(require('node:child_process').execFileSync('node',['reference/shared-engine-cases.cjs','--batch','hppower'],{maxBuffer:64*1024*1024}));
 assert.deepEqual(normalizeStacks(JSON.parse(JSON.stringify(actual))),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/hp-power.json.gz')))));
 assert.equal(actual.cases.length,54);assert.equal(actual.cases.reduce((n,c)=>n+c.steps.length,0),64);assert.ok(actual.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
 for(const c of actual.cases)for(const side of ['p1','p2'])for(const mon of c.setup[side])assert.equal(new Set(mon.moves.map(m=>ps.Dex.toID(m))).size,mon.moves.length,c.id+' unique moves');
 const c=id=>actual.cases.find(c=>c.id==='G4-hp-power-'+id),p=(id,i=0)=>c(id).steps[i].after.sides[0].pokemon[0],damage=id=>c(id).initial.sides[1].pokemon[0].hp-parseInt(c(id).steps[0].events.p2.find(e=>e.message.startsWith('|-damage|p2a:')).message.split('|')[3],10);
 for(const [key,move]of [['eruption','Eruption'],['waterspout','Water Spout']]){
  assert.equal(damage(key+'-hp1'),1);assert.ok(damage(key+'-hp175')>damage(key+'-hp87'));
  assert.equal(damage(key+'-technician-hp70'),damage(key+'-technician-hp71'));assert.ok(damage(key+'-technician-hp71')>damage(key+'-technician-hp72'));
  assert.ok(damage(key+'-ability-hp58')>damage(key+'-ability-hp59'));
  assert.equal(p(key+'-slower').hp,79);assert.equal(p(key+'-sitrus').hp,122);assert.ok(damage(key+'-sitrus')>damage(key+'-slower'));
  assert.ok(c(key+'-sleep-talk').steps[1].events.p1.some(e=>e.message.includes('|'+move+'|')&&e.message.includes('[from] move: Sleep Talk')));
  assert.equal(p(key+'-custap').item,'');assert.equal(p(key+'-custap').speed,240);
  assert.ok(!c(key+'-protect').steps[0].draws.some(d=>d.semanticContext==='damage'));
  assert.ok(!c(key+'-absorbed').steps[0].draws.some(d=>d.semanticContext==='damage'));
  assert.equal(c(key+'-terminal').steps[0].after.terminal,true);
 }
 assert.deepEqual(spec.membershipWitnesses().map(w=>w.set.species),['Typhlosion','Wailord']);
});
