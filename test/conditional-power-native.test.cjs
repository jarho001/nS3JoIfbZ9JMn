'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),zlib=require('node:zlib');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native-only Facade and Acrobatics preserve Gen5 burn handling and live item-dependent power',()=>{
 require('../reference/neutral-turn-oracle.cjs').assertReady();
 const ps=require('../reference/vendor/pokemon-showdown/dist/sim'),{runCase}=require('../reference/shared-engine-cases.cjs'),spec=require('../reference/conditional-power-cases.cjs');
 const actual={pin:require('../reference-lock.json').commit,witnesses:spec.membershipWitnesses(),cases:spec().map(c=>runCase(ps,c))};
 assert.deepEqual(normalizeStacks(JSON.parse(JSON.stringify(actual))),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/conditional-power-precheck.json.gz')))));
 assert.equal(actual.cases.length,28);assert.equal(actual.cases.reduce((n,c)=>n+c.steps.length,0),36);assert.ok(actual.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
 for(const c of actual.cases)for(const side of ['p1','p2'])for(const mon of c.setup[side])assert.equal(new Set(mon.moves.map(m=>ps.Dex.toID(m))).size,mon.moves.length,c.id+' unique moves');
 const c=id=>actual.cases.find(c=>c.id==='G4-conditional-power-'+id),hp=id=>Number(c(id).steps.at(-1).events.p2.find(e=>e.message.startsWith('|-damage|p2a:')).message.split('|')[3].split(/[ /]/)[0]);
 assert.equal(hp('facade-brn'),hp('facade-base'));assert.ok(hp('facade-psn')<hp('facade-base'));assert.ok(hp('facade-guts-brn')<hp('facade-brn'));assert.equal(hp('facade-sleep-talk'),hp('facade-base'));
 assert.equal(c('facade-magicguard').steps[0].after.sides[0].pokemon[0].hp,175);
 assert.ok(hp('acrobatics-empty')<hp('acrobatics-leftovers'));assert.equal(hp('acrobatics-technician-empty'),hp('acrobatics-empty'));assert.ok(hp('acrobatics-technician-held')<hp('acrobatics-leftovers'));
 assert.equal(c('acrobatics-lum').steps[0].after.sides[0].pokemon[0].item,'');assert.equal(c('acrobatics-lum').steps[0].after.sides[0].pokemon[0].status,'');
 assert.equal(c('acrobatics-trick-lose').steps[0].after.sides[0].pokemon[0].item,'');assert.equal(c('acrobatics-trick-gain').steps[0].after.sides[0].pokemon[0].item,'leftovers');
 assert.deepEqual(actual.witnesses.map(w=>w.set.species),['Raticate','Drifblim']);
});
