'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),zlib=require('node:zlib');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native Custap priority preserves queue-time consumption, priority bands and Unburden caching',()=>{
 require('../reference/neutral-turn-oracle.cjs').assertReady();
 const ps=require('../reference/vendor/pokemon-showdown/dist/sim'),{runCase}=require('../reference/shared-engine-cases.cjs'),spec=require('../reference/custap-priority-cases.cjs');
 const actual=JSON.parse(require('node:child_process').execFileSync('node',['reference/shared-engine-cases.cjs','--batch','custappriority'],{maxBuffer:64*1024*1024}));
 assert.deepEqual(normalizeStacks(JSON.parse(JSON.stringify(actual))),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/custap-priority.json.gz')))));
 assert.equal(actual.cases.length,33);assert.equal(actual.cases.reduce((n,c)=>n+c.steps.length,0),45);assert.ok(actual.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
 for(const c of actual.cases)for(const side of ['p1','p2'])for(const mon of c.setup[side])assert.equal(new Set(mon.moves.map(m=>ps.Dex.toID(m))).size,mon.moves.length,c.id+' unique moves');
 const c=id=>actual.cases.find(c=>c.id==='G4-custap-priority-'+id),p=(id,i=0)=>c(id).steps[i].after.sides[0].pokemon[0],first=(id,i=0)=>c(id).steps[i].events.p1.find(e=>e.message.startsWith('|move|')).message.split('|')[2];
 assert.equal(first('slow-active'),'p1a: Mew');assert.equal(first('above-threshold'),'p2a: Mew');assert.equal(first('above-threshold',1),'p1a: Mew');
 assert.equal(p('above-threshold').item,'custapberry');assert.equal(p('above-threshold',1).item,'');assert.equal(p('exact-quarter').item,'');assert.equal(p('above-exact-quarter').item,'custapberry');
 for(const id of ['slow-active','quickattack','protect','counter','recover','rest','struggle-slot4']){
  assert.equal(p(id).item,'');const ev=c(id).steps[0].events.p1.map(e=>e.message);const eat=ev.findIndex(e=>e.includes('|Custap Berry|[eat]')),activate=ev.findIndex(e=>e.includes('item: Custap Berry|[consumed]')),move=ev.findIndex(e=>e.startsWith('|move|'));assert.ok(eat>=0&&activate>eat&&move>activate,id);
 }
 assert.equal(first('counter'),'p2a: Mew');assert.equal(first('positive-target-priority'),'p2a: Mew');
 assert.equal(first('unburden-both'),'p1a: Mew');assert.equal(p('unburden-both').speed,240);
 assert.equal(p('recycle-unburden',1).item,'custapberry');assert.equal(p('recycle-unburden',1).speed,120);assert.deepEqual(p('recycle-unburden',2).volatiles.unburden,p('recycle-unburden').volatiles.unburden);
 assert.equal(p('switch-no-eat').item,'custapberry');assert.equal(p('suppressed-before-eat',1).item,'custapberry');assert.equal(p('suppressed-before-eat',2).item,'');assert.equal(p('suppressed-before-eat',2).volatiles.unburden,undefined);
 assert.equal(p('struggle-slot4').hp,0);assert.equal(p('struggle-slot4').volatiles.counter,undefined);
 for(const ability of ['pressure','unburden']){assert.equal(p('recharge-'+ability).item,'custapberry');assert.equal(p('recharge-'+ability,1).item,'');assert.ok(c('recharge-'+ability).steps[1].events.p1.some(e=>e.message==='|cant|p1a: Mew|recharge'));}
 assert.equal(c('fractional-counter-pivot').steps[0].after.queue.find(q=>q.choice==='move').priority,-4.9);
 assert.ok(p('legacy-counter-pivot').volatiles.counter.damage>0);
 assert.deepEqual(spec.membershipWitnesses().map(w=>w.set.species),['Golem','Drifblim']);
});
