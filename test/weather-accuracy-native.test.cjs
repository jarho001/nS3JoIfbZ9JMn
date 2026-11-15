'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),zlib=require('node:zlib');
const {normalizeStacks}=require('./native-stack-support.cjs');
test('native-only Thunder and Blizzard weather accuracy preserves stage-query and hit RNG',()=>{
 require('../reference/neutral-turn-oracle.cjs').assertReady();
 const ps=require('../reference/vendor/pokemon-showdown/dist/sim'),{runCase}=require('../reference/shared-engine-cases.cjs'),spec=require('../reference/weather-accuracy-cases.cjs');
 const actual={pin:require('../reference-lock.json').commit,witnesses:spec.membershipWitnesses(),cases:spec().map(c=>runCase(ps,c))};
 assert.deepEqual(normalizeStacks(JSON.parse(JSON.stringify(actual))),normalizeStacks(JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/weather-accuracy-precheck.json.gz')))));
 assert.equal(actual.cases.length,51);assert.equal(actual.cases.reduce((n,c)=>n+c.steps.length,0),84);assert.ok(actual.cases.every(c=>c.steps.every(s=>s.accepted.every(a=>a.accepted))));
 for(const c of actual.cases)for(const side of ['p1','p2'])for(const mon of c.setup[side])assert.equal(new Set(mon.moves.map(m=>ps.Dex.toID(m))).size,mon.moves.length,c.id+' unique moves');
 const c=id=>actual.cases.find(c=>c.id==='G4-weather-accuracy-'+id),draws=(id,i=0)=>c(id).steps[i].draws,events=(id,i=0)=>c(id).steps[i].events.p1.map(e=>e.message);
 for(const [key,weather]of [['thunder','raindance'],['blizzard','hail']]){
  for(const [id,i]of [[key+'-'+weather,1],[key+'-'+weather+'-midturn',0],[key+'-weather-stages',1],[key+'-weather-unaware',1]]){assert.ok(!draws(id,i).some(d=>d.semanticContext==='accuracy'),id);assert.ok(draws(id,i).some(d=>d.semanticContext==='damage'),id);}
  assert.ok(draws(key+'-weather-victory-tie',1).some(d=>d.semanticContext==='modify-accuracy-sort'));assert.ok(!draws(key+'-weather-victory-tie',1).some(d=>d.semanticContext==='accuracy'));
  assert.ok(draws(key+'-weather-unaware',1).filter(d=>d.semanticContext==='modify-boost-sort').length>0);
  assert.ok(events(key+'-weather-sleep-talk',1).some(e=>e.includes('|'+(key==='thunder'?'Thunder':'Blizzard')+'|')&&e.includes('[from] move: Sleep Talk')));
  assert.ok(events(key+'-weather-protect',1).some(e=>e.includes('|Protect')));assert.ok(!draws(key+'-weather-protect',1).some(d=>d.semanticContext==='damage'));
  assert.equal(c(key+'-terminal').steps[0].after.terminal,true);
  assert.ok(!draws(key+'-dig').some(d=>d.semanticContext==='damage'));
 }
 assert.ok(draws('thunder-fly').some(d=>d.semanticContext==='damage'));assert.ok(!draws('blizzard-fly').some(d=>d.semanticContext==='damage'));
 assert.ok(!draws('thunder-immune').some(d=>d.semanticContext==='accuracy'));assert.ok(events('thunder-absorb').some(e=>e.includes('Volt Absorb')));
 assert.deepEqual(actual.witnesses.map(w=>w.set.species),['Kyogre','Abomasnow']);
});
