'use strict';
// One-off adapter: long-lived uninstrumented pinned Showdown, fixed saved choices.
const fs = require('node:fs');
const readline = require('node:readline');
const {performance} = require('node:perf_hooks');
const {assertReady} = require('../reference/neutral-turn-oracle.cjs');
assertReady();
const ps = require('../reference/vendor/pokemon-showdown/dist/sim');
const cases = JSON.parse(fs.readFileSync(process.argv[2], 'utf8')).cases;
function fresh(copies) {
  return Array.from({length: copies}, () => cases.map(c => {
    const s = c.setup;
    if (s.setup) throw Error('Unexpected trusted mutation in this fixed pack');
    return new ps.Battle({formatid:'gen5randombattle', seed:s.seedWords ?? [s.seed,2,3,4],
      p1:{name:'p1',team:structuredClone(s.p1)},p2:{name:'p2',team:structuredClone(s.p2)}});
  }));
}
function simulate(packs) {
  for (const pack of packs) for (let i=0;i<cases.length;i++) {
    const battle=pack[i];
    for (const step of cases[i].steps) for (const [side,choice] of Object.entries(step.choices)) battle.choose(side,choice);
  }
}
function verify(packs) {
  for (const pack of packs) for (let i=0;i<cases.length;i++) {
    const b=pack[i], expected=cases[i].steps.at(-1).after;
    if (!b.ended || b.winner!==`p${expected.winner}` || b.prng.getSeed()!==expected.rng.join(',')) throw Error('Native terminal/RNG mismatch '+cases[i].id);
  }
}
readline.createInterface({input:process.stdin}).on('line', line => {
 try {
  const copies=JSON.parse(line).copies, initialization=[], samples=[];
  for(let repetition=-1;repetition<3;repetition++) {
    let t=performance.now();const packs=fresh(copies);initialization.push((performance.now()-t)/1000);
    t=performance.now();simulate(packs);const elapsed=(performance.now()-t)/1000;
    verify(packs); // All assertions, cleanup, IPC and I/O outside timing.
    for (const pack of packs) for (const b of pack) b.destroy();
    if(repetition<0) var warm=elapsed;else samples.push(elapsed);
  }
  console.log(JSON.stringify({backend:'showdown',copies,samples_seconds:samples,warmup_seconds:warm,initialization_seconds:initialization,node:process.version,threads:fs.readdirSync('/proc/self/task').length}));
 } catch(error) {console.log(JSON.stringify({error:error.stack}));}
});
