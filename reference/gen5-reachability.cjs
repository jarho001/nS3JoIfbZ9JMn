#!/usr/bin/env node
'use strict';
// Pinned *inventory*, not a claim that a string scan proves reachability.
// The conservative data universe and unresolved generator/callback edges are
// retained until independent witnesses or source proofs discharge them.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const {execFileSync} = require('node:child_process');
const {assertReady} = require('./neutral-turn-oracle.cjs');
const ROOT = path.resolve(__dirname, '..');
const UPSTREAM = path.join(__dirname, 'vendor/pokemon-showdown');
const PIN = require('../reference-lock.json').commit;
const CHAIN = ['gen5', 'gen6', 'gen7', 'gen8', 'base'];
const FILES = {Pokedex:'pokedex', Moves:'moves', Abilities:'abilities', Items:'items',
  Conditions:'conditions', Rulesets:'rulesets', TypeChart:'typechart', Natures:'natures', Scripts:'scripts'};
const hash = value => crypto.createHash('sha256').update(value).digest('hex');
const jsonHash = value => hash(JSON.stringify(value));
// Partial mappings record implemented branches; they do not discharge the
// reachability/interaction audit or upgrade an entire native callback to done.
const STATUS_OWNERS=new Set(['conditions:slp','conditions:frz','conditions:confusion',
  'conditions:brn','conditions:par','conditions:psn','conditions:tox','moves:rest',
  'moves:protect','abilities:synchronize','items:leftovers','items:lumberry','rules:sleepclausemod']);
const FIELD_OWNERS=new Set(['conditions:raindance','conditions:sunnyday','conditions:sandstorm',
  'conditions:hail','moves:reflect','moves:lightscreen','moves:stealthrock','moves:spikes']);
const HOOK_OWNERS=new Set([
  ...['pressure','intimidate','prankster','hugepower','purepower','adaptability','technician',
    'hustle','compoundeyes','superluck','sniper','guts','quickfeet','marvelscale','battlearmor',
    'shellarmor','thickfat','filter','solidrock','tintedlens','drizzle','drought','sandstream',
    'snowwarning','swiftswim','chlorophyll','sandrush','icebody'].map(id=>`abilities:${id}`),
  ...['choiceband','choicespecs','choicescarf','lifeorb','expertbelt','muscleband','wiseglasses',
    'silkscarf','mysticwater','scopelens','eviolite'].map(id=>`items:${id}`),
  'conditions:choicelock',
]);
const HIT_OWNERS=new Set(['skilllink','rockhead','roughskin','ironbarbs','liquidooze'].map(id=>`abilities:${id}`).concat(['items:rockyhelmet']));
const UNMAPPED_BRANCHES=new Set(['conditions:frz:onAfterMoveSecondary']);
const PHASE_OWNERS=new Set(['conditions:lockedmove','conditions:twoturnmove','conditions:flinch',...['solarbeam','skyattack','skullbash','razorwind','fly','dig','dive','bounce','taunt','encore','disable','torment'].map(id=>`moves:${id}`),'items:powerherb']);
const TRAP_OWNERS=new Set(['shadowtag','arenatrap','magnetpull','frisk'].map(id=>'abilities:'+id));
function partialBehavior(owner,field) {
  if(UNMAPPED_BRANCHES.has(`${owner}:${field}`))return null;
  const family=owner==='abilities:dryskin'?'dry_skin':['abilities:shielddust','abilities:serenegrace'].includes(owner)?'secondary_ability':owner==='abilities:flashfire'?'flash_fire':owner==='abilities:sapsipper'&&field==='onTryHit'?'sap_sipper':['abilities:stormdrain','abilities:lightningrod','abilities:motordrive'].includes(owner)&&field==='onTryHit'?'type_boost':['abilities:waterabsorb','abilities:voltabsorb'].includes(owner)?'absorption':['abilities:overgrow','abilities:blaze','abilities:torrent','abilities:swarm'].includes(owner)?'low_hp':owner==='abilities:magicguard'?'magic_guard':owner==='abilities:mummy'?'mummy':owner==='abilities:trace'?'trace':owner==='abilities:wonderguard'?'wonder_guard':owner==='moves:magiccoat'?'magic_coat':owner==='abilities:magicbounce'?'reflection':['moves:copycat','moves:mirrormove'].includes(owner)?'copy_move':owner==='moves:naturepower'?'nature_power':owner==='moves:sleeptalk'?'called_move':['moves:yawn','abilities:owntempo'].includes(owner)||['abilities:insomnia','abilities:vitalspirit'].includes(owner)&&field==='onTryAddVolatile'?'sleep_control':owner==='moves:gastroacid'&&['onTryHit','condition.onStart'].includes(field)?'suppression':['moves:worryseed','moves:simplebeam','abilities:simple','abilities:contrary'].includes(owner)?'ability_create':['abilities:immunity','abilities:limber','abilities:waterveil','abilities:magmaarmor','abilities:insomnia','abilities:vitalspirit'].includes(owner)&&['onUpdate','onSetStatus','onImmunity'].includes(field)?'status_immunity':['moves:roleplay','moves:skillswap','moves:entrainment'].includes(owner)?'ability_change':['moves:relicsong','abilities:zenmode'].includes(owner)?'trigger_form':['abilities:forecast','abilities:flowergift'].includes(owner)?'form':owner==='abilities:illusion'?'illusion':['moves:transform','abilities:imposter'].includes(owner)?'identity':['moves:trick','moves:switcheroo','moves:knockoff','moves:recycle','abilities:stickyhold'].includes(owner)?'item_lifecycle':owner==='moves:pursuit'?'pursuit':['abilities:suctioncups','abilities:soundproof'].includes(owner)?'phazing':owner==='conditions:partiallytrapped'?'partial_trap':owner==='moves:substitute'?'substitute':TRAP_OWNERS.has(owner)?'trapping':PHASE_OWNERS.has(owner)?'lock_phase':owner==='conditions:mustrecharge'?'recharge':owner==='conditions:frz'&&['onModifyMove','onDamagingHit'].includes(field)?'hit':STATUS_OWNERS.has(owner)?'status':FIELD_OWNERS.has(owner)?'field':HOOK_OWNERS.has(owner)?'hook':HIT_OWNERS.has(owner)?'hit':
    owner==='abilities:immunity'&&field==='onSetStatus'?'status':null;
  if(!family)return null;
  const weatherHook=(owner==='abilities:sandrush'&&field==='onImmunity')||
    (owner==='abilities:icebody'&&['onImmunity','onWeather'].includes(field));
  const fieldModifier=(['moves:reflect','moves:lightscreen'].includes(owner)&&field==='condition.onAnyModifyDamage')||
    (owner==='conditions:sandstorm'&&field==='onModifySpD');
  return {family,file:family==='dry_skin'?'simulator/status_immunity.py':family==='secondary_ability'?'simulator/status_effects.py':family==='flash_fire'?'simulator/flash_fire.py':family==='sap_sipper'?'simulator/status_immunity.py':family==='type_boost'?'simulator/status_immunity.py':family==='absorption'?'simulator/status_immunity.py':family==='low_hp'?'simulator/hook_effects.py':family==='magic_guard'?'simulator/shared_engine.py':family==='mummy'?'simulator/mummy.py':family==='trace'?'simulator/trace_ability.py':family==='wonder_guard'?'simulator/status_immunity.py':family==='magic_coat'?'simulator/called_moves.py':family==='reflection'?'simulator/called_moves.py':family==='copy_move'?'simulator/called_moves.py':family==='nature_power'?'simulator/called_moves.py':family==='called_move'?'simulator/called_moves.py':family==='sleep_control'?'simulator/sleep_control.py':family==='suppression'?'simulator/suppression.py':family==='ability_create'?'simulator/ability_creation.py':family==='status_immunity'?'simulator/status_immunity.py':family==='ability_change'?'simulator/ability_changes.py':family==='trigger_form'?'simulator/trigger_forms.py':family==='form'?'simulator/forms.py':family==='illusion'?'simulator/illusion.py':family==='identity'?'simulator/identity.py':family==='item_lifecycle'?'simulator/item_lifecycle.py':family==='pursuit'?'simulator/pursuit.py':family==='phazing'?'simulator/phazing.py':family==='partial_trap'?'simulator/partial_trap.py':family==='substitute'?'simulator/substitute.py':family==='trapping'?(owner==='abilities:frisk'?'simulator/hook_effects.py':'simulator/trapping.py'):family==='lock_phase'||family==='recharge'?'simulator/phase_effects.py':family==='hit'?(owner==='abilities:rockhead'||owner==='conditions:frz'&&field==='onDamagingHit'?'simulator/shared_engine.py':owner==='conditions:frz'?'simulator/status_effects.py':'simulator/hit_effects.py'):family==='status'?'simulator/status_effects.py':weatherHook||family==='field'&&!fieldModifier?'simulator/field_effects.py':'simulator/hook_effects.py',
    scheduler:'simulator/shared_engine.py',
    scope:'Implemented branches under explicit current capability masks only; complete callback/indirect/interaction validation pending'};
}
function familyEvidence(family) {
  return [{test:`test/test_${family}_engine.py`,
    native:`reference/${family==='dry_skin'?'dry-skin':family==='secondary_ability'?'secondary-ability':family==='flash_fire'?'flash-fire':family==='sap_sipper'?'sap-sipper':family==='type_boost'?'type-boost':family==='absorption'?'absorption':family==='low_hp'?'low-hp':family==='magic_guard'?'magic-guard':family==='mummy'?'mummy':family==='trace'?'trace':family==='wonder_guard'?'wonder-guard':family==='magic_coat'?'magic-coat':family==='reflection'?'reflection':family==='copy_move'?'copy-move':family==='nature_power'?'nature-power':family==='called_move'?'called-move':family==='sleep_control'?'sleep-control':family==='suppression'?'suppression':family==='ability_create'?'ability-create':family==='status_immunity'?'status-immunity':family==='ability_change'?'ability-change':family==='trigger_form'?'trigger-form':family==='form'?'form':family==='illusion'?'illusion':family==='identity'?'identity':family==='item_lifecycle'?'item-lifecycle':family==='pursuit'?'pursuit':family==='phazing'?'phazing':family==='partial_trap'?'partial-trap':family==='substitute'?'substitute':family==='trapping'?'trapping':family==='lock_phase'?'lock-phase':family==='recharge'?'recharge':family==='status'?'status-duration':family==='field'?'field-boost':family==='hit'?'hit-effect':'ability-item'}-cases.cjs`,
    fixture:`test-logs/g4-cpu-conformance/${family==='dry_skin'?'dry-skin':family==='secondary_ability'?'secondary-ability':family==='flash_fire'?'flash-fire':family==='sap_sipper'?'sap-sipper':family==='type_boost'?'type-boost':family==='absorption'?'absorption':family==='low_hp'?'low-hp':family==='magic_guard'?'magic-guard':family==='mummy'?'mummy':family==='trace'?'trace':family==='wonder_guard'?'wonder-guard':family==='magic_coat'?'magic-coat':family==='reflection'?'reflection':family==='copy_move'?'copy-moves':family==='nature_power'?'nature-power':family==='called_move'?'called-moves':family==='sleep_control'?'sleep-control':family==='suppression'?'suppression':family==='ability_create'?'ability-create':family==='status_immunity'?'status-immunity':family==='ability_change'?'ability-change':family==='trigger_form'?'trigger-forms':family==='form'?'forms':family==='illusion'?'illusion':family==='identity'?'identity':family==='item_lifecycle'?'items':family==='pursuit'?'pursuit':family==='phazing'?'phazing':family==='partial_trap'?'partial-trap':family==='substitute'?'substitute':family==='trapping'?'trapping':family==='lock_phase'?'locks':family==='recharge'?'recharge':family==='status'?'status-v5':family==='field'?'field':family==='hit'?'hits':'hooks'}.json.gz`,
    scope:family==='dry_skin'?'Independent32 synthetic plus2 retained historical weather scenarios/68 boundaries covering water/fire/weather; remaining interactions and full-format acceptance pending':family==='secondary_ability'?'Independent29 synthetic scenarios/33 boundaries, supported target status/flinch secondaries only; self-secondary and remaining full-format validation pending':family==='flash_fire'?'Independent29 synthetic scenarios/43 directed boundaries with saved continuation; full-format/capacity and remaining modifier dependencies pending':family==='sap_sipper'?'Independent17 synthetic scenarios/29 directed boundaries; Gen5 removes ally-side callback; full-format/remaining dependency validation pending':family==='type_boost'?'Independent54 synthetic scenarios/87 directed boundaries; TryHit only, redirect callback closure and complete-game/full-format validation pending':family==='absorption'?'Independent54 synthetic scenarios/71 directed request boundaries only; no complete-game or generator-membership validation; full-format acceptance pending':'Independent directed family cases and complete constructed games; not generator membership or full-format acceptance'}];
}
function canonical(value) {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === 'object') return Object.fromEntries(
    Object.keys(value).sort().map(key => [key, canonical(value[key])]));
  return value;
}
function buildInventory() {
  assertReady();
  const sources = {};
  function source(file) {
    if (!sources[file]) sources[file] = {sha256:hash(fs.readFileSync(path.join(UPSTREAM,file)))};
    return file;
  }
  // Snapshot function identity before Dex.loadData mutates/merges its modules.
  // A callback's owner is the original module, including nested conditions;
  // an overridden or null callback cannot accidentally count as inherited.
  const owners = new Map();
  function visit(value, file, field) {
    if (typeof value === 'function') {
      const list = owners.get(value) || [];
      list.push({file,field}); owners.set(value,list); return;
    }
    if (value && typeof value === 'object') for (const [key,child] of Object.entries(value)) {
      visit(child,file,field ? `${field}.${key}` : key);
    }
  }
  for (const mod of CHAIN) for (const [table,name] of Object.entries(FILES)) {
    const prefix = mod === 'base' ? 'data' : `data/mods/${mod}`;
    const file = `${prefix}/${name}.ts`;
    const built = path.join(UPSTREAM,'dist',prefix,`${name}.js`);
    if (!fs.existsSync(built)) continue;
    source(file); visit(require(built)[table],file,'');
  }
  const ps = require('./vendor/pokemon-showdown/dist/sim');
  const dex = ps.Dex.mod('gen5');
  const generator = ps.Teams.getGenerator('gen5randombattle',[1,2,3,4]);
  const setsFile = source('data/random-battles/gen5/sets.json');
  const sets = JSON.parse(fs.readFileSync(path.join(UPSTREAM,setsFile),'utf8'));
  const rootRows = [];
  for (const id of Object.keys(sets).sort()) for (const [roleIndex,set] of sets[id].sets.entries()) {
    const species = dex.species.get(id);
    assert.ok(species.exists,`Missing species root ${id}`);
    rootRows.push({id:`generator:${id}:${roleIndex}`,species:species.id,role:set.role,
      level:sets[id].level,moves:set.movepool.map(name => ({input:name,id:dex.moves.get(name).id})),
      abilities:set.abilities.map(name => dex.abilities.get(name).id),
      source:{file:setsFile,field:`${id}.sets.${roleIndex}`},
      reachability:'candidate',proof:null});
  }
  // Resolved method ownership is recorded separately from mechanics inheritance.
  const generatorMethods = {};
  const generatorOwners = new Map();
  for (const gen of [5,6,7,8,9]) {
    const file=source(`data/random-battles/gen${gen}/teams.ts`);
    const ctor=require(path.join(UPSTREAM,`dist/data/random-battles/gen${gen}/teams.js`)).default;
    generatorOwners.set(ctor.prototype,file);
  }
  for (let proto=Object.getPrototypeOf(generator);proto && proto!==Object.prototype;proto=Object.getPrototypeOf(proto)) {
    const file=generatorOwners.get(proto); assert.ok(file,'Unknown generator prototype');
    for (const name of Object.getOwnPropertyNames(proto).sort()) {
      if (name==='constructor' || generatorMethods[name] || typeof proto[name]!=='function') continue;
      generatorMethods[name]={file,sha256:hash(proto[name].toString()),
        status:'unaudited',calls:[...new Set([...proto[name].toString().matchAll(/this\.(\w+)\s*\(/g)].map(m=>m[1]))].sort(),
        callAnalysis:'syntactic candidate edges; dynamic calls and branch preconditions still require audit'};
    }
  }
  const callbacks = {};
  function serialize(value, owner, field='') {
    if (typeof value === 'function') {
      const id=`${owner}:${field}`;
      const definitions=owners.get(value) || [];
      // No guessed source ownership: fail the export for an unlocated function.
      assert.ok(definitions.length,`Unlocated effective callback ${id}`);
      const behavior=partialBehavior(owner,field);
      callbacks[id]={id,owner,field,sha256:hash(value.toString()),definitions,
        incoming:[{entry:owner,field,precondition:'owner is active and native callback event applies; transitive reachability audit pending'}],
        reachability:'candidate',implementation:behavior,evidence:behavior?familyEvidence(behavior.family):[],
        status:behavior?'partial':'unsupported'};
      return {$callback:id};
    }
    if (value === undefined) return undefined;
    if (typeof value==='number' && !Number.isFinite(value)) return {$number:String(value)};
    if (Array.isArray(value)) return value.map((v,i)=>serialize(v,owner,`${field}.${i}`));
    if (value && typeof value==='object') {
      return Object.fromEntries(Object.keys(value).sort().filter(k=>value[k]!==undefined &&
        !['desc','shortDesc','ruleTable'].includes(k)).map(k=>[k,serialize(value[k],owner,field?`${field}.${k}`:k)]));
    }
    return value;
  }
  const data = {schemaVersion:1,pin:PIN,format:'gen5randombattle',modInheritance:CHAIN,
    note:'Resolved immutable data, including a conservative candidate superset. Callback tokens are audit references, never executable fallbacks.',
    species:[],moves:[],abilities:[],items:[],conditions:[],types:[],natures:[],rules:[],scripts:null};
  const exclusions=[];
  for (const [kind,accessor] of [['species',dex.species],['moves',dex.moves],['abilities',dex.abilities],['items',dex.items]]) {
    const all=[...accessor.all()].sort((a,b)=>a.id.localeCompare(b.id,'en'));
    const seen=new Set();
    for (const entity of all) {
      if (seen.has(entity.id)) continue; seen.add(entity.id);
      const reason=!entity.exists?'does not exist':entity.gen>5?'introduced after generation 5':
        entity.isNonstandard && entity.isNonstandard!=='Past' ? `nonstandard: ${entity.isNonstandard}` : null;
      if (reason) {exclusions.push({id:`${kind}:${entity.id}`,reason,
        proof:'resolved gen5 Dex metadata; excluded from the initial legal data universe, not a proof against indirect effects'});continue;}
      data[kind].push(serialize(entity,`${kind}:${entity.id}`));
    }
  }
  for (const id of Object.keys(dex.data.Conditions).sort()) data.conditions.push({
    id,data:serialize(dex.data.Conditions[id],`conditions:${id}`)});
  for (const type of dex.types.all().filter(t=>t.exists && !t.isNonstandard).sort((a,b)=>a.name.localeCompare(b.name,'en'))) {
    data.types.push(serialize(type,`types:${type.id}`));
  }
  for (const nature of [...dex.natures.all()].sort((a,b)=>a.id.localeCompare(b.id,'en'))) {
    data.natures.push(serialize(nature,`natures:${nature.id}`));
  }
  const format=ps.Dex.formats.get('gen5randombattle');
  const rules=dex.formats.getRuleTable(format);
  for (const [id,from] of rules) data.rules.push({id,from,
    data:serialize(dex.formats.get(id),`rules:${id}`)});
  data.rules.sort((a,b)=>a.id.localeCompare(b.id,'en'));
  data.scripts=serialize(dex.data.Scripts,'scripts:gen5');
  for (const file of ['sim/battle.ts','sim/battle-actions.ts','sim/battle-queue.ts',
    'sim/field.ts','sim/pokemon.ts','sim/side.ts','sim/prng.ts','sim/dex.ts',
    'sim/dex-species.ts','sim/dex-moves.ts','sim/dex-items.ts','sim/dex-abilities.ts',
    'sim/dex-conditions.ts','sim/dex-formats.ts','sim/teams.ts']) source(file);
  const entries=[];
  const dataMappings={species:'BASE_STATS/SPECIES_TYPES; decode_sets/derive_set_stats',
    moves:'MOVE_TYPE/MOVE_POWER/MOVE_CATEGORY/MOVE_PRIORITY/MOVE_PP',
    abilities:'ABILITIES canonical IDs; battle callback coverage pending',
    items:'ITEMS canonical IDs; battle callback coverage pending',
    types:'TYPE_CHART/HP_TYPE_IDS; type_matchup/hidden_power',
    natures:'NATURE_MULTIPLIERS; derive_stored_stats'};
  const direct=new Set(rootRows.flatMap(row=>[
    `species:${row.species}`,...row.moves.map(m=>`moves:${m.id}`),...row.abilities.map(a=>`abilities:${a}`)]));
  for (const kind of ['species','moves','abilities','items','conditions','types','natures','rules']) {
    for (const entity of data[kind]) {
      const id=`${kind}:${entity.id}`;
      const behavior=partialBehavior(id,'');
      const implementation=dataMappings[kind]?{file:'simulator/gen5_tables.py',mapping:dataMappings[kind],
        scope:'immutable data/pre-entry values only; complete entity and interaction behavior pending'}:null;
      if(behavior){
        if(implementation)implementation.behavior=behavior;
      }
      entries.push({id,kind,reachability:'candidate',incoming:[{kind:direct.has(id)?'generator-root':'conservative-superset',
        precondition:direct.has(id)?'generator role/set branches pending audit':'indirect reachability pending closure'}],
        definition:{resolver:`Dex.mod('gen5').${kind}`,sha256:jsonHash(entity)},
        implementation:implementation||behavior,
        evidence:[...(dataMappings[kind]?[{test:'test/test_gen5_tables.py',native:'reference/gen5-data-cases.cjs',
          scope:'438 pre-entry sets, HP bitplanes and type chart; not every callback or generator branch',
          record:'test-logs/g4-cpu-conformance/data-cpu.log',source:'6857d91'}]:[]),...(behavior?familyEvidence(behavior.family):[])],
        status:dataMappings[kind]||behavior?'partial':'unsupported'});
    }
  }
  const ledger={schemaVersion:1,pin:PIN,format:'gen5randombattle',
    status:'incomplete',dataSha256:jsonHash(canonical(data)),sources,roots:rootRows,
    generatorMethods,entries,callbacks:Object.values(callbacks).sort((a,b)=>a.id.localeCompare(b.id,'en')),
    initialDataExclusions:exclusions,
    gaps:[
      {id:'generator-branches',status:'unresolved',detail:'Eligibility, role/move/ability filters, item return branches, forme/stat choices and inherited helper preconditions require source-backed audit.'},
      {id:'indirect-closure',status:'unresolved',detail:'Resolve called/copied effects, Transform/Imposter, changed abilities, item exchange/creation, forms and recursively referenced conditions to a fixed point. The exported superset does not prove closure.'},
      {id:'engine-contract',status:'unresolved',detail:'Scheduler, requests/retries, visibility, active rules/terminal paths and capacities require implementation/evidence entries.'},
    ]};
  return {data:canonical(data),ledger:canonical(require('./gen5-closure-audit.cjs').classify(ledger))};
}
function inventory() {
  // Dex mutates require-cached raw tables while inheriting. An isolated reader
  // preserves original definitions even when the caller already used Dex or
  // requests the inventory twice in one process.
  return JSON.parse(execFileSync(process.execPath,[__filename,'--worker'],{
    encoding:'utf8',maxBuffer:16*1024*1024}));
}
function main(args) {
  if(args.length===1 && args[0]==='--worker') {
    process.stdout.write(JSON.stringify(buildInventory()));return;
  }
  let output,dataOutput;
  for(let i=0;i<args.length;i++) {
    if(args[i]==='--output') output=args[++i];
    else if(args[i]==='--data-output') dataOutput=args[++i];
    else throw Error(`Unknown argument ${args[i]}`);
  }
  const result=buildInventory();
  if(dataOutput) fs.writeFileSync(dataOutput,JSON.stringify(result.data,null,2)+'\n',{flag:'wx'});
  if(output) fs.writeFileSync(output,JSON.stringify(result.ledger,null,2)+'\n',{flag:'wx'});
  else process.stdout.write(JSON.stringify(result.ledger,null,2)+'\n');
}
if(require.main===module) main(process.argv.slice(2));
module.exports={inventory,canonical,jsonHash};
