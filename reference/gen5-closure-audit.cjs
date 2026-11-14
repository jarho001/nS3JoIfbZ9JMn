'use strict';
// Source-backed bound investigation, not a replacement for the full ledger.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const {assertReady}=require('./neutral-turn-oracle.cjs');
function audit(ledger) {
assertReady();const ps=require('./vendor/pokemon-showdown/dist/sim');
const dex=ps.Dex.mod('gen5'),gen=ps.Teams.getGenerator('gen5randombattle',[1,2,3,4]);
const sha=s=>crypto.createHash('sha256').update(s).digest('hex');
const moves=[...new Set(ledger.roots.flatMap(r=>r.moves.map(m=>m.id)))].sort();
const abilities=[...new Set(ledger.roots.flatMap(r=>r.abilities))].sort();
const acorn=require('./vendor/pokemon-showdown/node_modules/acorn');
const itemMethods=['getPriorityItem','getItem'];
const literalItems=new Set(),methodEvidence={};
const requiredItems=[...new Set(ledger.roots.flatMap(root=>dex.species.get(root.species).requiredItems??[]))].sort();
function methodAst(name){const text=gen[name].toString();return {text,body:acorn.parse('({'+text+'})',{ecmaVersion:'latest'}).body[0].expression.properties[0].value.body};}
function walk(node,visit){if(!node||typeof node!=='object')return;visit(node);for(const value of Object.values(node))if(Array.isArray(value))value.forEach(n=>walk(n,visit));else if(value&&typeof value==='object')walk(value,visit);}
for(const name of itemMethods){
 const {text,body}=methodAst(name),returns=[];
 function bound(node){
  if(node?.type==='ConditionalExpression')return [...bound(node.consequent),...bound(node.alternate)];
  if(node?.type==='Literal'&&typeof node.value==='string')return [node.value];
  if(node?.type==='CallExpression'&&text.slice(node.start-2,node.end-2)==='this.sample(species.requiredItems)')return requiredItems;
  if(!node||node.type==='Identifier'&&node.name==='undefined')return [];
  throw Error('Unaudited item return expression: '+text.slice(node.start-2,node.end-2));
 }
 walk(body,node=>{if(node.type==='ReturnStatement'){
  const names=bound(node.argument);for(const value of names){const item=dex.items.get(value);if(!item.exists)throw Error('Unknown item '+value);literalItems.add(item.id);}
  returns.push({expression:node.argument?text.slice(node.argument.start-2,node.argument.end-2):'undefined',itemIds:names.map(n=>dex.items.get(n).id).sort()});
 }});
 methodEvidence[name]={sha256:sha(text),owner:ledger.generatorMethods[name],status:'all_return_expressions_bounded',returns};
}
{
 const {text,body}=methodAst('randomSet'),assignments=[];
 walk(body,node=>{if(node.type==='AssignmentExpression'&&node.left.type==='Identifier'&&node.left.name==='item'){
  const value=text.slice(node.right.start-2,node.right.end-2);
  if(node.right.type==='CallExpression'&&node.right.callee.type==='MemberExpression'&&['getPriorityItem','getItem'].includes(node.right.callee.property.name))assignments.push({expression:value,edge:node.right.callee.property.name});
  else if(node.right.type==='Literal'&&node.right.value==='Black Sludge'){literalItems.add('blacksludge');assignments.push({expression:value,edge:'leftovers_to_blacksludge_for_poison'});}
  else throw Error('Unaudited randomSet item assignment '+value);
 }});
 if(assignments.length!==3)throw Error('Expected all three pinned item assignments');
 methodEvidence.randomSet={sha256:sha(text),owner:ledger.generatorMethods.randomSet,status:'all_item_assignments_bounded',assignments};
}
const speciesForms=new Set();
for(const root of ledger.roots){
 const s=dex.species.get(root.species);speciesForms.add(s.id);
 for(const item of s.requiredItems??[])literalItems.add(dex.items.get(item).id);
 for(const forme of s.cosmeticFormes??[])speciesForms.add(dex.species.get(forme).id);
 if(typeof s.battleOnly==='string')speciesForms.add(dex.species.get(s.battleOnly).id);
 if(s.baseSpecies==='Basculin')speciesForms.add(dex.species.get('Basculin-Blue-Striped').id);
 if(s.baseSpecies==='Keldeo')speciesForms.add(dex.species.get('Keldeo-Resolute').id);
}
const calls={};
for(const id of moves){
 const move=dex.moves.get(id);const entries=[];
 function visit(v,key=''){
  if(typeof v==='function'){
   const text=v.toString();
   if(/\b(?:useMove|runMove|transformInto|setAbility|setItem|formeChange)\s*\(/.test(text))entries.push({field:key,sha256:sha(text),body:text});
  }else if(v&&typeof v==='object')for(const[k,x]of Object.entries(v))if(!['ruleTable'].includes(k))visit(x,key?key+'.'+k:k);
 }
 // Resolved move objects contain immutable effect/data values, not battle objects.
 visit(move);if(entries.length)calls[id]=entries;
}
const abilityEdges={};
for(const id of abilities){
 const ability=dex.abilities.get(id),entries=[];
 for(const [field,value] of Object.entries(ability))if(typeof value==='function'){
  const text=value.toString();
  if(/\b(?:useMove|runMove|transformInto|setAbility|setItem|formeChange)\s*\(|\.item\s*=/.test(text))entries.push({field,sha256:sha(text),body:text});
 }
 if(entries.length)abilityEdges[id]=entries;
}
// Conservative dependency fixed point. Dynamic arguments remain explicit gaps;
// callback/reference discovery is not itself an unreachability proof.
const conditionMethods=new Set(['addVolatile','addSideCondition','addSlotCondition',
 'addPseudoWeather','setWeather','setTerrain','setStatus','trySetStatus']);
const conditionIds=new Set(['brn','par','psn','tox','slp','frz','confusion','strugglerecoil']);
const conditionEdges=[],dynamicConditionEdges=[],seenOwners=new Set(),queue=[];
function strings(node){
 if(!node)return [];
 if(node.type==='Literal'&&typeof node.value==='string')return [node.value];
 if(node.type==='ArrayExpression')return node.elements.flatMap(strings);
 if(node.type==='ConditionalExpression')return [...strings(node.consequent),...strings(node.alternate)];
 if(node.type==='CallExpression'&&node.callee.type==='MemberExpression'&&['sample','sampleIfArray'].includes(node.callee.property.name))return strings(node.arguments[0]);
 return [];
}
function inspectConditionRefs(value,owner,field=''){
 if(typeof value==='function'){
  const text=value.toString();let tree;
  try{tree=acorn.parse('({'+text+'})',{ecmaVersion:'latest'});}catch{tree=acorn.parse('('+text+')',{ecmaVersion:'latest'});}
  walk(tree,node=>{
   if(node.type!=='CallExpression'||node.callee.type!=='MemberExpression'||!conditionMethods.has(node.callee.property.name))return;
   const method=node.callee.property.name,ids=strings(node.arguments[0]);
   if(ids.length)for(const id of ids){conditionIds.add(ps.toID(id));conditionEdges.push({owner,field,method,condition:id,callbackSha256:sha(text)});}
   else dynamicConditionEdges.push({owner,field,method,callbackSha256:sha(text),expression:node.arguments[0]?.type??'missing'});
  });return;
 }
 if(value&&typeof value==='object')for(const[key,child]of Object.entries(value)){
  if(['volatileStatus','sideCondition','slotCondition','weather','terrain','pseudoWeather','status'].includes(key)&&typeof child==='string'&&child){conditionIds.add(ps.toID(child));conditionEdges.push({owner,field:field?field+'.'+key:key,condition:child,method:'declarative'});}
  if(key!=='ruleTable')inspectConditionRefs(child,owner,field?field+'.'+key:key);
 }
}
for(const id of moves)queue.push(['moves:'+id,dex.moves.get(id)]);
for(const id of abilities)queue.push(['abilities:'+id,dex.abilities.get(id)]);
for(const id of literalItems)queue.push(['items:'+id,dex.items.get(id)]);
for(const id of speciesForms)if(dex.data.Conditions[id])conditionIds.add(id);
for(const row of require('../rules/gen5-data-v1.json').rules)queue.push(['rules:'+row.id,dex.formats.get(row.id)]);
for(let cursor=0;cursor<queue.length;cursor++){
 const [owner,value]=queue[cursor];if(seenOwners.has(owner))continue;seenOwners.add(owner);
 inspectConditionRefs(value,owner);
 for(const id of conditionIds)if(!seenOwners.has('conditions:'+id)&&!queue.some(([key])=>key==='conditions:'+id))queue.push(['conditions:'+id,dex.conditions.get(id)]);
}
const dynamicProofs={
 'moves:triattack:secondary.onHit':{sha256:'6c68bee2aa4fd5023f3868d24061684126b36c3d7abcbe379e5389e8d28d682c',ids:['brn','par','frz'],reason:'Local status is sampled from the literal three-entry array before trySetStatus.'},
 'moves:triattack:secondaries.0.onHit':{sha256:'6c68bee2aa4fd5023f3868d24061684126b36c3d7abcbe379e5389e8d28d682c',ids:['brn','par','frz'],reason:'Alias of the same resolved secondary callback.'},
 'abilities:synchronize:onAfterSetStatus':{sha256:'8756bb4c6ee180c5fdfb4d0c80afd3d3f51ff64896bc48a800f737289fa7e0ec',ids:['brn','par','psn','tox'],reason:'Copies the applied status; sleep/freeze exit before the call. No new status identity.'},
 'conditions:twoturnmove:onStart':{sha256:'88b7dea7a80770b682af442a9b26c1f3d30d59e75c5d7dba710886680ddb96b2',ids:moves.filter(id=>dex.moves.get(id).flags.charge),reason:'Native hit pipeline installs this condition with its active charge move. Closed root charge flag bound is Solar Beam; called/copied move closure remains separately pending.'},
};
for(const edge of dynamicConditionEdges){
 const proof=dynamicProofs[edge.owner+':'+edge.field];
 if(proof&&proof.sha256===edge.callbackSha256){edge.resolution=proof;for(const id of proof.ids)conditionIds.add(id);}
}
// Inspect conditions newly bounded by the reviewed dynamic arguments too.
let priorOwners=-1;
while(priorOwners!==seenOwners.size){
 priorOwners=seenOwners.size;
 for(const id of conditionIds){const owner='conditions:'+id;if(!seenOwners.has(owner)){seenOwners.add(owner);inspectConditionRefs(dex.conditions.get(id),owner);}}
 for(const edge of dynamicConditionEdges)if(!edge.resolution){const proof=dynamicProofs[edge.owner+':'+edge.field];if(proof&&proof.sha256===edge.callbackSha256){edge.resolution=proof;for(const id of proof.ids)conditionIds.add(id);}}
}
const conditionClosure={status:'provisional_dynamic_edges_and_engine_paths_pending',
 ids:[...conditionIds].sort(),edges:conditionEdges,dynamicEdges:dynamicConditionEdges,unresolvedDynamicEdges:dynamicConditionEdges.filter(e=>!e.resolution),
 note:'Static methods/declarative fields are a conservative discovery fixed point, not complete dependency acceptance. Audit computed arguments, state-installed species/rule hooks and all engine entry paths before exclusions.'};
// Architecture audit: distinguish executed ability roots from engine queries.
// Querying a species' alternate trapping ability does not grant that ability.
const queryAbilities=[];
for(const id of speciesForms){
 const species=dex.species.get(id);
 for(const name of Object.values(species.abilities??{})){
  const ability=dex.abilities.get(name);
  if(typeof ability.onFoeMaybeTrapPokemon==='function')queryAbilities.push({species:id,ability:ability.id,
   callbackSha256:sha(ability.onFoeMaybeTrapPokemon.toString()),actualRoot:abilities.includes(ability.id)});
 }
}
const mutationMethods=new Set(['useMove','runMove','transformInto','setAbility','setItem','takeItem',
 'formeChange','setSpecies','setType','addType','clearVolatile','addVolatile','addSideCondition','addSlotCondition']);
const mutationSites=[];
function frontier(value,owner,field=''){
 if(typeof value==='function'){
  const text=value.toString();let tree;
  try{tree=acorn.parse('({'+text+'})',{ecmaVersion:'latest'});}catch{tree=acorn.parse('('+text+')',{ecmaVersion:'latest'});}
  walk(tree,node=>{
   if(node.type!=='CallExpression'||node.callee.type!=='MemberExpression')return;
   const name=node.callee.property.name;if(!mutationMethods.has(name))return;
   mutationSites.push({owner,field,method:name,callbackSha256:sha(text),
    literalArguments:strings(node.arguments[0]),argumentKind:node.arguments[0]?.type??'missing'});
  });return;
 }
 if(value&&typeof value==='object')for(const [key,child]of Object.entries(value))if(key!=='ruleTable')frontier(child,owner,field?field+'.'+key:key);
}
for(const id of moves)frontier(dex.moves.get(id),'moves:'+id);
for(const id of abilities)frontier(dex.abilities.get(id),'abilities:'+id);
for(const id of literalItems)frontier(dex.items.get(id),'items:'+id);
for(const id of conditionIds)frontier(dex.conditions.get(id),'conditions:'+id);
for(const row of require('../rules/gen5-data-v1.json').rules)frontier(dex.formats.get(row.id),'rules:'+row.id);
const engineFiles=['sim/battle.ts','sim/battle-actions.ts','sim/pokemon.ts','sim/side.ts','sim/field.ts'];
const designFrontier={status:'source_frontier_not_complete_closure_proof',
 engineSources:Object.fromEntries(engineFiles.map(file=>[file,sha(fs.readFileSync(path.join(__dirname,'vendor/pokemon-showdown',file)))])),
 implicitMoveRoots:['struggle'],queryAbilities,mutationSites,
 sleepTalkEligibility:Object.fromEntries(moves.map(id=>[id,!dex.moves.get(id).flags.nosleeptalk&&!dex.moves.get(id).flags.charge])),
 absentInitialMoveNames:['metronome','assist','sketch','copycat','mimic','mirrormove','mefirst','instruct'].filter(id=>!moves.includes(id)),
 absentInitialAbilityNames:['dancer'].filter(id=>!abilities.includes(id)),
 note:'Initial absence is checked, not promoted to transitive exclusion. Resolve every mutation site, helper and engine/rule edge before closing the ledger. Query-only abilities add callback obligations, not transferable ability identities.'};
const output={status:'provisional_source_bound_not_full_reachability_acceptance',pin:require('../reference-lock.json').commit,
 designFrontier,conditionClosure,rootMoveIds:moves,rootAbilityIds:abilities,initialFormBound:[...speciesForms].sort(),initialItemReturnBound:[...literalItems].sort(),methodEvidence,
 indirectMoveCallbacks:calls,indirectAbilityCallbacks:abilityEdges,
 findings:['230 set-file spellings resolve to223 move IDs. The only direct calling/copy roots among the candidate pools are Nature Power, Sleep Talk and Transform; broad Metronome/Assist/Sketch/Copycat expansion is not justified by these roots.',
 'Pinned Gen5 Nature Power calls Earthquake. Sleep Talk selects from the user set; Transform/Imposter copy opponent moves. Pursuit and Magic Bounce reuse existing move IDs. These observations still require a complete source/engine/ability/item closure audit before exclusions enter the accepted ledger.',
 'All pinned getPriorityItem/getItem return expressions and all randomSet item assignments are structurally bounded by the recorded expressions and species.requiredItems. This initial item upper bound is not proof of generator eligibility or transitive closure.'],
 remaining:['Generator branch/source audit and initial eligibility','All127 ability callback bodies and item transfer/creation edges','Forms from Forecast/Flower Gift/Relic Song/Multitype and Transform','Engine/rule injection and complete dependency fixed point']};
return output;
}
if(require.main===module){
 const output=audit(require('../rules/gen5-reachable-v1.json'));
 if(!process.argv[2])throw Error('Pass a fresh output path');
 fs.writeFileSync(process.argv[2],JSON.stringify(output,null,2)+'\n',{flag:'wx'});
 console.log(JSON.stringify({moveIds:output.rootMoveIds.length,abilityIds:output.rootAbilityIds.length,initialItemReturnBound:output.initialItemReturnBound}));
}
module.exports={audit};

// Identity bounds are narrower than behavioral closure. Keep conditions, forms,
// alternate-ability queries and individual callback activation open explicitly.
function classify(ledger) {
 const sha=s=>crypto.createHash('sha256').update(s).digest('hex');
 const output=audit(ledger);
 const ps=require('./vendor/pokemon-showdown/dist/sim'),dex=ps.Dex.mod('gen5');
 const assert=require('node:assert/strict');
 const moveBound=new Set([...output.rootMoveIds,'struggle']);
 const itemBound=new Set(output.initialItemReturnBound);
 for(const root of ledger.roots){
  const required=dex.species.get(root.species).requiredMove;
  if(required)assert.ok(moveBound.has(dex.moves.get(required).id),'new required move');
 }
 // Inspect the larger legal ability universe as a safety bound even though
 // alternate identity eligibility remains open. Extra identity mutation sites
 // are Pickup/Pickpocket (existing items) and Zen Mode (forms), not new moves.
 const supplementalAbilityCallbacks=[];
 for(const ability of dex.abilities.all()){
  if(ability.gen>5||ability.isNonstandard||output.rootAbilityIds.includes(ability.id))continue;
  for(const [field,value] of Object.entries(ability))if(typeof value==='function'&&
   /useMove|runMove|setItem|setAbility|formeChange|transformInto|\.item\s*=|moveSlots\s*=/.test(value.toString())){
   assert.ok(['pickup','pickpocket','zenmode'].includes(ability.id),'Unreviewed alternate ability mutation: '+ability.id);
   supplementalAbilityCallbacks.push({owner:'abilities:'+ability.id,field,sha256:sha(value.toString()),body:value.toString()});
  }
 }
 const seed=[1,2,3,4];
 const team=ps.Teams.getGenerator('gen5randombattle',seed).getTeam();
 const required=new Set(team.flatMap(s=>[
  'species:'+dex.species.get(s.species).id,'abilities:'+dex.abilities.get(s.ability).id,
  'items:'+dex.items.get(s.item).id,...s.moves.map(m=>'moves:'+dex.moves.get(m).id)]));
 const refs={
  generator:['data/random-battles/gen5/teams.ts:103-494 (cullMovePool/randomMoveset)',
   'data/random-battles/gen5/teams.ts:496-566 (ability filters/fallback)',
   'data/random-battles/gen5/teams.ts:568-816 (item returns and randomSet)',
   'data/random-battles/gen5/teams.ts:850-1014 (team eligibility)',
   'data/random-battles/gen9/teams.ts:675-693 (addMove)',
   'data/random-battles/gen9/teams.ts:1442-1497 (getLevel/getForme)',
   'data/random-battles/gen9/teams.ts:1641-1683 (getPokemonPool)'],
  engine:['sim/pokemon.ts:1291-1395 (Transform copies slots/ability, not item)',
   'sim/pokemon.ts:1448-1510 (only permanent forme changes replace ability)',
   'sim/pokemon.ts:1800-1983 (consume/take/set item and ability)',
   'sim/pokemon.ts:1125 (Struggle request)',
   'sim/battle-actions.ts:267-408 (OverrideAction, Z/Max and Dancer guards)'],
 };
 const proofs={
  'move-identity-bound':{
   conclusion:'proven_unreachable',kind:'moves',bound:[...moveBound].sort(),references:[...refs.generator,...refs.engine],
   reasoning:'Initial moves come only from role pools plus requiredMove (checked within the bound). Sleep Talk chooses current slots; Transform/Imposter copy existing opponent slots. Gen5 Nature Power calls Earthquake, already in the bound; Pursuit and Magic Bounce reuse the current ID. Encore/locked requests reuse slots; Struggle is the engine exception. No reachable callback introduces Metronome/Assist/Sketch/Mimic/Copycat/Mirror Move/Me First. Z/Max/Mega and post-Gen5 ability injections lack their generation/item/species prerequisites. By induction these paths cannot introduce another move ID; synthetic trusted installs are outside this proof.'},
  'item-identity-bound':{
   conclusion:'proven_unreachable',kind:'items',bound:[...itemBound].sort(),references:[...refs.generator,...refs.engine],
   reasoning:'All getPriorityItem/getItem return expressions (including requiredItems) and all randomSet item assignments are structurally bounded. Black Sludge conversion is included. Trick/Switcheroo transfer existing items, with failure restoration; consumption/Knock Off remove them; Harvest restores lastItem only if a berry. Transform copies no item. No initial/called move adds Recycle/Bestow/Thief/Covet/Fling or an item-creation path; no reachable ability creates a new item ID. Thus item identity remains inside this upper bound. This does not prove each of the 45 branches feasible.'},
 };
 const lowHpWitnesses=require('./low-hp-cases.cjs').membershipWitnesses();
 const lowHpAbilities=new Set(lowHpWitnesses.map(w=>'abilities:'+w.ability));
 for(const id of lowHpAbilities)required.add(id);
 proofs['low-hp-ability-positive-witness']={conclusion:'required',kind:'abilities',bound:[...lowHpAbilities].sort(),references:refs.generator,witnesses:lowHpWitnesses,reasoning:'Unmodified pinned generator outputs positively witness these four abilities only. Other set identities and callback activation remain independently unresolved; absence supplies no exclusions.'};
 const absorptionWitnesses=require('./absorption-cases.cjs').membershipWitnesses();
 const absorptionAbilities=new Set(absorptionWitnesses.map(w=>'abilities:'+w.ability));
 for(const id of absorptionAbilities)required.add(id);
 proofs['type-absorption-positive-witness']={conclusion:'required',kind:'abilities',bound:[...absorptionAbilities].sort(),references:refs.generator,witnesses:absorptionWitnesses,reasoning:'Unmodified pinned generator outputs positively witness Water Absorb and Volt Absorb memberships only; callback activation and other set identities remain independently unresolved.'};
 const typeBoostWitnesses=require('./type-boost-cases.cjs').membershipWitnesses();
 const typeBoostAbilities=new Set(typeBoostWitnesses.map(w=>'abilities:'+w.ability));
 for(const id of typeBoostAbilities)required.add(id);
 proofs['type-boost-positive-witness']={conclusion:'required',kind:'abilities',bound:[...typeBoostAbilities].sort(),references:refs.generator,witnesses:typeBoostWitnesses,reasoning:'Unmodified pinned generator outputs witness Storm Drain, Lightning Rod and Motor Drive memberships only; redirect/activation branches and other identities remain unresolved.'};
 const sapSipperWitnesses=require('./sap-sipper-cases.cjs').membershipWitnesses();
 const sapSipperAbilities=new Set(sapSipperWitnesses.map(w=>'abilities:'+w.ability));
 for(const id of sapSipperAbilities)required.add(id);
 proofs['sap-sipper-positive-witness']={conclusion:'required',kind:'abilities',bound:[...sapSipperAbilities].sort(),references:refs.generator,witnesses:sapSipperWitnesses,reasoning:'Pinned generator seed[10,2,3,4] positively witnesses Sap Sipper Girafarig; no other exclusion/activation claim. Gen5 abilities override removes onAllyTryHitSide.'};
 const flashFireWitnesses=require('./flash-fire-cases.cjs').membershipWitnesses();
 const flashFireAbilities=new Set(flashFireWitnesses.map(w=>'abilities:'+w.ability));
 for(const id of flashFireAbilities)required.add(id);
 proofs['flash-fire-positive-witness']={conclusion:'required',kind:'abilities',bound:[...flashFireAbilities].sort(),references:refs.generator,witnesses:flashFireWitnesses,reasoning:'Pinned generator seed[6,2,3,4] positively witnesses Flash Fire Chandelure; no complete callback activation or unrelated membership claim.'};
 const secondaryAbilityWitnesses=require('./secondary-ability-cases.cjs').membershipWitnesses();
 const secondaryAbilities=new Set(secondaryAbilityWitnesses.map(w=>'abilities:'+w.ability));
 for(const id of secondaryAbilities)required.add(id);
 proofs['secondary-ability-positive-witness']={conclusion:'required',kind:'abilities',bound:[...secondaryAbilities].sort(),references:refs.generator,witnesses:secondaryAbilityWitnesses,reasoning:'Pinned generator outputs witness Shield Dust Dustox and Serene Grace Dunsparce only; supported target secondaries are directed cases, not complete effect/dependency closure.'};
 const drySkinWitnesses=require('./dry-skin-cases.cjs').membershipWitnesses();
 const drySkinAbilities=new Set(drySkinWitnesses.map(w=>'abilities:'+w.ability));
 for(const id of drySkinAbilities)required.add(id);
 proofs['dry-skin-positive-witness']={conclusion:'required',kind:'abilities',bound:[...drySkinAbilities].sort(),references:refs.generator,witnesses:drySkinWitnesses,reasoning:'Pinned generator seed[23,2,3,4] positively witnesses Dry Skin Jynx only; water/fire/weather callbacks have directed synthetic evidence, not full-format closure.'};
 const recoveryWitnesses=require('./switch-recovery-cases.cjs').membershipWitnesses();
 const recoveryAbilities=new Set(recoveryWitnesses.map(w=>'abilities:'+w.ability));
 for(const id of recoveryAbilities)required.add(id);
 proofs['switch-recovery-positive-witness']={conclusion:'required',kind:'abilities',bound:[...recoveryAbilities].sort(),references:refs.generator,witnesses:recoveryWitnesses,reasoning:'Pinned generator seeds[3,2,3,4] Slowking Regenerator and[24,2,3,4] Celebi Natural Cure positively witness required abilities. Natural Cure onCheckShow returns for active.length1: only singles no-op branch applies; remaining full-format closure pending.'};
 const rainRecoveryWitnesses=require('./rain-recovery-cases.cjs').membershipWitnesses();
 const rainRecoveryAbilities=new Set(rainRecoveryWitnesses.map(w=>'abilities:'+w.ability));
 for(const id of rainRecoveryAbilities)required.add(id);
 proofs['rain-recovery-positive-witness']={conclusion:'required',kind:'abilities',bound:[...rainRecoveryAbilities].sort(),references:refs.generator,witnesses:rainRecoveryWitnesses,reasoning:'Pinned generator seeds[59,2,3,4] Pelipper Rain Dish and[1,2,3,4] Swanna Hydration prove membership. Directed synthetic weather/residual cases cover implemented branches, not complete dependency/format closure.'};
 const residualRecoveryWitnesses=require('./residual-recovery-cases.cjs').membershipWitnesses();
 const residualRecoveryAbilities=new Set(residualRecoveryWitnesses.map(w=>'abilities:'+w.ability));
 for(const id of residualRecoveryAbilities)required.add(id);
 proofs['residual-recovery-positive-witness']={conclusion:'required',kind:'abilities',bound:[...residualRecoveryAbilities].sort(),references:refs.generator,witnesses:residualRecoveryWitnesses,reasoning:'Pinned generator seeds[4,2,3,4] Arbok Shed Skin and[38,2,3,4] Breloom Poison Heal prove membership; directed status/residual evidence is not complete callback/format closure.'};
 for(const entry of ledger.entries){
  entry.reachability='unresolved';
  entry.reachabilityAudit='unresolved-membership';
  const key=entry.kind==='moves'?'move-identity-bound':entry.kind==='items'?'item-identity-bound':null;
  if(key&&!proofs[key].bound.includes(entry.id.split(':')[1])){
   entry.reachability='proven_unreachable';entry.reachabilityAudit=key;
  }
  if(required.has(entry.id)){
   assert.notEqual(entry.reachability,'proven_unreachable',entry.id);
   entry.reachability='required';entry.reachabilityAudit=residualRecoveryAbilities.has(entry.id)?'residual-recovery-positive-witness':rainRecoveryAbilities.has(entry.id)?'rain-recovery-positive-witness':recoveryAbilities.has(entry.id)?'switch-recovery-positive-witness':drySkinAbilities.has(entry.id)?'dry-skin-positive-witness':secondaryAbilities.has(entry.id)?'secondary-ability-positive-witness':flashFireAbilities.has(entry.id)?'flash-fire-positive-witness':sapSipperAbilities.has(entry.id)?'sap-sipper-positive-witness':typeBoostAbilities.has(entry.id)?'type-boost-positive-witness':lowHpAbilities.has(entry.id)?'low-hp-ability-positive-witness':absorptionAbilities.has(entry.id)?'type-absorption-positive-witness':'native-team-positive-witness';
  }
 }
 for(const root of ledger.roots){
  const witness=team.find(s=>s.speciesId===root.species&&s.role===root.role&&s.moves.every(m=>root.moves.some(x=>x.id===dex.moves.get(m).id)));
  root.reachability=witness?'required':'unresolved';
  root.proof=witness?'native-team-positive-witness':null;
 }
 const entries=new Map(ledger.entries.map(e=>[e.id,e]));
 for(const cb of ledger.callbacks){
  const owner=entries.get(cb.owner);
  cb.reachability=owner?.reachability==='proven_unreachable'?'proven_unreachable':'unresolved';
  cb.reachabilityAudit=cb.reachability==='proven_unreachable'?owner.reachabilityAudit:'unresolved-callback-activation';
 }
 function counts(rows){return Object.fromEntries(['required','proven_unreachable','unresolved'].map(k=>[k,rows.filter(r=>r.reachability===k).length]));}
 ledger.reachabilityAudit={date:'2026-11-13',status:'partial_identity_bounds_behavioral_closure_open',
  classificationNotes:{'unresolved-membership':'Candidate membership is insufficient; no positive branch witness or complete exclusion proof.', 'unresolved-callback-activation':'Required owner membership does not prove each callback trigger or nested branch.'},
  proofs,positiveWitness:{id:'native-team-positive-witness',seed,format:ledger.format,team,
   reason:'Unmodified pinned generator output is positive membership evidence only; absence supplies no exclusions.'},
  references:refs,supplementalAbilityCallbacks,initialFormBound:output.initialFormBound,
  indirectMoveCallbacks:output.indirectMoveCallbacks,indirectAbilityCallbacks:output.indirectAbilityCallbacks,
  itemMethodEvidence:output.methodEvidence,
  summary:{entries:counts(ledger.entries),callbacks:counts(ledger.callbacks),roots:counts(ledger.roots)},
  unresolved:[
   'Feasibility of every role/move/ability/item branch, including teamDetails and ordered item precedence; no upper bound is promoted to required.',
   'Species/form exclusions: Forecast, Flower Gift, Relic Song, freeze Shaymin-Sky reversion, Multitype/plate initialization and Transform targets; permanent reversion can reset ability.',
   'Abilities outside initial pools remain unresolved, including engine species-based FoeMaybeTrapPokemon queries; queries do not grant transferable abilities.',
   'Condition/rule/engine callback activation and all behavioral branches, RNG/request/event/visibility/terminal/capacity obligations remain unresolved.'
  ]};
 return ledger;
}
module.exports.classify=classify;
