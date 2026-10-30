#!/usr/bin/env node
'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');
const ROOT = path.resolve(__dirname, '..');
const LOCK = require(path.join(ROOT, 'reference-lock.json'));
const FORMAT_ID = 'gen5randombattle';
const SEED = [101, 202, 303, 404];

function restore(object, key, descriptor) {
	if (descriptor) Object.defineProperty(object, key, descriptor);
	else delete object[key];
}

function createBattle(ps, { species = 'Pikachu', ability = 'Static', item = '' } = {}) {
	const makeSet = (name, pokemonAbility, heldItem, moves) => ({
		species: name, ability: pokemonAbility, item: heldItem, level: 50, moves, nature: 'Hardy',
	});
	const battle = new ps.Battle({
		formatid: FORMAT_ID, seed: [...SEED],
		p1: { name: 'composition-p1', team: [makeSet(species, ability, item, ['Tackle', 'Thunderbolt'])] },
		p2: { name: 'composition-p2', team: [makeSet('Bulbasaur', 'Overgrow', '', ['Tackle'])] },
	});
	assert.equal(battle.gen, 5);
	assert.equal(battle.started, true);
	return battle;
}

function capture(battle, config) {
	const source = battle.p1.pokemon[0];
	const target = battle.p2.pokemon[0];
	const battleKeys = ['randomizer', 'random', 'modify', 'runEvent', 'events'];
	const battleDescriptors = battleKeys.map(key => Object.getOwnPropertyDescriptor(battle, key));
	const effectivenessDescriptor = Object.getOwnPropertyDescriptor(target, 'runEffectiveness');
	const modifyDamageDescriptor = Object.getOwnPropertyDescriptor(battle.actions, 'modifyDamage');
	const priorityEventDescriptor = Object.getOwnPropertyDescriptor(battle, 'priorityEvent');
	const previousStatus = source.status;
	const move = config.typeless ? 50 : battle.dex.getActiveMove(config.move || 'Tackle');
	if (config.captureEffectiveCore && typeof move === 'number') {
		throw new TypeError('effective-core capture requires a real move object');
	}
	const moveKeys = typeof move === 'number' ? [] : ['forceSTAB', ...(config.captureEffectiveCore ? ['willCrit'] : [])];
	const moveDescriptors = moveKeys.map(key => Object.getOwnPropertyDescriptor(move, key));
	const inputDescriptors = config.captureEffectiveCore ? {
		level: Object.getOwnPropertyDescriptor(source, 'level'),
		attack: Object.getOwnPropertyDescriptor(source.storedStats, 'atk'),
		defense: Object.getOwnPropertyDescriptor(target.storedStats, 'def'),
		basePower: typeof move === 'number' ? undefined : Object.getOwnPropertyDescriptor(move, 'basePower'),
	} : undefined;
	const original = {
		randomizer: battle.randomizer, random: battle.random, modify: battle.modify,
		runEvent: battle.runEvent, runEffectiveness: target.runEffectiveness,
		priorityEvent: battle.priorityEvent, modifyDamage: battle.actions.modifyDamage,
	};
	const trace = {
		randomizerCalls: 0, damageDraws: 0, modifyStabCalls: 0, effectivenessCalls: 0,
		burnCalls: 0, finalCalls: 0, finalModifierCalls: 0,
		...(config.captureEffectiveCore ? { baseCoreCalls: 0, weatherEventCalls: 0 } : {}),
	};
	let inRandomizer = false;
	let pendingStab;
	try {
		if (config.captureEffectiveCore) {
			move.willCrit = false;
			if (config.level !== undefined) source.level = config.level;
			if (config.attack !== undefined) source.storedStats.atk = config.attack;
			if (config.defense !== undefined) target.storedStats.def = config.defense;
			if (config.basePower !== undefined) move.basePower = config.basePower;
			battle.actions.modifyDamage = function (baseDamage, ...args) {
				trace.baseCoreCalls++;
				assert.equal(trace.baseCoreCalls, 1, 'getDamage must reach the core exactly once');
				trace.baseCore = baseDamage;
				trace.coreMove = args[2];
				assert.equal(args[0], source);
				assert.equal(args[1], target);
				assert.equal(target.getMoveHitData(trace.coreMove).crit, false, 'controlled composition path excludes critical hits');
				assert.equal(Boolean(trace.coreMove.spreadHit), false, 'spread damage is outside this composition');
				assert.notEqual(trace.coreMove.multihitType, 'parentalbond', 'Parental Bond is outside this composition');
				return original.modifyDamage.apply(this, [baseDamage, ...args]);
			};
			battle.priorityEvent = function (event, ...args) {
				const result = original.priorityEvent.call(this, event, ...args);
				if (event === 'WeatherModifyDamage') {
					trace.weatherEventCalls++;
					assert.equal(trace.weatherEventCalls, 1, 'neutral modifyDamage must dispatch weather exactly once');
					assert.equal(args[0], source);
					assert.equal(args[1], target);
					trace.weatherInput = args[args.length - 1];
					trace.weatherOutput = result;
					trace.weatherHandlerCount = this.findEventHandlers(args[0], event, args[1]).length;
				}
				return result;
			};
		}
		if (config.status === 'brn') source.status = 'brn';
		if (config.ability) assert.equal(source.hasAbility(config.ability), true, 'battle ability is resolved during initialization');
		if (typeof move !== 'number' && config.forceSTAB !== undefined) move.forceSTAB = config.forceSTAB;
		if (config.finalModifier !== undefined) {
			assert.equal(battle.findEventHandlers(source, 'ModifyDamage', target).length, 0);
			battle.events = { ...battle.events };
			battle.onEvent('ModifyDamage', battle.format, function () {
				trace.syntheticModifierHandlerCalls++;
				this.chainModify(config.finalModifier, 4096);
			});
			trace.syntheticModifierHandlerCalls = 0;
		}
		battle.random = function (...args) {
			if (inRandomizer) {
				trace.damageDraws++;
				assert.equal(args[0], 16, 'the randomizer must draw exactly once from random(16)');
				trace.roll = config.roll === undefined ? original.random.apply(this, args) : config.roll;
				return trace.roll;
			}
			return original.random.apply(this, args);
		};
		battle.randomizer = function (input) {
			trace.randomizerCalls++;
			assert.equal(trace.randomizerCalls, 1, 'getDamage must invoke the pinned randomizer once');
			trace.preRollDamage = input;
			const controlledInput = config.preRollDamage === undefined ? input : config.preRollDamage;
			inRandomizer = true;
			try {
				trace.randomizerInput = controlledInput;
				trace.randomizerOutput = original.randomizer.call(this, controlledInput);
				if (config.failAfterRandomizer) throw new Error('injected composition capture failure');
				return trace.randomizerOutput;
			} finally {
				inRandomizer = false;
			}
		};
		battle.modify = function (input, numerator, denominator = 1) {
			if (pendingStab !== undefined) {
				trace.stab = { input, numerator, denominator, output: original.modify.call(this, input, numerator, denominator) };
				pendingStab = undefined;
				return trace.stab.output;
			}
			if (numerator === 0.5 && denominator === 1) {
				trace.burnCalls++;
				trace.afterType = input;
				trace.afterBurn = original.modify.call(this, input, numerator, denominator);
				return trace.afterBurn;
			}
			if (this.event && this.event.id === 'ModifyDamage') {
				trace.finalModifierCalls++;
				trace.finalModifierObserved = numerator * 4096 / denominator;
			}
			return original.modify.call(this, input, numerator, denominator);
		};
		battle.runEvent = function (event, ...args) {
			if (event === 'ModifySTAB') trace.modifyStabCalls++;
			if (event === 'ModifyDamage') {
				trace.finalCalls++;
				const [pokemon, defender, actualMove, value] = args;
				assert.equal(pokemon, source);
				assert.equal(defender, target);
				assert.equal(!!target.getMoveHitData(actualMove).bypassProtect, false, 'bypassProtect is outside this composition');
				trace.preFinal = value;
				trace.finalMoveType = actualMove.type;
				trace.finalMoveCategory = actualMove.category;
				trace.stabEligible = actualMove.type !== '???' && !!(
					actualMove.forceSTAB || source.hasType(actualMove.type) || source.getTypes(false, true).includes(actualMove.type)
				);
			}
			const result = original.runEvent.call(this, event, ...args);
			if (config.captureEffectiveCore && event === 'BasePower') trace.effectiveBasePower = result;
			if (config.captureEffectiveCore && ['ModifyAtk', 'ModifySpA'].includes(event)) trace.effectiveAttack = result;
			if (config.captureEffectiveCore && ['ModifyDef', 'ModifySpD'].includes(event)) trace.effectiveDefense = result;
			if (event === 'ModifySTAB') pendingStab = result;
			return result;
		};
		target.runEffectiveness = function (...args) {
			trace.effectivenessCalls++;
			if (config.typeExponent !== undefined) {
				trace.effectivenessInput = args[0].id;
				trace.typeExponent = config.typeExponent;
				return trace.typeExponent;
			}
			trace.typeExponent = original.runEffectiveness.apply(this, args);
			return trace.typeExponent;
		};

		trace.result = battle.actions.getDamage(source, target, move, true);
		if (config.captureEffectiveCore) {
			trace.effectiveLevel = source.level;
			assert.equal(trace.baseCoreCalls, 1);
			assert.equal(trace.weatherEventCalls, 1);
			assert.equal(trace.weatherInput, trace.baseCore + 2, 'WeatherModifyDamage receives core plus two');
			assert.equal(trace.randomizerInput, trace.weatherOutput, 'weather result feeds the original randomizer');
			assert.equal(trace.weatherHandlerCount, 0, 'neutral format has no WeatherModifyDamage handlers');
		}
		assert.equal(trace.randomizerCalls, 1);
		assert.equal(trace.damageDraws, 1, 'exactly one random(16) draw belongs to the randomizer');
		assert.equal(trace.effectivenessCalls, 1, 'getDamage must resolve move effectiveness exactly once');
		assert.equal(trace.finalCalls, 1, 'getDamage must use the real final ModifyDamage event');
		assert.equal(trace.finalModifierCalls, 1, 'the real ModifyDamage event must use one encoded modifier');
		assert.ok(Number.isInteger(trace.finalModifierObserved), 'the final encoded modifier is observed in ModifyDamage');
		if (config.finalModifier !== undefined) assert.equal(trace.finalModifierObserved, config.finalModifier);
		assert.equal(trace.modifyStabCalls, trace.finalMoveType === '???' ? 0 : 1);
		if (trace.finalMoveType !== '???') {
			const expectedStab = trace.stabEligible ? (source.hasAbility('adaptability') ? 2 : 1.5) : 1;
			assert.equal(trace.stab.numerator, expectedStab);
			assert.equal(trace.stab.denominator, 1);
		}
		if (config.finalModifier !== undefined) assert.equal(trace.syntheticModifierHandlerCalls, 1);
		trace.input = {
			preRollDamage: config.preRollDamage ?? trace.preRollDamage,
			roll: trace.roll,
			stabEligible: trace.stabEligible,
			adaptability: source.hasAbility('adaptability'),
			typeless: !!config.typeless,
			typeExponent: trace.typeExponent ?? 0,
			burned: source.status === 'brn',
			physical: trace.finalMoveCategory === 'Physical',
			hasGuts: source.hasAbility('guts'),
			finalModifier: trace.finalModifierObserved,
		};
		if (config.captureEffectiveCore) {
			trace.input.level = trace.effectiveLevel;
			trace.input.basePower = trace.effectiveBasePower;
			trace.input.effectiveAttack = trace.effectiveAttack;
			trace.input.effectiveDefense = trace.effectiveDefense;
		}
		if (config.forceSTAB !== undefined) assert.equal(trace.input.stabEligible, config.forceSTAB);
		if (config.ability) assert.equal(trace.input.adaptability, config.ability === 'Adaptability');
		if (config.status) assert.equal(trace.input.burned, config.status === 'brn');
		if (config.typeless) {
			assert.equal(trace.finalMoveType, '???');
			assert.equal(trace.modifyStabCalls, 0, 'typeless moves bypass the STAB event');
		}
		if (trace.stab) trace.afterStab = trace.stab.output;
		return trace;
	} finally {
		battleKeys.forEach((key, index) => restore(battle, key, battleDescriptors[index]));
		restore(target, 'runEffectiveness', effectivenessDescriptor);
		restore(battle.actions, 'modifyDamage', modifyDamageDescriptor);
		restore(battle, 'priorityEvent', priorityEventDescriptor);
		moveKeys.forEach((key, index) => restore(move, key, moveDescriptors[index]));
		if (inputDescriptors) {
			restore(source, 'level', inputDescriptors.level);
			restore(source.storedStats, 'atk', inputDescriptors.attack);
			restore(target.storedStats, 'def', inputDescriptors.defense);
			restore(move, 'basePower', inputDescriptors.basePower);
		}
		source.status = previousStatus;
	}
}

function ordinaryCases(ps) {
	const cases = [];
	for (const cfg of [
		{ label: 'ordinary-stab-life-orb-special-burned', species: 'Pikachu', ability: 'Static', item: 'Life Orb', move: 'Thunderbolt', status: 'brn', roll: 7 },
		{ label: 'ordinary-typed-nonstab', species: 'Pikachu', ability: 'Static', move: 'Tackle', roll: 12 },
		{ label: 'ordinary-burned-physical', species: 'Pikachu', ability: 'Static', move: 'Tackle', status: 'brn', roll: 5 },
		{ label: 'ordinary-burned-physical-guts', species: 'Pikachu', ability: 'Guts', move: 'Tackle', status: 'brn', roll: 9 },
		{ label: 'ordinary-adaptability-stab', species: 'Eevee', ability: 'Adaptability', move: 'Tackle', roll: 3 },
	]) {
		const battle = createBattle(ps, cfg);
		try {
			const result = capture(battle, cfg);
			result.label = cfg.label;
			result.contract = 'real move, type lookup, ability/status, and Life Orb event where configured; random(16) is controlled while the pinned randomizer remains active';
			cases.push(result);
		} finally { battle.destroy(); }
	}
	return cases;
}

function syntheticCases(ps) {
	const configs = Array.from({ length: 16 }, (_, roll) => ({
		label: `all-rolls-${roll}`, preRollDamage: 12345, roll, forceSTAB: true,
		typeExponent: 1, status: 'brn', finalModifier: 4096,
	}));
	configs.push(
		{ label: 'order-anchor-odd-low', preRollDamage: 1, roll: 0, forceSTAB: true, typeExponent: 2, status: 'brn', finalModifier: 6144 },
		{ label: 'zero-before-final-zero-modifier', preRollDamage: 1, roll: 15, typeExponent: -6, status: 'brn', finalModifier: 0 },
		{ label: 'type-exponent-low-clamp', preRollDamage: 33554431, roll: 15, forceSTAB: true, typeExponent: -2147483648, status: 'brn', finalModifier: 4096 },
		{ label: 'type-exponent-high-clamp', preRollDamage: 33554431, roll: 0, forceSTAB: true, typeExponent: 2147483647, status: 'brn', finalModifier: 4096 },
		{ label: 'bounded-preroll-and-all-max-amplifiers', preRollDamage: 33554431, roll: 0, forceSTAB: true,
			ability: 'Adaptability', typeExponent: 6, status: 'brn', finalModifier: 65535 },
		{ label: 'typeless-burned-physical-low', preRollDamage: 1, roll: 15, typeless: true, status: 'brn', typeExponent: 0, finalModifier: 4096 },
		{ label: 'typeless-uint16-zero-boundary', preRollDamage: 65536, roll: 0, typeless: true, typeExponent: 0, finalModifier: 4096 },
		{ label: 'typeless-max-preroll-and-type-exponent', preRollDamage: 33554431, roll: 0, typeless: true, typeExponent: 6, finalModifier: 4096 },
		{ label: 'zero-type-change-and-zero-final-modifier', preRollDamage: 0, roll: 0, typeless: true, typeExponent: 0, finalModifier: 0 },
	);
	return configs.map(config => {
		const battle = createBattle(ps, { species: config.ability === 'Adaptability' ? 'Eevee' : 'Pikachu', ability: config.ability || 'Static' });
		try {
			const result = capture(battle, config);
			result.label = config.label;
			result.contract = 'synthetic pre-roll replaces only original Battle.randomizer input; original randomizer and random(16) draw execute; effectiveness/final modifier event are explicitly controlled';
			return result;
		} finally { battle.destroy(); }
	});
}

function generateFixture(ps) {
	return {
		schemaVersion: 1,
		kind: 'gen5-bounded-post-core-damage-composition-oracle',
		upstream: LOCK,
		formatId: FORMAT_ID,
		generation: 5,
		battleSeed: [...SEED],
		method: 'BattleActions.getDamage with real Battle.randomizer, ModifySTAB, type effectiveness, burn, and ModifyDamage finalization',
		sourcePaths: ['sim/battle-actions.ts:1809-1897', 'sim/battle.ts:2337-2359', 'sim/dex.ts:364-369'],
		inputDomain: { preRollDamage: [0, 33554431], roll: [0, 15], typeExponent: [-2147483648, 2147483647], finalModifier: [0, 65535] },
		scope: 'Post-core, immediately before randomizer arithmetic, through final Gen 5 damage. No core, +2, weather, critical, spread, RNG generation, runtime lookup kernel, event engine, bypassProtect, or full damage calculator.',
		ordinaryContract: 'Initialized Gen 5 Pokemon and real moves with real type/ability/status resolution and original randomizer; random(16) damage draws are controlled for determinism. Includes real Life Orb ModifyDamage event.',
		syntheticContract: 'Only the input to original Battle.randomizer is replaced; the pinned randomizer and one controlled random(16) damage draw remain active. Effectiveness and one registered ModifyDamage modifier are intercepted explicitly. Temporary descriptors, event registry, and status restore in finally.',
		ordinaryCases: ordinaryCases(ps),
		syntheticCases: syntheticCases(ps),
	};
}

function main(args = process.argv.slice(2)) {
	if (args.length > 1 || (args.length && args[0] !== 'fixture')) throw new Error('Usage: node reference/damage-composition-oracle.cjs [fixture]');
	const oracle = require('./oracle.cjs');
	oracle.assertReady();
	const ps = require('./vendor/pokemon-showdown/dist/sim');
	process.stdout.write(`${JSON.stringify(generateFixture(ps), null, 2)}\n`);
}

if (require.main === module) {
	main();
}

module.exports = { capture, createBattle, generateFixture, syntheticCases };
