#!/usr/bin/env node
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const ROOT = path.resolve(__dirname, '..');
const LOCK = JSON.parse(fs.readFileSync(path.join(ROOT, 'reference-lock.json'), 'utf8'));
const FORMAT_ID = 'gen5randombattle';
const BATTLE_SEED = [101, 202, 303, 404];
const ROLLS = Array.from({ length: 16 }, (_, roll) => roll);
const SPECIES = ['Pikachu', 'Bulbasaur', 'Shuckle', 'Blissey', 'Wobbuffet', 'Chansey', 'Deoxys-Attack', 'Shedinja'];
const LEVELS = [1, 2, 4, 5, 17, 50, 99, 100];
const BASE_POWERS = [1, 2, 50, 255, 65535, 16, 100, 300];
const SYNTHETIC_INPUTS = [
	{ label: 'zero-attack', level: 50, basePower: 50, attack: 0, defense: 1 },
	{ label: 'zero-defense-positive-numerator', level: 50, basePower: 50, attack: 1, defense: 0 },
	{ label: 'zero-defense-zero-numerator', level: 50, basePower: 50, attack: 0, defense: 0 },
	{ label: 'minimum-floor', level: 1, basePower: 1, attack: 1, defense: 1 },
	{ label: 'ordinary-synthetic', level: 50, basePower: 50, attack: 255, defense: 255 },
	{ label: 'zero-core-before-plus-two', level: 50, basePower: 1, attack: 1, defense: 65535 },
	{ label: 'postroll-zero-before-gen5-minimum-one', level: 2, basePower: 441, attack: 59443, defense: 1 },
	{ label: 'core-product-wrap-below', level: 100, basePower: 1559, attack: 65535, defense: 1 },
	{ label: 'core-product-wrap-edge', level: 100, basePower: 1560, attack: 65535, defense: 1 },
	{ label: 'core-product-wrap-above', level: 100, basePower: 1561, attack: 65535, defense: 1 },
	{ label: 'post-event-large-roll-input', level: 100, basePower: 1560, attack: 65535, defense: 1 },
	{ label: 'high-power-high-defense', level: 100, basePower: 65535, attack: 65535, defense: 65535 },
	{ label: 'unit-defense-high-power', level: 17, basePower: 65535, attack: 255, defense: 1 },
];
const SOURCE_PATHS = [
	'sim/battle-actions.ts:1769-1786',
	'sim/battle-actions.ts:1800',
	'sim/battle-actions.ts:1809-1812',
	'sim/battle.ts:934-939',
	'sim/battle.ts:2348-2359',
	'sim/battle.ts:2407-2410',
	'sim/dex.ts:364-369',
];
const STAT_IDS = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];

function makeSet(dex, speciesName, level, pattern) {
	const species = dex.species.get(speciesName);
	assert.ok(species.exists, `Unknown pinned species ${speciesName}`);
	const ability = species.abilities['0'] || Object.values(species.abilities)[0];
	const ivValues = [0, 3, 4, 30, 31];
	const evValues = [0, 3, 4, 252, 255];
	const ivs = {};
	const evs = {};
	for (const [index, stat] of STAT_IDS.entries()) {
		ivs[stat] = ivValues[(pattern + index) % ivValues.length];
		evs[stat] = evValues[(pattern * 2 + index) % evValues.length];
	}
	return { species: species.name, ability, moves: ['Tackle'], nature: 'Hardy', level, ivs, evs };
}

function createBattle(ps, dex, p1Team, p2Team, suffix) {
	const battle = new ps.Battle({
		formatid: FORMAT_ID,
		seed: [...BATTLE_SEED],
		p1: { name: `neutral-${suffix}`, team: p1Team },
		p2: { name: `target-${suffix}`, team: p2Team },
	});
	assert.equal(battle.gen, 5);
	assert.equal(battle.started, true);
	assert.equal(battle.ruleTable.has('overflowstatmod'), false);
	return battle;
}

function findCore(coreFixture, input) {
	const rows = [...coreFixture.ordinaryCases, ...coreFixture.syntheticCases];
	const result = rows.find(row => row.level === input.level && row.basePower === input.basePower && row.attack === input.attack && row.defense === input.defense);
	return result?.baseDamage;
}

function restore(object, key, ownDescriptor) {
	if (ownDescriptor) Object.defineProperty(object, key, ownDescriptor);
	else delete object[key];
}

function invokeComposition(battle, source, target, basePower, expectedCore, roll) {
	const actions = battle.actions;
	const actionsDescriptor = Object.getOwnPropertyDescriptor(actions, 'modifyDamage');
	const randomizerDescriptor = Object.getOwnPropertyDescriptor(battle, 'randomizer');
	const randomDescriptor = Object.getOwnPropertyDescriptor(battle, 'random');
	const priorityDescriptor = Object.getOwnPropertyDescriptor(battle, 'priorityEvent');
	const originalModifyDamage = actions.modifyDamage;
	const originalRandomizer = battle.randomizer;
	const originalPriorityEvent = battle.priorityEvent;
	let modifyDamageCalls = 0;
	let randomizerCalls = 0;
	let randomCalls = 0;
	let modifyDamageInput;
	let randomizerInput;
	let randomizerOutput;
	let capturedMove;
	let weatherEvent;
	try {
		actions.modifyDamage = function (baseDamage, actualSource, actualTarget, move, suppressMessages) {
			modifyDamageCalls++;
			assert.equal(modifyDamageCalls, 1, 'getDamage must call modifyDamage exactly once');
			if (expectedCore !== undefined) assert.equal(baseDamage, expectedCore, 'modifyDamage must receive the real core');
			assert.equal(actualSource, source);
			assert.equal(actualTarget, target);
			assert.equal(suppressMessages, true);
			modifyDamageInput = baseDamage;
			capturedMove = move;
			return originalModifyDamage.call(this, baseDamage, actualSource, actualTarget, move, suppressMessages);
		};
		battle.randomizer = function (baseDamage) {
			randomizerCalls++;
			assert.equal(randomizerCalls, 1, 'neutral getDamage must call randomizer once');
			randomizerInput = baseDamage;
			randomizerOutput = originalRandomizer.call(this, baseDamage);
			return randomizerOutput;
		};
		battle.random = bound => {
			randomCalls++;
			assert.equal(bound, 16, 'Battle.randomizer must request random(16)');
			return roll;
		};
		battle.priorityEvent = function (event, ...args) {
			const handlerCount = event === 'WeatherModifyDamage' ? this.findEventHandlers(args[0], event, args[1]).length : 0;
			const result = originalPriorityEvent.call(this, event, ...args);
			if (event === 'WeatherModifyDamage') weatherEvent = { input: args[args.length - 1], output: result, handlerCount };
			return result;
		};
		const finalDamage = actions.getDamage(source, target, basePower, true);
		assert.equal(modifyDamageCalls, 1);
		assert.equal(randomizerCalls, 1);
		assert.equal(randomCalls, 1);
		assert.equal(weatherEvent.input, modifyDamageInput + 2, 'original modifyDamage must add two before WeatherModifyDamage');
		assert.equal(weatherEvent.handlerCount, 0, 'neutral format must have no WeatherModifyDamage handlers');
		assert.equal(weatherEvent.output, battle.modify(weatherEvent.input, 1), 'neutral event must retain pinned Battle.modify identity arithmetic');
		assert.equal(randomizerInput, weatherEvent.output, 'original modifyDamage must pass neutral weather output to randomizer');
		return {
			roll,
			modifyDamageInput,
			weatherInput: weatherEvent.input,
			weatherOutput: weatherEvent.output,
			weatherHandlerCount: weatherEvent.handlerCount,
			randomizerInput,
			randomizerOutput,
			finalDamage,
			randomCalls,
			move: capturedMove,
		};
	} finally {
		restore(actions, 'modifyDamage', actionsDescriptor);
		restore(battle, 'randomizer', randomizerDescriptor);
		restore(battle, 'random', randomDescriptor);
		restore(battle, 'priorityEvent', priorityDescriptor);
	}
}

function assertNeutral(battle, source, target, move, basePower, attack, defense) {
	assert.equal(source.boosts.atk, 0);
	assert.equal(target.boosts.def, 0);
	assert.equal(source.status, '');
	assert.equal(battle.field.weather, '');
	assert.deepEqual(Object.keys(battle.field.pseudoWeather).sort(), ['sleepclausemod']);
	assert.equal(move.type, '???');
	assert.equal(move.category, 'Physical');
	assert.equal(move.willCrit, false);
	assert.equal(Boolean(move.spreadHit), false);
	assert.notEqual(move.multihitType, 'parentalbond');
	assert.equal(battle.runEvent('BasePower', source, target, move, basePower, true), basePower);
	const calculatedAttack = source.calculateStat('atk', 0, 1, source);
	const calculatedDefense = target.calculateStat('def', 0, 1, target);
	assert.equal(battle.runEvent('ModifyAtk', source, target, move, calculatedAttack), attack);
	assert.equal(battle.runEvent('ModifyDef', target, source, move, calculatedDefense), defense);
	assert.equal(calculatedAttack, attack);
	assert.equal(calculatedDefense, defense);
}

function captureCases(battle, sources, targets, descriptors, coreFixture, kind) {
	return descriptors.map((descriptor, index) => {
		const source = sources[index];
		const target = targets[index];
		if (kind === 'synthetic') {
			source.level = descriptor.level;
			source.storedStats.atk = descriptor.attack;
			target.storedStats.def = descriptor.defense;
		}
		const input = {
			level: source.level,
			basePower: descriptor.basePower,
			attack: source.calculateStat('atk', 0, 1, source),
			defense: target.calculateStat('def', 0, 1, target),
		};
		assert.equal(input.level, descriptor.level);
		if (kind === 'synthetic') {
			assert.equal(input.attack, descriptor.attack);
			assert.equal(input.defense, descriptor.defense);
		}
		let baseCore = findCore(coreFixture, input);
		const outcomes = ROLLS.map(roll => {
			const captured = invokeComposition(battle, source, target, descriptor.basePower, baseCore, roll);
			if (baseCore === undefined) baseCore = captured.modifyDamageInput;
			assert.equal(captured.modifyDamageInput, baseCore, 'core must not depend on controlled damage roll');
			assertNeutral(battle, source, target, captured.move, descriptor.basePower, input.attack, input.defense);
			return {
				roll,
				modifyDamageInput: captured.modifyDamageInput,
				weatherInput: captured.weatherInput,
				weatherOutput: captured.weatherOutput,
				weatherHandlerCount: captured.weatherHandlerCount,
				randomizerInput: captured.randomizerInput,
				randomizerOutput: captured.randomizerOutput,
				finalDamage: captured.finalDamage,
			};
		});
		return {
			label: descriptor.label || `${descriptor.sourceSpecies}-vs-${descriptor.targetSpecies}`,
			level: input.level,
			basePower: descriptor.basePower,
			attack: input.attack,
			defense: input.defense,
			baseCore,
			outcomes,
		};
	});
}

function ordinaryCases(ps, dex, coreFixture) {
	const cases = [];
	for (let offset = 0; offset < LEVELS.length; offset += 4) {
		const descriptors = LEVELS.slice(offset, offset + 4).map((level, local) => ({
			level,
			species: SPECIES[offset + local],
			basePower: BASE_POWERS[offset + local],
			label: `${SPECIES[offset + local]}-level-${level}`,
		}));
		const p1 = descriptors.map((row, local) => makeSet(dex, row.species, row.level, offset + local));
		const p2 = descriptors.map((row, local) => makeSet(dex, SPECIES[(offset + local + 3) % SPECIES.length], row.level, offset + local + 2));
		const battle = createBattle(ps, dex, p1, p2, `ordinary-${offset}`);
		try {
			const inputs = descriptors.map((row, local) => ({
				...row,
				sourceSpecies: row.species,
				targetSpecies: SPECIES[(offset + local + 3) % SPECIES.length],
				basePower: row.basePower,
			}));
			cases.push(...captureCases(battle, battle.p1.pokemon, battle.p2.pokemon, inputs, coreFixture, 'ordinary'));
		} finally {
			battle.destroy();
		}
	}
	return cases;
}

function syntheticCases(ps, dex, coreFixture) {
	const p1 = [makeSet(dex, 'Pikachu', 50, 0)];
	const p2 = [makeSet(dex, 'Bulbasaur', 50, 1)];
	const battle = createBattle(ps, dex, p1, p2, 'synthetic');
	try {
		const sources = SYNTHETIC_INPUTS.map(() => battle.p1.pokemon[0]);
		const targets = SYNTHETIC_INPUTS.map(() => battle.p2.pokemon[0]);
		return captureCases(battle, sources, targets, SYNTHETIC_INPUTS, coreFixture, 'synthetic');
	} finally {
		battle.destroy();
	}
}

function fixedpointIdentityBoundaries(ps, dex) {
	const battle = createBattle(ps, dex, [makeSet(dex, 'Pikachu', 50, 0)], [makeSet(dex, 'Bulbasaur', 50, 1)], 'fixedpoint-boundary');
	try {
		return [1048575, 1048576, 1048577].map(input => ({ input, output: battle.modify(input, 1) }));
	} finally {
		battle.destroy();
	}
}

function generateFixture(ps) {
	const format = ps.Dex.forFormat(ps.Dex.formats.get(FORMAT_ID));
	assert.equal(format.gen, 5);
	const coreFixture = require('./base-damage-oracle.cjs').generateFixture(ps);
	const cases = [...ordinaryCases(ps, format, coreFixture), ...syntheticCases(ps, format, coreFixture)];
	return {
		schemaVersion: 1,
		kind: 'gen5-randombattle-neutral-damage-after-roll-oracle',
		upstream: { repository: LOCK.repository, commit: LOCK.commit },
		formatId: FORMAT_ID,
		generation: format.gen,
		method: 'BattleActions.getDamage -> BattleActions.modifyDamage -> Battle.randomizer',
		sourcePaths: SOURCE_PATHS,
		battleSeed: [...BATTLE_SEED],
		inputDomain: { level: [1, 100], basePower: [1, 65535], attack: [0, 65535], defense: [0, 65535], roll: [0, 15], result: 'int32' },
		neutralWeatherIdentity: 'WeatherModifyDamage calls Battle.runEvent, which always applies Battle.modify(value, 1) to integer relay values; uint32 truncation of value*4096 means this neutral step masks to 20 bits before randomizer.',
		boundary: 'randomizerOutput is the target intermediate; finalDamage is recorded only to distinguish it from later Gen 5 minimum-one and final uint16 behavior.',
		control: 'modifyDamage and randomizer wrappers delegate to their original implementations; only Battle.random is controlled to supply each roll and must be called once with bound 16.',
		neutralConditions: 'gen5randombattle; no active weather or WeatherModifyDamage handlers, critical hit, spread, Parental Bond, or status; BasePower, ModifyAtk, and ModifyDef are identity events; numeric move has type ??? and does not receive STAB.',
		fixedpointIdentityBoundaries: fixedpointIdentityBoundaries(ps, format),
		ordinaryCases: cases.slice(0, LEVELS.length),
		syntheticCases: cases.slice(LEVELS.length),
	};
}

function stringifyFixture(fixture) {
	const rows = new Set(['ordinaryCases', 'syntheticCases']);
	const lines = ['{'];
	for (const [index, [key, value]] of Object.entries(fixture).entries()) {
		const last = index === Object.keys(fixture).length - 1;
		if (rows.has(key)) {
			lines.push(`  ${JSON.stringify(key)}: [`);
			for (const [rowIndex, row] of value.entries()) lines.push(`    ${JSON.stringify(row)}${rowIndex === value.length - 1 ? '' : ','}`);
			lines.push(`  ]${last ? '' : ','}`);
		} else {
			const rendered = JSON.stringify(value, null, 2).split('\n');
			lines.push(`  ${JSON.stringify(key)}: ${rendered[0]}`);
			for (const line of rendered.slice(1)) lines.push(`  ${line}`);
			if (!last) lines[lines.length - 1] += ',';
		}
	}
	lines.push('}');
	return `${lines.join('\n')}\n`;
}

function main(args = process.argv.slice(2)) {
	if (args.length > 1 || (args.length && args[0] !== 'fixture')) throw new Error('Usage: node reference/neutral-damage-oracle.cjs [fixture]');
	const oracle = require('./oracle.cjs');
	oracle.assertReady();
	const ps = require('./vendor/pokemon-showdown/dist/sim');
	process.stdout.write(stringifyFixture(generateFixture(ps)));
}

if (require.main === module) {
	try {
		main();
	} catch (error) {
		console.error(`Neutral-damage oracle failed: ${error.message}`);
		process.exitCode = 1;
	}
}

module.exports = { FORMAT_ID, ROLLS, generateFixture, main };
