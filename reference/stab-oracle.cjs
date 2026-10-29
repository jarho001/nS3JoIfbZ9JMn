#!/usr/bin/env node
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const ROOT = path.resolve(__dirname, '..');
const LOCK = JSON.parse(fs.readFileSync(path.join(ROOT, 'reference-lock.json'), 'utf8'));
const FORMAT_ID = 'gen5randombattle';
const SEED = [101, 202, 303, 404];
const SOURCE_PATHS = [
	'sim/battle-actions.ts:1809-1847',
	'data/abilities.ts:43-54',
	'sim/battle.ts:2337-2359',
	'sim/dex.ts:364-369',
];
const MODIFIERS = { typedNonStab: 4096, stab: 6144, adaptability: 8192 };
const WRAP_ANCHORS = [...new Set(Object.values(MODIFIERS).flatMap(modifier => {
	const boundary = Math.floor(2 ** 32 / modifier);
	return [boundary - 1, boundary, boundary + 1];
}))].sort((a, b) => a - b);
const SYNTHETIC_DAMAGE = [...new Set([0, 1, 2, 3, ...WRAP_ANCHORS, 2147483647])].sort((a, b) => a - b);

function createBattle(ps, ability = 'Static') {
	const team = (species, pokemonAbility) => [{
		species, ability: pokemonAbility, item: '', level: 50, moves: ['Tackle', 'Thunderbolt'], nature: 'Hardy',
	}];
	const battle = new ps.Battle({
		formatid: FORMAT_ID,
		seed: [...SEED],
		p1: { name: 'stab-p1', team: team('Pikachu', ability) },
		p2: { name: 'stab-p2', team: team('Bulbasaur', 'Overgrow') },
	});
	assert.equal(battle.gen, 5);
	assert.equal(battle.started, true);
	return battle;
}

function restore(object, key, descriptor) {
	if (descriptor) Object.defineProperty(object, key, descriptor);
	else delete object[key];
}

function activeMove(battle, config) {
	if (config.typeless) return 50;
	return battle.dex.getActiveMove(config.move || 'Tackle');
}

function capture(battle, config) {
	const source = battle.p1.pokemon[0];
	const target = battle.p2.pokemon[0];
	const randomizerDescriptor = Object.getOwnPropertyDescriptor(battle, 'randomizer');
	const modifyDescriptor = Object.getOwnPropertyDescriptor(battle, 'modify');
	const runEventDescriptor = Object.getOwnPropertyDescriptor(battle, 'runEvent');
	const effectivenessDescriptor = Object.getOwnPropertyDescriptor(target, 'runEffectiveness');
	const originalRandomizer = battle.randomizer;
	const originalModify = battle.modify;
	const originalRunEvent = battle.runEvent;
	const originalRunEffectiveness = target.runEffectiveness;
	const move = activeMove(battle, config);
	const moveDescriptors = typeof move === 'number' ? null : ['type', 'forceSTAB'].map(key => [key, Object.getOwnPropertyDescriptor(move, key)]);
	let randomizerCalls = 0;
	let randomizerOutput;
	let modifyStabCalls = 0;
	let preTypeModifyCalls = 0;
	let stabModifierCalls = 0;
	let typeEffectivenessStarted = false;
	let pendingStab;
	let stabModification;
	try {
		if (moveDescriptors) {
			if (config.type) move.type = config.type;
			if (config.forceSTAB) move.forceSTAB = true;
		}
		battle.randomizer = function (input) {
			randomizerCalls++;
			assert.equal(randomizerCalls, 1);
			if (config.damage === undefined) {
				randomizerOutput = originalRandomizer.call(this, input);
			} else {
				assert.ok(Number.isInteger(config.damage) && config.damage >= 0 && config.damage <= 2147483647,
					'synthetic damage must be an integer in [0, 2147483647]');
				randomizerOutput = config.damage;
			}
			return randomizerOutput;
		};
		battle.modify = function (input, numerator, denominator = 1) {
			if (randomizerCalls && !typeEffectivenessStarted) preTypeModifyCalls++;
			if (pendingStab !== undefined) {
				stabModifierCalls++;
				const output = originalModify.call(this, input, numerator, denominator);
				stabModification = { input, numerator, denominator, output, eventResult: pendingStab };
				pendingStab = undefined;
				return output;
			}
			return originalModify.call(this, input, numerator, denominator);
		};
		battle.runEvent = function (event, ...args) {
			if (event === 'ModifySTAB') modifyStabCalls++;
			const result = originalRunEvent.call(this, event, ...args);
			if (event === 'ModifySTAB') pendingStab = result;
			return result;
		};
		target.runEffectiveness = function (...args) {
			typeEffectivenessStarted = true;
			return originalRunEffectiveness.apply(this, args);
		};
		const result = battle.actions.getDamage(source, target, move, true);
		assert.equal(randomizerCalls, 1);
		const type = typeof move === 'number' ? '???' : move.type;
		const shouldModify = type !== '???';
		assert.equal(modifyStabCalls, shouldModify ? 1 : 0, `${config.label}: ModifySTAB event count`);
		assert.equal(stabModifierCalls, shouldModify ? 1 : 0, `${config.label}: STAB modifying call count`);
		assert.equal(!!stabModification, shouldModify, `${config.label}: STAB modifying call capture`);
		if (!shouldModify) assert.equal(preTypeModifyCalls, 0, `${config.label}: typeless must bypass all modifiers before type effectiveness`);
		if (shouldModify) {
			const expectedNumerator = config.stabEligible ? (config.adaptability ? 2 : 1.5) : 1;
			const observed = stabModification;
			assert.equal(observed.input, randomizerOutput, `${config.label}: STAB input must equal randomizer output`);
			assert.equal(observed.numerator, expectedNumerator, `${config.label}: pinned STAB event result`);
			assert.equal(observed.denominator, 1);
			assert.equal(observed.eventResult, expectedNumerator);
		}
		return {
			label: config.label,
			damage: randomizerOutput,
			move: typeof move === 'number' ? null : move.id,
			type,
			stabEligible: !!config.stabEligible,
			adaptability: !!config.adaptability,
			forceSTAB: !!config.forceSTAB,
			typeless: !!config.typeless,
			modifier: stabModification ? stabModification.numerator * 4096 / stabModification.denominator : null,
			modifyStabCalls,
			modifyCalls: stabModifierCalls,
			preTypeModifyCalls,
			modifiedInput: stabModification ? stabModification.input : randomizerOutput,
			modifiedOutput: stabModification ? stabModification.output : randomizerOutput,
			finalDamage: result,
		};
	} finally {
		restore(battle, 'randomizer', randomizerDescriptor);
		restore(battle, 'modify', modifyDescriptor);
		restore(battle, 'runEvent', runEventDescriptor);
		restore(target, 'runEffectiveness', effectivenessDescriptor);
		if (moveDescriptors) for (const [key, descriptor] of moveDescriptors) restore(move, key, descriptor);
	}
}

function captureWithAbility(ps, ability, config) {
	const battle = createBattle(ps, ability);
	try {
		return capture(battle, { ...config, adaptability: ability === 'Adaptability' });
	} finally {
		battle.destroy();
	}
}

function ordinaryCases(ps) {
	return [
		captureWithAbility(ps, 'Static', { label: 'ordinary-native-stab', move: 'Thunderbolt', stabEligible: true }),
		captureWithAbility(ps, 'Static', { label: 'ordinary-typed-non-stab', move: 'Tackle' }),
		captureWithAbility(ps, 'Adaptability', { label: 'ordinary-adaptability-stab', move: 'Thunderbolt', stabEligible: true }),
		captureWithAbility(ps, 'Adaptability', { label: 'ordinary-adaptability-non-stab', move: 'Tackle' }),
		captureWithAbility(ps, 'Static', { label: 'ordinary-force-stab', move: 'Tackle', stabEligible: true, forceSTAB: true }),
	];
}

function syntheticCases(ps) {
	const cases = [];
	for (const [label, ability, stabEligible] of [
		['typed-non-stab', 'Static', false],
		['typed-stab', 'Static', true],
		['typed-adaptability', 'Adaptability', true],
	]) {
		const battle = createBattle(ps, ability);
		try {
			for (const damage of SYNTHETIC_DAMAGE) {
				cases.push(capture(battle, {
					label: `${label}-${damage}`,
					damage,
					move: 'Tackle',
					stabEligible,
					forceSTAB: stabEligible,
					adaptability: ability === 'Adaptability',
				}));
			}
		} finally { battle.destroy(); }
	}
	const battle = createBattle(ps, 'Adaptability');
	try {
		for (const damage of [0, 1, 2, 3, 2147483647]) {
			cases.push(capture(battle, {
				label: `typeless-conflicting-flags-${damage}`,
				damage,
				typeless: true,
				stabEligible: true,
				adaptability: true,
			}));
		}
	} finally { battle.destroy(); }
	return cases;
}

function generateFixture(ps) {
	const ordinary = ordinaryCases(ps);
	const synthetic = syntheticCases(ps);
	return {
		schemaVersion: 1,
		kind: 'gen5-stab-arithmetic-oracle',
		upstream: { repository: LOCK.repository, commit: LOCK.commit },
		formatId: FORMAT_ID,
		generation: 5,
		battleSeed: [...SEED],
		method: "BattleActions.getDamage -> runEvent('ModifySTAB') -> Battle.modify",
		sourcePaths: SOURCE_PATHS,
		inputDomain: { damage: [0, 2147483647], flags: 'strict boolean', result: 'int32' },
		modifiers: MODIFIERS,
		wrapAnchors: WRAP_ANCHORS,
		scope: 'Gen 5 explicit STAB arithmetic only. Eligibility and effective Adaptability are supplied; type and ability lookup, event dispatch, Tera, type effectiveness, burn, finalization, and the full damage calculator are excluded.',
		ordinaryContract: 'Real initialized Gen 5 Pokemon and moves, actual getDamage, ModifySTAB, and Adaptability event path. A transparent Battle.modify wrapper records and delegates the STAB call.',
		syntheticContract: 'Only Battle.randomizer output is replaced. Real getDamage, ModifySTAB, Adaptability, and Battle.modify remain active. Synthetic Tackle cases set explicit forceSTAB to supply already resolved eligibility; a transparent wrapper records and delegates the typed modifier. A target.runEffectiveness boundary wrapper verifies ??? makes no intervening modify call. No arithmetic formula is cloned.',
		ordinaryCases: ordinary,
		syntheticCases: synthetic,
	};
}

function stringifyFixture(fixture) {
	return `${JSON.stringify(fixture, null, 2)}\n`;
}

function main(args = process.argv.slice(2)) {
	if (args.length > 1 || (args.length && args[0] !== 'fixture')) throw new Error('Usage: node reference/stab-oracle.cjs [fixture]');
	require('./oracle.cjs').assertReady();
	const ps = require('./vendor/pokemon-showdown/dist/sim');
	process.stdout.write(stringifyFixture(generateFixture(ps)));
}

if (require.main === module) main();

module.exports = { capture, createBattle, generateFixture, restore, stringifyFixture };
