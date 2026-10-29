#!/usr/bin/env node
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const ROOT = path.resolve(__dirname, '..');
const LOCK = JSON.parse(fs.readFileSync(path.join(ROOT, 'reference-lock.json'), 'utf8'));
const FORMAT_ID = 'gen5randombattle';
const SEED = [101, 202, 303, 404];
const SOURCE_PATHS = ['sim/battle-actions.ts:1870-1882', 'sim/battle.ts:2337-2359', 'sim/dex.ts:364-369'];
const ORDINARY_CASES = [
	{ label: 'burned-physical', move: 'Tackle', ability: 'Static', burned: true },
	{ label: 'burned-special', move: 'Water Gun', ability: 'Static', burned: true },
	{ label: 'burned-physical-guts', move: 'Tackle', ability: 'Guts', burned: true },
	{ label: 'burned-facade-gen5', move: 'Facade', ability: 'Static', burned: true },
	{ label: 'unburned-physical', move: 'Tackle', ability: 'Static', burned: false },
];
const SYNTHETIC_VALUES = [0, 1, 2, 3, 4, 5, 2 ** 21 - 1, 2 ** 21, 2 ** 21 + 1, 2147483647];

function createBattle(ps, ability = 'Static') {
	const team = (species, pokemonAbility) => [{
		species, ability: pokemonAbility, item: '', level: 50, moves: ['Tackle'], nature: 'Hardy',
	}];
	const battle = new ps.Battle({
		formatid: FORMAT_ID,
		seed: [...SEED],
		p1: { name: 'burn-p1', team: team('Pikachu', ability) },
		p2: { name: 'burn-p2', team: team('Bulbasaur', 'Overgrow') },
	});
	assert.equal(battle.gen, 5);
	assert.equal(battle.started, true);
	return battle;
}

function restore(object, key, descriptor) {
	if (descriptor) Object.defineProperty(object, key, descriptor);
	else delete object[key];
}

function captureOrdinary(ps, config) {
	const battle = createBattle(ps, config.ability);
	try {
		const source = battle.p1.pokemon[0];
		const target = battle.p2.pokemon[0];
		const modifyDescriptor = Object.getOwnPropertyDescriptor(battle, 'modify');
		const originalModify = battle.modify;
		const previousStatus = source.status;
		let modifyCalls = 0;
		let captured;
		try {
			source.status = config.burned ? 'brn' : '';
			battle.modify = function (input, numerator, denominator = 1) {
				if (numerator === 0.5 && denominator === 1) {
					modifyCalls++;
					captured = { input, output: originalModify.call(this, input, numerator, denominator) };
					return captured.output;
				}
				return originalModify.call(this, input, numerator, denominator);
			};
			const finalDamage = battle.actions.getDamage(source, target, config.move, true);
			const move = battle.dex.moves.get(config.move);
			const shouldReduce = config.burned && move.category === 'Physical' && config.ability !== 'Guts';
			assert.equal(modifyCalls, shouldReduce ? 1 : 0, `${config.label} burn modify call count`);
			return {
				label: config.label,
				move: move.id,
				category: move.category,
				ability: config.ability.toLowerCase(),
				burned: config.burned,
				modifyCalls,
				burnInput: captured ? captured.input : null,
				burnOutput: captured ? captured.output : null,
				finalDamage,
			};
		} finally {
			restore(battle, 'modify', modifyDescriptor);
			source.status = previousStatus;
		}
	} finally {
		battle.destroy();
	}
}

function captureSynthetic(battle, damage) {
	const source = battle.p1.pokemon[0];
	const target = battle.p2.pokemon[0];
	const randomizerDescriptor = Object.getOwnPropertyDescriptor(battle, 'randomizer');
	const modifyDescriptor = Object.getOwnPropertyDescriptor(battle, 'modify');
	const previousStatus = source.status;
	const originalModify = battle.modify;
	let randomizerCalls = 0;
	let burnCalls = 0;
	let captured;
	try {
		source.status = 'brn';
		battle.randomizer = function () {
			randomizerCalls++;
			assert.equal(randomizerCalls, 1);
			assert.ok(Number.isInteger(damage), 'synthetic damage must be an integer');
			return damage;
		};
		battle.modify = function (input, numerator, denominator = 1) {
			if (numerator === 0.5 && denominator === 1) {
				burnCalls++;
				captured = { input, output: originalModify.call(this, input, numerator, denominator) };
				return captured.output;
			}
			return originalModify.call(this, input, numerator, denominator);
		};
		const finalDamage = battle.actions.getDamage(source, target, 50, true);
		assert.equal(randomizerCalls, 1);
		assert.equal(burnCalls, 1);
		assert.equal(captured.input, damage);
		return {
			damage,
			burnInput: captured.input,
			burnOutput: captured.output,
			postBurnAndGen5FinalDamage: finalDamage,
		};
	} finally {
		restore(battle, 'randomizer', randomizerDescriptor);
		restore(battle, 'modify', modifyDescriptor);
		source.status = previousStatus;
	}
}

function generateFixture(ps) {
	const ordinaryCases = ORDINARY_CASES.map(config => captureOrdinary(ps, config));
	const battle = createBattle(ps);
	let syntheticCases;
	try {
		syntheticCases = SYNTHETIC_VALUES.map(damage => captureSynthetic(battle, damage));
	} finally {
		battle.destroy();
	}
	return {
		schemaVersion: 1,
		kind: 'gen5-physical-burn-reduction-oracle',
		upstream: { repository: LOCK.repository, commit: LOCK.commit },
		formatId: FORMAT_ID,
		generation: 5,
		battleSeed: [...SEED],
		method: 'BattleActions.getDamage -> Battle.modify(damage, 0.5)',
		sourcePaths: SOURCE_PATHS,
		inputDomain: { damage: [0, 2147483647], eligibility: 'boolean', result: 'int32' },
		predicate: 'pokemon.status === brn && move.category === Physical && !pokemon.hasAbility(guts)',
		scope: 'One physical-burn arithmetic step after type effectiveness and before the Gen 5 minimum-one check. Gen 5 has no Facade exemption. No runtime status/ability/category resolution or event dispatch is modeled.',
		ordinaryContract: 'Real initialized Gen 5 Pokemon and moves with real getDamage, type lookup, randomizer, and modifying arithmetic. A transparent Battle.modify wrapper records and delegates the burn call.',
		syntheticContract: 'Only Battle.randomizer output is controlled. A burned physical ??? move uses real type handling; a transparent Battle.modify wrapper records and delegates the 0.5 call. The post-burn capture is before Gen 5 minimum-one and later events. No arithmetic formula is cloned.',
		ordinaryCases,
		syntheticCases,
	};
}

function stringifyFixture(fixture) {
	return `${JSON.stringify(fixture, null, 2)}\n`;
}

function main(args = process.argv.slice(2)) {
	if (args.length > 1 || (args.length && args[0] !== 'fixture')) {
		throw new Error('Usage: node reference/burn-oracle.cjs [fixture]');
	}
	require('./oracle.cjs').assertReady();
	const ps = require('./vendor/pokemon-showdown/dist/sim');
	process.stdout.write(stringifyFixture(generateFixture(ps)));
}

if (require.main === module) main();

module.exports = { captureSynthetic, createBattle, generateFixture, restore, stringifyFixture };
