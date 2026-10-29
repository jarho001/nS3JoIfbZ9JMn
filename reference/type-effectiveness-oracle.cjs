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
	'sim/battle-actions.ts:1849-1865',
	'sim/battle-actions.ts:1870-1877',
	'sim/battle-actions.ts:1640-1775',
];
const CLAMPED_EXPONENTS = Array.from({ length: 13 }, (_, index) => index - 6);
const GRID_DAMAGE_VALUES = [0, 1, 63, 65, 33554431];
const SYNTHETIC_CASES = [
	...GRID_DAMAGE_VALUES.flatMap(damage => CLAMPED_EXPONENTS.map(exponent => ({
		label: `grid-damage-${damage}-exponent-${exponent}`,
		damage,
		exponent,
	}))),
	{ label: 'clamped-minimum-zero-result', damage: 1, exponent: -2147483648 },
	{ label: 'clamped-minimum-positive-result', damage: 127, exponent: -2147483648 },
	{ label: 'clamped-maximum', damage: 33554431, exponent: 2147483647 },
];

function createBattle(ps) {
	const team = (species, ability) => [{ species, ability, item: '', level: 50, moves: ['Tackle'], nature: 'Hardy' }];
	const battle = new ps.Battle({
		formatid: FORMAT_ID,
		seed: [...SEED],
		p1: { name: 'type-p1', team: team('Pikachu', 'Static') },
		p2: { name: 'type-p2', team: team('Bulbasaur', 'Overgrow') },
	});
	assert.equal(battle.gen, 5);
	assert.equal(battle.started, true);
	return battle;
}

function restore(object, key, descriptor) {
	if (descriptor) Object.defineProperty(object, key, descriptor);
	else delete object[key];
}

function captureOrdinary(battle, moveName) {
	const source = battle.p1.pokemon[0];
	const target = battle.p2.pokemon[0];
	const runEventDescriptor = Object.getOwnPropertyDescriptor(battle, 'runEvent');
	const randomizerDescriptor = Object.getOwnPropertyDescriptor(battle, 'randomizer');
	const originalRunEvent = battle.runEvent;
	const originalRandomizer = battle.randomizer;
	let captured;
	let damageBeforeType;
	let randomizerCalls = 0;
	let modifierEventCalls = 0;
	try {
		battle.randomizer = function (input) {
			randomizerCalls++;
			damageBeforeType = originalRandomizer.call(this, input);
			return damageBeforeType;
		};
		battle.runEvent = function (event, ...args) {
			if (event !== 'ModifyDamage') return originalRunEvent.call(this, event, ...args);
			modifierEventCalls++;
			const [actualSource, actualTarget, move, damage] = args;
			assert.equal(actualSource, source);
			assert.equal(actualTarget, target);
			captured = { damage, move, type: move.type, category: move.category };
			return originalRunEvent.call(this, event, ...args);
		};
		const result = battle.actions.getDamage(source, target, moveName, true);
		assert.ok(captured, 'ordinary getDamage must reach modifyDamage');
		assert.equal(randomizerCalls, 1, 'ordinary getDamage must use the real randomizer once');
		assert.equal(modifierEventCalls, 1, 'ordinary getDamage must reach ModifyDamage once');
		const typeMod = target.getMoveHitData(captured.move).typeMod;
		assert.equal(typeMod, battle.dex.getEffectiveness(captured.type, target));
		assert.ok(damageBeforeType > 0, 'ordinary randomizer output must be positive');
		if (typeMod < 0) {
			assert.ok(damageBeforeType >= 2 ** (-typeMod), 'resisted ordinary damage must remain nonzero before Gen 5 minimum-one');
		}
		assert.notEqual(source.status, 'brn', 'ordinary cases must not receive burn');
		assert.equal(source.hasType(captured.type), false, 'ordinary moves must not receive STAB');
		assert.equal(!!captured.move.forceSTAB, false);
		return {
			move: captured.move.id,
			moveType: captured.type,
			damageBeforeType,
			typeMod,
			damageAfterType: captured.damage,
			finalDamage: result,
		};
	} finally {
		restore(battle, 'runEvent', runEventDescriptor);
		restore(battle, 'randomizer', randomizerDescriptor);
	}
}

function captureSynthetic(battle, { label, damage, exponent }) {
	const source = battle.p1.pokemon[0];
	const target = battle.p2.pokemon[0];
	const randomizerDescriptor = Object.getOwnPropertyDescriptor(battle, 'randomizer');
	const effectivenessDescriptor = Object.getOwnPropertyDescriptor(target, 'runEffectiveness');
	const modifyDescriptor = Object.getOwnPropertyDescriptor(battle, 'modify');
	const originalModify = battle.modify;
	const previousStatus = source.status;
	let randomizerCalls = 0;
	let effectivenessCalls = 0;
	let burnBoundaryCalls = 0;
	let captured;
	let capturedMove;
	try {
		source.status = 'brn';
		battle.randomizer = function (input) {
			randomizerCalls++;
			assert.equal(randomizerCalls, 1);
			return damage;
		};
		target.runEffectiveness = function (move) {
			effectivenessCalls++;
			assert.equal(effectivenessCalls, 1);
			assert.ok(Number.isInteger(exponent), 'synthetic exponent must be an integer');
			capturedMove = move;
			return exponent;
		};
		battle.modify = function (input, numerator, denominator = 1) {
			if (numerator === 0.5 && denominator === 1 && source.status === 'brn') {
				burnBoundaryCalls++;
				captured = input;
			}
			return originalModify.call(this, input, numerator, denominator);
		};
		const finalDamage = battle.actions.getDamage(source, target, 50, true);
		assert.equal(randomizerCalls, 1);
		assert.equal(effectivenessCalls, 1);
		assert.equal(burnBoundaryCalls, 1, 'burn wrapper must capture one post-type value');
		assert.equal(target.getMoveHitData(capturedMove).typeMod, Math.max(-6, Math.min(6, exponent)));
		return {
			label,
			damage,
			exponent,
			clampedExponent: target.getMoveHitData(capturedMove).typeMod,
			result: captured,
			postBurnAndGen5FinalDamage: finalDamage,
		};
	} finally {
		restore(battle, 'randomizer', randomizerDescriptor);
		restore(target, 'runEffectiveness', effectivenessDescriptor);
		restore(battle, 'modify', modifyDescriptor);
		source.status = previousStatus;
	}
}

function generateFixture(ps) {
	const ordinaryBattle = createBattle(ps);
	let ordinaryCases;
	try {
		ordinaryCases = ['Tackle', 'Water Gun', 'Ember', 'Psychic'].map(move => ({
			move,
			...captureOrdinary(ordinaryBattle, move),
		}));
	} finally {
		ordinaryBattle.destroy();
	}

	const syntheticBattle = createBattle(ps);
	let syntheticCases;
	try {
		syntheticCases = SYNTHETIC_CASES.map(input => captureSynthetic(syntheticBattle, input));
	} finally {
		syntheticBattle.destroy();
	}

	return {
		schemaVersion: 1,
		kind: 'gen5-type-effectiveness-arithmetic-oracle',
		upstream: { repository: LOCK.repository, commit: LOCK.commit },
		formatId: FORMAT_ID,
		generation: 5,
		battleSeed: [...SEED],
		method: 'BattleActions.getDamage type exponent loop',
		sourcePaths: SOURCE_PATHS,
		inputDomain: { damage: [0, 33554431], exponent: [-2147483648, 2147483647], result: 'int32 [0,2147483584]' },
		scope: 'Type-exponent arithmetic only. Type lookup, immunity, STAB, burn, modifiers, and full damage calculation are excluded.',
		ordinaryContract: 'Real initialized Gen 5 Pokemon, real moves, real type lookup, real randomizer, and real damage path; no arithmetic or type lookup overrides.',
		syntheticContract: 'Only randomizer output and target.runEffectiveness are controlled. A burned physical ??? move lets the temporary Battle.modify wrapper capture the value immediately after type arithmetic and before burn; the original burn, Gen 5 minimum-one, modifiers, and truncation all run afterward. No arithmetic formula is cloned.',
		ordinaryCases,
		syntheticCases,
	};
}

function stringifyFixture(fixture) {
	return `${JSON.stringify(fixture, null, 2)}\n`;
}

function main(args = process.argv.slice(2)) {
	if (args.length > 1 || (args.length && args[0] !== 'fixture')) {
		throw new Error('Usage: node reference/type-effectiveness-oracle.cjs [fixture]');
	}
	const oracle = require('./oracle.cjs');
	oracle.assertReady();
	const ps = require('./vendor/pokemon-showdown/dist/sim');
	process.stdout.write(stringifyFixture(generateFixture(ps)));
}

if (require.main === module) main();

module.exports = { captureSynthetic, createBattle, generateFixture, restore, stringifyFixture };
