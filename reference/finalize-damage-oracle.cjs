#!/usr/bin/env node
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const LOCK = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'reference-lock.json'), 'utf8'));
const FORMAT_ID = 'gen5randombattle';
const SEED = [101, 202, 303, 404];
const VALUES = [...Array.from({ length: 33 }, (_, i) => i), 32767, 32768, 65534, 65535, 65536, 65537, 1048575, 1048576, 1048577, 2147483647];
const MODIFIERS = [0, 1, 2047, 2048, 2049, 4096, 5324, 6144, 8192, 65535];

function createBattle(ps, item = '') {
	const team = (species, ability, heldItem) => [{ species, ability, item: heldItem, level: 50, moves: ['Tackle'], nature: 'Hardy' }];
	const battle = new ps.Battle({
		formatid: FORMAT_ID, seed: [...SEED],
		p1: { name: 'finalize-p1', team: team('Pikachu', 'Static', item) },
		p2: { name: 'finalize-p2', team: team('Bulbasaur', 'Overgrow', '') },
	});
	assert.equal(battle.gen, 5);
	assert.equal(battle.started, true);
	return battle;
}

function restore(object, key, descriptor) {
	if (descriptor) Object.defineProperty(object, key, descriptor);
	else delete object[key];
}

function capture(battle, { value, modifier, basePower = 50 }) {
	const source = battle.p1.pokemon[0];
	const target = battle.p2.pokemon[0];
	const synthetic = value !== undefined;
	const keys = ['randomizer', 'runEvent', 'modify', 'events'];
	const descriptors = keys.map(key => Object.getOwnPropertyDescriptor(battle, key));
	const original = { randomizer: battle.randomizer, runEvent: battle.runEvent, modify: battle.modify };
	let randomizerCalls = 0;
	let finalCalls = 0;
	let modifierCalls = 0;
	let handlerCalls = 0;
	let damage, minimumInput, modifiedDamage, encodedModifier;
	try {
		if (synthetic) {
			assert.equal(battle.findEventHandlers(source, 'ModifyDamage', target).length, 0);
			battle.events = { ...battle.events };
			battle.onEvent('ModifyDamage', battle.format, function () {
				handlerCalls++;
				this.chainModify(modifier, 4096);
			});
		}
		battle.randomizer = function (input) {
			randomizerCalls++;
			damage = synthetic ? value : original.randomizer.call(this, input);
			return damage;
		};
		battle.modify = function (input, numerator, denominator = 1) {
			if (this.event.id === 'ModifyDamage') {
				modifierCalls++;
				encodedModifier = numerator * 4096 / denominator;
				assert.equal(Number.isInteger(encodedModifier), true);
			}
			return original.modify.call(this, input, numerator, denominator);
		};
		battle.runEvent = function (event, ...args) {
			if (event !== 'ModifyDamage') return original.runEvent.call(this, event, ...args);
			finalCalls++;
			const [pokemon, defender, move, input] = args;
			assert.equal(pokemon, source);
			assert.equal(defender, target);
			assert.equal(move.type, '???'); // no STAB or type modifiers
			assert.equal(target.getMoveHitData(move).typeMod, 0);
			assert.equal(!!target.getMoveHitData(move).bypassProtect, false);
			assert.notEqual(source.status, 'brn');
			minimumInput = input;
			assert.equal(input, damage === 0 ? 1 : damage);
			modifiedDamage = original.runEvent.call(this, event, ...args);
			return modifiedDamage;
		};
		const result = battle.actions.getDamage(source, target, basePower, true);
		assert.equal(randomizerCalls, 1);
		assert.equal(finalCalls, 1);
		assert.equal(modifierCalls, 1);
		assert.equal(handlerCalls, synthetic ? 1 : 0);
		if (synthetic) assert.equal(encodedModifier, modifier);
		return { damage, encodedModifier, minimumInput, modifiedDamage, result };
	} finally {
		keys.forEach((key, i) => restore(battle, key, descriptors[i]));
	}
}

function generateFixture(ps) {
	const ordinaryCases = ['', 'Life Orb'].flatMap(item => {
		const battle = createBattle(ps, item);
		try {
			return [1, 50, 100, 300].map(basePower => {
				const captured = capture(battle, { basePower });
				assert.equal(captured.encodedModifier, item ? 5324 : 4096);
				return { item, basePower, ...captured };
			});
		} finally { battle.destroy(); }
	});
	const battle = createBattle(ps);
	try {
		return {
			schemaVersion: 1, kind: 'gen5-damage-finalization-oracle',
			upstream: LOCK, formatId: FORMAT_ID, generation: 5, battleSeed: SEED,
			sourcePaths: ['sim/battle-actions.ts:1876-1897', 'sim/battle.ts:934-939', 'sim/battle.ts:2337-2359', 'sim/dex.ts:364-369'],
			inputDomain: { damage: [0, 2147483647], encodedModifier: [0, 65535], result: 'int32 [0,65535]' },
			scope: 'After type and burn: Gen 5 zero-to-one, real ModifyDamage event, final 16-bit truncation. No bypassProtect. Not a full damage calculator.',
			ordinaryContract: 'Initialized Pokemon, real getDamage and randomizer, no arithmetic injection. Empty item and real Life Orb; encodedModifier observed inside the real event.',
			syntheticContract: 'Explicit arithmetic boundary cases: only randomizer output is replaced with damage; ??? type, neutral effectiveness, no burn. One registered event calls real chainModify(modifier,4096); real runEvent, modifyDamage tail, modify and trunc execute unchanged. No finalization formula clone.',
			ordinaryCases,
			controlledGrid: {
				values: VALUES, encodedModifiers: MODIFIERS,
				cases: VALUES.map(value => MODIFIERS.map(modifier => capture(battle, { value, modifier }))),
			},
		};
	} finally { battle.destroy(); }
}

function main(args = process.argv.slice(2)) {
	if (args.length > 1 || (args.length && args[0] !== 'fixture')) throw new Error('Usage: node reference/finalize-damage-oracle.cjs [fixture]');
	const oracle = require('./oracle.cjs');
	oracle.assertReady();
	const ps = require('./vendor/pokemon-showdown/dist/sim');
	const fixture = generateFixture(ps);
	const rows = fixture.controlledGrid.cases;
	fixture.controlledGrid.cases = '__COMPACT_ROWS__';
	const rendered = JSON.stringify(fixture, null, 2).replace('"__COMPACT_ROWS__"', `[\n${rows.map(row => `      ${JSON.stringify(row)}`).join(',\n')}\n    ]`);
	process.stdout.write(`${rendered}\n`);
}

if (require.main === module) {
	try { main(); } catch (error) {
		console.error(`Damage finalization oracle failed: ${error.message}`);
		process.exitCode = 1;
	}
}

module.exports = { capture, createBattle, generateFixture, main };
