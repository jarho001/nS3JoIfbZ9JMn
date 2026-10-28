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
const CONTROLLED_DAMAGE = [
	...Array.from({ length: 1025 }, (_, damage) => damage),
	65535, 16777217,
	42949671, 42949672, 42949673, 42949674,
	50529026, 50529027, 50529028,
	100000000, 2147483647,
];
const LIVE_DAMAGE = [0, 1, 100, 65535, 42949673, 50529028, 2147483647];
const SOURCE_PATHS = ['sim/battle.ts:2407-2410', 'sim/dex.ts:364-369', 'sim/battle-actions.ts:1809-1812'];

function createBattle(ps, suffix) {
	const emptyEvs = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
	const perfectIvs = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
	const team = species => [{ species, ability: species === 'Pikachu' ? 'Static' : 'Overgrow', moves: ['Tackle'], nature: 'Hardy', evs: emptyEvs, ivs: perfectIvs }];
	const battle = new ps.Battle({
		formatid: FORMAT_ID,
		seed: [...BATTLE_SEED],
		p1: { name: `roll-${suffix}`, team: team('Pikachu') },
		p2: { name: `ref-${suffix}`, team: team('Bulbasaur') },
	});
	assert.equal(battle.gen, 5);
	assert.equal(battle.started, true);
	return battle;
}

function restoreMethod(object, name, ownDescriptor) {
	if (ownDescriptor) Object.defineProperty(object, name, ownDescriptor);
	else delete object[name];
}

function controlledGrid(battle) {
	const originalOwnDescriptor = Object.getOwnPropertyDescriptor(battle, 'random');
	let selectedRoll = 0;
	let callsForEntry = 0;
	let totalCalls = 0;
	try {
		battle.random = bound => {
			assert.equal(bound, 16, 'Battle.randomizer must request random(16)');
			callsForEntry++;
			totalCalls++;
			return selectedRoll;
		};
		const results = CONTROLLED_DAMAGE.map(baseDamage => ROLLS.map(roll => {
			selectedRoll = roll;
			callsForEntry = 0;
			const result = battle.randomizer(baseDamage);
			assert.equal(callsForEntry, 1, 'Battle.randomizer must draw exactly once');
			return result;
		}));
		assert.equal(totalCalls, CONTROLLED_DAMAGE.length * ROLLS.length);
		return { baseDamageValues: CONTROLLED_DAMAGE, rolls: ROLLS, results, callsPerResult: 1 };
	} finally {
		restoreMethod(battle, 'random', originalOwnDescriptor);
	}
}

function seededDraws(battle) {
	const originalOwnDescriptor = Object.getOwnPropertyDescriptor(battle, 'random');
	const originalRandom = battle.random;
	const draws = [];
	try {
		battle.random = function (bound) {
			const result = originalRandom.call(this, bound);
			draws.push({ bound, roll: result });
			return result;
		};
		const cases = LIVE_DAMAGE.map(baseDamage => {
			const drawsBefore = draws.length;
			const result = battle.randomizer(baseDamage);
			assert.equal(draws.length, drawsBefore + 1, 'real Battle.randomizer must make one seeded draw');
			assert.equal(draws[drawsBefore].bound, 16);
			return { baseDamage, roll: draws[drawsBefore].roll, result };
		});
		return {
			battleSeed: [...BATTLE_SEED],
			cases,
			drawContract: 'wrapper only logs and returns the original seeded Battle.random result after Battle initialization; these are not asserted to be the seed PRNG first draws',
		};
	} finally {
		restoreMethod(battle, 'random', originalOwnDescriptor);
	}
}

function generateFixture(ps) {
	const controlledBattle = createBattle(ps, 'controlled');
	let controlled;
	try {
		controlled = controlledGrid(controlledBattle);
	} finally {
		controlledBattle.destroy();
	}
	const liveBattle = createBattle(ps, 'seeded');
	let liveSeeded;
	try {
		liveSeeded = seededDraws(liveBattle);
	} finally {
		liveBattle.destroy();
	}
	return {
		schemaVersion: 1,
		kind: 'gen5-randombattle-damage-randomizer-oracle',
		upstream: { repository: LOCK.repository, commit: LOCK.commit },
		formatId: FORMAT_ID,
		generation: 5,
		method: 'Battle.randomizer',
		sourcePaths: SOURCE_PATHS,
		inputDomain: { baseDamage: [0, 2147483647], roll: [0, 15], result: 'int32' },
		drawContract: 'Battle.randomizer invokes Battle.random(16) exactly once; the caller supplies the resulting roll.',
		controlledGrid: {
			purpose: 'Controlled arithmetic oracle: Battle.random alone is replaced temporarily to provide each roll; Battle.randomizer and truncation are the pinned implementations.',
			...controlled,
		},
		liveSeeded,
	};
}

function stringifyFixture(fixture) {
	const compactRows = new Set(['results', 'baseDamageValues', 'rolls']);
	const lines = ['{'];
	const entries = Object.entries(fixture);
	for (const [index, [key, value]] of entries.entries()) {
		const isLast = index === entries.length - 1;
		const indent = '  ';
		const renderCompactObject = (object, objectIndent) => {
			const objectLines = ['{'];
			const fields = Object.entries(object);
			for (const [fieldIndex, [field, fieldValue]] of fields.entries()) {
				const fieldLast = fieldIndex === fields.length - 1;
				if (compactRows.has(field)) {
					objectLines.push(`${objectIndent}  ${JSON.stringify(field)}: [`);
					for (const [rowIndex, row] of fieldValue.entries()) {
						if (Array.isArray(row)) objectLines.push(`${objectIndent}    ${JSON.stringify(row)}${rowIndex === fieldValue.length - 1 ? '' : ','}`);
						else objectLines.push(`${objectIndent}    ${JSON.stringify(row)}${rowIndex === fieldValue.length - 1 ? '' : ','}`);
					}
					objectLines.push(`${objectIndent}  ]${fieldLast ? '' : ','}`);
					continue;
				}
				const rendered = JSON.stringify(fieldValue, null, 2).split('\n');
				objectLines.push(`${objectIndent}  ${JSON.stringify(field)}: ${rendered[0]}`);
				for (const line of rendered.slice(1)) objectLines.push(`${objectIndent}  ${line}`);
				if (!fieldLast) objectLines[objectLines.length - 1] += ',';
			}
			objectLines.push(`${objectIndent}}`);
			return objectLines;
		};
		if (key === 'controlledGrid' || key === 'liveSeeded') {
			const rendered = renderCompactObject(value, `${indent}  `);
			lines.push(`${indent}${JSON.stringify(key)}: ${rendered[0]}`);
			lines.push(...rendered.slice(1));
			if (!isLast) lines[lines.length - 1] += ',';
		} else {
			const rendered = JSON.stringify(value, null, 2).split('\n');
			lines.push(`${indent}${JSON.stringify(key)}: ${rendered[0]}`);
			for (const line of rendered.slice(1)) lines.push(`${indent}${line}`);
			if (!isLast) lines[lines.length - 1] += ',';
		}
	}
	lines.push('}');
	return `${lines.join('\n')}\n`;
}

function main(args = process.argv.slice(2)) {
	if (args.length > 1 || (args.length && args[0] !== 'fixture')) throw new Error('Usage: node reference/damage-roll-oracle.cjs [fixture]');
	const oracle = require('./oracle.cjs');
	oracle.assertReady();
	const ps = require('./vendor/pokemon-showdown/dist/sim');
	process.stdout.write(stringifyFixture(generateFixture(ps)));
}

if (require.main === module) {
	try {
		main();
	} catch (error) {
		console.error(`Damage-roll oracle failed: ${error.message}`);
		process.exitCode = 1;
	}
}

module.exports = { FORMAT_ID, ROLLS, CONTROLLED_DAMAGE, generateFixture, main };
