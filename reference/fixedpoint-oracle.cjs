#!/usr/bin/env node
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const ROOT = path.resolve(__dirname, '..');
const LOCK = JSON.parse(fs.readFileSync(path.join(ROOT, 'reference-lock.json'), 'utf8'));
const FORMAT_ID = 'gen5randombattle';
const SEED = [101, 202, 303, 404];
const VALUES = Array.from({ length: 1025 }, (_, value) => value);
const MODIFIERS = [0, 1, 2047, 2048, 2049, 3072, 4095, 4096, 4097, 5324, 6144, 8192, 65535];
const TARGETED_PAIRS = [
	{ label: 'half-below', value: 1, modifier: 2047 },
	{ label: 'half-tie', value: 1, modifier: 2048 },
	{ label: 'half-above', value: 1, modifier: 2049 },
	{ label: 'quotient-half-tie', value: 3, modifier: 2048 },
	{ label: 'quotient-round-up', value: 3, modifier: 2049 },
	{ label: 'identity-max-before-wrap', value: 1048575, modifier: 4096 },
	{ label: 'identity-first-wrap', value: 1048576, modifier: 4096 },
	{ label: 'identity-after-wrap', value: 1048577, modifier: 4096 },
	{ label: 'uint32-product-max-rounding-carry', value: 65537, modifier: 65535 },
	{ label: 'int32-value-high-modifier', value: 2147483647, modifier: 65535 },
	{ label: 'int32-value-identity', value: 2147483647, modifier: 4096 },
];
const SOURCE_PATHS = ['sim/battle.ts:2348-2359', 'sim/dex.ts:364-369'];

function createBattle(ps) {
	const empty = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
	const perfect = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
	const team = (species, ability) => [{ species, ability, moves: ['Tackle'], nature: 'Hardy', evs: empty, ivs: perfect }];
	const battle = new ps.Battle({
		formatid: FORMAT_ID,
		seed: [...SEED],
		p1: { name: 'fixedpoint-p1', team: team('Pikachu', 'Static') },
		p2: { name: 'fixedpoint-p2', team: team('Bulbasaur', 'Overgrow') },
	});
	assert.equal(battle.gen, 5);
	assert.equal(battle.started, true);
	return battle;
}

function generateFixture(ps) {
	const format = ps.Dex.forFormat(ps.Dex.formats.get(FORMAT_ID));
	assert.equal(format.gen, 5);
	const battle = createBattle(ps);
	try {
		return {
			schemaVersion: 1,
			kind: 'gen5-fixedpoint-modifier-oracle',
			upstream: { repository: LOCK.repository, commit: LOCK.commit },
			formatId: FORMAT_ID,
			generation: battle.gen,
			battleSeed: [...SEED],
			method: 'Battle.modify(value, modifier, 4096)',
			sourcePaths: SOURCE_PATHS,
			inputDomain: { value: [0, 2147483647], encodedModifier: [0, 65535], result: 'int32' },
			modifierContract: 'The supplied modifier is already encoded. Passing it as numerator with denominator 4096 makes Battle.modify encode it unchanged; no fraction encoding is tested.',
			roundingContract: 'Battle.trunc(value * modifier) wraps to uint32; Battle.modify then adds 2047 as a JavaScript number and floors by 4096. Remainder 2048 ties round down.',
			controlledGrid: {
				purpose: 'Direct calls to pinned Battle.modify; no arithmetic or event overrides.',
				values: VALUES,
				encodedModifiers: MODIFIERS,
				results: VALUES.map(value => MODIFIERS.map(modifier => battle.modify(value, modifier, 4096))),
			},
			targetedCases: TARGETED_PAIRS.map(pair => ({
				...pair,
				result: battle.modify(pair.value, pair.modifier, 4096),
			})),
		};
	} finally {
		battle.destroy();
	}
}

function stringifyFixture(fixture) {
	const lines = ['{'];
	for (const [index, [key, value]] of Object.entries(fixture).entries()) {
		const last = index === Object.keys(fixture).length - 1;
		if (key === 'controlledGrid') {
			lines.push('  "controlledGrid": {');
			for (const [gridIndex, [gridKey, gridValue]] of Object.entries(value).entries()) {
				const gridLast = gridIndex === Object.keys(value).length - 1;
				const rendered = JSON.stringify(gridValue);
				lines.push(`    ${JSON.stringify(gridKey)}: ${rendered}${gridLast ? '' : ','}`);
			}
			lines.push(`  }${last ? '' : ','}`);
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
	if (args.length > 1 || (args.length && args[0] !== 'fixture')) throw new Error('Usage: node reference/fixedpoint-oracle.cjs [fixture]');
	const oracle = require('./oracle.cjs');
	oracle.assertReady();
	const ps = require('./vendor/pokemon-showdown/dist/sim');
	process.stdout.write(stringifyFixture(generateFixture(ps)));
}

if (require.main === module) {
	try {
		main();
	} catch (error) {
		console.error(`Fixed-point oracle failed: ${error.message}`);
		process.exitCode = 1;
	}
}

module.exports = { FORMAT_ID, MODIFIERS, VALUES, generateFixture, main };
