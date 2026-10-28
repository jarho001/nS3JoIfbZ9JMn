#!/usr/bin/env node
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const ROOT = path.resolve(__dirname, '..');
const LOCK = JSON.parse(fs.readFileSync(path.join(ROOT, 'reference-lock.json'), 'utf8'));
const FORMAT_ID = 'gen5customgame';
const SOURCE_PATH = 'sim/pokemon.ts:581-612';
const FIXTURE_STATS = [0, 1, 2, 3, 5, 7, 9, 15, 31, 127, 255, 257, 1023, 1024, 65534, 65535];
const FIXTURE_STAGES = [-7, -6, -5, -3, -2, -1, 0, 1, 2, 3, 5, 6, 7];
const GRID_STAGES = [-2147483648, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 2147483647];

function setupBattle(ps) {
	const makeSet = species => ({
		species,
		ability: 'Static',
		moves: ['Tackle'],
		nature: 'Hardy',
		evs: { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 },
		ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 },
	});
	const battle = new ps.Battle({
		formatid: FORMAT_ID,
		seed: [101, 202, 303, 404],
		p1: { name: 'p1', team: [makeSet('Pikachu')] },
		p2: { name: 'p2', team: [makeSet('Bulbasaur')] },
	});
	assert.equal(battle.gen, 5, 'oracle format must resolve to Gen 5');
	assert.equal(battle.started, true, 'controlled battle must be initialized');
	assert.equal('wonderroom' in battle.field.pseudoWeather, false, 'oracle battle must not have Wonder Room');
	const pokemon = battle.p1.pokemon[0];
	const probe = { atk: 3 };
	assert.deepEqual(battle.runEvent('ModifyBoost', pokemon, null, null, probe), probe, 'controlled battle must not modify boost stages');
	return { battle, pokemon };
}

function calculate(pokemon, stat, stage) {
	pokemon.storedStats.atk = stat;
	return pokemon.calculateStat('atk', stage, 1);
}

function generateFixture(ps) {
	const { battle, pokemon } = setupBattle(ps);
	try {
		const cases = [];
		for (const storedStat of FIXTURE_STATS) {
			for (const stage of FIXTURE_STAGES) {
				cases.push({ storedStat, stage, result: calculate(pokemon, storedStat, stage) });
			}
		}
		return {
			schemaVersion: 1,
			kind: 'gen5-stat-stage-scaling-oracle',
			upstream: { repository: LOCK.repository, commit: LOCK.commit },
			formatId: FORMAT_ID,
			generation: battle.gen,
			method: 'Pokemon.calculateStat',
			sourcePath: SOURCE_PATH,
			conditions: {
				modifyBoost: 'identity for the tested stage',
				fieldStatSelection: 'no Wonder Room; selected stat is atk',
				modifier: 1,
				storedStatDomain: [0, 65535],
				stageInput: 'integer; Pokemon.calculateStat clamps to [-6, 6]',
			},
			cases,
		};
	} finally {
		battle.destroy();
	}
}

function generateGrid(ps) {
	const { battle, pokemon } = setupBattle(ps);
	try {
		const cases = [];
		for (let storedStat = 0; storedStat <= 1024; storedStat++) {
			for (let stage = -6; stage <= 6; stage++) {
				cases.push([storedStat, stage, calculate(pokemon, storedStat, stage)]);
			}
		}
		for (const storedStat of [1025, 65534, 65535]) {
			for (const stage of GRID_STAGES) {
				cases.push([storedStat, stage, calculate(pokemon, storedStat, stage)]);
			}
		}
		return { schemaVersion: 1, upstreamCommit: LOCK.commit, formatId: FORMAT_ID, sourcePath: SOURCE_PATH, cases };
	} finally {
		battle.destroy();
	}
}

function main(args = process.argv.slice(2)) {
	const mode = args[0] || 'fixture';
	if (args.length > 1 || !['fixture', 'grid'].includes(mode)) {
		throw new Error('Usage: node reference/stat-stage-oracle.cjs [fixture|grid]');
	}
	const oracle = require('./oracle.cjs');
	oracle.assertReady();
	const ps = require('./vendor/pokemon-showdown/dist/sim');
	const result = mode === 'grid' ? generateGrid(ps) : generateFixture(ps);
	process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
}

if (require.main === module) {
	try {
		main();
	} catch (error) {
		console.error(`Stat-stage oracle failed: ${error.message}`);
		process.exitCode = 1;
	}
}

module.exports = { FORMAT_ID, generateFixture, generateGrid, main };
