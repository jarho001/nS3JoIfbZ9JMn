#!/usr/bin/env node
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const ROOT = path.resolve(__dirname, '..');
const LOCK = JSON.parse(fs.readFileSync(path.join(ROOT, 'reference-lock.json'), 'utf8'));
const FORMAT_ID = 'gen5randombattle';
const STAT_IDS = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
const STAGES = [-2147483648, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 2147483647];
const LEVELS = [1, 17, 50, 99, 100];
const IVS = [0, 3, 4, 30, 31];
const EVS = [0, 3, 4, 252, 255];
const SPECIES = ['Pikachu', 'Blissey', 'Shuckle', 'Shedinja', 'Deoxys-Attack', 'Chansey', 'Wobbuffet', 'Bulbasaur'];
const SOURCE_PATHS = ['sim/pokemon.ts:581-612', 'sim/pokemon.ts:1408-1436', 'sim/battle.ts:2362-2394'];

function toStats(values) {
	return STAT_IDS.map(stat => values[stat]);
}

function natureMultipliers(dex, name) {
	const nature = dex.natures.get(name);
	return STAT_IDS.map(stat => nature.plus === stat ? 11 : nature.minus === stat ? 9 : 10);
}

function makeSet(dex, descriptor, index) {
	const species = dex.species.get(descriptor.species);
	assert.ok(species.exists, `Unknown pinned species ${descriptor.species}`);
	const ability = species.abilities['0'] || Object.values(species.abilities)[0];
	const ivs = {};
	const evs = {};
	for (const [slot, stat] of STAT_IDS.entries()) {
		ivs[stat] = IVS[(index + slot) % IVS.length];
		evs[stat] = EVS[(index * 2 + slot) % EVS.length];
	}
	return {
		species: species.name,
		ability,
		moves: ['Tackle'],
		nature: descriptor.nature,
		level: descriptor.level,
		ivs,
		evs,
	};
}

function createBattle(ps, sets, suffix) {
	return new ps.Battle({
		formatid: FORMAT_ID,
		seed: [101, 202, 303, 404],
		p1: { name: `atk-${suffix}`, team: sets },
		p2: { name: `ref-${suffix}`, team: [{ species: 'Bulbasaur', ability: 'Overgrow', moves: ['Tackle'], nature: 'Hardy' }] },
	});
}

function assertNeutralConditions(battle, pokemon) {
	assert.equal(battle.gen, 5);
	assert.equal(battle.ruleTable.has('overflowstatmod'), false);
	assert.equal('wonderroom' in battle.field.pseudoWeather, false);
	for (const stage of STAGES) {
		const probe = { atk: stage };
		const result = battle.runEvent('ModifyBoost', pokemon, null, null, probe);
		assert.deepEqual(probe, { atk: stage }, `ModifyBoost must not mutate its probe for ${pokemon.species.name} at stage ${stage}`);
		assert.deepEqual(result, { atk: stage }, `ModifyBoost must be identity for ${pokemon.species.name} at stage ${stage}`);
	}
}

function ordinaryCases(ps, dex) {
	const natures = dex.natures.all().map(item => item.name).sort();
	assert.equal(natures.length, 25, 'Gen 5 should expose all 25 natures');
	const descriptors = natures.map((nature, index) => ({
		nature,
		species: SPECIES[(index * 3) % SPECIES.length],
		level: LEVELS[index % LEVELS.length],
	}));
	const cases = [];
	for (let offset = 0; offset < descriptors.length; offset += 5) {
		const batch = descriptors.slice(offset, offset + 5);
		const battle = createBattle(ps, batch.map((item, index) => makeSet(dex, item, offset + index)), `ordinary-${offset}`);
		try {
			assert.equal(battle.started, true);
			for (const [localIndex, descriptor] of batch.entries()) {
				const pokemon = battle.p1.pokemon[localIndex];
				assertNeutralConditions(battle, pokemon);
				const storedAttack = pokemon.storedStats.atk;
				const expected = STAGES.map(stage => pokemon.calculateStat('atk', stage, 1));
				cases.push({
					species: pokemon.species.name,
					nature: descriptor.nature,
					natureMultipliers: natureMultipliers(dex, descriptor.nature),
					level: pokemon.level,
					baseStats: toStats(pokemon.species.baseStats),
					ivs: toStats(pokemon.set.ivs),
					evs: toStats(pokemon.set.evs),
					storedAttack,
					stages: STAGES,
					expected,
				});
			}
		} finally {
			battle.destroy();
		}
	}
	return cases;
}

function syntheticZeroCase(ps, dex) {
	const descriptor = { nature: 'Adamant', species: 'Pikachu', level: 100 };
	const set = makeSet(dex, descriptor, 0);
	set.ivs.atk = 30;
	set.evs.atk = 255;
	const battle = createBattle(ps, [set], 'synthetic-zero');
	try {
		const pokemon = battle.p1.pokemon[0];
		assertNeutralConditions(battle, pokemon);
		const baseStats = toStats(pokemon.species.baseStats);
		baseStats[1] = 249;
		const syntheticBaseStats = Object.fromEntries(STAT_IDS.map((stat, index) => [stat, baseStats[index]]));
		const neutralSet = { ...pokemon.set, nature: 'Hardy' };
		const attackBeforeNature = battle.statModify(syntheticBaseStats, neutralSet, 'atk');
		const storedAttack = battle.statModify(syntheticBaseStats, pokemon.set, 'atk');
		assert.equal(attackBeforeNature, 596);
		assert.equal(storedAttack, 0);
		pokemon.storedStats.atk = storedAttack;
		return {
			label: 'arithmetic-only-synthetic-zero',
			species: pokemon.species.name,
			nature: descriptor.nature,
			natureMultipliers: natureMultipliers(dex, descriptor.nature),
			level: pokemon.level,
			baseStats,
			ivs: toStats(pokemon.set.ivs),
			evs: toStats(pokemon.set.evs),
			storedAttack,
			attackBeforeNature,
			stages: STAGES,
			expected: STAGES.map(stage => pokemon.calculateStat('atk', stage, 1)),
			boundary: 'Battle.statModify derives zero from synthetic base Attack 249, IV 30, EV 255 at level 100; that zero is then injected into this real Pokemon solely to exercise calculateStat stage arithmetic. Not an ordinary initialized species stat.',
		};
	} finally {
		battle.destroy();
	}
}

function generateFixture(ps) {
	const format = ps.Dex.forFormat(ps.Dex.formats.get(FORMAT_ID));
	assert.equal(format.gen, 5);
	const ruleTable = format.formats.getRuleTable(format.formats.get(FORMAT_ID));
	assert.equal(ruleTable.has('overflowstatmod'), false, 'target format must not enable optional nature-stat caps');
	const dex = format;
	return {
		schemaVersion: 1,
		kind: 'gen5-randombattle-attack-stage-composition-oracle',
		upstream: { repository: LOCK.repository, commit: LOCK.commit },
		formatId: FORMAT_ID,
		generation: format.gen,
		statOrder: STAT_IDS,
		stageInputs: STAGES,
		sourcePaths: SOURCE_PATHS,
		conditions: {
			modifyBoost: 'checked by real Battle.runEvent for each Pokemon and stage; identity',
			fieldStatSelection: 'no Wonder Room',
			modifier: 1,
			overflowstatmod: ruleTable.has('overflowstatmod'),
			ordinaryCases: 'initialized Pokemon; storedStats are not overwritten',
			storedStatRange: [0, 65535],
			output: "Pokemon.calculateStat('atk', stage, 1)",
		},
		cases: ordinaryCases(ps, dex),
		syntheticZeroCase: syntheticZeroCase(ps, dex),
	};
}

function stringifyFixture(fixture) {
	const compactRows = new Set(['cases']);
	const lines = ['{'];
	const entries = Object.entries(fixture);
	for (const [index, [key, value]] of entries.entries()) {
		const isLast = index === entries.length - 1;
		if (compactRows.has(key)) {
			lines.push(`  ${JSON.stringify(key)}: [`);
			for (const [rowIndex, row] of value.entries()) lines.push(`    ${JSON.stringify(row)}${rowIndex === value.length - 1 ? '' : ','}`);
			lines.push(`  ]${isLast ? '' : ','}`);
			continue;
		}
		const rendered = JSON.stringify(value, null, 2).split('\n');
		lines.push(`  ${JSON.stringify(key)}: ${rendered[0]}`);
		for (const line of rendered.slice(1)) lines.push(`  ${line}`);
		if (!isLast) lines[lines.length - 1] += ',';
	}
	lines.push('}');
	return `${lines.join('\n')}\n`;
}

function main(args = process.argv.slice(2)) {
	if (args.length > 1 || (args.length && args[0] !== 'fixture')) throw new Error('Usage: node reference/attack-oracle.cjs [fixture]');
	const oracle = require('./oracle.cjs');
	oracle.assertReady();
	const ps = require('./vendor/pokemon-showdown/dist/sim');
	process.stdout.write(stringifyFixture(generateFixture(ps)));
}

if (require.main === module) {
	try {
		main();
	} catch (error) {
		console.error(`Attack oracle failed: ${error.message}`);
		process.exitCode = 1;
	}
}

module.exports = { FORMAT_ID, STAGES, generateFixture, main };
