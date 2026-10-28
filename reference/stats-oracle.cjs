#!/usr/bin/env node
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const ROOT = path.resolve(__dirname, '..');
const LOCK = JSON.parse(fs.readFileSync(path.join(ROOT, 'reference-lock.json'), 'utf8'));
const FORMAT_ID = 'gen5randombattle';
const STAT_IDS = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
const NATURE_MULTIPLIERS = { minus: 9, neutral: 10, plus: 11 };
const LEVELS = [1, 17, 50, 99, 100];
const IV_VALUES = [0, 3, 4, 30, 31];
const EV_VALUES = [0, 3, 4, 252, 255];
const SPECIES_PROFILES = ['Pikachu', 'Blissey', 'Shuckle', 'Shedinja', 'Deoxys-Attack', 'Chansey', 'Wobbuffet', 'Bulbasaur'];
const SYNTHETIC_NATURES = ['Adamant', 'Modest', 'Hardy'];
const SYNTHETIC_ATK_PROFILES = [
	{ label: 'pre-nature-595', base: 248, iv: 31 },
	{ label: 'pre-nature-596', base: 249, iv: 30 },
	{ label: 'base-stat-upper-bound', base: 255, iv: 31 },
];

function natureMultipliers(dex, natureName) {
	const nature = dex.natures.get(natureName);
	return STAT_IDS.map(stat => nature.plus === stat ? NATURE_MULTIPLIERS.plus :
		nature.minus === stat ? NATURE_MULTIPLIERS.minus : NATURE_MULTIPLIERS.neutral);
}

function statValues(pokemon) {
	return [pokemon.maxhp, ...STAT_IDS.slice(1).map(stat => pokemon.storedStats[stat])];
}

function toStatArray(values) {
	return STAT_IDS.map(stat => values[stat]);
}

function makeSet(dex, speciesName, nature, level, pattern, { attackIV = null, evs = null } = {}) {
	const species = dex.species.get(speciesName);
	if (!species.exists) throw new Error(`Unknown pinned oracle species ${speciesName}.`);
	const ability = species.abilities['0'] || Object.values(species.abilities)[0];
	const ivTable = {};
	const evTable = {};
	for (const [index, stat] of STAT_IDS.entries()) {
		ivTable[stat] = attackIV !== null && stat === 'atk' ? attackIV : IV_VALUES[(pattern + index) % IV_VALUES.length];
		evTable[stat] = evs ? evs[stat] : EV_VALUES[(pattern + index) % EV_VALUES.length];
	}
	return {
		species: species.name,
		ability,
		moves: ['Tackle'],
		nature,
		level,
		ivs: ivTable,
		evs: evTable,
	};
}

function createBattle(ps, dex, team, suffix = 'oracle') {
	return new ps.Battle({
		formatid: FORMAT_ID,
		seed: [101, 202, 303, 404],
		p1: { name: `p1-${suffix}`, team },
		p2: { name: `p2-${suffix}`, team: [makeSet(dex, 'Bulbasaur', 'Hardy', 50, 0)] },
	});
}

function generatePokemonCases(ps, dex) {
	const natureNames = dex.natures.all().map(nature => nature.name).sort();
	assert.equal(natureNames.length, 25, 'Gen 5 should expose all 25 natures');
	const descriptors = [];
	for (const [natureIndex, nature] of natureNames.entries()) {
		for (let pattern = 0; pattern < IV_VALUES.length; pattern++) {
			const level = LEVELS[(natureIndex + pattern) % LEVELS.length];
			const species = SPECIES_PROFILES[(natureIndex * 2 + pattern) % SPECIES_PROFILES.length];
			descriptors.push({ nature, level, species, pattern });
		}
	}

	const results = [];
	for (let offset = 0; offset < descriptors.length; offset += 6) {
		const batch = descriptors.slice(offset, offset + 6);
		const team = batch.map(entry => makeSet(dex, entry.species, entry.nature, entry.level, entry.pattern));
		const battle = createBattle(ps, dex, team, `stats-${offset}`);
		try {
			assert.equal(battle.gen, 5);
			assert.equal(battle.ruleTable.has('overflowstatmod'), false);
			assert.equal('wonderroom' in battle.field.pseudoWeather, false);
			for (const [index, descriptor] of batch.entries()) {
				const pokemon = battle.p1.pokemon[index];
				const expectedSpecies = dex.species.get(descriptor.species);
				assert.equal(pokemon.species.id, expectedSpecies.id);
				const isShedinja = pokemon.species.id === 'shedinja';
				assert.equal(Boolean(pokemon.species.maxHP === 1), isShedinja, 'only Shedinja has the Gen 5 maxHP override in these cases');
				results.push({
					species: pokemon.species.name,
					nature: descriptor.nature,
					natureMultipliers: natureMultipliers(dex, descriptor.nature),
					level: pokemon.level,
					baseStats: toStatArray(pokemon.species.baseStats),
					ivs: toStatArray(pokemon.set.ivs),
					evs: toStatArray(pokemon.set.evs),
					shedinja: isShedinja,
					expected: statValues(pokemon),
				});
			}
		} finally {
			battle.destroy();
		}
	}
	return results;
}

function generateSyntheticNatureCases(ps, dex) {
	const descriptors = [];
	for (const nature of SYNTHETIC_NATURES) {
		for (const profile of SYNTHETIC_ATK_PROFILES) descriptors.push({ nature, profile });
	}
	const cases = [];
	for (let offset = 0; offset < descriptors.length; offset += 6) {
		const batch = descriptors.slice(offset, offset + 6);
		const sets = batch.map(({ nature, profile }) => {
			const evs = Object.fromEntries(STAT_IDS.map(stat => [stat, 255]));
			return makeSet(dex, 'Pikachu', nature, 100, 0, { attackIV: profile.iv, evs });
		});
		const battle = createBattle(ps, dex, sets, `synthetic-${offset}`);
		try {
			assert.equal(battle.gen, 5);
			assert.equal(battle.ruleTable.has('overflowstatmod'), false);
			for (const [index, descriptor] of batch.entries()) {
				const pokemon = battle.p1.pokemon[index];
				const baseStats = Object.fromEntries(STAT_IDS.map(stat => [stat, 255]));
				baseStats.atk = descriptor.profile.base;
				const expected = STAT_IDS.map(stat => battle.statModify(baseStats, pokemon.set, stat));
				const ivs = toStatArray(pokemon.set.ivs);
				const evs = toStatArray(pokemon.set.evs);
				const caseRecord = {
					label: descriptor.profile.label,
					nature: descriptor.nature,
					natureMultipliers: natureMultipliers(dex, descriptor.nature),
					level: pokemon.level,
					baseStats: toStatArray(baseStats),
					ivs,
					evs,
					shedinja: false,
					expected,
				};
				const neutralSet = { ...pokemon.set, nature: 'Hardy' };
				caseRecord.attackBeforeNature = battle.statModify(baseStats, neutralSet, 'atk');
				cases.push(caseRecord);
			}
		} finally {
			battle.destroy();
		}
	}
	return cases;
}

function generateFixture(ps) {
	const format = ps.Dex.forFormat(ps.Dex.formats.get(FORMAT_ID));
	const ruleTable = format.formats.getRuleTable(format.formats.get(FORMAT_ID));
	assert.equal(format.gen, 5);
	assert.equal(ruleTable.has('overflowstatmod'), false, 'target format must not enable the optional 595/728 caps');
	return {
		schemaVersion: 1,
		kind: 'gen5-stored-stat-derivation-oracle',
		upstream: { repository: LOCK.repository, commit: LOCK.commit },
		formatId: FORMAT_ID,
		generation: format.gen,
		statOrder: STAT_IDS,
		sourcePaths: ['sim/pokemon.ts:407-431', 'sim/pokemon.ts:1408-1436', 'sim/battle.ts:2362-2394', 'sim/dex.ts:365-367'],
		formatFacts: { overflowstatmod: ruleTable.has('overflowstatmod'), natureProductTruncation: 'unsigned low 16 bits before division by 100' },
		inputDomain: { baseStats: [1, 255], level: [1, 100], ivs: [0, 31], evs: [0, 255], natureMultipliers: [9, 10, 11] },
		cases: generatePokemonCases(ps, format),
		syntheticStatModifyCases: generateSyntheticNatureCases(ps, format),
	};
}

function stringifyFixture(fixture) {
	const compactRows = new Set(['cases', 'syntheticStatModifyCases']);
	const lines = ['{'];
	const entries = Object.entries(fixture);
	for (const [index, [key, value]] of entries.entries()) {
		const isLast = index === entries.length - 1;
		if (compactRows.has(key)) {
			lines.push(`  ${JSON.stringify(key)}: [`);
			for (const [rowIndex, row] of value.entries()) {
				lines.push(`    ${JSON.stringify(row)}${rowIndex === value.length - 1 ? '' : ','}`);
			}
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
	if (args.length > 1 || (args.length && args[0] !== 'fixture')) throw new Error('Usage: node reference/stats-oracle.cjs [fixture]');
	const oracle = require('./oracle.cjs');
	oracle.assertReady();
	const ps = require('./vendor/pokemon-showdown/dist/sim');
	process.stdout.write(stringifyFixture(generateFixture(ps)));
}

if (require.main === module) {
	try {
		main();
	} catch (error) {
		console.error(`Stored-stat oracle failed: ${error.message}`);
		process.exitCode = 1;
	}
}

module.exports = { FORMAT_ID, STAT_IDS, generateFixture, main };
