#!/usr/bin/env node
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const ROOT = path.resolve(__dirname, '..');
const LOCK = JSON.parse(fs.readFileSync(path.join(ROOT, 'reference-lock.json'), 'utf8'));
const FORMAT_ID = 'gen5randombattle';
const SEED = [101, 202, 303, 404];
const STAT_IDS = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
const ORDINARY_LEVELS = [1, 2, 4, 5, 17, 50, 99, 100];
const SPECIES = ['Pikachu', 'Bulbasaur', 'Shuckle', 'Blissey', 'Wobbuffet', 'Chansey', 'Deoxys-Attack', 'Shedinja'];
const BASE_POWERS = [1, 2, 50, 255, 65535];
const ATTACK_VALUES = [0, 1, 2, 255, 65535];
const DEFENSE_VALUES = [0, 1, 2, 3, 255, 65535];
const SOURCE_PATHS = ['sim/battle-actions.ts:1769-1775', 'sim/battle-actions.ts:1778-1786', 'sim/dex.ts:364-369'];

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

function createBattle(ps, p1Team, p2Team, suffix) {
	const battle = new ps.Battle({
		formatid: FORMAT_ID,
		seed: [...SEED],
		p1: { name: `base-${suffix}`, team: p1Team },
		p2: { name: `def-${suffix}`, team: p2Team },
	});
	assert.equal(battle.gen, 5);
	assert.equal(battle.started, true);
	assert.equal(battle.ruleTable.has('overflowstatmod'), false);
	return battle;
}

function invokeCore(battle, source, target, basePower) {
	const actions = battle.actions;
	const originalOwnDescriptor = Object.getOwnPropertyDescriptor(actions, 'modifyDamage');
	let calls = 0;
	let captured;
	try {
		actions.modifyDamage = function (baseDamage, actualSource, actualTarget, move, suppressMessages) {
			calls++;
			assert.equal(calls, 1, 'getDamage must invoke modifyDamage exactly once');
			captured = { baseDamage, source: actualSource, target: actualTarget, move, suppressMessages };
			return baseDamage;
		};
		const result = actions.getDamage(source, target, basePower, true);
		assert.equal(calls, 1);
		assert.equal(result, captured.baseDamage);
	} finally {
		if (originalOwnDescriptor) Object.defineProperty(actions, 'modifyDamage', originalOwnDescriptor);
		else delete actions.modifyDamage;
	}
	assert.equal(captured.source, source);
	assert.equal(captured.target, target);
	assert.equal(captured.suppressMessages, true);
	assert.equal(captured.move.type, '???');
	assert.equal(captured.move.category, 'Physical');
	assert.equal(captured.move.willCrit, false);
	assert.equal(captured.move.basePower, basePower);
	return captured;
}

function assertNeutralEvents(battle, source, target, move, basePower, attack, defense) {
	assert.equal(source.boosts.atk, 0);
	assert.equal(target.boosts.def, 0);
	assert.equal(battle.runEvent('BasePower', source, target, move, basePower, true), basePower);
	const calculatedAttack = source.calculateStat('atk', 0, 1, source);
	const calculatedDefense = target.calculateStat('def', 0, 1, target);
	assert.equal(battle.runEvent('ModifyAtk', source, target, move, calculatedAttack), attack);
	assert.equal(battle.runEvent('ModifyDef', target, source, move, calculatedDefense), defense);
	assert.equal(calculatedAttack, attack);
	assert.equal(calculatedDefense, defense);
}

function ordinaryCases(ps, dex) {
	const cases = [];
	for (let offset = 0; offset < ORDINARY_LEVELS.length; offset += 4) {
		const descriptors = ORDINARY_LEVELS.slice(offset, offset + 4).map((level, local) => ({
			level,
			species: SPECIES[offset + local],
			basePower: [1, 2, 50, 255, 65535, 16, 100, 300][offset + local],
		}));
		const p1Team = descriptors.map((item, local) => makeSet(dex, item.species, item.level, offset + local));
		const p2Team = descriptors.map((item, local) => makeSet(dex, SPECIES[(offset + local + 3) % SPECIES.length], item.level, offset + local + 2));
		const battle = createBattle(ps, p1Team, p2Team, `ordinary-${offset}`);
		try {
			for (const [local, descriptor] of descriptors.entries()) {
				const source = battle.p1.pokemon[local];
				const target = battle.p2.pokemon[local];
				const attack = source.calculateStat('atk', 0, 1, source);
				const defense = target.calculateStat('def', 0, 1, target);
				const captured = invokeCore(battle, source, target, descriptor.basePower);
				assertNeutralEvents(battle, source, target, captured.move, descriptor.basePower, attack, defense);
				cases.push({
					sourceSpecies: source.species.name,
					targetSpecies: target.species.name,
					level: source.level,
					basePower: descriptor.basePower,
					attack,
					defense,
					baseDamage: captured.baseDamage,
				});
			}
		} finally {
			battle.destroy();
		}
	}
	return cases;
}

function syntheticDescriptors() {
	const cases = [];
	for (const level of ORDINARY_LEVELS) {
		for (const basePower of BASE_POWERS) {
			for (const attack of ATTACK_VALUES) {
				for (const defense of DEFENSE_VALUES) cases.push({ level, basePower, attack, defense });
			}
		}
	}
	for (const basePower of [1559, 1560, 1561]) {
		for (const defense of [1, 2, 50, 65535]) cases.push({ level: 100, basePower, attack: 65535, defense });
	}
	for (const basePower of [2978, 2979]) {
		for (const defense of [1, 2, 50]) cases.push({ level: 50, basePower, attack: 65535, defense });
	}
	return cases;
}

function syntheticCases(ps, dex) {
	const battle = createBattle(ps, [makeSet(dex, 'Pikachu', 50, 0)], [makeSet(dex, 'Bulbasaur', 50, 1)], 'synthetic');
	try {
		const source = battle.p1.pokemon[0];
		const target = battle.p2.pokemon[0];
		return syntheticDescriptors().map(input => {
			source.level = input.level;
			source.storedStats.atk = input.attack;
			target.storedStats.def = input.defense;
			const captured = invokeCore(battle, source, target, input.basePower);
			assertNeutralEvents(battle, source, target, captured.move, input.basePower, input.attack, input.defense);
			return { ...input, baseDamage: captured.baseDamage };
		});
	} finally {
		battle.destroy();
	}
}

function generateFixture(ps) {
	const format = ps.Dex.forFormat(ps.Dex.formats.get(FORMAT_ID));
	assert.equal(format.gen, 5);
	return {
		schemaVersion: 1,
		kind: 'gen5-randombattle-base-damage-core-oracle',
		upstream: { repository: LOCK.repository, commit: LOCK.commit },
		formatId: FORMAT_ID,
		generation: format.gen,
		method: 'BattleActions.getDamage',
		coreExpression: 'tr(tr(tr(tr(2 * level / 5 + 2) * basePower * attack) / defense) / 50)',
		sourcePaths: SOURCE_PATHS,
		inputDomain: { level: [1, 100], basePower: [1, 65535], attack: [0, 65535], defense: [0, 65535], result: 'int32' },
		truncation: 'Battle.trunc uses Dex.trunc default unsigned 32-bit conversion after the level expression, product, defense division, and final division by 50.',
		context: {
			moveInput: 'numeric base power produces a Physical ??? move with willCrit=false',
			statSelection: 'actual source Attack and target Defense; initialized ordinary Pokemon have their stored stats untouched',
			syntheticRows: 'controlled level and stored Attack/Defense are injected into actual Pokemon only to probe arithmetic boundaries; these are not ordinary initialized Pokemon cases',
			neutralEvents: 'BasePower, ModifyAtk, and ModifyDef verified identity; zero stage boosts; no critical hit',
			captureBoundary: 'modifyDamage is temporarily replaced to capture and return getDamage core output before downstream modifiers; restored in finally',
			zeroDefense: 'Battle.trunc(Infinity or NaN) converts to unsigned 32-bit zero',
		},
		ordinaryCases: ordinaryCases(ps, format),
		syntheticCases: syntheticCases(ps, format),
	};
}

function stringifyFixture(fixture) {
	const compactRows = new Set(['ordinaryCases', 'syntheticCases']);
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
	if (args.length > 1 || (args.length && args[0] !== 'fixture')) throw new Error('Usage: node reference/base-damage-oracle.cjs [fixture]');
	const oracle = require('./oracle.cjs');
	oracle.assertReady();
	const ps = require('./vendor/pokemon-showdown/dist/sim');
	process.stdout.write(stringifyFixture(generateFixture(ps)));
}

if (require.main === module) {
	try {
		main();
	} catch (error) {
		console.error(`Base-damage oracle failed: ${error.message}`);
		process.exitCode = 1;
	}
}

module.exports = { FORMAT_ID, ORDINARY_LEVELS, generateFixture, main };
