#!/usr/bin/env node
'use strict';

const fs = require('node:fs');
const path = require('node:path');
const ROOT = path.resolve(__dirname, '..');
const LOCK = require(path.join(ROOT, 'reference-lock.json'));
const { capture, createBattle } = require('./damage-composition-oracle.cjs');
const ROLLS = Array.from({ length: 16 }, (_, roll) => roll);

const ORDINARY = [
	{ label: 'ordinary-stab-life-orb-special-burned', species: 'Pikachu', ability: 'Static', item: 'Life Orb', move: 'Thunderbolt', status: 'brn' },
	{ label: 'ordinary-typed-nonstab', species: 'Pikachu', ability: 'Static', move: 'Tackle' },
	{ label: 'ordinary-burned-physical', species: 'Pikachu', ability: 'Static', move: 'Tackle', status: 'brn' },
	{ label: 'ordinary-burned-physical-guts', species: 'Pikachu', ability: 'Guts', move: 'Tackle', status: 'brn' },
	{ label: 'ordinary-adaptability-stab', species: 'Eevee', ability: 'Adaptability', move: 'Tackle' },
];

const SYNTHETIC = [
	{ label: 'synthetic-core-product-wrap', level: 100, basePower: 1560, attack: 65535, defense: 1 },
	{ label: 'synthetic-zero-defense', level: 50, basePower: 50, attack: 1, defense: 0 },
	{ label: 'synthetic-neutral-event-below-wrap', level: 2, basePower: 403, attack: 65048, defense: 1 },
	{ label: 'synthetic-neutral-event-wrap', level: 2, basePower: 441, attack: 59443, defense: 1 },
	{ label: 'synthetic-neutral-event-above-wrap', level: 2, basePower: 427, attack: 61392, defense: 1 },
	{ label: 'synthetic-max-domain', level: 100, basePower: 65535, attack: 65535, defense: 1 },
];

function captureRows(ps, descriptor, synthetic) {
	const battle = createBattle(ps, descriptor);
	try {
		return ROLLS.map(roll => {
			const trace = capture(battle, {
				...descriptor,
				roll,
				captureEffectiveCore: true,
				...(synthetic ? { typeExponent: 0, finalModifier: 4096 } : {}),
			});
			return {
				roll,
				level: trace.effectiveLevel,
				basePower: trace.effectiveBasePower,
				attack: trace.effectiveAttack,
				defense: trace.effectiveDefense,
				baseCore: trace.baseCore,
				weatherInput: trace.weatherInput,
				weatherHandlerCount: trace.weatherHandlerCount,
				preRollDamage: trace.weatherOutput,
				randomizerOutput: trace.randomizerOutput,
				input: trace.input,
				result: trace.result,
				randomizerCalls: trace.randomizerCalls,
				damageDraws: trace.damageDraws,
				finalCalls: trace.finalCalls,
				finalModifierCalls: trace.finalModifierCalls,
				finalModifierObserved: trace.finalModifierObserved,
				finalMoveType: trace.finalMoveType,
				finalMoveCategory: trace.finalMoveCategory,
				baseCoreCalls: trace.baseCoreCalls,
				weatherEventCalls: trace.weatherEventCalls,
			};
		});
	} finally {
		battle.destroy();
	}
}

function generateFixture(ps) {
	const ordinaryCases = ORDINARY.map(descriptor => ({
		label: descriptor.label,
		kind: 'ordinary',
		outcomes: captureRows(ps, descriptor, false),
	}));
	const syntheticCases = SYNTHETIC.map(descriptor => ({
		label: descriptor.label,
		kind: 'synthetic',
		requested: descriptor,
		outcomes: captureRows(ps, { species: 'Pikachu', ability: 'Static', move: 'Tackle', ...descriptor }, true),
	}));
	return {
		schemaVersion: 1,
		kind: 'gen5-neutral-full-damage-composition-oracle',
		upstream: LOCK,
		formatId: 'gen5randombattle',
		generation: 5,
		method: 'pinned BattleActions.getDamage through actual modifyDamage, neutral WeatherModifyDamage, Battle.randomizer, and final tail',
		sourcePaths: ['sim/battle-actions.ts:1769-1786', 'sim/battle-actions.ts:1809-1897', 'sim/battle.ts:934-939', 'sim/battle.ts:2348-2359'],
		inputDomain: { level: [1, 100], basePower: [1, 65535], effectiveAttack: [0, 65535], effectiveDefense: [0, 65535], roll: [0, 15], preRollDamage: [0, 1048575] },
		ordinaryContract: 'Initialized Pokemon and real moves; actual BasePower and category-appropriate ModifyAtk/ModifySpA and ModifyDef/ModifySpD event outputs are captured as core inputs. willCrit is forced false; real neutral weather, randomizer, type, STAB, burn, and final events delegate to pinned implementations; only random(16) is controlled.',
		syntheticContract: 'Only level, basePower, and stored Attack/Defense are changed to exercise explicit core/event boundaries. Effective inputs are captured from real pre-core event outputs. Neutral type effectiveness and final modifier are controlled in the downstream Task016 oracle path. Temporary state restores in finally.',
		orderContract: 'The actual source/target and core result are observed at one modifyDamage call; it feeds one WeatherModifyDamage event after +2, with zero handlers. Weather output feeds the original randomizer exactly once, followed by real downstream finalization. Critical is false; no spread, Parental Bond, or bypassProtect.',
		ordinaryCases,
		syntheticCases,
	};
}

function main(args = process.argv.slice(2)) {
	if (args.length > 1 || (args.length && args[0] !== 'fixture')) throw new Error('Usage: node reference/neutral-full-damage-oracle.cjs [fixture]');
	const oracle = require('./oracle.cjs');
	oracle.assertReady();
	const ps = require('./vendor/pokemon-showdown/dist/sim');
	process.stdout.write(`${JSON.stringify(generateFixture(ps), null, 2)}\n`);
}

if (require.main === module) main();

module.exports = { generateFixture, main };
