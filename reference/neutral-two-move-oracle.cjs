#!/usr/bin/env node
'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');
const {
	assertReady,
	projectEvents,
	snapshot: baseSnapshot,
	withObservedBattleDraws,
} = require('./neutral-turn-oracle.cjs');

const API = path.join(__dirname, 'vendor', 'pokemon-showdown', 'dist', 'sim');
const FORMAT_ID = 'gen5randombattle';
const VARIANT = 'gen5-neutral-two-move-mew-1v1-v4';
const MOVE_IDS = { tackle: 33, pound: 1 };
const MOVE_POWER = { tackle: 50, pound: 40 };
const SCENARIOS = [
	{
		id: 'tackle-tackle-p2-first-critical',
		choices: [1, 1],
		seed: [9, 2, 3, 4],
		p1: { level: 50, speedEvs: 0 },
		p2: { level: 50, speedEvs: 252 },
	},
	{
		id: 'tackle-pound-p1-first',
		choices: [1, 2],
		seed: [31, 41, 59, 26],
		p1: { level: 50, speedEvs: 252 },
		p2: { level: 50, speedEvs: 0 },
	},
	{
		id: 'pound-tackle-p2-first',
		choices: [2, 1],
		seed: [1, 2, 3, 4],
		p1: { level: 50, speedEvs: 0 },
		p2: { level: 50, speedEvs: 252 },
	},
	{
		id: 'pound-pound-p1-first',
		choices: [2, 2],
		seed: [0, 65535, 0, 65535],
		p1: { level: 50, speedEvs: 252 },
		p2: { level: 50, speedEvs: 0 },
	},
	{
		id: 'tied-mixed-choice',
		choices: [1, 2],
		seed: [0, 0, 0, 0],
		p1: { level: 50, speedEvs: 0 },
		p2: { level: 50, speedEvs: 0 },
	},
	{
		id: 'first-action-ko',
		choices: [2, 2],
		seed: [1, 2, 3, 4],
		p1: { level: 1, speedEvs: 0 },
		p2: { level: 100, speedEvs: 0 },
	},
	{
		id: 'last-tackle-pp',
		choices: [1, 2],
		seed: [1, 2, 3, 4],
		p1: { level: 50, speedEvs: 0 },
		p2: { level: 50, speedEvs: 252 },
		setup: { player: 'p1', slot: 1, pp: 1, regenerateRequest: true },
	},
];

function makeSet(descriptor) {
	const empty = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
	const perfect = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
	return {
		species: 'Mew',
		ability: 'Synchronize',
		item: '',
		level: descriptor.level,
		moves: ['Tackle', 'Pound'],
		nature: 'Hardy',
		ivs: perfect,
		evs: { ...empty, spe: descriptor.speedEvs },
	};
}

function createBattle(ps, scenario) {
	const battle = new ps.Battle({
		formatid: FORMAT_ID,
		seed: [...scenario.seed],
		p1: { name: 'p1', team: [makeSet(scenario.p1)] },
		p2: { name: 'p2', team: [makeSet(scenario.p2)] },
	});
	assert.equal(battle.gen, 5);
	assert.equal(battle.gameType, 'singles');
	assert.equal(battle.p1.pokemon.length, 1);
	assert.equal(battle.p2.pokemon.length, 1);
	assert.equal(battle.p1.active[0].moves.length, 2);
	assert.equal(battle.p2.active[0].moves.length, 2);
	for (const side of [battle.p1, battle.p2]) {
		assert.deepEqual(side.active[0].moves, ['tackle', 'pound']);
		assert.equal(side.active[0].item, '');
	}
	return battle;
}

function snapshot(battle) {
	const state = baseSnapshot(battle);
	state.pp = [battle.p1, battle.p2].map(side => side.active[0].moveSlots.map(slot => slot.pp));
	state.moveIds = [battle.p1, battle.p2].map(side => side.active[0].moveSlots.map(slot => slot.id));
	return state;
}

function projectRequest(side) {
	const request = side.activeRequest;
	assert.ok(request, `${side.id} must have a native active request`);
	assert.equal(request.active.length, 1);
	assert.equal(request.side.id, side.id);
	const rows = request.active[0].moves;
	assert.equal(rows.length, 2, 'normal request must retain both move-slot rows');
	assert.deepEqual(rows.map(move => move.id), ['tackle', 'pound']);
	const moves = rows.map((move, index) => ({
		slot: index + 1,
		name: move.move,
		id: move.id,
		pp: move.pp,
		disabled: Boolean(move.disabled),
	}));
	const own = request.side.pokemon.find(mon => mon.active);
	assert.ok(own, `${side.id} request identifies its active Pokemon`);
	return {
		phase: 'move',
		legalActions: moves.filter(move => !move.disabled).map(move => `move ${move.slot}`),
		own: {
			species: own.details.split(',')[0],
			level: side.active[0].level,
			hpCondition: own.condition,
			stats: { atk: own.stats.atk, def: own.stats.def, spe: own.stats.spe },
			moves,
		},
	};
}

function sideWinner(battle) {
	if (!battle.ended) return -1;
	if (battle.winner === 'p1') return 1;
	if (battle.winner === 'p2') return 2;
	return 0;
}

function captureTurn(ps, battle, scenario) {
	assert.equal(battle.ended, false);
	const before = snapshot(battle);
	const requests = { p1: projectRequest(battle.p1), p2: projectRequest(battle.p2) };
	const logStart = battle.log.length;
	const command = scenario.choices.map(choice => `move ${choice}`);
	const { draws } = withObservedBattleDraws(ps, battle, () => battle.makeChoices(...command));
	const logDelta = battle.log.slice(logStart);
	const events = {
		p1: projectEvents(logDelta, 'p1'),
		p2: projectEvents(logDelta, 'p2'),
	};
	const after = snapshot(battle);
	const moveEvents = events.p1.filter(event => event.type === 'move');
	const actionOrder = moveEvents.map(event => event.actor);
	assert.ok(actionOrder.length >= 1 && actionOrder.length <= 2);
	for (const event of moveEvents) {
		const actor = event.actor === 'p1' ? 0 : 1;
		const expected = scenario.choices[actor] === 1 ? 'tackle' : 'pound';
		assert.equal(event.move, expected, `${scenario.id} must execute the selected native move identity`);
	}
	assert.equal(
		draws.filter(draw => ['accuracy', 'critical', 'damage'].includes(draw.context)).length,
		moveEvents.length * 3,
		'each executed move retains its three ordinary action draws',
	);
	assert.equal(draws.length, moveEvents.length * 3 + draws.filter(draw => draw.context === 'speed-sort').length);
	assert.deepEqual(draws[0].before, before.rng);
	assert.deepEqual(draws.at(-1).after, after.rng);
	const nextRequests = battle.ended
		? { p1: null, p2: null }
		: { p1: projectRequest(battle.p1), p2: projectRequest(battle.p2) };
	return {
		choices: [...scenario.choices],
		before,
		requests,
		draws,
		actionOrder,
		events,
		after,
		nextRequests,
		winner: sideWinner(battle),
		setup: scenario.setup ? { ...scenario.setup } : null,
	};
}

function scenarioFixture(ps, scenario) {
	const battle = createBattle(ps, scenario);
	try {
		let setupEvidence = null;
		if (scenario.setup) {
			const setupRngBefore = [...snapshot(battle).rng];
			const side = battle[scenario.setup.player];
			side.active[0].moveSlots[scenario.setup.slot - 1].pp = scenario.setup.pp;
			battle.makeRequest('move');
			const setupRngAfter = [...snapshot(battle).rng];
			assert.deepEqual(setupRngAfter, setupRngBefore, 'native request regeneration must not advance the PRNG');
			setupEvidence = { rngBefore: setupRngBefore, rngAfter: setupRngAfter, engineApi: 'battle.makeRequest(\'move\')' };
		}
		const initial = {
			state: snapshot(battle),
			requests: { p1: projectRequest(battle.p1), p2: projectRequest(battle.p2) },
		};
		const turn = captureTurn(ps, battle, scenario);
		if (scenario.id === 'tackle-tackle-p2-first-critical') {
			assert.equal(turn.actionOrder[0], 'p2');
			assert.ok(turn.events.p1.some(event => event.type === 'critical'), 'the pinned seed must produce an actual critical');
		}
		if (scenario.id === 'tied-mixed-choice') {
			assert.equal(turn.before.speed[0], turn.before.speed[1]);
			assert.ok(turn.draws.some(draw => draw.context === 'speed-sort'));
		}
		if (scenario.id === 'first-action-ko') {
			assert.equal(turn.actionOrder.length, 1, 'native first-action KO skips the other move');
			assert.equal(turn.after.terminal, true);
			assert.equal(turn.after.hp.includes(0), true);
		}
		if (scenario.id === 'last-tackle-pp') {
			assert.equal(setupEvidence.engineApi, 'battle.makeRequest(\'move\')');
			assert.equal(initial.state.pp[0][0], 1);
			assert.equal(turn.after.pp[0][0], 0);
			assert.equal(turn.after.pp[0][1], initial.state.pp[0][1]);
			assert.equal(turn.nextRequests.p1.own.moves[0].disabled, true);
			assert.equal(turn.nextRequests.p1.own.moves[1].disabled, false);
		}
		return {
			id: scenario.id,
			seed: [...scenario.seed],
			initial,
			turn,
			setupEvidence,
		};
	} finally {
		battle.destroy();
	}
}

function assertMoveFacts(ps) {
	const facts = {};
	for (const id of ['tackle', 'pound']) {
		const move = ps.Dex.mod('gen5').moves.get(id);
		assert.equal(move.id, id);
		assert.equal(move.type, 'Normal');
		assert.equal(move.category, 'Physical');
		assert.equal(move.basePower, MOVE_POWER[id]);
		assert.equal(move.accuracy, 100);
		assert.equal(move.pp, 35);
		assert.equal(move.critRatio, 1);
		assert.equal(move.priority, 0);
		facts[id] = {
			moveId: MOVE_IDS[id],
			type: move.type,
			category: move.category,
			basePower: move.basePower,
			accuracy: move.accuracy,
			basePp: move.pp,
			criticalRatio: move.critRatio,
			priority: move.priority,
		};
	}
	return facts;
}

function generateFixture(ps) {
	return {
		schemaVersion: 1,
		kind: 'gen5-restricted-neutral-two-move-turn-oracle',
		upstream: { repository: 'smogon/pokemon-showdown', commit: '1242c1e2d9763ba5b85d33f13aecb89bae19ae3d' },
		formatId: FORMAT_ID,
		generation: 5,
		variant: VARIANT,
		moveOrder: ['tackle', 'pound'],
		moveFacts: assertMoveFacts(ps),
		initialization: 'Each case uses a normal pinned Gen 5 Random Battle with directly supplied one-Mew Synchronize/Hardy/perfect-IV/itemless sets and both actual moves in slot order. The PP boundary changes slot 1 PP through the live Pokemon and regenerates normal requests with battle.makeRequest(\'move\').',
		drawContract: 'The imported guarded helper records and returns original PRNG.prototype.random results, including exact bounds, contexts and before/after limbs. Executed moves retain accuracy, critical and damage draws; tied turns retain native speed-sort draws.',
		requestContract: 'Both native move rows remain present in every nonterminal request; PP zero is represented by disabled:true while another move remains legal.',
		scenarios: SCENARIOS.map(scenario => scenarioFixture(ps, scenario)),
	};
}

function main() {
	assertReady();
	const ps = require(API);
	process.stdout.write(`${JSON.stringify(generateFixture(ps), null, 2)}\n`);
}

if (require.main === module) main();

module.exports = { generateFixture, main };
