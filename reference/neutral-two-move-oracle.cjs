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
const DUEL_SCENARIOS = [
	{
		id: 'complete-p1-faster',
		seed: [31, 41, 59, 26],
		p1: { level: 50, speedEvs: 252 },
		p2: { level: 50, speedEvs: 0 },
	},
	{
		id: 'complete-p2-faster',
		seed: [1, 2, 3, 4],
		p1: { level: 50, speedEvs: 0 },
		p2: { level: 50, speedEvs: 252 },
	},
	{
		id: 'complete-tied-speed',
		seed: [0, 0, 0, 0],
		p1: { level: 50, speedEvs: 0 },
		p2: { level: 50, speedEvs: 0 },
	},
	{
		id: 'complete-last-tackle-pp',
		seed: [9, 2, 3, 4],
		p1: { level: 50, speedEvs: 252 },
		p2: { level: 50, speedEvs: 0 },
		setup: { player: 'p1', slot: 1, pp: 1, regenerateRequest: true },
	},
];
const DUEL_TURN_LIMIT = 56;

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

function chooseRequestedSlot(request, plannedSlot, scenarioId, player) {
	const rows = request.own.moves;
	const planned = rows[plannedSlot - 1];
	assert.ok(planned, `${scenarioId} ${player} planned slot must exist`);
	if (!planned.disabled) return plannedSlot;
	const fallbackSlot = plannedSlot === 1 ? 2 : 1;
	assert.equal(rows[fallbackSlot - 1].disabled, false,
		`${scenarioId} ${player} must have a legal native slot when its plan is unavailable`);
	return fallbackSlot;
}

function captureDuelTurn(ps, battle, scenario, turnIndex) {
	assert.equal(battle.ended, false);
	const before = snapshot(battle);
	const requests = { p1: projectRequest(battle.p1), p2: projectRequest(battle.p2) };
	const plannedChoices = [1 + (turnIndex % 2), 1 + (turnIndex % 2)];
	const choices = [
		chooseRequestedSlot(requests.p1, plannedChoices[0], scenario.id, 'p1'),
		chooseRequestedSlot(requests.p2, plannedChoices[1], scenario.id, 'p2'),
	];
	const commands = choices.map(choice => `move ${choice}`);
	for (const [index, side] of [battle.p1, battle.p2].entries()) {
		assert.ok(side.activeRequest, `${scenario.id} p${index + 1} command must use a current native request`);
		assert.ok(requests[`p${index + 1}`].legalActions.includes(commands[index]),
			`${scenario.id} p${index + 1} command must be legal in its current native request`);
	}
	const logStart = battle.log.length;
	const { draws } = withObservedBattleDraws(ps, battle, () => battle.makeChoices(...commands));
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
		const expected = choices[actor] === 1 ? 'tackle' : 'pound';
		assert.equal(event.move, expected, `${scenario.id} turn ${turnIndex + 1} must execute its submitted native move`);
	}
	assert.equal(
		draws.filter(draw => ['accuracy', 'critical', 'damage'].includes(draw.context)).length,
		moveEvents.length * 3,
		'each executed move retains its three ordinary action draws',
	);
	assert.equal(draws.length, moveEvents.length * 3 + draws.filter(draw => draw.context === 'speed-sort').length);
	if (draws.length) {
		assert.deepEqual(draws[0].before, before.rng);
		assert.deepEqual(draws.at(-1).after, after.rng);
	}
	const nextRequests = battle.ended
		? { p1: null, p2: null }
		: { p1: projectRequest(battle.p1), p2: projectRequest(battle.p2) };
	return {
		index: turnIndex,
		plannedChoices,
		choices,
		commands,
		before,
		requests,
		draws,
		actionOrder,
		events,
		after,
		nextRequests,
		winner: sideWinner(battle),
	};
}

function duelScenarioFixture(ps, scenario) {
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
		const turns = [];
		while (!battle.ended && turns.length < DUEL_TURN_LIMIT) {
			turns.push(captureDuelTurn(ps, battle, scenario, turns.length));
		}
		assert.equal(battle.ended, true, `${scenario.id} must end in the real native battle within ${DUEL_TURN_LIMIT} turns`);
		assert.ok(battle.winner === 'p1' || battle.winner === 'p2', `${scenario.id} must have a p1/p2 winner`);
		assert.ok(turns.length <= DUEL_TURN_LIMIT);
		assert.ok(snapshot(battle).pp.every(slots => slots.some(pp => pp > 0)), `${scenario.id} must retain a move slot for both sides`);
		for (const player of [0, 1]) {
			const sequence = turns.flatMap(turn => turn.events.p1
				.filter(event => event.type === 'move' && event.actor === `p${player + 1}`)
				.map(event => event.move));
			assert.ok(sequence.indexOf('tackle') >= 0 && sequence.indexOf('pound') > sequence.indexOf('tackle'),
				`${scenario.id} p${player + 1} must execute Tackle followed by Pound on later turns`);
		}
		if (scenario.id === 'complete-tied-speed') {
			assert.equal(initial.state.speed[0], initial.state.speed[1]);
			assert.ok(turns.some(turn => turn.draws.some(draw => draw.context === 'speed-sort')),
				'tied native speeds must retain the native speed-sort draw');
		}
		if (scenario.id === 'complete-last-tackle-pp') {
			assert.equal(setupEvidence.engineApi, 'battle.makeRequest(\'move\')');
			assert.equal(initial.state.pp[0][0], 1);
			assert.ok(initial.state.pp[0][1] > 1, 'Pound must remain available after the disclosed Tackle PP setup');
			assert.equal(turns[0].choices[0], 1);
			assert.equal(turns[0].after.pp[0][0], 0);
			assert.equal(turns[0].nextRequests.p1.own.moves[0].disabled, true);
			assert.equal(turns[0].nextRequests.p1.own.moves[1].disabled, false);
			assert.equal(turns[1].choices[0], 2);
			assert.ok(turns[1].events.p1.some(event => event.type === 'move' && event.actor === 'p1' && event.move === 'pound'));
		}
		const terminal = {
			ended: battle.ended,
			winner: sideWinner(battle),
			state: snapshot(battle),
			requests: { p1: null, p2: null },
		};
		return {
			id: scenario.id,
			seed: [...scenario.seed],
			turnLimit: DUEL_TURN_LIMIT,
			initial,
			turns,
			terminal,
			setupEvidence,
		};
	} finally {
		battle.destroy();
	}
}

function generateDuelFixture(ps) {
	const scenarios = DUEL_SCENARIOS.map(scenario => duelScenarioFixture(ps, scenario));
	return {
		schemaVersion: 1,
		kind: 'gen5-restricted-neutral-two-move-complete-duel-oracle',
		upstream: { repository: 'smogon/pokemon-showdown', commit: '1242c1e2d9763ba5b85d33f13aecb89bae19ae3d' },
		formatId: FORMAT_ID,
		generation: 5,
		variant: VARIANT,
		moveOrder: ['tackle', 'pound'],
		moveFacts: assertMoveFacts(ps),
		initialization: 'Each case uses actual pinned Gen 5 Random Battle requests and direct one-Mew Synchronize/Hardy/perfect-IV/itemless sets with both moves in slot order. Each battle uses native Battle.makeChoices until its genuine terminal winner. The PP boundary changes p1 slot 1 through the live Pokemon and regenerates normal requests with battle.makeRequest(\'move\').',
		drawContract: 'Every turn records the original PRNG.prototype.random results, exact bounds, contexts, before/after limbs, and recipient-split native events. Executed moves retain accuracy, critical and damage draws; tied turns retain native speed-sort draws.',
		choiceContract: 'Each side plans Tackle on odd-numbered turns and Pound on even-numbered turns, falling back only to the other currently legal native request slot if its planned slot is disabled.',
		turnLimit: DUEL_TURN_LIMIT,
		scenarios,
	};
}

const SESSION_PAIR_SCENARIOS = [
	{ id: 'tackle-tackle-unequal', choices: [1, 1], seed: [31, 41, 59, 26], p1: { level: 50, speedEvs: 252 }, p2: { level: 50, speedEvs: 0 } },
	{ id: 'tackle-pound-unequal', choices: [1, 2], seed: [1, 2, 3, 4], p1: { level: 50, speedEvs: 0 }, p2: { level: 50, speedEvs: 252 } },
	{ id: 'pound-tackle-tied', choices: [2, 1], seed: [0, 0, 0, 0], p1: { level: 50, speedEvs: 0 }, p2: { level: 50, speedEvs: 0 } },
	{ id: 'pound-pound-tied', choices: [2, 2], seed: [9, 2, 3, 4], p1: { level: 50, speedEvs: 0 }, p2: { level: 50, speedEvs: 0 } },
];
const SESSION_SCENARIOS = [
	...SESSION_PAIR_SCENARIOS.flatMap(pair => ['p1', 'p2'].map(firstPlayer => ({
		...pair,
		id: `${pair.id}-first-${firstPlayer}`,
		firstPlayer,
	}))),
	{
		id: 'first-action-ko', firstPlayer: 'p1', choices: [2, 2], seed: [1, 2, 3, 4],
		p1: { level: 1, speedEvs: 0 }, p2: { level: 100, speedEvs: 0 },
	},
	{
		id: 'depleted-tackle-legal-pound', firstPlayer: 'p1', choices: [2, 1], seed: [9, 2, 3, 4],
		p1: { level: 50, speedEvs: 252 }, p2: { level: 50, speedEvs: 0 },
		setup: { player: 'p1', slot: 1, pp: 0, regenerateRequest: true },
	},
];

function captureStagedSessionTurn(ps, battle, scenario) {
	assert.equal(battle.ended, false);
	const before = snapshot(battle);
	const requests = { p1: projectRequest(battle.p1), p2: projectRequest(battle.p2) };
	const logStart = battle.log.length;
	const firstPlayer = scenario.firstPlayer;
	const firstCapture = withObservedBattleDraws(
		ps,
		battle,
		() => battle.choose(firstPlayer, `move ${scenario.choices[firstPlayer === 'p1' ? 0 : 1]}`),
	);
	assert.equal(firstCapture.result, true, `${scenario.id} ${firstPlayer} first commitment must be accepted`);
	assert.deepEqual(firstCapture.draws, [], `${scenario.id} first commitment must consume no RNG`);
	const afterFirst = snapshot(battle);
	assert.deepEqual(afterFirst, before, `${scenario.id} first commitment must preserve battle, PP and RNG`);
	const requestsAfterFirst = { p1: projectRequest(battle.p1), p2: projectRequest(battle.p2) };
	assert.deepEqual(requestsAfterFirst, requests, `${scenario.id} first commitment must leave both native requests unchanged`);
	const firstEvents = {
		p1: projectEvents(battle.log.slice(logStart), 'p1'),
		p2: projectEvents(battle.log.slice(logStart), 'p2'),
	};
	assert.deepEqual(firstEvents, { p1: [], p2: [] }, `${scenario.id} first commitment must emit no battle events`);
	assert.equal(battle.p1.choice.actions.length + battle.p2.choice.actions.length, 1, 'exactly one native private choice is held');

	const secondPlayer = firstPlayer === 'p1' ? 'p2' : 'p1';
	const secondCapture = withObservedBattleDraws(
		ps,
		battle,
		() => battle.choose(secondPlayer, `move ${scenario.choices[secondPlayer === 'p1' ? 0 : 1]}`),
	);
	assert.equal(secondCapture.result, true, `${scenario.id} second commitment must resolve the native turn`);
	const after = snapshot(battle);
	const events = {
		p1: projectEvents(battle.log.slice(logStart), 'p1'),
		p2: projectEvents(battle.log.slice(logStart), 'p2'),
	};
	const actionOrder = events.p1.filter(event => event.type === 'move').map(event => event.actor);
	assert.ok(actionOrder.length >= 1 && actionOrder.length <= 2);
	for (const event of events.p1.filter(event => event.type === 'move')) {
		const actor = event.actor === 'p1' ? 0 : 1;
		assert.equal(event.move, scenario.choices[actor] === 1 ? 'tackle' : 'pound');
	}
	const nextRequests = battle.ended ? { p1: null, p2: null } : {
		p1: projectRequest(battle.p1),
		p2: projectRequest(battle.p2),
	};
	assert.deepEqual(secondCapture.draws.length ? secondCapture.draws[0].before : before.rng, before.rng);
	if (secondCapture.draws.length) assert.deepEqual(secondCapture.draws.at(-1).after, after.rng);
	if (scenario.id === 'first-action-ko') {
		assert.equal(actionOrder.length, 1, 'first-action KO must skip the remaining move');
		assert.equal(after.terminal, true);
		assert.ok(after.hp.includes(0));
	}
	if (scenario.id === 'depleted-tackle-legal-pound') {
		assert.equal(scenario.setup.engineApi, 'battle.makeRequest(\'move\')');
		assert.equal(before.pp[0][0], 0);
		assert.equal(requests.p1.own.moves[0].disabled, true);
		assert.equal(requests.p1.own.moves[1].disabled, false);
		assert.deepEqual(
			events.p1.filter(event => event.type === 'move' && event.actor === 'p1').map(event => event.move),
			['pound'],
		);
	}
	return {
		firstPlayer,
		firstCommit: { accepted: firstCapture.result, draws: firstCapture.draws, before, after: afterFirst, requests, requestsAfter: requestsAfterFirst, events: firstEvents },
		secondPlayer,
		choices: [...scenario.choices],
		secondCommit: { accepted: secondCapture.result, draws: secondCapture.draws },
		actionOrder,
		events,
		after,
		nextRequests,
	};
}

function sessionScenarioFixture(ps, scenario) {
	const battle = createBattle(ps, scenario);
	try {
		let setupEvidence = null;
		if (scenario.setup) {
			const rngBefore = [...snapshot(battle).rng];
			battle[scenario.setup.player].active[0].moveSlots[scenario.setup.slot - 1].pp = scenario.setup.pp;
			battle.makeRequest('move');
			const rngAfter = [...snapshot(battle).rng];
			assert.deepEqual(rngAfter, rngBefore, 'request regeneration must not advance original PRNG limbs');
			setupEvidence = {
				player: scenario.setup.player,
				slot: scenario.setup.slot,
				pp: scenario.setup.pp,
				engineApi: 'battle.makeRequest(\'move\')',
				rngBefore,
				rngAfter,
			};
			scenario = { ...scenario, setup: setupEvidence };
		}
		const initial = {
			state: snapshot(battle),
			requests: { p1: projectRequest(battle.p1), p2: projectRequest(battle.p2) },
		};
		const turn = captureStagedSessionTurn(ps, battle, scenario);
		return { id: scenario.id, seed: [...scenario.seed], setup: setupEvidence, initial, turn };
	} finally {
		battle.destroy();
	}
}

function generateSessionFixture(ps) {
	const scenarios = SESSION_SCENARIOS.map(scenario => sessionScenarioFixture(ps, scenario));
	for (const pair of SESSION_PAIR_SCENARIOS) {
		const p1First = scenarios.find(scenario => scenario.id === `${pair.id}-first-p1`).turn;
		const p2First = scenarios.find(scenario => scenario.id === `${pair.id}-first-p2`).turn;
		assert.deepEqual(p1First.after, p2First.after, `${pair.id} resolution must not depend on first submitter`);
		assert.deepEqual(p1First.secondCommit.draws, p2First.secondCommit.draws, `${pair.id} draws must not depend on first submitter`);
		assert.deepEqual(p1First.actionOrder, p2First.actionOrder);
		assert.deepEqual(p1First.events, p2First.events);
		assert.deepEqual(p1First.nextRequests, p2First.nextRequests);
	}
	return {
		schemaVersion: 1,
		kind: 'gen5-restricted-neutral-two-move-staged-session-oracle',
		upstream: { repository: 'smogon/pokemon-showdown', commit: '1242c1e2d9763ba5b85d33f13aecb89bae19ae3d' },
		formatId: FORMAT_ID,
		generation: 5,
		variant: VARIANT,
		moveOrder: ['tackle', 'pound'],
		moveFacts: assertMoveFacts(ps),
		captureContract: 'Capture actual separate Battle.choose submissions and return original PRNG values unchanged. The first accepted commit leaves the full battle snapshot, both native requests, draw trace, and recipient events unchanged. The second accepted commit records exact state, all PP, draw bounds/contexts/limbs, action order, move identities, recipient event projections, and next requests. Paired same-state choices resolve identically in either submission order.',
		setupContract: 'The depleted Tackle case sets p1 slot 1 PP to zero through the live Pokemon, regenerates with battle.makeRequest(\'move\'), confirms RNG limbs are unchanged, and submits legal move 2 Pound. No invalid native command is issued.',
		scenarios,
	};
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

function main(argv = process.argv.slice(2)) {
	assertReady();
	const ps = require(API);
	if (argv.length === 1 && argv[0] === '--sessions') {
		process.stdout.write(`${JSON.stringify(generateSessionFixture(ps), null, 2)}\n`);
		return;
	}
	if (argv.length === 1 && argv[0] === '--duels') {
		process.stdout.write(`${JSON.stringify(generateDuelFixture(ps), null, 2)}\n`);
		return;
	}
	assert.deepEqual(argv, [], 'supported arguments: --duels or --sessions');
	process.stdout.write(`${JSON.stringify(generateFixture(ps), null, 2)}\n`);
}

if (require.main === module) main();

module.exports = { generateDuelFixture, generateFixture, main };
