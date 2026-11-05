#!/usr/bin/env node
'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');
const { assertReady, snapshot: baseSnapshot, withObservedBattleDraws } = require('./neutral-turn-oracle.cjs');
const { projectMegaPunchEvents } = require('./neutral-mega-punch-oracle.cjs');

const ROOT = path.resolve(__dirname, '..');
const LOCK = require(path.join(ROOT, 'reference-lock.json'));
const API = path.join(__dirname, 'vendor', 'pokemon-showdown', 'dist', 'sim');
const FORMAT_ID = 'gen5randombattle';
const VARIANT = 'gen5-accuracy-choice-mew-1v1-v6';
const SCENARIOS = [
	{ id: 'tackle-tackle-p1-faster', seed: [0, 2, 3, 4], p1: { level: 50, speedEvs: 252 }, p2: { level: 50, speedEvs: 0 }, choices: { p1: 1, p2: 1 } },
	{ id: 'tackle-mega-punch-p1-faster', seed: [1, 2, 3, 4], p1: { level: 50, speedEvs: 252 }, p2: { level: 50, speedEvs: 0 }, choices: { p1: 1, p2: 2 } },
	{ id: 'mega-punch-tackle-p2-faster', seed: [1, 2, 3, 4], p1: { level: 50, speedEvs: 0 }, p2: { level: 50, speedEvs: 252 }, choices: { p1: 2, p2: 1 } },
	{ id: 'mega-punch-mega-punch-p2-faster', seed: [1, 2, 3, 4], p1: { level: 50, speedEvs: 0 }, p2: { level: 50, speedEvs: 252 }, choices: { p1: 2, p2: 2 } },
	{ id: 'last-mega-punch-misses-then-tackle-remains', seed: [0, 2, 3, 4], p1: { level: 50, speedEvs: 252, pp: { megapunch: 1 } }, p2: { level: 50, speedEvs: 0 }, choices: { p1: 2, p2: 1 }, continuation: { p1: 1, p2: 2 } },
	{ id: 'mega-punch-critical-seed-21', seed: [21, 2, 3, 4], p1: { level: 50, speedEvs: 0 }, p2: { level: 50, speedEvs: 252 }, choices: { p1: 1, p2: 2 } },
	{ id: 'first-action-ko-skips-other-choice', seed: [21, 2, 3, 4], p1: { level: 100, speedEvs: 252 }, p2: { level: 1, speedEvs: 0 }, choices: { p1: 2, p2: 1 } },
	{ id: 'last-tackle-pp-leaves-mega-punch', seed: [1, 2, 3, 4], p1: { level: 50, speedEvs: 252, pp: { tackle: 1 } }, p2: { level: 50, speedEvs: 0 }, choices: { p1: 1, p2: 2 } },
];

function assertMoveFacts(ps) {
	assert.equal(LOCK.commit, '1242c1e2d9763ba5b85d33f13aecb89bae19ae3d');
	const tackle = ps.Dex.mod('gen5').moves.get('tackle');
	const megaPunch = ps.Dex.mod('gen5').moves.get('megapunch');
	assert.deepEqual(
		[tackle.id, tackle.name, tackle.type, tackle.category, tackle.basePower, tackle.accuracy, tackle.pp],
		['tackle', 'Tackle', 'Normal', 'Physical', 50, 100, 35],
	);
	assert.deepEqual(
		[megaPunch.id, megaPunch.name, megaPunch.type, megaPunch.category, megaPunch.basePower, megaPunch.accuracy, megaPunch.pp, megaPunch.critRatio],
		['megapunch', 'Mega Punch', 'Normal', 'Physical', 80, 85, 20, 1],
	);
	return {
		tackle: { id: tackle.id, basePower: tackle.basePower, accuracy: tackle.accuracy, basePp: tackle.pp, maxPp: 56 },
		megaPunch: { id: megaPunch.id, basePower: megaPunch.basePower, accuracy: megaPunch.accuracy, basePp: megaPunch.pp, maxPp: 32, criticalRatio: megaPunch.critRatio },
	};
}

function makeSet(descriptor) {
	const empty = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
	const perfect = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
	return {
		species: 'Mew', ability: 'Synchronize', item: '', level: descriptor.level,
		moves: ['Tackle', 'Mega Punch'], nature: 'Hardy', ivs: perfect,
		evs: { ...empty, spe: descriptor.speedEvs },
	};
}

function createBattle(ps, scenario) {
	const battle = new ps.Battle({
		formatid: FORMAT_ID, seed: [...scenario.seed],
		p1: { name: 'p1', team: [makeSet(scenario.p1)] },
		p2: { name: 'p2', team: [makeSet(scenario.p2)] },
	});
	assert.equal(battle.gen, 5);
	assert.equal(battle.gameType, 'singles');
	assert.equal(battle.ruleTable.has('hppercentagemod'), true);
	for (const side of [battle.p1, battle.p2]) {
		assert.equal(side.pokemon.length, 1);
		assert.deepEqual(side.active[0].moves, ['tackle', 'megapunch']);
		assert.equal(side.active[0].item, '');
		assert.equal(side.active[0].ability, 'synchronize');
		assert.equal(side.pokemon[0].moveSlots[0].pp, 56, 'Tackle receives three PP Ups');
		assert.equal(side.pokemon[0].moveSlots[1].pp, 32, 'Mega Punch receives three PP Ups');
	}
	return battle;
}

function snapshot(battle) {
	const result = baseSnapshot(battle);
	const pokemon = [battle.p1.pokemon[0], battle.p2.pokemon[0]];
	return { ...result, pp: pokemon.map(mon => mon.moveSlots.map(slot => slot.pp)) };
}

function projectRequest(side) {
	const request = side.activeRequest;
	assert.ok(request, `${side.id} must have a live request`);
	assert.equal(request.active.length, 1);
	assert.equal(request.side.id, side.id);
	const own = request.side.pokemon.find(mon => mon.active);
	assert.ok(own, `${side.id} request must identify its own active Mew`);
	const rows = request.active[0].moves;
	const slots = side.pokemon[0].moveSlots;
	if (slots.every(slot => slot.pp === 0)) {
		assert.equal(rows.length, 1);
		assert.equal(rows[0].id, 'struggle');
		assert.equal(rows[0].disabled, false);
		return { phase: 'move', boundary: 'native-struggle-no-pp', legalActions: ['move 1'], moves: [{ id: 'struggle', name: 'Struggle', pp: rows[0].pp, disabled: false }] };
	}
	assert.equal(rows.length, 2, 'ordinary requests retain both move rows');
	const moveRows = rows.map((move, index) => {
		assert.equal(move.id, ['tackle', 'megapunch'][index]);
		assert.equal(move.pp, slots[index].pp);
		assert.equal(move.disabled, slots[index].pp === 0);
		return { id: move.id, name: move.move, pp: move.pp, disabled: move.disabled };
	});
	return {
		phase: 'move', boundary: 'accuracy-choice',
		legalActions: moveRows.flatMap((move, index) => move.disabled ? [] : [`move ${index + 1}`]),
		own: { species: own.details.split(',')[0], level: side.active[0].level, hpCondition: own.condition, stats: { atk: side.active[0].storedStats.atk, def: side.active[0].storedStats.def, spe: side.active[0].speed } },
		moves: moveRows,
	};
}

function choiceToken(value) {
	assert.ok(value === 1 || value === 2);
	return `move ${value}`;
}

function applyPpSetup(battle, scenario) {
	const configured = ['p1', 'p2'].some(id => scenario[id].pp);
	if (!configured) return;
	const before = snapshot(battle);
	for (const id of ['p1', 'p2']) {
		const settings = scenario[id].pp;
		if (!settings) continue;
		const side = battle[id];
		for (const [moveId, pp] of Object.entries(settings)) {
			const slot = side.pokemon[0].moveSlots.find(candidate => candidate.id === moveId);
			assert.ok(slot);
			slot.pp = pp;
		}
	}
	battle.makeRequest('move');
	assert.deepEqual(snapshot(battle).rng, before.rng, 'live PP/request setup must leave original PRNG unchanged');
}

function captureTurn(ps, battle, choices, scenarioId) {
	assert.equal(battle.ended, false);
	const before = snapshot(battle);
	const requests = { p1: projectRequest(battle.p1), p2: projectRequest(battle.p2) };
	const logStart = battle.log.length;
	const { draws } = withObservedBattleDraws(ps, battle, () => battle.makeChoices(choiceToken(choices.p1), choiceToken(choices.p2)));
	const logDelta = battle.log.slice(logStart);
	const events = { p1: projectMegaPunchEvents(logDelta, 'p1'), p2: projectMegaPunchEvents(logDelta, 'p2') };
	const after = snapshot(battle);
	const actionOrder = events.p1.filter(event => event.type === 'move').map(event => event.actor);
	const misses = new Set(events.p1.filter(event => event.type === 'miss').map(event => event.actor));
	const contexts = [];
	for (const actor of actionOrder) {
		contexts.push('accuracy');
		if (!misses.has(actor)) contexts.push('critical', 'damage');
	}
	assert.equal(draws.length, contexts.length, `${scenarioId} draw count follows attempted selected actions`);
	for (let index = 0; index < draws.length; index++) {
		assert.equal(draws[index].context, contexts[index]);
		assert.equal(draws[index].bound, { accuracy: 100, critical: 16, damage: 16 }[contexts[index]]);
	}
	if (draws.length) {
		assert.deepEqual(draws[0].before, before.rng);
		assert.deepEqual(draws.at(-1).after, after.rng);
	}
	const nextRequests = battle.ended ? { p1: null, p2: null } : { p1: projectRequest(battle.p1), p2: projectRequest(battle.p2) };
	return { before, requests, choices, actionOrder, draws, events, after, nextRequests };
}

function scenarioFixture(ps, scenario) {
	const battle = createBattle(ps, scenario);
	try {
		applyPpSetup(battle, scenario);
		const turn = captureTurn(ps, battle, scenario.choices, scenario.id);
		let continuation = null;
		if (scenario.continuation) {
			assert.equal(turn.after.terminal, false);
			assert.equal(turn.after.pp[0][1], 0, 'follow-up begins with exhausted Mega Punch');
			assert.ok(turn.after.pp[0][0] > 0, 'follow-up must retain Tackle');
			continuation = captureTurn(ps, battle, scenario.continuation, `${scenario.id}-follow-up`);
			assert.equal(continuation.before.pp[0][1], 0);
			assert.equal(continuation.choices.p1, 1, 'native follow-up selects the remaining Tackle');
		}
		return { id: scenario.id, seed: [...scenario.seed], choices: scenario.choices, sets: { p1: scenario.p1, p2: scenario.p2 }, turn, continuation };
	} finally {
		battle.destroy();
	}
}

function allPpZeroBoundary(ps) {
	const scenario = { seed: [1, 2, 3, 4], p1: { level: 50, speedEvs: 252 }, p2: { level: 50, speedEvs: 0 } };
	const battle = createBattle(ps, scenario);
	try {
		const before = snapshot(battle);
		for (const slot of battle.p1.pokemon[0].moveSlots) slot.pp = 0;
		battle.makeRequest('move');
		assert.deepEqual(snapshot(battle).rng, before.rng);
		const request = projectRequest(battle.p1);
		assert.equal(request.boundary, 'native-struggle-no-pp');
		return { nativeRequest: request, jaxBoundary: 'STATUS_UNSUPPORTED with unchanged typed v6 state; Struggle is not executed' };
	} finally {
		battle.destroy();
	}
}

function generateFixture(ps) {
	const moveFacts = assertMoveFacts(ps);
	const scenarios = SCENARIOS.map(scenario => scenarioFixture(ps, scenario));
	const miss = scenarios.find(scenario => scenario.id === 'last-mega-punch-misses-then-tackle-remains');
	assert.ok(miss.turn.events.p1.some(event => event.type === 'miss' && event.actor === 'p1'));
	assert.ok(miss.turn.events.p1.some(event => event.type === 'move' && event.actor === 'p2' && event.move === 'tackle'));
	assert.equal(miss.turn.events.p1.some(event => event.type === 'miss' && event.actor === 'p2'), false, 'Tackle hits at 100 accuracy on the same accuracy roll');
	assert.equal(miss.turn.draws[0].bound, 100);
	assert.equal(miss.turn.draws[0].context, 'accuracy');
	assert.equal(miss.turn.draws[0].value, 93, 'seed [0,2,3,4] first accuracy is 93');
	const tackleAtSameRoll = scenarios.find(scenario => scenario.id === 'tackle-tackle-p1-faster');
	assert.equal(tackleAtSameRoll.turn.draws[0].value, miss.turn.draws[0].value);
	assert.equal(tackleAtSameRoll.turn.events.p1.some(event => event.type === 'miss'), false);
	assert.ok(miss.continuation);
	const critical = scenarios.find(scenario => scenario.id === 'mega-punch-critical-seed-21');
	assert.ok(critical.turn.events.p1.some(event => event.type === 'critical'));
	assert.ok(critical.turn.events.p2.some(event => event.type === 'critical'));
	const ko = scenarios.find(scenario => scenario.id === 'first-action-ko-skips-other-choice');
	assert.equal(ko.turn.actionOrder.length, 1);
	const lastTackle = scenarios.find(scenario => scenario.id === 'last-tackle-pp-leaves-mega-punch');
	assert.equal(lastTackle.turn.after.pp[0][0], 0);
	assert.ok(lastTackle.turn.after.pp[0][1] > 0);
	return {
		schemaVersion: 1,
		kind: 'gen5-restricted-accuracy-choice-turn-oracle',
		upstream: { repository: LOCK.repository, commit: LOCK.commit },
		formatId: FORMAT_ID,
		generation: 5,
		variant: VARIANT,
		moveFacts,
		initialization: 'Unmodified pinned gen5randombattle with directly supplied one-Mew Synchronize, Hardy, perfect-IV, itemless sets; random team generation bypassed. Each normal request retains Tackle and Mega Punch in slots 1 and 2 with live PP 56 and 32.',
		requestContract: 'Each ordinary request preserves both rows and disables a slot after its live PP reaches zero. At both slots exhausted, native requests expose Struggle while v6 returns typed STATUS_UNSUPPORTED and never executes Struggle.',
		drawContract: 'The guarded native PRNG observer returns original values unchanged. Each attempted action consumes random(100); a miss consumes no critical or damage call, while a hit consumes random(16) critical and random(16) damage. Native split-selected move, miss, critical, damage, faint, turn and winner events are retained for both recipients.',
		scenarios,
		allPpZeroBoundary: allPpZeroBoundary(ps),
	};
}

function main() {
	assertReady();
	const ps = require(API);
	process.stdout.write(`${JSON.stringify(generateFixture(ps), null, 2)}\n`);
}

if (require.main === module) main();
module.exports = { generateFixture, main };
