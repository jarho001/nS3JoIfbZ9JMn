#!/usr/bin/env node
'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');
const { assertReady, projectEvents, snapshot, withObservedBattleDraws } = require('./neutral-turn-oracle.cjs');

const ROOT = path.resolve(__dirname, '..');
const LOCK = require(path.join(ROOT, 'reference-lock.json'));
const CHECKOUT = path.join(__dirname, 'vendor', 'pokemon-showdown');
const API = path.join(CHECKOUT, 'dist', 'sim');
const FORMAT_ID = 'gen5randombattle';

const SCENARIOS = [
	{
		id: 'p2-first-surviving', seed: [1, 2, 3, 4],
		p1: { level: 50, speedEvs: 0 }, p2: { level: 50, speedEvs: 252 },
	},
	{
		id: 'p1-first-surviving', seed: [31, 41, 59, 26],
		p1: { level: 50, speedEvs: 252 }, p2: { level: 50, speedEvs: 0 },
	},
	{
		id: 'equal-speed-tied-turn', seed: [1, 2, 3, 4],
		p1: { level: 50, speedEvs: 0 }, p2: { level: 50, speedEvs: 0 },
	},
	{
		id: 'native-critical-hit', seed: [9, 2, 3, 4],
		p1: { level: 50, speedEvs: 0 }, p2: { level: 50, speedEvs: 252 },
	},
	{
		id: 'first-action-ko', seed: [1, 2, 3, 4],
		p1: { level: 1, speedEvs: 0 }, p2: { level: 100, speedEvs: 0 },
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
		moves: ['Pound'],
		nature: 'Hardy',
		ivs: perfect,
		evs: { ...empty, spe: descriptor.speedEvs },
	};
}

function assertPoundDexFacts(ps) {
	const move = ps.Dex.mod('gen5').moves.get('pound');
	assert.equal(move.id, 'pound');
	assert.equal(move.num, 1);
	assert.equal(move.name, 'Pound');
	assert.equal(move.basePower, 40);
	assert.equal(move.accuracy, 100);
	assert.equal(move.category, 'Physical');
	assert.equal(move.type, 'Normal');
	assert.equal(move.pp, 35);
	assert.equal(move.critRatio, 1, 'Gen 5 default critical ratio is 1/16');
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
	assert.equal(battle.ruleTable.has('hppercentagemod'), true);
	for (const side of [battle.p1, battle.p2]) {
		assert.equal(side.active[0].moves.length, 1);
		assert.equal(side.active[0].moves[0], 'pound');
		assert.equal(side.active[0].item, '');
	}
	return battle;
}

function projectPoundRequest(side) {
	const request = side.activeRequest;
	assert.ok(request, `${side.id} must have a normal active request`);
	assert.equal(request.active.length, 1);
	assert.equal(request.side.id, side.id);
	const moves = request.active[0].moves;
	assert.equal(moves.length, 1, 'the restricted request must expose only Pound');
	const move = moves[0];
	assert.equal(move.id, 'pound');
	assert.equal(move.move, 'Pound');
	assert.equal(move.disabled, false);
	assert.ok(move.pp > 0);
	const own = request.side.pokemon.find(mon => mon.active);
	assert.ok(own, `${side.id} request must identify its active Mew`);
	return {
		phase: 'move',
		legalActions: ['move 1'],
		own: {
			species: own.details.split(',')[0],
			level: side.active[0].level,
			hpCondition: own.condition,
			stats: { atk: own.stats.atk, def: own.stats.def, spe: own.stats.spe },
			move: { name: move.move, id: move.id, pp: move.pp, disabled: move.disabled },
		},
	};
}

function captureTurn(ps, battle, scenarioId) {
	assert.equal(battle.ended, false);
	const before = snapshot(battle);
	const requests = { p1: projectPoundRequest(battle.p1), p2: projectPoundRequest(battle.p2) };
	const logStart = battle.log.length;
	const observed = withObservedBattleDraws(ps, battle, () => battle.makeChoices('move 1', 'move 1'));
	const logDelta = battle.log.slice(logStart);
	const events = { p1: projectEvents(logDelta, 'p1'), p2: projectEvents(logDelta, 'p2') };
	const after = snapshot(battle);
	const actionOrder = events.p1.filter(event => event.type === 'move').map(event => event.actor);
	assert.ok(actionOrder.length >= 1 && actionOrder.length <= 2, `${scenarioId} executes one or two actions`);
	assert.deepEqual(events.p2.filter(event => event.type === 'move').map(event => event.actor), actionOrder);
	for (const event of events.p1.filter(event => event.type === 'move')) assert.equal(event.move, 'pound');
	const actionDraws = observed.draws.filter(draw => draw.to === undefined);
	assert.equal(actionDraws.length, actionOrder.length * 3, 'each executed Pound consumes three action draws');
	for (let index = 0; index < actionDraws.length; index++) {
		const draw = actionDraws[index];
		assert.equal(draw.bound, [100, 16, 16][index % 3]);
		assert.equal(draw.context, ['accuracy', 'critical', 'damage'][index % 3]);
	}
	const tieDraws = observed.draws.filter(draw => draw.to !== undefined);
	if (before.speed[0] === before.speed[1]) {
		assert.ok(tieDraws.length > 0, 'equal speeds must produce native speed-tie draws');
		assert.ok(observed.draws.length <= 16, 'the tied turn fits the accepted fixed trace');
		for (const draw of tieDraws) {
			assert.equal(draw.bound, 0);
			assert.equal(draw.to, 2);
			assert.equal(draw.context, 'speed-sort');
			assert.notEqual(draw.phase, 'unsupported');
		}
	} else {
		assert.deepEqual(tieDraws, [], 'unequal speeds must not consume speed-tie draws');
	}
	assert.deepEqual(observed.draws[0].before, before.rng);
	assert.deepEqual(observed.draws.at(-1).after, after.rng);
	const nextRequests = battle.ended ? { p1: null, p2: null } : {
		p1: projectPoundRequest(battle.p1),
		p2: projectPoundRequest(battle.p2),
	};
	return {
		before,
		requests,
		choices: { p1: 'move 1', p2: 'move 1' },
		actionOrder,
		draws: observed.draws.map(draw => ({ ...draw, phase: draw.phase ?? draw.context })),
		events,
		after,
		nextRequests,
		views: {
			p1: { request: requests.p1, events: events.p1 },
			p2: { request: requests.p2, events: events.p2 },
		},
	};
}

function scenarioFixture(ps, scenario) {
	const battle = createBattle(ps, scenario);
	try {
		const initial = {
			state: snapshot(battle),
			requests: { p1: projectPoundRequest(battle.p1), p2: projectPoundRequest(battle.p2) },
		};
		const turn = captureTurn(ps, battle, scenario.id);
		if (scenario.id === 'native-critical-hit') {
			assert.ok(turn.events.p1.some(event => event.type === 'critical'), 'seed [9,2,3,4] must produce an actual Pound critical');
		}
		if (scenario.id === 'first-action-ko') {
			assert.equal(turn.actionOrder.length, 1, 'the faster level-100 Mew must knock out the level-1 Mew first');
			assert.equal(turn.after.terminal, true);
			assert.equal(turn.events.p1.some(event => event.type === 'faint'), true);
		}
		return { id: scenario.id, seed: [...scenario.seed], initial, turns: [turn] };
	} finally {
		battle.destroy();
	}
}

function generateFixture(ps) {
	assertPoundDexFacts(ps);
	return {
		schemaVersion: 1,
		kind: 'gen5-restricted-neutral-pound-turn-oracle',
		upstream: { repository: LOCK.repository, commit: LOCK.commit },
		formatId: FORMAT_ID,
		generation: 5,
		variant: 'gen5-neutral-pound-mew-1v1-v3',
		initialization: 'Each case uses a normal pinned Gen 5 Random Battle initialized from directly supplied one-Mew sets. Random team generation is bypassed; each Hardy Mew has Synchronize, no item, perfect IVs, and Pound only. Each ordinary turn submits actual move 1 for both sides.',
		mechanics: 'Pinned Gen 5 Pound is Normal, physical, 40 base power, accuracy 100, 35 base PP, and default critical ratio 1/16. Both Mew are Psychic, so Pound is neutral and never STAB. No status, item, weather, boost, type modifier, or relevant passive effect is present.',
		drawContract: 'PRNG.prototype.random is wrapped only to record and return each original result. Every executed Pound consumes accuracy random(100), critical random(16), and damage random(16). Equal-speed native queue and phase draws retain their original bounds, contexts, stack and before/after limbs.',
		viewContract: 'Each fixture records the actual normal Pound-only requests and the recipient-selected split-event projection for p1 and p2 before and after the turn. Trusted state, raw packets, and unprojected protocol are not actor views.',
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
