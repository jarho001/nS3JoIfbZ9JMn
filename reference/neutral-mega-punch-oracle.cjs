#!/usr/bin/env node
'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');
const {
	assertReady,
	snapshot: baseSnapshot,
	withObservedBattleDraws,
} = require('./neutral-turn-oracle.cjs');

const ROOT = path.resolve(__dirname, '..');
const LOCK = require(path.join(ROOT, 'reference-lock.json'));
const API = path.join(__dirname, 'vendor', 'pokemon-showdown', 'dist', 'sim');
const FORMAT_ID = 'gen5randombattle';
const VARIANT = 'gen5-neutral-mega-punch-mew-1v1-v5';
const SCENARIOS = [
	{ id: 'p1-first-normal-hit', seed: [1, 2, 3, 4], p1: { level: 50, speedEvs: 252 }, p2: { level: 50, speedEvs: 0 } },
	{ id: 'p2-first-critical-hit', seed: [21, 2, 3, 4], p1: { level: 50, speedEvs: 0 }, p2: { level: 50, speedEvs: 252 } },
	{ id: 'p1-first-misses', seed: [0, 2, 3, 4], p1: { level: 50, speedEvs: 252 }, p2: { level: 50, speedEvs: 0 } },
	{ id: 'p2-first-misses', seed: [0, 2, 3, 4], p1: { level: 50, speedEvs: 0 }, p2: { level: 50, speedEvs: 252 } },
	{ id: 'first-action-ko-skips-p2', seed: [21, 2, 3, 4], p1: { level: 100, speedEvs: 0 }, p2: { level: 1, speedEvs: 0 } },
	{ id: 'last-pp-miss-then-opponent-acts', seed: [0, 2, 3, 4], p1: { level: 50, speedEvs: 252, pp: 1 }, p2: { level: 50, speedEvs: 0 } },
];

function assertMegaPunchFacts(ps) {
	assert.equal(LOCK.commit, '1242c1e2d9763ba5b85d33f13aecb89bae19ae3d');
	const move = ps.Dex.mod('gen5').moves.get('megapunch');
	assert.equal(move.name, 'Mega Punch');
	assert.equal(move.type, 'Normal');
	assert.equal(move.category, 'Physical');
	assert.equal(move.basePower, 80);
	assert.equal(move.accuracy, 85);
	assert.equal(move.pp, 20);
	assert.equal(move.critRatio, 1);
	return {
		name: move.name,
		id: move.id,
		type: move.type,
		category: move.category,
		basePower: move.basePower,
		accuracy: move.accuracy,
		basePp: move.pp,
		maxPp: move.pp * 1.6,
		criticalRatio: move.critRatio,
	};
}

function makeSet(descriptor) {
	const empty = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
	const perfect = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
	return {
		species: 'Mew',
		ability: 'Synchronize',
		item: '',
		level: descriptor.level,
		moves: ['Mega Punch'],
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
	assert.equal(battle.ruleTable.has('hppercentagemod'), true);
	for (const side of [battle.p1, battle.p2]) {
		assert.equal(side.pokemon.length, 1);
		assert.equal(side.active[0].moves.length, 1);
		assert.equal(side.active[0].moves[0], 'megapunch');
		assert.equal(side.active[0].item, '');
		assert.equal(side.active[0].ability, 'synchronize');
	}
	return battle;
}

function snapshot(battle) {
	return baseSnapshot(battle);
}

function projectRequest(side) {
	const request = side.activeRequest;
	assert.ok(request, `${side.id} must have an active request before the turn`);
	assert.equal(request.active.length, 1);
	assert.equal(request.side.id, side.id);
	const moveRows = request.active[0].moves;
	assert.equal(moveRows.length, 1, 'restricted request must contain exactly one move row');
	const move = moveRows[0];
	const own = request.side.pokemon.find(mon => mon.active);
	assert.ok(own, `${side.id} request must identify its own active Mew`);
	const struggleBoundary = move.id === 'struggle';
	if (struggleBoundary) {
		assert.equal(move.move, 'Struggle');
		assert.equal(move.disabled, false);
		assert.equal(side.pokemon[0].moveSlots.every(candidate => candidate.pp === 0), true);
	} else {
		assert.equal(move.id, 'megapunch');
		assert.equal(move.disabled, false);
		assert.ok(move.pp > 0);
		assert.equal(move.pp, side.pokemon[0].moveSlots[0].pp);
	}
	return {
		phase: 'move',
		legalActions: ['move 1'],
		boundary: struggleBoundary ? 'struggle-no-pp' : 'mega-punch',
		own: {
			species: own.details.split(',')[0],
			level: side.active[0].level,
			hpCondition: own.condition,
			stats: { atk: own.stats.atk, def: own.stats.def, spe: own.stats.spe },
			move: { name: move.move, id: move.id, pp: move.pp, disabled: move.disabled },
		},
	};
}

function sideFromIdent(ident) {
	if (typeof ident !== 'string') return null;
	if (ident.startsWith('p1')) return 'p1';
	if (ident.startsWith('p2')) return 'p2';
	return null;
}

function semanticEvent(line) {
	const fields = line.split('|');
	switch (fields[1]) {
	case 'move':
		return { type: 'move', actor: sideFromIdent(fields[2]), move: fields[3].toLowerCase(), target: sideFromIdent(fields[4]) };
	case '-miss':
		return { type: 'miss', actor: sideFromIdent(fields[2]), target: sideFromIdent(fields[3]) };
	case '-crit':
		return { type: 'critical', target: sideFromIdent(fields[2]) };
	case '-damage':
		return { type: 'damage', target: sideFromIdent(fields[2]), condition: fields[3] };
	case 'faint':
		return { type: 'faint', target: sideFromIdent(fields[2]) };
	case 'turn':
		return { type: 'turn', number: Number(fields[2]) };
	case 'win':
		return { type: 'winner', side: fields[2] };
	case 'tie':
		return { type: 'winner', side: 'tie' };
	default:
		return null;
	}
}

function projectMegaPunchEvents(lines, audience) {
	const projected = [];
	for (let index = 0; index < lines.length; index++) {
		let line = lines[index];
		if (line.startsWith('|split|')) {
			const owner = line.slice('|split|'.length);
			assert.ok(index + 2 < lines.length, 'split branch must contain both recipient records');
			line = audience === owner ? lines[index + 1] : lines[index + 2];
			index += 2;
		}
		if (line.startsWith('|t:|') || line === '|' || line === '|upkeep') continue;
		const event = semanticEvent(line);
		if (event) projected.push(event);
	}
	return projected;
}

function captureTurn(ps, battle, scenario) {
	assert.equal(battle.ended, false);
	if (scenario.id === 'last-pp-miss-then-opponent-acts') {
		const beforeRequest = snapshot(battle);
		battle.p1.pokemon[0].moveSlots[0].pp = 1;
		battle.makeRequest('move');
		assert.deepEqual(snapshot(battle).rng, beforeRequest.rng, 'last-PP setup must not advance native RNG');
	}
	const actualBefore = snapshot(battle);
	const actualRequests = { p1: projectRequest(battle.p1), p2: projectRequest(battle.p2) };
	const logStart = battle.log.length;
	const { draws } = withObservedBattleDraws(ps, battle, () => battle.makeChoices('move 1', 'move 1'));
	const logDelta = battle.log.slice(logStart);
	const events = { p1: projectMegaPunchEvents(logDelta, 'p1'), p2: projectMegaPunchEvents(logDelta, 'p2') };
	const after = snapshot(battle);
	const actionOrder = events.p1.filter(event => event.type === 'move').map(event => event.actor);
	assert.ok(actionOrder.length >= 1 && actionOrder.length <= 2);
	const misses = new Set(events.p1.filter(event => event.type === 'miss').map(event => event.actor));
	const expectedContexts = [];
	for (const actor of actionOrder) {
		expectedContexts.push('accuracy');
		if (!misses.has(actor)) expectedContexts.push('critical', 'damage');
	}
	assert.equal(draws.length, expectedContexts.length);
	for (let index = 0; index < draws.length; index++) {
		assert.deepEqual(draws[index].bound, { accuracy: 100, critical: 16, damage: 16 }[expectedContexts[index]]);
		assert.equal(draws[index].context, expectedContexts[index]);
	}
	assert.deepEqual(draws[0].before, actualBefore.rng);
	assert.deepEqual(draws.at(-1).after, after.rng);
	const nextRequests = battle.ended ? { p1: null, p2: null } : {
		p1: projectRequest(battle.p1),
		p2: projectRequest(battle.p2),
	};
	return {
		before: actualBefore,
		requests: actualRequests,
		choices: { p1: 'move 1', p2: 'move 1' },
		actionOrder,
		draws,
		events,
		after,
		nextRequests,
		views: {
			p1: { request: actualRequests.p1, events: events.p1 },
			p2: { request: actualRequests.p2, events: events.p2 },
		},
	};
}

function scenarioFixture(ps, scenario) {
	const battle = createBattle(ps, scenario);
	try {
		return {
			id: scenario.id,
			seed: [...scenario.seed],
			sets: { p1: scenario.p1, p2: scenario.p2 },
			turn: captureTurn(ps, battle, scenario),
		};
	} finally {
		battle.destroy();
	}
}

function generateFixture(ps) {
	const moveFacts = assertMegaPunchFacts(ps);
	const scenarios = SCENARIOS.map(scenario => scenarioFixture(ps, scenario));
	const byId = new Map(scenarios.map(scenario => [scenario.id, scenario.turn]));
	assert.ok(byId.get('p1-first-misses').events.p1.some(event => event.type === 'miss' && event.actor === 'p1'));
	assert.ok(byId.get('p2-first-misses').events.p1.some(event => event.type === 'miss' && event.actor === 'p2'));
	assert.ok(byId.get('p2-first-critical-hit').events.p1.some(event => event.type === 'critical'));
	assert.equal(byId.get('first-action-ko-skips-p2').actionOrder.length, 1);
	assert.ok(byId.get('first-action-ko-skips-p2').after.terminal);
	assert.ok(byId.get('last-pp-miss-then-opponent-acts').events.p1.some(event => event.type === 'miss' && event.actor === 'p1'));
	assert.equal(byId.get('last-pp-miss-then-opponent-acts').actionOrder.length, 2);
	assert.equal(byId.get('last-pp-miss-then-opponent-acts').nextRequests.p1.own.move.id, 'struggle');
	return {
		schemaVersion: 1,
		kind: 'gen5-restricted-neutral-mega-punch-turn-oracle',
		upstream: { repository: LOCK.repository, commit: LOCK.commit },
		formatId: FORMAT_ID,
		generation: 5,
		variant: VARIANT,
		moveFacts,
		initialization: 'Each fixture is a live unmodified pinned gen5randombattle Battle initialized with directly supplied one-Mew Synchronize, Hardy, perfect-IV, itemless sets. Random team generation is bypassed. Every turn uses actual normal activeRequest objects and battle.makeChoices(move 1, move 1). The last-PP case changes a live moveSlot to one PP, calls battle.makeRequest(move), and asserts that request setup does not advance RNG.',
		mechanics: 'Mega Punch is Normal Physical, power 80, accuracy 85, base PP 20 / maximum PP 32, and default critical ratio 1/16. Distinct positive speeds avoid tie draws. Normal against Psychic is neutral, with no STAB, status, item, weather, boost, or relevant passive effect.',
		drawContract: 'The imported withObservedBattleDraws helper wraps PRNG.prototype.random and returns each original value unchanged. Every attempted Mega Punch consumes random(100). Misses stop there; hits consume random(16) critical then random(16) damage. Ordered contexts, bounds, values, and four-limb pre/post states are retained.',
		requestContract: 'Normal requests expose the sole Mega Punch move 1 with own HP condition and stats. At the declared last-PP unsupported continuation boundary, native getMoveRequestData exposes Struggle; this is recorded as a boundary and is not treated as Mega Punch support or executed by the JAX v5 API.',
		eventContract: 'Recipient event projection preserves the recipient-selected split branch and includes native -miss actor and target alongside move, critical, damage, faint, turn, and winner events. Both recipient projections are compared independently.',
		scenarios,
	};
}

function main() {
	assertReady();
	const ps = require(API);
	process.stdout.write(`${JSON.stringify(generateFixture(ps), null, 2)}\n`);
}

if (require.main === module) main();

module.exports = { generateFixture, main };
