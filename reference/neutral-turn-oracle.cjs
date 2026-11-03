#!/usr/bin/env node
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { assertReferenceIdentity, inspectCheckout } = require('./setup.cjs');

const ROOT = path.resolve(__dirname, '..');
const LOCK = require(path.join(ROOT, 'reference-lock.json'));
const CHECKOUT = path.join(__dirname, 'vendor', 'pokemon-showdown');
const API = path.join(CHECKOUT, 'dist', 'sim');
const FORMAT_ID = 'gen5randombattle';
const SEEDS = [
	[0, 0, 0, 0],
	[1, 2, 3, 4],
	[65535, 65535, 65535, 65535],
	[0, 65535, 0, 65535],
	[65535, 0, 65535, 0],
	[0x5D58, 0x8B65, 0x6C07, 0x8965],
];
const BOUNDS = [1, 2, 16, 100, 65535];
const SCENARIOS = [
	{
		id: 'p2-first-surviving',
		seed: [1, 2, 3, 4],
		p1: { level: 50, speedEvs: 0 },
		p2: { level: 50, speedEvs: 252 },
		turnLimit: 3,
	},
	{
		id: 'p1-first-surviving',
		seed: [31, 41, 59, 26],
		p1: { level: 50, speedEvs: 252 },
		p2: { level: 50, speedEvs: 0 },
		turnLimit: 3,
	},
	{
		id: 'native-critical-hit',
		seed: [9, 2, 3, 4],
		p1: { level: 50, speedEvs: 0 },
		p2: { level: 50, speedEvs: 252 },
		turnLimit: 1,
	},
	{
		id: 'first-action-ko-skip',
		seed: [1, 2, 3, 4],
		p1: { level: 1, speedEvs: 0 },
		p2: { level: 100, speedEvs: 0 },
		turnLimit: 1,
	},
];

function assertReady() {
	let actual;
	try {
		actual = inspectCheckout(CHECKOUT);
	} catch (error) {
		throw new Error(`Pinned Showdown is unavailable at ${CHECKOUT}. Run "node reference/setup.cjs" first. (${error.message})`);
	}
	assertReferenceIdentity(LOCK, actual);
	assert.ok(fs.existsSync(path.join(API, 'index.js')), 'Pinned Showdown must be built; run "node reference/setup.cjs" first.');
}

function makeSet(ps, descriptor) {
	const empty = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
	const perfect = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
	return {
		species: 'Mew',
		ability: 'Synchronize',
		item: '',
		level: descriptor.level,
		moves: ['Tackle'],
		nature: 'Hardy',
		ivs: perfect,
		evs: { ...empty, spe: descriptor.speedEvs },
	};
}

function createBattle(ps, scenario) {
	const battle = new ps.Battle({
		formatid: FORMAT_ID,
		seed: [...scenario.seed],
		p1: { name: 'p1', team: [makeSet(ps, scenario.p1)] },
		p2: { name: 'p2', team: [makeSet(ps, scenario.p2)] },
	});
	assert.equal(battle.gen, 5);
	assert.equal(battle.gameType, 'singles');
	assert.equal(battle.p1.pokemon.length, 1);
	assert.equal(battle.p2.pokemon.length, 1);
	assert.equal(battle.ruleTable.has('hppercentagemod'), true);
	assert.equal(battle.p1.active[0].moves.length, 1);
	assert.equal(battle.p2.active[0].moves.length, 1);
	assert.equal(battle.p1.active[0].moves[0], 'tackle');
	assert.equal(battle.p2.active[0].moves[0], 'tackle');
	assert.equal(battle.p1.active[0].item, '');
	assert.equal(battle.p2.active[0].item, '');
	return battle;
}

function seedArray(prng) {
	return prng.getSeed().split(',').map(Number);
}

function sideWinner(battle) {
	if (!battle.ended) return -1;
	if (battle.winner === 'p1') return 1;
	if (battle.winner === 'p2') return 2;
	return 0;
}

function snapshot(battle) {
	const pokemon = [battle.p1.pokemon[0], battle.p2.pokemon[0]];
	const activeRequests = [battle.p1.activeRequest, battle.p2.activeRequest];
	return {
		hp: pokemon.map(mon => mon.hp),
		maxHp: pokemon.map(mon => mon.maxhp),
		pp: pokemon.map(mon => mon.moveSlots[0].pp),
		level: pokemon.map(mon => mon.level),
		attack: pokemon.map(mon => mon.storedStats.atk),
		defense: pokemon.map(mon => mon.storedStats.def),
		speed: pokemon.map(mon => mon.speed),
		rng: seedArray(battle.prng),
		turn: battle.turn,
		requestPhase: battle.ended ? 'terminal' : 'move',
		terminal: battle.ended,
		winner: sideWinner(battle),
		requestsPresent: activeRequests.map(Boolean),
	};
}

function projectRequest(side) {
	const request = side.activeRequest;
	assert.ok(request, `${side.id} must have a normal active request before a turn`);
	assert.equal(request.active.length, 1);
	assert.equal(request.side.id, side.id);
	const moveRows = request.active[0].moves;
	assert.equal(moveRows.length, 1, 'the restricted request must contain exactly one move');
	const move = moveRows[0];
	assert.equal(move.id, 'tackle');
	assert.equal(move.disabled, false);
	assert.ok(move.pp > 0);
	const own = request.side.pokemon.find(mon => mon.active);
	assert.ok(own, `${side.id} normal request must identify its own active Pokemon`);
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

function projectEvents(lines, audience) {
	const projected = [];
	for (let index = 0; index < lines.length; index++) {
		let line = lines[index];
		if (line.startsWith('|split|')) {
			const owner = line.slice('|split|'.length);
			assert.ok(index + 2 < lines.length, 'split visibility branch must have both records');
			line = audience === owner ? lines[index + 1] : lines[index + 2];
			index += 2;
		}
		if (line.startsWith('|t:|') || line === '|' || line === '|upkeep') continue;
		const event = semanticEvent(line);
		if (event) projected.push(event);
	}
	return projected;
}

function withObservedBattleDraws(ps, battle, action) {
	const descriptor = Object.getOwnPropertyDescriptor(ps.PRNG.prototype, 'random');
	assert.ok(descriptor && typeof descriptor.value === 'function');
	const original = descriptor.value;
	const draws = [];
	ps.PRNG.prototype.random = function (bound, to) {
		const isBattleRng = this === battle.prng;
		const before = isBattleRng ? seedArray(this) : null;
		const value = original.call(this, bound, to);
		if (isBattleRng) {
			const stack = new Error().stack.split('\n').slice(2, 8).map(frame => frame.trim());
			const context = stack.some(frame => frame.includes('hitStepAccuracy'))
				? 'accuracy'
				: stack.some(frame => frame.includes('randomizer'))
					? 'damage'
					: stack.some(frame => frame.includes('getDamage'))
						? 'critical'
						: 'unsupported';
			draws.push({ bound, value, before, after: seedArray(this), context, stack });
		}
		return value;
	};
	try {
		return { result: action(), draws };
	} finally {
		Object.defineProperty(ps.PRNG.prototype, 'random', descriptor);
	}
}

function captureTurn(ps, battle) {
	assert.equal(battle.ended, false);
	const before = snapshot(battle);
	const requests = { p1: projectRequest(battle.p1), p2: projectRequest(battle.p2) };
	const logStart = battle.log.length;
	const { draws } = withObservedBattleDraws(ps, battle, () => battle.makeChoices('move 1', 'move 1'));
	const logDelta = battle.log.slice(logStart);
	const events = {
		p1: projectEvents(logDelta, 'p1'),
		p2: projectEvents(logDelta, 'p2'),
	};
	const after = snapshot(battle);
	const actionOrder = events.p1.filter(event => event.type === 'move').map(event => event.actor);
	assert.ok(actionOrder.length >= 1 && actionOrder.length <= 2);
	assert.equal(draws.length, actionOrder.length * 3, 'each executed Tackle must consume accuracy, critical, and damage draws');
	for (let index = 0; index < draws.length; index++) {
		assert.deepEqual(draws[index].bound, [100, 16, 16][index % 3]);
		assert.equal(draws[index].context, ['accuracy', 'critical', 'damage'][index % 3]);
	}
	assert.deepEqual(draws[0].before, before.rng);
	assert.deepEqual(draws.at(-1).after, after.rng);
	return {
		before,
		requests,
		choices: { p1: 'move 1', p2: 'move 1' },
		actionOrder,
		draws,
		events,
		after,
		views: {
			p1: { request: requests.p1, events: events.p1 },
			p2: { request: requests.p2, events: events.p2 },
		},
	};
}

function scenarioFixture(ps, scenario) {
	const battle = createBattle(ps, scenario);
	try {
		const turns = [];
		for (let index = 0; index < scenario.turnLimit && !battle.ended; index++) turns.push(captureTurn(ps, battle));
		return { id: scenario.id, turns };
	} finally {
		battle.destroy();
	}
}

function captureRngVectors(ps) {
	return SEEDS.map(seed => ({
		seed,
		rows: BOUNDS.map(bound => {
			const prng = new ps.PRNG(seed);
			const before = seedArray(prng);
			const value = prng.random(bound);
			return { bound, before, value, after: seedArray(prng) };
		}),
	}));
}

function generateFixture(ps) {
	return {
		schemaVersion: 1,
		kind: 'gen5-restricted-neutral-tackle-turn-oracle',
		upstream: { repository: LOCK.repository, commit: LOCK.commit },
		formatId: FORMAT_ID,
		generation: 5,
		variant: 'gen5-neutral-tackle-mew-1v1-v1',
		initialization: 'Pinned Battle initialized from two directly supplied one-Mew sets; normal activeRequest/getRequestData captured before each actual Battle.makeChoices move 1/move 1 submission. The format rule table is unmodified Gen 5 Random Battle; random team generation is bypassed. Each Mew has Synchronize, no item, Hardy nature, perfect IVs, only Tackle, and only the declared Speed EV difference or level difference.',
		mechanics: 'Tackle is pinned Gen 5 base power 50, accuracy 100, and default critical ratio 1/16. No status, item, weather, boost, type modifier, or relevant passive effect is present. Speed is distinct; Tackle is neutral non-STAB Normal damage against Psychic Mew.',
		drawContract: 'PRNG.prototype.random is wrapped only to record and return the exact original result. Each executed Tackle makes accuracy random(100), critical random(16), then damage random(16); the draw trace records actual bound, context stack, and four-limb pre/post seed. Speed ties and all other draw contexts are rejected by the oracle.',
		viewContract: 'Views are semantic projections of pinned normal requests and the recipient-selected branch of split events. The damaged owner receives exact HP conditions; the opponent receives HP Percentage Mod conditions. Raw packets and trusted internal snapshots are not included in either player view.',
		rngVectors: captureRngVectors(ps),
		scenarios: SCENARIOS.map(scenario => scenarioFixture(ps, scenario)),
	};
}

function main() {
	assertReady();
	const ps = require(API);
	const fixture = generateFixture(ps);
	process.stdout.write(`${JSON.stringify(fixture, null, 2)}\n`);
}

if (require.main === module) main();

module.exports = { generateFixture };
