'use strict';

const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');
const path = require('node:path');
const test = require('node:test');

const ROOT = path.resolve(__dirname, '..');
const ORACLE = path.join(ROOT, 'reference', 'neutral-turn-oracle.cjs');

function loadGuardedOracle() {
	return JSON.parse(execFileSync(process.execPath, [ORACLE], { cwd: ROOT, encoding: 'utf8' }));
}

test('guarded oracle is pinned and captures actual normal requests and RNG vectors', () => {
	const fixture = loadGuardedOracle();
	assert.equal(fixture.schemaVersion, 1);
	assert.equal(fixture.kind, 'gen5-restricted-neutral-tackle-turn-oracle');
	assert.equal(fixture.upstream.commit, require('../reference-lock.json').commit);
	assert.equal(fixture.formatId, 'gen5randombattle');
	assert.equal(fixture.variant, 'gen5-neutral-tackle-mew-1v1-v1');
	assert.match(fixture.initialization, /normal activeRequest\/getRequestData/);
	assert.match(fixture.drawContract, /record and return the exact original result/);
	assert.equal(fixture.rngVectors.length, 6);
	for (const vector of fixture.rngVectors) {
		assert.equal(vector.rows.length, 5);
		for (const row of vector.rows) {
			assert.deepEqual(row.before, vector.seed);
			assert.equal(row.after.length, 4);
			assert.ok(row.after.every(limb => Number.isInteger(limb) && limb >= 0 && limb <= 65535));
		}
	}
});

test('both native speed orders, actual critical, survival continuation, and first-action KO are captured', () => {
	const fixture = loadGuardedOracle();
	const scenarios = new Map(fixture.scenarios.map(scenario => [scenario.id, scenario]));
	const p2First = scenarios.get('p2-first-surviving');
	const p1First = scenarios.get('p1-first-surviving');
	assert.equal(p2First.turns.length, 3);
	assert.equal(p1First.turns.length, 3);
	assert.deepEqual(p2First.turns[0].actionOrder, ['p2', 'p1']);
	assert.deepEqual(p1First.turns[0].actionOrder, ['p1', 'p2']);
	for (const turn of [...p2First.turns, ...p1First.turns]) {
		assert.equal(turn.requests.p1.phase, 'move');
		assert.equal(turn.requests.p2.phase, 'move');
		assert.deepEqual(turn.requests.p1.legalActions, ['move 1']);
		assert.deepEqual(turn.requests.p2.legalActions, ['move 1']);
		assert.deepEqual(turn.choices, { p1: 'move 1', p2: 'move 1' });
		assert.deepEqual(turn.draws.map(draw => draw.context), ['accuracy', 'critical', 'damage', 'accuracy', 'critical', 'damage']);
		assert.deepEqual(turn.draws.map(draw => draw.bound), [100, 16, 16, 100, 16, 16]);
		for (let index = 0; index < turn.draws.length; index++) {
			assert.deepEqual(turn.draws[index].before, index ? turn.draws[index - 1].after : turn.before.rng);
		}
		assert.deepEqual(turn.draws.at(-1).after, turn.after.rng);
	}
	for (const scenario of [p2First, p1First]) {
		for (let index = 0; index < scenario.turns.length - 1; index++) {
			assert.deepEqual(scenario.turns[index].after.rng, scenario.turns[index + 1].before.rng);
			assert.deepEqual(scenario.turns[index].after.hp, scenario.turns[index + 1].before.hp);
			assert.deepEqual(scenario.turns[index].after.pp, scenario.turns[index + 1].before.pp);
			assert.equal(scenario.turns[index].after.turn, scenario.turns[index + 1].before.turn);
		}
	}

	const critical = scenarios.get('native-critical-hit').turns[0];
	assert.ok(critical.draws.some(draw => draw.context === 'critical' && draw.value === 0));
	assert.ok(critical.views.p1.events.some(event => event.type === 'critical'));
	assert.ok(critical.views.p2.events.some(event => event.type === 'critical'));

	const ko = scenarios.get('first-action-ko-skip').turns[0];
	assert.deepEqual(ko.actionOrder, ['p2']);
	assert.equal(ko.draws.length, 3);
	assert.deepEqual(ko.before.pp, [56, 56]);
	assert.deepEqual(ko.after.pp, [56, 55]);
	assert.deepEqual(ko.after.hp, [0, 341]);
	assert.equal(ko.after.turn, 1, 'a terminal first-action KO does not open another turn');
	assert.equal(ko.after.terminal, true);
	assert.equal(ko.after.winner, 2);
	assert.deepEqual(ko.after.requestsPresent, [false, false]);
});

test('recipient projections preserve exact own HP and percentage opponent HP without trusted state', () => {
	const fixture = loadGuardedOracle();
	const turn = fixture.scenarios.find(scenario => scenario.id === 'p2-first-surviving').turns[0];
	const p1Condition = turn.views.p1.events.find(event => event.type === 'damage' && event.target === 'p1').condition;
	const p2Condition = turn.views.p2.events.find(event => event.type === 'damage' && event.target === 'p1').condition;
	assert.equal(p1Condition, '154/175');
	assert.equal(p2Condition, '88/100');
	assert.notEqual(p1Condition, p2Condition);
	for (const view of [turn.views.p1, turn.views.p2]) {
		assert.deepEqual(Object.keys(view).sort(), ['events', 'request']);
		assert.ok(!Object.hasOwn(view, 'hp'));
		assert.ok(!Object.hasOwn(view, 'rng'));
		assert.ok(!Object.hasOwn(view, 'draws'));
		assert.ok(!Object.hasOwn(view, 'rawPackets'));
	}
	const ko = fixture.scenarios.find(scenario => scenario.id === 'first-action-ko-skip').turns[0];
	assert.equal(ko.views.p2.events.find(event => event.type === 'damage').condition, '0 fnt');
});
