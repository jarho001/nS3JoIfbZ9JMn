'use strict';

const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');
const path = require('node:path');
const test = require('node:test');

const ROOT = path.resolve(__dirname, '..');
const ORACLE = path.join(ROOT, 'reference', 'neutral-turn-oracle.cjs');

function loadDuelOracle() {
	return JSON.parse(execFileSync(process.execPath, [ORACLE, '--duels'], { cwd: ROOT, encoding: 'utf8' }));
}

test('explicit full-duel oracle captures bounded complete native battles', () => {
	const fixture = loadDuelOracle();
	assert.equal(fixture.schemaVersion, 1);
	assert.equal(fixture.kind, 'gen5-restricted-neutral-tackle-full-duel-oracle');
	assert.equal(fixture.upstream.commit, require('../reference-lock.json').commit);
	assert.equal(fixture.variant, 'gen5-neutral-tackle-mew-1v1-v1');
	assert.equal(fixture.turnLimit, 56);
	assert.match(fixture.initialBoundary, /normal activeRequest/);
	assert.match(fixture.captureContract, /return values/);
	assert.equal(fixture.scenarios.length, 4);
	assert.equal(new Set(fixture.scenarios.map(row => row.seed.join(','))).size, 4);

	const scenarios = new Map(fixture.scenarios.map(row => [row.id, row]));
	const p1First = scenarios.get('p1-first-full-duel');
	const p2First = scenarios.get('p2-first-full-duel');
	assert.deepEqual(p1First.turns[0].actionOrder, ['p1', 'p2']);
	assert.deepEqual(p2First.turns[0].actionOrder, ['p2', 'p1']);
	assert.ok(p1First.turns.length > 1);
	assert.ok(p2First.turns.length > 1);

	for (const scenario of fixture.scenarios) {
		assert.deepEqual(scenario.initial.state.rng, scenario.seed, scenario.id);
		assert.equal(scenario.initial.state.terminal, false, scenario.id);
		assert.equal(scenario.initial.requests.p1.phase, 'move', scenario.id);
		assert.equal(scenario.initial.requests.p2.phase, 'move', scenario.id);
		assert.deepEqual(scenario.initial.requests.p1.legalActions, ['move 1'], scenario.id);
		assert.deepEqual(scenario.initial.requests.p2.legalActions, ['move 1'], scenario.id);
		assert.ok(scenario.turns.length > 0 && scenario.turns.length <= fixture.turnLimit, scenario.id);
		assert.equal(scenario.outcome.terminal, true, scenario.id);
		assert.ok(['p1', 'p2'].includes(scenario.outcome.winner), scenario.id);
		assert.equal(scenario.outcome.state.terminal, true, scenario.id);
		assert.deepEqual(scenario.outcome.state.requestsPresent, [false, false], scenario.id);
		for (const [index, turn] of scenario.turns.entries()) {
			assert.deepEqual(turn.choices, { p1: 'move 1', p2: 'move 1' }, [scenario.id, index]);
			assert.equal(turn.before.requestPhase, 'move', [scenario.id, index]);
			assert.equal(turn.requests.p1.phase, 'move', [scenario.id, index]);
			assert.equal(turn.requests.p2.phase, 'move', [scenario.id, index]);
			assert.equal(turn.draws.length, turn.actionOrder.length * 3, [scenario.id, index]);
			assert.deepEqual(turn.draws.map(draw => draw.context), turn.actionOrder.flatMap(() => ['accuracy', 'critical', 'damage']), [scenario.id, index]);
			assert.deepEqual(turn.draws.map(draw => draw.bound), turn.actionOrder.flatMap(() => [100, 16, 16]), [scenario.id, index]);
			for (let drawIndex = 0; drawIndex < turn.draws.length; drawIndex++) {
				const draw = turn.draws[drawIndex];
				assert.deepEqual(draw.before, drawIndex ? turn.draws[drawIndex - 1].after : turn.before.rng, [scenario.id, index, drawIndex]);
				assert.equal(draw.after.length, 4, [scenario.id, index, drawIndex]);
				assert.ok(draw.after.every(limb => Number.isInteger(limb) && limb >= 0 && limb <= 65535), [scenario.id, index, drawIndex]);
			}
			for (const view of [turn.views.p1, turn.views.p2]) {
				assert.deepEqual(Object.keys(view).sort(), ['events', 'request'], [scenario.id, index]);
				assert.ok(!Object.hasOwn(view, 'rng'));
				assert.ok(!Object.hasOwn(view, 'draws'));
				assert.ok(!Object.hasOwn(view, 'rawPackets'));
			}
			if (index + 1 < scenario.turns.length) {
				const next = scenario.turns[index + 1].before;
				assert.deepEqual(turn.after.hp, next.hp, [scenario.id, index, 'hp continuation']);
				assert.deepEqual(turn.after.pp, next.pp, [scenario.id, index, 'pp continuation']);
				assert.deepEqual(turn.after.rng, next.rng, [scenario.id, index, 'rng continuation']);
				assert.equal(turn.after.turn, next.turn, [scenario.id, index, 'turn continuation']);
			}
		}
	}

	const critical = scenarios.get('critical-full-duel');
	assert.ok(critical.turns.some(turn => turn.draws.some(draw => draw.context === 'critical' && draw.value === 0)));
	assert.ok(critical.turns.some(turn => turn.views.p1.events.some(event => event.type === 'critical')));
	assert.ok(critical.turns.some(turn => turn.views.p2.events.some(event => event.type === 'critical')));

	const ko = scenarios.get('one-action-ko-full-duel');
	assert.equal(ko.turns.length, 1);
	assert.deepEqual(ko.turns[0].actionOrder, ['p2']);
	assert.equal(ko.turns[0].draws.length, 3);
	assert.equal(ko.outcome.winner, 'p2');
	assert.equal(ko.outcome.sideWinner, 2);
});

test('no-argument mode still emits the original restricted-turn fixture', () => {
	const fixture = JSON.parse(execFileSync(process.execPath, [ORACLE], { cwd: ROOT, encoding: 'utf8' }));
	assert.equal(fixture.kind, 'gen5-restricted-neutral-tackle-turn-oracle');
	assert.deepEqual(fixture.scenarios.map(row => row.id), [
		'p2-first-surviving',
		'p1-first-surviving',
		'native-critical-hit',
		'first-action-ko-skip',
	]);
	assert.deepEqual(fixture.scenarios.map(row => row.turns.length), [3, 3, 1, 1]);
});
