'use strict';

const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');
const path = require('node:path');
const test = require('node:test');

const ROOT = path.resolve(__dirname, '..');
const ORACLE = path.join(ROOT, 'reference', 'neutral-turn-oracle.cjs');

function loadFixture() {
	return JSON.parse(execFileSync(process.execPath, [ORACLE, '--ties'], { cwd: ROOT, encoding: 'utf8' }));
}

test('pinned speed-tie oracle records complete native duels and every RNG phase', () => {
	const fixture = loadFixture();
	assert.equal(fixture.schemaVersion, 1);
	assert.equal(fixture.kind, 'gen5-restricted-neutral-tackle-speed-tie-full-duel-oracle');
	assert.equal(fixture.upstream.commit, require('../reference-lock.json').commit);
	assert.equal(fixture.variant, 'gen5-neutral-tackle-speed-ties-mew-1v1-v2');
	assert.equal(fixture.turnLimit, 56);
	assert.equal(fixture.scenarios.length, 3);
	assert.equal(new Set(fixture.scenarios.map(row => row.seed.join(','))).size, 3);

	for (const scenario of fixture.scenarios) {
		assert.deepEqual(scenario.initial.state.speed, [120, 120], scenario.id);
		assert.equal(scenario.initial.requests.p1.phase, 'move', scenario.id);
		assert.equal(scenario.initial.requests.p2.phase, 'move', scenario.id);
		assert.ok(scenario.turns.length > 1 && scenario.turns.length <= fixture.turnLimit, scenario.id);
		assert.equal(scenario.outcome.terminal, true, scenario.id);
		for (const [turnIndex, turn] of scenario.turns.entries()) {
			assert.deepEqual(turn.before.speed, [120, 120], [scenario.id, turnIndex]);
			const expectedPhases = ['queue-sort', 'before-turn', 'turn-start-update', 'accuracy', 'critical', 'damage', 'move-hit-update'];
			if (turn.actionOrder.length === 2) {
				expectedPhases.push('move-hit-update', 'action-update', 'accuracy', 'critical', 'damage');
				expectedPhases.push(...(turn.after.terminal
					? ['move-hit-update']
					: ['move-hit-update', 'move-hit-update', 'action-update', 'residual-update']));
			}
			assert.deepEqual(turn.draws.map(draw => draw.phase), expectedPhases, [scenario.id, turnIndex, 'native phase schedule']);
			const expectedDrawCount = turn.after.terminal ? (turn.actionOrder.length === 1 ? 7 : 13) : 16;
			assert.equal(turn.draws.length, expectedDrawCount, [scenario.id, turnIndex]);
			assert.ok(turn.draws.length <= 16, [scenario.id, turnIndex]);
			assert.deepEqual(turn.draws[0].phase, 'queue-sort', [scenario.id, turnIndex]);
			assert.deepEqual(turn.draws[0].before, turn.before.rng, [scenario.id, turnIndex, 'initial draw']);
			assert.deepEqual(turn.draws.at(-1).after, turn.after.rng, [scenario.id, turnIndex, 'final draw']);
			for (const [drawIndex, draw] of turn.draws.entries()) {
				assert.deepEqual(draw.before, drawIndex ? turn.draws[drawIndex - 1].after : turn.before.rng, [scenario.id, turnIndex, drawIndex]);
				assert.equal(draw.after.length, 4, [scenario.id, turnIndex, drawIndex]);
				assert.ok(draw.after.every(limb => Number.isInteger(limb) && limb >= 0 && limb <= 65535), [scenario.id, turnIndex, drawIndex]);
				if (draw.context === 'speed-sort') {
					assert.equal(draw.bound, 0, [scenario.id, turnIndex, drawIndex]);
					assert.equal(draw.to, 2, [scenario.id, turnIndex, drawIndex]);
					assert.ok(['queue-sort', 'before-turn', 'turn-start-update', 'move-hit-update', 'action-update', 'residual-update'].includes(draw.phase));
				} else {
					assert.equal(draw.to, undefined, [scenario.id, turnIndex, drawIndex]);
					assert.ok(['accuracy', 'critical', 'damage'].includes(draw.phase), [scenario.id, turnIndex, drawIndex]);
				}
			}
			assert.equal(turn.events.p1.filter(event => event.type === 'move').length, turn.actionOrder.length, [scenario.id, turnIndex]);
			assert.equal(turn.events.p2.filter(event => event.type === 'move').length, turn.actionOrder.length, [scenario.id, turnIndex]);
			for (const view of [turn.views.p1, turn.views.p2]) {
				assert.deepEqual(Object.keys(view).sort(), ['events', 'request'], [scenario.id, turnIndex]);
				assert.ok(!Object.hasOwn(view, 'rng'));
				assert.ok(!Object.hasOwn(view, 'draws'));
			}
		}
	}

	const first = fixture.scenarios[0];
	assert.deepEqual(first.turns[0].actionOrder, ['p1', 'p2']);
	assert.deepEqual(first.turns.at(-1).actionOrder, ['p2']);
	assert.equal(first.turns.at(-1).draws.length, 7);
	assert.ok(first.turns.at(-1).events.p1.some(event => event.type === 'faint'));
	assert.deepEqual(fixture.scenarios[1].turns[0].actionOrder, ['p2', 'p1']);
	assert.ok(fixture.scenarios[2].turns.some(turn => turn.draws.some(draw => draw.phase === 'critical' && draw.value === 0)));
});
