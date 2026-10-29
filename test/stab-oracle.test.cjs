'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const oracle = require('../reference/stab-oracle.cjs');
require('../reference/oracle.cjs').assertReady();

test('STAB oracle reproduces guarded ordinary and synthetic cases', () => {
	const fixture = oracle.generateFixture(require('../reference/vendor/pokemon-showdown/dist/sim'));
	assert.equal(oracle.stringifyFixture(fixture), fs.readFileSync(path.join(__dirname, '../reference/fixtures/stab-v1.json'), 'utf8'));
	assert.equal(fixture.upstream.commit, '1242c1e2d9763ba5b85d33f13aecb89bae19ae3d');
	assert.equal(fixture.formatId, 'gen5randombattle');
	assert.deepEqual(fixture.ordinaryCases.map(row => row.modifier), [6144, 4096, 8192, 4096, 6144]);
	assert.deepEqual(fixture.ordinaryCases.map(row => row.modifyStabCalls), [1, 1, 1, 1, 1]);
	assert.deepEqual(fixture.ordinaryCases.map(row => row.modifyCalls), [1, 1, 1, 1, 1]);
	assert.equal(fixture.syntheticCases.filter(row => row.typeless).length, 5);
	assert.ok(fixture.syntheticCases.filter(row => row.typeless).every(row => row.modifyCalls === 0 && row.preTypeModifyCalls === 0));
	assert.deepEqual(fixture.wrapAnchors, [524287, 524288, 524289, 699049, 699050, 699051, 1048575, 1048576, 1048577]);
});

test('STAB oracle restores wrappers and temporary move state after success and failure', () => {
	const ps = require('../reference/vendor/pokemon-showdown/dist/sim');
	const battle = oracle.createBattle(ps);
	try {
		const source = battle.p1.pokemon[0];
		const target = battle.p2.pokemon[0];
		const beforeStatus = source.status;
		const beforeAbility = source.ability;
		const before = Object.fromEntries(['randomizer', 'modify', 'runEvent'].map(key => [
			key, Object.getOwnPropertyDescriptor(battle, key),
		]));
		const beforeEffectiveness = Object.getOwnPropertyDescriptor(target, 'runEffectiveness');
		const assertRestored = () => {
			for (const key of ['randomizer', 'modify', 'runEvent']) {
				assert.deepEqual(Object.getOwnPropertyDescriptor(battle, key), before[key]);
			}
			assert.equal(source.status, beforeStatus);
			assert.equal(source.ability, beforeAbility);
			assert.deepEqual(Object.getOwnPropertyDescriptor(target, 'runEffectiveness'), beforeEffectiveness);
		};
		assert.equal(oracle.capture(battle, {
			label: 'success', damage: 65, move: 'Tackle', stabEligible: true, forceSTAB: true,
		}).modifiedOutput, 97);
		assertRestored();
		assert.throws(() => oracle.capture(battle, { label: 'failure', damage: NaN, move: 'Tackle', stabEligible: true }), /synthetic damage must be an integer/);
		assertRestored();
		const originalGetDamage = battle.actions.getDamage;
		const actionDescriptor = Object.getOwnPropertyDescriptor(battle.actions, 'getDamage');
		const freshMove = battle.dex.getActiveMove('Tackle');
		const freshMoveDescriptors = ['type', 'forceSTAB'].map(key => Object.getOwnPropertyDescriptor(freshMove, key));
		let capturedMove;
		battle.actions.getDamage = function (_source, _target, move) {
			capturedMove = move;
			throw new Error('injected getDamage failure');
		};
		assert.throws(() => oracle.capture(battle, {
			label: 'move-state-failure', damage: 65, move: 'Tackle', stabEligible: true, forceSTAB: true,
		}), /injected getDamage failure/);
		assert.deepEqual(['type', 'forceSTAB'].map(key => Object.getOwnPropertyDescriptor(capturedMove, key)), freshMoveDescriptors);
		if (actionDescriptor) Object.defineProperty(battle.actions, 'getDamage', actionDescriptor);
		else delete battle.actions.getDamage;
		assert.equal(battle.actions.getDamage, originalGetDamage);
		assertRestored();
	} finally {
		battle.destroy();
	}
});
