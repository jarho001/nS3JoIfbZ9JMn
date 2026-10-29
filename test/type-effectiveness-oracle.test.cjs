'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const oracle = require('../reference/type-effectiveness-oracle.cjs');
require('../reference/oracle.cjs').assertReady();

test('synthetic oracle wrappers are restored after a guarded failure', () => {
	const ps = require('../reference/vendor/pokemon-showdown/dist/sim');
	const battle = oracle.createBattle(ps);
	try {
		const target = battle.p2.pokemon[0];
		const source = battle.p1.pokemon[0];
		const originalStatus = source.status;
		const before = {
			randomizer: Object.getOwnPropertyDescriptor(battle, 'randomizer'),
			runEffectiveness: Object.getOwnPropertyDescriptor(target, 'runEffectiveness'),
			modify: Object.getOwnPropertyDescriptor(battle, 'modify'),
		};
		const assertRestored = () => {
			assert.deepEqual(Object.getOwnPropertyDescriptor(battle, 'randomizer'), before.randomizer);
			assert.deepEqual(Object.getOwnPropertyDescriptor(target, 'runEffectiveness'), before.runEffectiveness);
			assert.deepEqual(Object.getOwnPropertyDescriptor(battle, 'modify'), before.modify);
			assert.equal(source.status, originalStatus);
		};
		const successful = oracle.captureSynthetic(battle, {
			label: 'successful-restoration', damage: 65, exponent: -2,
		});
		assert.equal(successful.result, 16);
		assertRestored();
		assert.throws(() => oracle.captureSynthetic(battle, {
			label: 'forced-restoration-error', damage: 5, exponent: NaN,
		}), /synthetic exponent must be an integer/);
		assertRestored();
	} finally {
		battle.destroy();
	}
});
