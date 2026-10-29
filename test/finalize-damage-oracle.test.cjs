'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const { capture, createBattle } = require('../reference/finalize-damage-oracle.cjs');
const oracle = require('../reference/oracle.cjs');

test('finalization oracle restores wrappers and registered events on success and failure', () => {
	oracle.assertReady();
	const ps = require('../reference/vendor/pokemon-showdown/dist/sim');
	const battle = createBattle(ps);
	const keys = ['randomizer', 'runEvent', 'modify', 'events'];
	const before = keys.map(key => Object.getOwnPropertyDescriptor(battle, key));
	const originalGetDamage = battle.actions.getDamage;
	try {
		assert.equal(capture(battle, { value: 0, modifier: 0 }).result, 0);
		keys.forEach((key, i) => assert.deepEqual(Object.getOwnPropertyDescriptor(battle, key), before[i]));
		battle.actions.getDamage = () => { throw new Error('deliberate oracle failure'); };
		assert.throws(() => capture(battle, { value: 1, modifier: 4096 }), /deliberate oracle failure/);
		keys.forEach((key, i) => assert.deepEqual(Object.getOwnPropertyDescriptor(battle, key), before[i]));
	} finally {
		battle.actions.getDamage = originalGetDamage;
		battle.destroy();
	}
});
