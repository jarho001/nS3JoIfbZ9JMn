'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const ROOT = path.resolve(__dirname, '..');
const oracle = require('../reference/oracle.cjs');
const { capture, createBattle } = require('../reference/damage-composition-oracle.cjs');
const { generateFixture } = require('../reference/neutral-full-damage-oracle.cjs');

function descriptors(object, keys) {
	return Object.fromEntries(keys.map(key => [key, Object.getOwnPropertyDescriptor(object, key)]));
}

test('neutral full-damage oracle reproduces guarded core-to-final results', () => {
	oracle.assertReady();
	const ps = require('../reference/vendor/pokemon-showdown/dist/sim');
	const fixture = JSON.parse(fs.readFileSync(path.join(ROOT, 'reference/fixtures/neutral-full-damage-v1.json'), 'utf8'));
	assert.deepEqual(generateFixture(ps), fixture);
	assert.equal(fixture.ordinaryCases.length, 5);
	assert.equal(fixture.syntheticCases.length, 6);
	for (const row of fixture.ordinaryCases.concat(fixture.syntheticCases).flatMap(group => group.outcomes)) {
		assert.equal(row.weatherInput, row.baseCore + 2);
		assert.equal(row.preRollDamage, row.weatherInput & 0xfffff);
		assert.equal(row.input.preRollDamage, row.preRollDamage);
		assert.equal(row.input.roll, row.roll);
		assert.equal(row.randomizerCalls, 1);
		assert.equal(row.damageDraws, 1);
		assert.equal(row.finalCalls, 1);
		assert.equal(row.finalModifierCalls, 1);
		assert.equal(row.finalModifierObserved, row.input.finalModifier);
		assert.equal(row.baseCoreCalls, 1);
		assert.equal(row.weatherEventCalls, 1);
		assert.equal(row.weatherHandlerCount, 0);
	}
	assert.equal(fixture.syntheticCases[2].outcomes[0].weatherInput, 1048575);
	assert.equal(fixture.syntheticCases[3].outcomes[0].weatherInput, 1048576);
	assert.equal(fixture.syntheticCases[4].outcomes[0].weatherInput, 1048577);
});

test('optional effective-core instrumentation restores temporary state on success and failure', () => {
	oracle.assertReady();
	const ps = require('../reference/vendor/pokemon-showdown/dist/sim');
	const battle = createBattle(ps);
	const source = battle.p1.pokemon[0];
	const target = battle.p2.pokemon[0];
	const keys = ['randomizer', 'random', 'modify', 'runEvent', 'events', 'priorityEvent'];
	const beforeBattle = descriptors(battle, keys);
	const beforeActions = Object.getOwnPropertyDescriptor(battle.actions, 'modifyDamage');
	const beforeEffectiveness = Object.getOwnPropertyDescriptor(target, 'runEffectiveness');
	const before = {
		level: Object.getOwnPropertyDescriptor(source, 'level'),
		attack: Object.getOwnPropertyDescriptor(source.storedStats, 'atk'),
		defense: Object.getOwnPropertyDescriptor(target.storedStats, 'def'),
		status: source.status,
	};
	function verify() {
		for (const key of keys) assert.deepEqual(Object.getOwnPropertyDescriptor(battle, key), beforeBattle[key]);
		assert.deepEqual(Object.getOwnPropertyDescriptor(battle.actions, 'modifyDamage'), beforeActions);
		assert.deepEqual(Object.getOwnPropertyDescriptor(target, 'runEffectiveness'), beforeEffectiveness);
		assert.deepEqual(Object.getOwnPropertyDescriptor(source, 'level'), before.level);
		assert.deepEqual(Object.getOwnPropertyDescriptor(source.storedStats, 'atk'), before.attack);
		assert.deepEqual(Object.getOwnPropertyDescriptor(target.storedStats, 'def'), before.defense);
		assert.equal(source.status, before.status);
	}
	try {
		const observed = capture(battle, { captureEffectiveCore: true, level: 100, basePower: 1560, attack: 65535, defense: 1, roll: 7 });
		assert.equal(observed.effectiveBasePower, 1560);
		assert.equal(observed.effectiveAttack, 65535);
		assert.equal(observed.effectiveDefense, 1);
		assert.equal(observed.weatherInput, observed.baseCore + 2);
		verify();
		assert.throws(() => capture(battle, {
			captureEffectiveCore: true, level: 100, basePower: 1560, attack: 65535, defense: 1,
			roll: 7, failAfterRandomizer: true,
		}), /injected composition capture failure/);
		verify();
	} finally {
		battle.destroy();
	}
});
