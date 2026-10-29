'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const ROOT = path.resolve(__dirname, '..');
const oracle = require('../reference/oracle.cjs');
const { capture, createBattle, generateFixture } = require('../reference/damage-composition-oracle.cjs');

function ownDescriptors(object, keys) {
	return Object.fromEntries(keys.map(key => [key, Object.getOwnPropertyDescriptor(object, key)]));
}

test('guarded fixture reproduces actual getDamage output and stage order', () => {
	oracle.assertReady();
	const ps = require('../reference/vendor/pokemon-showdown/dist/sim');
	const fixture = JSON.parse(fs.readFileSync(path.join(ROOT, 'reference/fixtures/damage-composition-v1.json'), 'utf8'));
	assert.deepEqual(generateFixture(ps), fixture);
	assert.equal(fixture.ordinaryCases.length, 5);
	assert.equal(fixture.syntheticCases.length, 25);
	for (const row of fixture.ordinaryCases.concat(fixture.syntheticCases)) {
		assert.equal(row.finalCalls, 1);
		assert.equal(row.finalModifierCalls, 1);
		assert.equal(row.finalModifierObserved, row.input.finalModifier);
		assert.equal(Number.isInteger(row.finalModifierObserved), true);
		assert.equal(row.randomizerCalls, 1);
		assert.equal(row.damageDraws, 1);
		assert.equal(row.effectivenessCalls, 1);
		assert.equal(row.modifyStabCalls, row.input.typeless ? 0 : 1);
		assert.equal(row.finalMoveType === '???', row.input.typeless);
	}
	const high = fixture.syntheticCases.find(row => row.label === 'bounded-preroll-and-all-max-amplifiers');
	assert.equal(high.randomizerInput, 33554431);
	assert.equal(high.input.adaptability, true);
	assert.equal(high.input.typeExponent, 6);
	const lifeOrb = fixture.ordinaryCases.find(row => row.label === 'ordinary-stab-life-orb-special-burned');
	assert.equal(lifeOrb.finalModifierObserved, 5324);
	assert.equal(lifeOrb.input.burned, true);
	const zero = fixture.syntheticCases.find(row => row.label === 'zero-before-final-zero-modifier');
	assert.equal(zero.afterBurn, 0);
	assert.equal(zero.preFinal, 1);
	assert.equal(zero.result, 0);
});

test('temporary oracle patches and state restore after success and injected failure', () => {
	oracle.assertReady();
	const ps = require('../reference/vendor/pokemon-showdown/dist/sim');
	const battle = createBattle(ps);
	const source = battle.p1.pokemon[0];
	const target = battle.p2.pokemon[0];
	const battleKeys = ['randomizer', 'random', 'modify', 'runEvent', 'events'];
	const beforeBattle = ownDescriptors(battle, battleKeys);
	const beforeEffectiveness = Object.getOwnPropertyDescriptor(target, 'runEffectiveness');
	const beforeStatus = source.status;
	const beforeAbility = source.ability;
	try {
		capture(battle, { preRollDamage: 111, roll: 3, forceSTAB: true, typeExponent: 1, status: 'brn', finalModifier: 6144 });
		for (const key of battleKeys) assert.deepEqual(Object.getOwnPropertyDescriptor(battle, key), beforeBattle[key]);
		assert.deepEqual(Object.getOwnPropertyDescriptor(target, 'runEffectiveness'), beforeEffectiveness);
		assert.equal(source.status, beforeStatus);
		assert.equal(source.ability, beforeAbility);

		assert.throws(() => capture(battle, {
			preRollDamage: 222, roll: 5, forceSTAB: true, typeExponent: 2, status: 'brn', finalModifier: 6144,
			failAfterRandomizer: true,
		}), /injected composition capture failure/);
		for (const key of battleKeys) assert.deepEqual(Object.getOwnPropertyDescriptor(battle, key), beforeBattle[key]);
		assert.deepEqual(Object.getOwnPropertyDescriptor(target, 'runEffectiveness'), beforeEffectiveness);
		assert.equal(source.status, beforeStatus);
		assert.equal(source.ability, beforeAbility);
	} finally {
		battle.destroy();
	}
});
