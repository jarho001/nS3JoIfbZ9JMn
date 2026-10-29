'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const oracle = require('../reference/burn-oracle.cjs');
require('../reference/oracle.cjs').assertReady();

test('burn oracle reproduces guarded ordinary and synthetic cases', () => {
	const fixture = oracle.generateFixture(require('../reference/vendor/pokemon-showdown/dist/sim'));
	assert.equal(oracle.stringifyFixture(fixture), require('node:fs').readFileSync(
		require('node:path').join(__dirname, '../reference/fixtures/burn-v1.json'), 'utf8'));
	assert.equal(fixture.upstream.commit, '1242c1e2d9763ba5b85d33f13aecb89bae19ae3d');
	assert.equal(fixture.formatId, 'gen5randombattle');
	assert.equal(fixture.inputDomain.damage[1], 2147483647);
	const ordinary = Object.fromEntries(fixture.ordinaryCases.map(row => [row.label, row]));
	assert.equal(ordinary['burned-physical'].modifyCalls, 1);
	assert.equal(ordinary['burned-special'].modifyCalls, 0);
	assert.equal(ordinary['burned-physical-guts'].modifyCalls, 0);
	assert.equal(ordinary['burned-facade-gen5'].modifyCalls, 1);
	assert.equal(ordinary['unburned-physical'].modifyCalls, 0);
	assert.deepEqual(fixture.syntheticCases.map(row => row.damage), [
		0, 1, 2, 3, 4, 5, 2 ** 21 - 1, 2 ** 21, 2 ** 21 + 1, 2147483647,
	]);
	assert.equal(fixture.syntheticCases[1].burnOutput, 0);
	assert.equal(fixture.syntheticCases[7].burnOutput, 0);
	assert.equal(fixture.syntheticCases[7].postBurnAndGen5FinalDamage, 1);
});

test('synthetic oracle restores wrappers and status after success and failure', () => {
	const ps = require('../reference/vendor/pokemon-showdown/dist/sim');
	const battle = oracle.createBattle(ps);
	try {
		const source = battle.p1.pokemon[0];
		const beforeStatus = source.status;
		const before = {
			randomizer: Object.getOwnPropertyDescriptor(battle, 'randomizer'),
			modify: Object.getOwnPropertyDescriptor(battle, 'modify'),
		};
		const assertRestored = () => {
			assert.deepEqual(Object.getOwnPropertyDescriptor(battle, 'randomizer'), before.randomizer);
			assert.deepEqual(Object.getOwnPropertyDescriptor(battle, 'modify'), before.modify);
			assert.equal(source.status, beforeStatus);
		};
		assert.equal(oracle.captureSynthetic(battle, 65).burnOutput, 32);
		assertRestored();
		assert.throws(() => oracle.captureSynthetic(battle, NaN), /synthetic damage must be an integer/);
		assertRestored();
	} finally {
		battle.destroy();
	}
});
