'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const oracle = require('../reference/oracle.cjs');
const setup = require('../reference/setup.cjs');
const lock = require('../reference-lock.json');

const fixturePath = name => path.join(__dirname, '..', 'reference', 'fixtures', name);

test('reference identity guard accepts only the pinned clean checkout', () => {
	assert.doesNotThrow(() => oracle.assertReferenceIdentity(lock, {
		origin: lock.repository,
		commit: lock.commit,
		dirty: '',
	}));
	assert.throws(() => oracle.assertReferenceIdentity(lock, {
		origin: lock.repository,
		commit: '0'.repeat(40),
		dirty: '',
	}), /commit mismatch/);
	assert.throws(() => oracle.assertReferenceIdentity(lock, {
		origin: 'https://example.invalid/pokemon-showdown.git',
		commit: lock.commit,
		dirty: '',
	}), /origin mismatch/);
	assert.throws(() => oracle.assertReferenceIdentity(lock, {
		origin: lock.repository,
		commit: lock.commit,
		dirty: ' M sim/battle.ts',
	}), /checkout is modified/);
});

test('real Git checkout inspection rejects changed source, untracked files, wrong commit, and wrong origin', t => {
	const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'showdown-guard-test-'));
	t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
	const git = (...args) => execFileSync('git', ['-c', `safe.directory=${directory}`, ...args], {
		cwd: directory,
		encoding: 'utf8',
		stdio: ['ignore', 'pipe', 'pipe'],
	}).trim();
	git('init');
	git('config', 'user.name', 'Reference guard test');
	git('config', 'user.email', 'reference-guard@example.invalid');
	fs.mkdirSync(path.join(directory, 'src'));
	fs.writeFileSync(path.join(directory, 'src', 'engine.js'), 'module.exports = 1;\n');
	git('add', 'src/engine.js');
	git('commit', '-m', 'fixture');
	git('remote', 'add', 'origin', lock.repository);
	const expected = { repository: lock.repository, commit: git('rev-parse', 'HEAD') };
	assert.deepEqual(setup.inspectCheckout(directory), {
		origin: lock.repository,
		commit: expected.commit,
		dirty: '',
	});
	assert.doesNotThrow(() => setup.assertReferenceIdentity(expected, setup.inspectCheckout(directory)));

	fs.writeFileSync(path.join(directory, 'src', 'engine.js'), 'module.exports = 2;\n');
	assert.throws(() => setup.assertReferenceIdentity(expected, setup.inspectCheckout(directory)), /checkout is modified/);
	git('checkout', '--', 'src/engine.js');

	fs.writeFileSync(path.join(directory, 'src', 'extra.js'), 'module.exports = 3;\n');
	assert.throws(() => setup.assertReferenceIdentity(expected, setup.inspectCheckout(directory)), /checkout is modified/);
	fs.unlinkSync(path.join(directory, 'src', 'extra.js'));

	fs.writeFileSync(path.join(directory, 'src', 'engine.js'), 'module.exports = 4;\n');
	git('add', 'src/engine.js');
	git('commit', '-m', 'different pin');
	assert.throws(() => setup.assertReferenceIdentity(expected, setup.inspectCheckout(directory)), /commit mismatch/);
	git('remote', 'set-url', 'origin', 'https://example.invalid/showdown.git');
	assert.throws(() => setup.assertReferenceIdentity(expected, setup.inspectCheckout(directory)), /origin mismatch/);
});

test('resolved format facts match the pinned Showdown manifest', () => {
	oracle.assertReady();
	const ps = require('../reference/vendor/pokemon-showdown/dist/sim');
	const manifest = oracle.formatManifest(ps);
	assert.deepEqual(manifest, JSON.parse(fs.readFileSync(fixturePath('format-v1.json'), 'utf8')));
	assert.equal(manifest.formatId, 'gen5randombattle');
	assert.equal(manifest.generation, 5);
	assert.equal(manifest.gameType, 'singles');
	assert.equal(manifest.teamGenerator, 'random');
	assert.equal(manifest.teamPreview, false);
	assert.ok(manifest.declaredRules.includes('Illusion Level Mod'));
	assert.ok(manifest.effectiveRules.some(rule => rule.id === 'sleepclausemod'));
	assert.ok(manifest.effectiveValueRules.some(rule => rule.id === 'evlimit'));
	assert.ok(manifest.modInheritance.includes('gen6'));
	assert.equal(manifest.upstreamCommit, lock.commit);
});

test('seeded teams and initial requests match the pinned semantic fixture', async () => {
	oracle.assertReady();
	const ps = require('../reference/vendor/pokemon-showdown/dist/sim');
	const actual = await oracle.smoke(ps, oracle.DEFAULT_SEEDS);
	const repeated = await oracle.smoke(ps, oracle.DEFAULT_SEEDS);
	const expected = JSON.parse(fs.readFileSync(fixturePath('smoke-v1.json'), 'utf8'));
	assert.deepEqual(actual, expected);
	assert.deepEqual(repeated, actual, 'independent initializations with the same seeds should match');
	for (const team of actual.teams) {
		assert.equal(team.length, 6);
		for (const pokemon of team) assert.equal(pokemon.moves.length, 4);
	}
	assert.deepEqual(Object.keys(actual.initialBattle.playerRequests).sort(), ['p1', 'p2']);
	assert.equal(actual.initialBattle.generation, 5);
	assert.equal(actual.initialBattle.gameType, 'singles');
	assert.equal(actual.initialBattle.teamPreview, false);
});

test('smoke seeds are separate and must each contain four integers', () => {
	assert.deepEqual(oracle.parseSmokeArgs([
		'--team-seed-p1=13,14,15,16',
		'--team-seed-p2', '17,18,19,20',
		'--battle-seed=21,22,23,24',
	]), {
		teamP1: [13, 14, 15, 16],
		teamP2: [17, 18, 19, 20],
		battle: [21, 22, 23, 24],
	});
	assert.deepEqual(oracle.parseSeed('0,1,65534,65535', 'test'), [0, 1, 65534, 65535]);
	for (const invalid of ['', '1,2', '1,,3,4', '0x10,2,3,4', '1e2,2,3,4', '1,2,3,65536', '1,2,3,99999']) {
		assert.throws(() => oracle.parseSeed(invalid, 'test'), /Gen 5 seed limbs/);
	}
	assert.throws(() => oracle.parseSmokeArgs(['--battle-seed=1,2,3,4=5']), /Invalid seed option/);
	assert.throws(() => oracle.parseSmokeArgs(['--battle-seed=1,2']), /Gen 5 seed limbs/);
});

test('initial battle facts come from protocol events and requests match submitted teams', () => {
	const teams = ['p1', 'p2'].map(() => Array.from({ length: 6 }, (_, i) => ({ species: `Testmon${i + 1}` })));
	const requests = Object.fromEntries(['p1', 'p2'].map((side, sideIndex) => [side, {
		side: { pokemon: teams[sideIndex].map((pokemon, i) => ({ details: `${pokemon.species}, L50` })) },
	}]));
	const protocol = ['|gen|5', '|gametype|singles', '|teamsize|p1|6', '|teamsize|p2|6'];
	assert.deepEqual(oracle.validateInitialBattle(protocol, requests, teams, value => value.toLowerCase()), {
		generation: 5,
		gameType: 'singles',
		teamPreview: false,
	});
	assert.throws(() => oracle.validateInitialBattle(protocol.map(line => line === '|gen|5' ? '|gen|6' : line), requests, teams, value => value.toLowerCase()), /Expected Gen 5/);
	assert.throws(() => oracle.validateInitialBattle([...protocol, '|teampreview'], requests, teams, value => value.toLowerCase()), /Unexpected Team Preview/);
	const mismatchedRequests = structuredClone(requests);
	mismatchedRequests.p1.side.pokemon[0].details = 'Othermon, L50';
	assert.throws(() => oracle.validateInitialBattle(protocol, mismatchedRequests, teams, value => value.toLowerCase()), /does not match submitted team/);
});

test('initialization reports stream errors/end and destroys the stream on every path', async () => {
	const teams = ['p1', 'p2'].map(side => Array.from({ length: 6 }, (_, i) => ({ species: `${side}mon${i + 1}` })));
	const makePs = packet => {
		let destroyed = false;
		return {
			ps: {
				BattleStream: class {
					write() {}
					[Symbol.asyncIterator]() { return { next: async () => packet }; }
					destroy() { destroyed = true; }
				},
				Teams: { pack: () => 'fixture' },
				toID: value => value.toLowerCase(),
			},
			destroyed: () => destroyed,
		};
	};
	const failed = makePs({ value: 'error\ninvalid battle', done: false });
	await assert.rejects(oracle.initialBattle(failed.ps, teams, [1, 2, 3, 4]), /Showdown stream error/);
	assert.equal(failed.destroyed(), true);
	const ended = makePs({ value: undefined, done: true });
	await assert.rejects(oracle.initialBattle(ended.ps, teams, [1, 2, 3, 4]), /stream ended/);
	assert.equal(ended.destroyed(), true);

	let resolvePending;
	let destroyed = false;
	const hangingPs = {
		BattleStream: class {
			write() {}
			[Symbol.asyncIterator]() { return { next: () => new Promise(resolve => { resolvePending = resolve; }) }; }
			destroy() { destroyed = true; resolvePending?.({ done: true }); }
		},
		Teams: { pack: () => 'fixture' },
		toID: value => value.toLowerCase(),
	};
	await assert.rejects(oracle.initialBattle(hangingPs, teams, [1, 2, 3, 4], 5), /timed out after 5 ms/);
	assert.equal(destroyed, true);
});
