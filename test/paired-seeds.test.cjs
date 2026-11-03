'use strict';

const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const path = require('node:path');
const test = require('node:test');

const oracle = require('../reference/oracle.cjs');
oracle.assertReady();
const ps = require('../reference/vendor/pokemon-showdown/dist/sim');
const referenceLock = require('../reference-lock.json');
const { generateManifest, generatePair, parseArgs, parseRootSeed } = require('../environment/paired-seeds.cjs');
const { runBattle } = require('../environment/battle-runner.cjs');

const ROOT = path.resolve(__dirname, '..');
const CLI = path.join(ROOT, 'environment/paired-seeds.cjs');

test('derivation and pair IDs match a frozen independent SHA-256 vector', () => {
	const manifest = generateManifest([1, 2, 3, 4], 1);
	assert.equal(manifest.derivation.id, 'sha256-json-domain-v1');
	assert.equal(manifest.derivation.domain, 'gen5-paired-seeds-v1');
	assert.equal(manifest.pairs[0].pairId, 'pair-000001-12d2fb155c77fba5');
	assert.deepEqual(manifest.pairs[0].teams.A.seed, [44745, 2544, 49753, 30028]);
	assert.deepEqual(manifest.pairs[0].teams.B.seed, [38670, 47315, 18808, 18778]);
	assert.deepEqual(manifest.pairs[0].runs[0].seeds.battle, [14129, 19999, 37816, 13386]);
	assert.deepEqual(manifest.pairs[0].runs[0].seeds.policyP1, [48610, 58908, 33605, 29270]);
	assert.deepEqual(manifest.pairs[0].runs[0].seeds.policyP2, [40788, 37485, 58001, 62354]);
});

test('manifest emits expanded base and swapped tuples with independent seed arrays and stable prefixes', () => {
	const shortManifest = generateManifest([17, 0, 4, 9], 1);
	const longManifest = generateManifest([17, 0, 4, 9], 3);
	const pair = longManifest.pairs[0];
	const [base, swapped] = pair.runs;
	assert.deepEqual(shortManifest.pairs[0], pair);
	assert.equal(shortManifest.pairs[0].pairId, longManifest.pairs[0].pairId);
	assert.equal(base.runId, `${pair.pairId}-A-p1`);
	assert.equal(swapped.runId, `${pair.pairId}-B-p1`);
	assert.deepEqual(base.seats, { p1Team: 'A', p2Team: 'B' });
	assert.deepEqual(swapped.seats, { p1Team: 'B', p2Team: 'A' });
	assert.deepEqual(base.policySeats, { p1: 'policyA', p2: 'policyB' });
	assert.deepEqual(swapped.policySeats, { p1: 'policyB', p2: 'policyA' });
	assert.deepEqual(base.seeds.teamP1, swapped.seeds.teamP2);
	assert.deepEqual(base.seeds.teamP2, swapped.seeds.teamP1);
	assert.deepEqual(base.seeds.policyP1, swapped.seeds.policyP2);
	assert.deepEqual(base.seeds.policyP2, swapped.seeds.policyP1);
	assert.deepEqual(base.seeds.battle, swapped.seeds.battle);
	assert.notEqual(base.seeds.teamP1, swapped.seeds.teamP2);
	assert.notEqual(base.seeds.battle, swapped.seeds.battle);
	assert.notEqual(pair.teams.A.seed, base.seeds.teamP1);
	assert.notEqual(pair.teams.A.seed, swapped.seeds.teamP2);
	assert.deepEqual(longManifest.upstream, { repository: referenceLock.repository, commit: referenceLock.commit });
	assert.equal(longManifest.formatId, 'gen5randombattle');
	assert.equal(longManifest.pairCount, 3);
});

test('single-pair selection matches the manifest without sharing the caller root seed', () => {
	for (const pairIndex of [0, 4, 9]) {
		const rootSeed = [0, 65_535, pairIndex + 1, 19];
		const originalRootSeed = [...rootSeed];
		const direct = generatePair(rootSeed, pairIndex);
		assert.deepEqual(direct, generateManifest(originalRootSeed, pairIndex + 1).pairs[pairIndex]);
		assert.notEqual(direct.teams.A.seed, rootSeed);
		rootSeed[0] = 1;
		assert.notDeepEqual(direct.teams.A.seed, generatePair(rootSeed, pairIndex).teams.A.seed);
	}
	for (const rootSeed of [new Array(4), [0, , 2, 3], [1, 2, 3], [1, 2, 3, 65_536]]) {
		assert.throws(() => generatePair(rootSeed, 0));
	}
	for (const pairIndex of [-1, 1.5, Number.NaN, 100_000]) assert.throws(() => generatePair([1, 2, 3, 4], pairIndex));
});

test('CLI output is byte-reproducible across fresh processes and agrees with the library', () => {
	const args = ['--root-seed', '1,2,3,4', '--pairs', '2'];
	const first = spawnSync(process.execPath, [CLI, ...args], { cwd: ROOT, encoding: 'utf8' });
	const second = spawnSync(process.execPath, [CLI, '--root-seed=1,2,3,4', '--pairs=2'], { cwd: ROOT, encoding: 'utf8' });
	assert.equal(first.status, 0, first.stderr);
	assert.equal(second.status, 0, second.stderr);
	assert.equal(first.stdout, second.stdout);
	assert.equal(first.stdout, `${JSON.stringify(generateManifest([1, 2, 3, 4], 2), null, 2)}\n`);
	assert.deepEqual(JSON.parse(first.stdout).pairs.map(pair => pair.pairId), generateManifest([1, 2, 3, 4], 2).pairs.map(pair => pair.pairId));
});

test('input parsing rejects missing, malformed, duplicate, and out-of-domain values', () => {
	for (const seed of [[1, 2, 3], [1, 2, 3, 65_536], [1, -1, 3, 4], [1, 2, 3, 4.5], '1,2,3,04', '1, 2,3,4']) {
		assert.throws(() => parseRootSeed(seed));
	}
	for (const args of [
		[], ['--root-seed', '1,2,3,4'], ['--pairs', '2'],
		['--root-seed', '1,2,3,4', '--pairs', '0'],
		['--root-seed', '1,2,3,4', '--pairs', '-1'],
		['--root-seed', '1,2,3,4', '--pairs', '1.5'],
		['--root-seed', '1,2,3,4', '--pairs', '100001'],
		['--root-seed', '1,2,3,4', '--pairs', '1', '--pairs', '2'],
		['--root-seed', '1,2,3,4', '--pairs', '1', '--surprise'],
	]) {
		assert.throws(() => parseArgs(args));
	}
	for (const seed of [new Array(4), [0, , 2, 3], [0, 1, 2], [0, 1, 2, 65_536]]) {
		assert.throws(() => generateManifest(seed, 1));
	}
	for (const pairCount of [undefined, 0, -1, 1.5, Number.NaN, Number.POSITIVE_INFINITY, 100_001]) {
		assert.throws(() => generateManifest([1, 2, 3, 4], pairCount));
	}
	assert.deepEqual(generateManifest([0, 65_535, 0, 65_535], 1).rootSeed, [0, 65_535, 0, 65_535]);
	assert.deepEqual(parseArgs(['--root-seed=1,2,3,4', '--pairs=1']), { rootSeed: [1, 2, 3, 4], pairCount: 1 });
});

function recordingStreamFactory(teamWrites) {
	return () => {
		const stream = new ps.BattleStream({ keepAlive: true });
		return {
			write(command) {
				const match = /^>player (p[12]) (.+)$/.exec(command);
				if (match) {
					const player = match[1];
					const payload = JSON.parse(match[2]);
					assert.equal(payload.name, player);
					teamWrites.push({ player, packedTeam: payload.team });
				}
				return stream.write(command);
			},
			destroy() { return stream.destroy(); },
			[Symbol.asyncIterator]() { return stream[Symbol.asyncIterator](); },
		};
	};
}

function packedTeamsBySeat(writes) {
	assert.deepEqual(writes.map(write => write.player), ['p1', 'p2']);
	return Object.fromEntries(writes.map(write => [write.player, write.packedTeam]));
}

test('real pinned paired battles complete and regenerate identical teams in exchanged seats', { timeout: 180_000 }, async () => {
	const manifest = generateManifest([1, 2, 3, 4], 1);
	const pair = manifest.pairs[0];
	const actualWrites = [];
	const results = [];
	for (const run of pair.runs) {
		const writes = [];
		const result = await runBattle({
			ps,
			seeds: run.seeds,
			streamFactory: recordingStreamFactory(writes),
		});
		assert.equal(result.resultRecord.status, 'completed');
		assert.ok(result.summary.outcome.turns > 0);
		assert.equal(result.summary.upstreamCommit, referenceLock.commit);
		assert.equal(result.summary.formatId, manifest.formatId);
		actualWrites.push(packedTeamsBySeat(writes));
		results.push(result);
	}
	assert.deepEqual(actualWrites[0].p1, actualWrites[1].p2);
	assert.deepEqual(actualWrites[0].p2, actualWrites[1].p1);
	assert.deepEqual(actualWrites[0].p1, ps.Teams.pack(ps.Teams.generate(manifest.formatId, { seed: pair.teams.A.seed })));
	assert.deepEqual(actualWrites[0].p2, ps.Teams.pack(ps.Teams.generate(manifest.formatId, { seed: pair.teams.B.seed })));
	assert.deepEqual(actualWrites[1].p1, ps.Teams.pack(ps.Teams.generate(manifest.formatId, { seed: pair.teams.B.seed })));
	assert.deepEqual(actualWrites[1].p2, ps.Teams.pack(ps.Teams.generate(manifest.formatId, { seed: pair.teams.A.seed })));
	assert.deepEqual(pair.runs[0].seeds.policyP1, pair.policySeeds.policyA);
	assert.deepEqual(pair.runs[0].seeds.policyP2, pair.policySeeds.policyB);
	assert.deepEqual(pair.runs[1].seeds.policyP1, pair.runs[0].seeds.policyP2);
	assert.deepEqual(pair.runs[1].seeds.policyP2, pair.runs[0].seeds.policyP1);
	assert.deepEqual(results.map(result => result.resultRecord.status), ['completed', 'completed']);
});
