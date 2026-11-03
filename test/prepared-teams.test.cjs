'use strict';

const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const test = require('node:test');

const oracle = require('../reference/oracle.cjs');
oracle.assertReady();
const ps = require('../reference/vendor/pokemon-showdown/dist/sim');
const { generateManifest } = require('../environment/paired-seeds.cjs');
const { runBattle } = require('../environment/battle-runner.cjs');
const { prepareTeam } = require('../environment/prepared-teams.cjs');
const { randomPolicy } = require('../environment/random-policy.cjs');
const referenceLock = require('../reference-lock.json');

const DEFAULTS = {
	teamP1: [1, 2, 3, 4], teamP2: [5, 6, 7, 8], battle: [9, 10, 11, 12],
	policyP1: [13, 14, 15, 16], policyP2: [17, 18, 19, 20],
};

function fakePs() {
	return {
		BattleStream: class {},
		Dex: { formats: { get(formatId) { return { id: formatId, exists: formatId === 'gen5randombattle' }; } } },
		Teams: {
			generate(formatId, { seed }) { return { formatId, seed: [...seed] }; },
			pack(team) { return `${team.formatId}:${team.seed.join(',')}`; },
		},
	};
}

async function failureOf(promise) {
	try {
		await promise;
	} catch (error) {
		return error;
	}
	assert.fail('Expected runner failure.');
}

function recordingStreamFactory(teamWrites) {
	return () => {
		const stream = new ps.BattleStream({ keepAlive: true });
		return {
			write(command) {
				const match = /^>player (p[12]) (.+)$/.exec(command);
				if (match) teamWrites[match[1]] = JSON.parse(match[2]).team;
				return stream.write(command);
			},
			destroy() { return stream.destroy(); },
			[Symbol.asyncIterator]() { return stream[Symbol.asyncIterator](); },
		};
	};
}

test('prepareTeam validates and copies four indexed uint16 limbs and produces immutable opaque handles', () => {
	const stub = fakePs();
	const sourceSeed = [0, 65_535, 3, 4];
	const handle = prepareTeam(stub, sourceSeed);
	sourceSeed[0] = 1;
	assert.deepEqual(handle.seed, [0, 65_535, 3, 4]);
	assert.equal(handle.formatId, 'gen5randombattle');
	assert.equal(handle.upstreamCommit, referenceLock.commit);
	assert.equal(handle.packedTeamSha256, crypto.createHash('sha256').update('gen5randombattle:0,65535,3,4').digest('hex'));
	assert.ok(Object.isFrozen(handle));
	assert.ok(Object.isFrozen(handle.seed));
	assert.throws(() => { handle.seed[0] = 5; }, TypeError);
	for (const seed of [new Array(4), [0, , 2, 3], [1, 2, 3], [0, 1, 2, 65_536], [0, -1, 2, 3]]) {
		assert.throws(() => prepareTeam(stub, seed), /four integer limbs/);
	}
	assert.throws(() => prepareTeam({ Teams: { generate: () => [], pack: () => '' }, Dex: { formats: { get: () => ({ exists: true }) } } }, [1, 2, 3, 4]), /invalid packed team/);
	assert.throws(() => prepareTeam({ Teams: { generate() {}, pack() {} }, Dex: { formats: { get: () => ({ exists: false }) } } }, [1, 2, 3, 4]), /Pinned Showdown format/);
});

test('runBattle rejects forged, cloned, mismatched, or incomplete handles before creating a stream', async () => {
	const firstPs = fakePs();
	const otherPs = fakePs();
	const p1 = prepareTeam(firstPs, DEFAULTS.teamP1);
	const p2 = prepareTeam(firstPs, DEFAULTS.teamP2);
	const foreignP2 = prepareTeam(otherPs, DEFAULTS.teamP2);
	let streamFactories = 0;
	const run = (showdown, preparedTeams, seeds = DEFAULTS) => runBattle({
		ps: showdown, seeds, preparedTeams,
		streamFactory: () => { streamFactories++; throw new Error('stream should not be initialized'); },
	});
	const mismatchSeeds = { ...DEFAULTS, teamP1: DEFAULTS.teamP2 };
	for (const [showdown, preparedTeams, seeds] of [
		[firstPs, { p1, p2: { ...p2 } }, DEFAULTS],
		[firstPs, { p1, p2: { ...p2, formatId: 'other-format' } }, DEFAULTS],
		[firstPs, { p1, p2: foreignP2 }, DEFAULTS],
		[firstPs, { p1: p2, p2 }, DEFAULTS],
		[firstPs, { p1 }, DEFAULTS],
		[firstPs, null, DEFAULTS],
		[firstPs, [p1, p2], DEFAULTS],
		[firstPs, { p1, p2 }, mismatchSeeds],
		[otherPs, { p1, p2 }, DEFAULTS],
	]) {
		const error = await failureOf(run(showdown, preparedTeams, seeds));
		assert.equal(error.resultRecord.failure.category, 'initialization');
		assert.equal(error.resultRecord.failure.code, 'invalid_prepared_teams');
		assert.equal(JSON.stringify(error.resultRecord).includes(p1.packedTeamSha256), false);
		assert.equal(Object.hasOwn(error.resultRecord, 'teamSource'), false);
	}
	assert.equal(streamFactories, 0);
});

test('pinned generated and prepared battles match while prepared teams are generated and packed once for both seats', { timeout: 180_000 }, async () => {
	const [pair] = generateManifest([1, 2, 3, 4], 1).pairs;
	const [base, swapped] = pair.runs;
	const policy = input => {
		assert.deepEqual(Object.keys(input).sort(), ['history', 'random', 'request']);
		return randomPolicy(input);
	};
	const policies = { p1: policy, p2: policy };
	const normal = await runBattle({ ps, seeds: base.seeds, policies });
	assert.equal(Object.hasOwn(normal.resultRecord, 'teamSource'), false);

	const originalGenerate = ps.Teams.generate;
	const originalPack = ps.Teams.pack;
	let generations = 0;
	let packs = 0;
	ps.Teams.generate = function (...args) {
		generations++;
		return originalGenerate.apply(this, args);
	};
	ps.Teams.pack = function (...args) {
		packs++;
		return originalPack.apply(this, args);
	};
	try {
		const teamA = prepareTeam(ps, pair.teams.A.seed);
		const teamB = prepareTeam(ps, pair.teams.B.seed);
		assert.equal(generations, 2);
		assert.equal(packs, 2);
		const writes = [{}, {}];
		const prepared = await runBattle({
			ps,
			seeds: base.seeds,
			policies,
			preparedTeams: { p1: teamA, p2: teamB },
			streamFactory: recordingStreamFactory(writes[0]),
		});
		const reversed = await runBattle({
			ps,
			seeds: swapped.seeds,
			policies,
			preparedTeams: { p1: teamB, p2: teamA },
			streamFactory: recordingStreamFactory(writes[1]),
		});
		assert.deepEqual(prepared.summary, normal.summary);
		assert.deepEqual(prepared.transcript, normal.transcript);
		assert.deepEqual(writes[1], { p1: writes[0].p2, p2: writes[0].p1 });
		assert.equal(crypto.createHash('sha256').update(writes[0].p1).digest('hex'), teamA.packedTeamSha256);
		assert.equal(crypto.createHash('sha256').update(writes[0].p2).digest('hex'), teamB.packedTeamSha256);
		assert.equal(crypto.createHash('sha256').update(writes[1].p1).digest('hex'), teamB.packedTeamSha256);
		assert.equal(crypto.createHash('sha256').update(writes[1].p2).digest('hex'), teamA.packedTeamSha256);
		assert.equal(prepared.resultRecord.status, 'completed');
		assert.equal(reversed.resultRecord.status, 'completed');
		assert.equal(prepared.resultRecord.teamSource, 'prepared');
		assert.equal(reversed.resultRecord.teamSource, 'prepared');
		assert.equal(generations, 2);
		assert.equal(packs, 2);
	} finally {
		ps.Teams.generate = originalGenerate;
		ps.Teams.pack = originalPack;
	}
});
