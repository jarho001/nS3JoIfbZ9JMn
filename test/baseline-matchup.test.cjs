'use strict';

const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const test = require('node:test');

const oracle = require('../reference/oracle.cjs');
oracle.assertReady();
const ps = require('../reference/vendor/pokemon-showdown/dist/sim');
const referenceLock = require('../reference-lock.json');
const { MAX_PAIRS, generatePair, generateManifest } = require('../environment/paired-seeds.cjs');
const { DEFAULT_LIMITS } = require('../environment/battle-runner.cjs');
const { createPolicyRng } = require('../environment/random-policy.cjs');
const { runBaselineMatchup, KIND } = require('../environment/baseline-matchup.cjs');
const { summarizeResults } = require('../environment/summarize-results.cjs');

const ROOT_SEED = [1, 2, 3, 4];
const FAILURE = 'Baseline paired matchup failed (baseline_matchup_failed).';

function moveRequest() {
	return {
		active: [{ moves: [{ move: 'Tackle', id: 'tackle', pp: 35, disabled: false }] }],
		side: { pokemon: [
			{ active: true, condition: '100/100', details: 'Ownmon' },
			{ active: false, condition: '100/100', details: 'Benchmon' },
			{ active: false, condition: '0 fnt', details: 'Faintedmon' },
		] },
	};
}

function terminalMetadata(winner) {
	return JSON.stringify({ winner, turns: 2, p1: 'p1', p2: 'p2', score: winner === 'p1' ? [1, 0] : [0, 1] });
}

function completedPackets() {
	return [
		'update\n|gen|5\n|gametype|singles\n|turn|1',
		`sideupdate\np1\n|request|${JSON.stringify(moveRequest())}`,
		`sideupdate\np2\n|request|${JSON.stringify(moveRequest())}`,
		'update\n|move|p1a: Ownmon|Tackle|p2a: Other\n|turn|2\n|win|p1',
		`end\n${terminalMetadata('p1')}`,
	];
}

class ControlledStream {
	constructor(packets, { release, destroyFailure, onDestroy } = {}) {
		this.packets = packets;
		this.release = release;
		this.destroyFailure = destroyFailure;
		this.onDestroy = onDestroy;
		this.writes = [];
		this.destroyCalls = 0;
		this.destroyed = false;
	}

	write(command) {
		this.writes.push(command);
	}

	[Symbol.asyncIterator]() {
		let index = 0;
		let paused = false;
		return {
			next: async () => {
				if (!paused && this.release) {
					paused = true;
					await this.release;
				}
				return index < this.packets.length
					? { value: this.packets[index++], done: false }
					: { value: undefined, done: true };
			},
		};
	}

	async destroy() {
		this.destroyCalls++;
		this.destroyed = true;
		this.onDestroy?.();
		if (this.destroyFailure) throw new Error('private cleanup detail');
	}
}

function controlledEnvironment({ packetsForCall, cleanupFailureCalls = new Set(), releaseFirst } = {}) {
	const counts = { generatedSeeds: [], packCalls: [], factoryCalls: 0, activeStreams: 0, maxActiveStreams: 0, streams: [] };
	const fakePs = {
		BattleStream: class {},
		Dex: { formats: { get(formatId) { return { id: formatId, exists: formatId === 'gen5randombattle' }; } } },
		Teams: {
			generate(formatId, { seed }) {
				counts.generatedSeeds.push([...seed]);
				return { formatId, seed: [...seed] };
			},
			pack(team) {
				counts.packCalls.push([...team.seed]);
				return `packed:${team.seed.join(',')}`;
			},
		},
	};
	const streamFactory = () => {
		counts.factoryCalls++;
		assert.equal(counts.activeStreams, 0, 'a second stream started before the first was destroyed');
		counts.activeStreams++;
		counts.maxActiveStreams = Math.max(counts.maxActiveStreams, counts.activeStreams);
		const call = counts.factoryCalls;
		const stream = new ControlledStream(
			packetsForCall ? packetsForCall(call) : completedPackets(),
			{
				release: call === 1 ? releaseFirst : undefined,
				destroyFailure: cleanupFailureCalls.has(call),
				onDestroy: () => { counts.activeStreams--; },
			},
		);
		counts.streams.push(stream);
		return stream;
	};
	return { ps: fakePs, counts, streamFactory };
}

function optionsFor(environment, overrides = {}) {
	return {
		rootSeed: [...ROOT_SEED],
		pairIndex: 0,
		policyA: 'legal-random-v1',
		policyB: 'move-first-v1',
		ps: environment.ps,
		streamFactory: environment.streamFactory,
		...overrides,
	};
}

async function assertFixedFailure(promise) {
	await assert.rejects(promise, error => {
		assert.equal(error.message, FAILURE);
		assert.deepEqual(Object.keys(error), []);
		assert.equal(Object.hasOwn(error, 'cause'), false);
		return true;
	});
}

function policyCommands(stream) {
	return stream.writes.filter(command => /^>p[12] /.test(command));
}

test('named matchup completes both exchanged seats with policy identities and role seeds together', async () => {
	const environment = controlledEnvironment();
	const artifact = await runBaselineMatchup(optionsFor(environment));

	assert.deepEqual(Object.keys(artifact).sort(), [
		'formatId', 'kind', 'limits', 'pairId', 'pairIndex', 'policiesByRole', 'policySeeds',
		'rootSeed', 'runs', 'schemaVersion', 'status', 'teams', 'upstream',
	].sort());
	assert.equal(artifact.schemaVersion, 1);
	assert.equal(artifact.kind, KIND);
	assert.equal(artifact.status, 'completed');
	assert.equal(artifact.formatId, 'gen5randombattle');
	assert.deepEqual(artifact.upstream, { repository: referenceLock.repository, commit: referenceLock.commit });
	assert.deepEqual(artifact.rootSeed, [1, 2, 3, 4]);
	assert.equal(artifact.pairIndex, 0);
	assert.equal(artifact.pairId, 'pair-000001-12d2fb155c77fba5');
	assert.deepEqual(artifact.policiesByRole, { policyA: 'legal-random-v1', policyB: 'move-first-v1' });
	assert.deepEqual(artifact.policySeeds, {
		policyA: [48610, 58908, 33605, 29270],
		policyB: [40788, 37485, 58001, 62354],
	});
	assert.deepEqual(artifact.limits, DEFAULT_LIMITS);
	assert.deepEqual(environment.counts.generatedSeeds, [
		[44745, 2544, 49753, 30028],
		[38670, 47315, 18808, 18778],
	]);
	assert.deepEqual(environment.counts.packCalls, environment.counts.generatedSeeds);
	assert.equal(environment.counts.factoryCalls, 2);
	assert.equal(environment.counts.maxActiveStreams, 1);
	assert.deepEqual(environment.counts.streams.map(stream => stream.destroyCalls), [1, 1]);
	assert.deepEqual(environment.counts.streams.map(policyCommands), [
		['>p1 switch 2', '>p2 move 1'],
		['>p1 move 1', '>p2 switch 2'],
	]);

	assert.deepEqual(artifact.runs.map(run => ({
		runId: run.runId,
		seats: run.seats,
		policySeats: run.policySeats,
		policyIdsBySeat: run.policyIdsBySeat,
		status: run.status,
	})), [
		{
			runId: 'pair-000001-12d2fb155c77fba5-A-p1',
			seats: { p1Team: 'A', p2Team: 'B' },
			policySeats: { p1: 'policyA', p2: 'policyB' },
			policyIdsBySeat: { p1: 'legal-random-v1', p2: 'move-first-v1' },
			status: 'completed',
		},
		{
			runId: 'pair-000001-12d2fb155c77fba5-B-p1',
			seats: { p1Team: 'B', p2Team: 'A' },
			policySeats: { p1: 'policyB', p2: 'policyA' },
			policyIdsBySeat: { p1: 'move-first-v1', p2: 'legal-random-v1' },
			status: 'completed',
		},
	]);
	assert.deepEqual(artifact.runs.map(run => run.seeds), [
		{
			teamP1: [44745, 2544, 49753, 30028], teamP2: [38670, 47315, 18808, 18778],
			battle: [14129, 19999, 37816, 13386],
			policyP1: [48610, 58908, 33605, 29270], policyP2: [40788, 37485, 58001, 62354],
		},
		{
			teamP1: [38670, 47315, 18808, 18778], teamP2: [44745, 2544, 49753, 30028],
			battle: [14129, 19999, 37816, 13386],
			policyP1: [40788, 37485, 58001, 62354], policyP2: [48610, 58908, 33605, 29270],
		},
	]);
	for (const run of artifact.runs) {
		assert.equal(run.resultRecord.kind, 'battle-run-result');
		assert.equal(run.resultRecord.status, 'completed');
		assert.equal(run.resultRecord.teamSource, 'prepared');
		assert.equal(run.resultRecord.failure, null);
		assert.equal(run.resultRecord.cleanupFailure, null);
		assert.deepEqual(run.resultRecord.limits, DEFAULT_LIMITS);
		assert.deepEqual(run.resultRecord.seeds, run.seeds);
		assert.deepEqual(run.resultRecord.outcome, { result: 'win', winner: 'p1', turns: 2 });
		assert.equal(Object.hasOwn(run, 'transcript'), false);
	}
	const sha256 = value => crypto.createHash('sha256').update(value, 'utf8').digest('hex');
	assert.deepEqual(artifact.teams, {
		A: { seed: [44745, 2544, 49753, 30028], packedTeamSha256: sha256('packed:44745,2544,49753,30028') },
		B: { seed: [38670, 47315, 18808, 18778], packedTeamSha256: sha256('packed:38670,47315,18808,18778') },
	});
	const expectedActions = [
		{
			move: { total: 1, byPlayer: { p1: 0, p2: 1 } },
			switch: { total: 1, byPlayer: { p1: 1, p2: 0 } },
		},
		{
			move: { total: 1, byPlayer: { p1: 1, p2: 0 } },
			switch: { total: 1, byPlayer: { p1: 0, p2: 1 } },
		},
	];
	for (const [index, run] of artifact.runs.entries()) {
		const counters = JSON.parse(JSON.stringify(run.resultRecord.counters));
		assert.equal(counters.objectPackets, 5);
		assert.deepEqual(counters.policyInvocations, { total: 2, byPlayer: { p1: 1, p2: 1 } });
		assert.deepEqual(counters.choiceSubmissions, { total: 2, byPlayer: { p1: 1, p2: 1 } });
		assert.deepEqual(counters.choiceResolutions, {
			accepted: { total: 2, byPlayer: { p1: 1, p2: 1 } },
			rejected: { total: 0, byPlayer: { p1: 0, p2: 0 } },
			pending: { total: 0, byPlayer: { p1: 0, p2: 0 } },
		});
		assert.deepEqual(counters.retryRequests, { total: 0, byPlayer: { p1: 0, p2: 0 } });
		assert.deepEqual(counters.retrySubmissions, { total: 0, byPlayer: { p1: 0, p2: 0 } });
		assert.deepEqual(counters.retryLimitExhaustions, { total: 0, byPlayer: { p1: 0, p2: 0 } });
		assert.deepEqual(counters.actions.selected, expectedActions[index]);
		assert.deepEqual(counters.actions.submitted, expectedActions[index]);
		assert.equal(counters.publicEvents, 6);
		assert.equal(counters.latestPublicTurn, 2);
		assert.equal(counters.terminalMetadataTurns, 2);
	}
	assert.equal(JSON.stringify(artifact).includes('packed:'), false);
	assert.equal(JSON.stringify(artifact).includes('transcript'), false);
	assert.equal(JSON.stringify(artifact).includes('Ownmon'), false);
});

test('named matchup accepts the reverse policy ordering and same-ID self-play with separate role streams', async () => {
	const reversed = controlledEnvironment();
	const reversedArtifact = await runBaselineMatchup(optionsFor(reversed, {
		policyA: 'move-first-v1', policyB: 'legal-random-v1',
	}));
	assert.deepEqual(reversed.counts.streams.map(policyCommands), [
		['>p1 move 1', '>p2 switch 2'],
		['>p1 switch 2', '>p2 move 1'],
	]);
	assert.deepEqual(reversedArtifact.runs.map(run => run.policyIdsBySeat), [
		{ p1: 'move-first-v1', p2: 'legal-random-v1' },
		{ p1: 'legal-random-v1', p2: 'move-first-v1' },
	]);

	const selfPlay = controlledEnvironment();
	const selfPlayArtifact = await runBaselineMatchup(optionsFor(selfPlay, {
		policyA: 'legal-random-v1', policyB: 'legal-random-v1',
	}));
	assert.deepEqual(selfPlayArtifact.policiesByRole, { policyA: 'legal-random-v1', policyB: 'legal-random-v1' });
	assert.notDeepEqual(selfPlayArtifact.policySeeds.policyA, selfPlayArtifact.policySeeds.policyB);
	assert.deepEqual(selfPlayArtifact.runs.map(run => run.policyIdsBySeat), [
		{ p1: 'legal-random-v1', p2: 'legal-random-v1' },
		{ p1: 'legal-random-v1', p2: 'legal-random-v1' },
	]);
	assert.deepEqual(selfPlayArtifact.runs.map(run => [run.seeds.policyP1, run.seeds.policyP2]), [
		[[48610, 58908, 33605, 29270], [40788, 37485, 58001, 62354]],
		[[40788, 37485, 58001, 62354], [48610, 58908, 33605, 29270]],
	]);
	assert.equal(selfPlay.counts.factoryCalls, 2);
});

test('snapshots caller configuration and forwards detached supported limits to both runs', async () => {
	let release;
	const gate = new Promise(resolve => { release = resolve; });
	const environment = controlledEnvironment({ releaseFirst: gate });
	const limits = { maxPackets: 500, timeoutMs: 30_000 };
	const options = optionsFor(environment, { limits });
	const pending = runBaselineMatchup(options);
	options.rootSeed[0] = 99;
	options.policyA = 'move-first-v1';
	limits.maxPackets = 1;
	limits.timeoutMs = 1;
	release();
	const artifact = await pending;
	assert.deepEqual(artifact.rootSeed, [1, 2, 3, 4]);
	assert.deepEqual(artifact.policiesByRole, { policyA: 'legal-random-v1', policyB: 'move-first-v1' });
	assert.deepEqual(artifact.limits, { ...DEFAULT_LIMITS, maxPackets: 500, timeoutMs: 30_000 });
	assert.deepEqual(artifact.runs.map(run => run.resultRecord.limits), [artifact.limits, artifact.limits]);
	assert.notEqual(artifact.rootSeed, options.rootSeed);
	assert.notEqual(artifact.limits, limits);
});

test('invalid configuration, hostile accessors, and foreign result-shaped fields fail before preparation', async () => {
	const invalidOverrides = [
		{ rootSeed: [1, 2, 3] },
		{ rootSeed: [1, 2, 3, 65_536] },
		{ pairIndex: -1 },
		{ pairIndex: MAX_PAIRS },
		{ policyA: 'random' },
		{ policyB: 'caller-defined-v1' },
		{ limits: { maxPackets: 0 } },
		{ limits: { inventedLimit: 1 } },
		{ streamFactory: {} },
		{ resultRecord: { kind: 'battle-run-result', status: 'completed', outcome: { winner: 'p1' } } },
	];
	for (const override of invalidOverrides) {
		const environment = controlledEnvironment();
		await assertFixedFailure(runBaselineMatchup(optionsFor(environment, override)));
		assert.equal(environment.counts.generatedSeeds.length, 0);
		assert.equal(environment.counts.packCalls.length, 0);
		assert.equal(environment.counts.factoryCalls, 0);
	}

	const environment = controlledEnvironment();
	let accessorCalls = 0;
	const getterOptions = optionsFor(environment);
	Object.defineProperty(getterOptions, 'policyA', {
		enumerable: true,
		get() { accessorCalls++; throw new Error('caller getter detail'); },
	});
	await assertFixedFailure(runBaselineMatchup(getterOptions));
	assert.equal(accessorCalls, 0);
	assert.equal(environment.counts.generatedSeeds.length, 0);
	assert.equal(environment.counts.factoryCalls, 0);

	let foreignFieldReads = 0;
	const thrown = new Error('foreign configuration trap');
	Object.defineProperty(thrown, 'resultRecord', {
		get() { foreignFieldReads++; return { kind: 'battle-run-result', outcome: { result: 'win', winner: 'p1' } }; },
	});
	const hostileProxy = new Proxy(optionsFor(environment), {
		ownKeys() { throw thrown; },
	});
	await assertFixedFailure(runBaselineMatchup(hostileProxy));
	assert.equal(foreignFieldReads, 0);
	assert.equal(environment.counts.generatedSeeds.length, 0);
	assert.equal(environment.counts.factoryCalls, 0);
});

test('preparation, first or second battle, cleanup, and reused-stream failures return no artifact', async () => {
	const preparation = controlledEnvironment();
	const originalPack = preparation.ps.Teams.pack;
	let packs = 0;
	preparation.ps.Teams.pack = team => {
		packs++;
		if (packs === 2) throw new Error('private pack failure');
		return originalPack(team);
	};
	await assertFixedFailure(runBaselineMatchup(optionsFor(preparation)));
	assert.equal(preparation.counts.factoryCalls, 0);
	assert.equal(preparation.counts.generatedSeeds.length, 2);

	const firstBattle = controlledEnvironment({ packetsForCall: () => ['error\nforeign battle details'] });
	await assertFixedFailure(runBaselineMatchup(optionsFor(firstBattle)));
	assert.equal(firstBattle.counts.factoryCalls, 1);
	assert.deepEqual(firstBattle.counts.streams.map(stream => stream.destroyCalls), [1]);

	const secondBattle = controlledEnvironment({
		packetsForCall: call => call === 1 ? completedPackets() : ['error\nforeign battle details'],
	});
	await assertFixedFailure(runBaselineMatchup(optionsFor(secondBattle)));
	assert.equal(secondBattle.counts.factoryCalls, 2);
	assert.deepEqual(secondBattle.counts.streams.map(stream => stream.destroyCalls), [1, 1]);

	const cleanup = controlledEnvironment({ cleanupFailureCalls: new Set([2]) });
	await assertFixedFailure(runBaselineMatchup(optionsFor(cleanup)));
	assert.equal(cleanup.counts.factoryCalls, 2);
	assert.deepEqual(cleanup.counts.streams.map(stream => stream.destroyCalls), [1, 1]);

	const duplicate = controlledEnvironment();
	const sameStream = new ControlledStream(completedPackets());
	let duplicateCalls = 0;
	const duplicateOptions = optionsFor(duplicate, {
		streamFactory() { duplicateCalls++; return sameStream; },
	});
	await assertFixedFailure(runBaselineMatchup(duplicateOptions));
	assert.equal(duplicateCalls, 2);
	assert.equal(sameStream.destroyCalls, 1);
});

test('legacy result summarizer rejects the new named artifact for its matching pair manifest', async () => {
	const environment = controlledEnvironment();
	const artifact = await runBaselineMatchup(optionsFor(environment));
	const manifest = generateManifest([1, 2, 3, 4], 1);
	assert.equal(manifest.pairs[0].pairId, artifact.pairId);
	assert.deepEqual(manifest.rootSeed, artifact.rootSeed);
	assert.throws(() => summarizeResults({ manifest, pairResults: [artifact] }));
});

test('real pinned mixed matchup exchanges prepared packed teams and retains canonical seed and outcome records', {
	timeout: 180_000,
}, async () => {
	const pair = generatePair([1, 2, 3, 4], 0);
	const writesByRun = [];
	const streamRecords = [];
	let activeStreams = 0;
	let factoryCalls = 0;
	const streamFactory = () => {
		assert.equal(activeStreams, 0, 'the prior pinned stream was not destroyed');
		const runIndex = factoryCalls++;
		const stream = new ps.BattleStream({ keepAlive: true });
		activeStreams++;
		const writes = { start: null, teams: [], choices: [] };
		writesByRun.push(writes);
		const streamRecord = { destroyCalls: 0, destroyed: false };
		streamRecords.push(streamRecord);
		return {
			write(command) {
				if (command.startsWith('>start ')) writes.start = JSON.parse(command.slice('>start '.length));
				const player = /^>player (p[12]) (.+)$/.exec(command);
				if (player) writes.teams.push({ player: player[1], packedTeam: JSON.parse(player[2]).team });
				if (/^>p[12] /.test(command)) writes.choices.push(command);
				return stream.write(command);
			},
			async destroy() {
				streamRecord.destroyCalls++;
				try { return await stream.destroy(); } finally {
					if (!streamRecord.destroyed) {
						streamRecord.destroyed = true;
						activeStreams--;
					}
				}
			},
			[Symbol.asyncIterator]() { return stream[Symbol.asyncIterator](); },
		};
	};

	const artifact = await runBaselineMatchup({
		ps,
		rootSeed: [1, 2, 3, 4],
		pairIndex: 0,
		policyA: 'legal-random-v1',
		policyB: 'move-first-v1',
		streamFactory,
	});
	assert.equal(factoryCalls, 2);
	assert.equal(activeStreams, 0);
	assert.deepEqual(streamRecords.map(stream => stream.destroyed), [true, true]);
	assert.deepEqual(streamRecords.map(stream => stream.destroyCalls), [1, 1]);
	assert.deepEqual(writesByRun.map(write => write.start.seed), [pair.runs[0].seeds.battle, pair.runs[1].seeds.battle]);
	assert.deepEqual(writesByRun.map(write => write.teams.map(entry => entry.player)), [['p1', 'p2'], ['p1', 'p2']]);
	assert.equal(writesByRun[0].teams[0].packedTeam, writesByRun[1].teams[1].packedTeam);
	assert.equal(writesByRun[0].teams[1].packedTeam, writesByRun[1].teams[0].packedTeam);
	const digest = value => crypto.createHash('sha256').update(value, 'utf8').digest('hex');
	assert.equal(digest(writesByRun[0].teams[0].packedTeam), artifact.teams.A.packedTeamSha256);
	assert.equal(digest(writesByRun[0].teams[1].packedTeam), artifact.teams.B.packedTeamSha256);
	assert.deepEqual(artifact.runs.map(run => run.policyIdsBySeat), [
		{ p1: 'legal-random-v1', p2: 'move-first-v1' },
		{ p1: 'move-first-v1', p2: 'legal-random-v1' },
	]);
	for (const [index, run] of artifact.runs.entries()) {
		const expected = pair.runs[index];
		assert.deepEqual(run.seeds, expected.seeds);
		assert.deepEqual(run.resultRecord.seeds, expected.seeds);
		assert.equal(run.resultRecord.status, 'completed');
		assert.equal(run.resultRecord.teamSource, 'prepared');
		assert.equal(run.resultRecord.failure, null);
		assert.equal(run.resultRecord.cleanupFailure, null);
		assert.ok(['win', 'tie'].includes(run.resultRecord.outcome.result));
		assert.ok(run.resultRecord.outcome.result === 'tie'
			? run.resultRecord.outcome.winner === null
			: ['p1', 'p2'].includes(run.resultRecord.outcome.winner));
		assert.ok(Number.isSafeInteger(run.resultRecord.outcome.turns));
	}
	assert.deepEqual(artifact.teams.A.seed, pair.teams.A.seed);
	assert.deepEqual(artifact.teams.B.seed, pair.teams.B.seed);
	assert.equal(writesByRun[0].teams[0].packedTeam,
		ps.Teams.pack(ps.Teams.generate('gen5randombattle', { seed: pair.teams.A.seed })));
	assert.equal(writesByRun[0].teams[1].packedTeam,
		ps.Teams.pack(ps.Teams.generate('gen5randombattle', { seed: pair.teams.B.seed })));
});
