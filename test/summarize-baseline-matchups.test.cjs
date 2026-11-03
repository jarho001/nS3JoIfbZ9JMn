'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const referenceLock = require('../reference-lock.json');
const oracle = require('../reference/oracle.cjs');
const { DEFAULT_LIMITS } = require('../environment/battle-runner.cjs');
const { generateManifest, generatePair } = require('../environment/paired-seeds.cjs');
const { runBaselineMatchup } = require('../environment/baseline-matchup.cjs');
const { summarizeBaselineMatchups } = require('../environment/summarize-baseline-matchups.cjs');
const { summarizeResults } = require('../environment/summarize-results.cjs');

const ROOT_SEED = [11, 22, 33, 44];
const LIMITS = { ...DEFAULT_LIMITS };
const INVALID_MESSAGE = 'Invalid named baseline outcome inputs (named_baseline_report_invalid).';

function outcome(result, winner = null, turns = 30) {
	return { result, winner, turns };
}

function battleRecord(seeds, resultOutcome) {
	return {
		schemaVersion: 1,
		kind: 'battle-run-result',
		upstreamCommit: referenceLock.commit,
		formatId: 'gen5randombattle',
		seeds: structuredClone(seeds),
		limits: { ...LIMITS },
		status: 'completed',
		outcome: structuredClone(resultOutcome),
		elapsedMs: 10,
		elapsedScope: 'battle_runner',
		counters: { producerOwned: ['finite', 1] },
		failure: null,
		cleanupFailure: null,
		teamSource: 'prepared',
	};
}

function controlledMatchup(pairIndex, policyIdsByRun = undefined, outcomes = undefined, policyIdsByRole = undefined) {
	const pair = generatePair(ROOT_SEED, pairIndex);
	return {
		schemaVersion: 1,
		kind: 'gen5-named-baseline-paired-matchup',
		status: 'completed',
		formatId: 'gen5randombattle',
		upstream: { repository: referenceLock.repository, commit: referenceLock.commit },
		rootSeed: [...ROOT_SEED],
		pairIndex,
		pairId: pair.pairId,
		policiesByRole: policyIdsByRole || { policyA: 'legal-random-v1', policyB: 'move-first-v1' },
		limits: { ...LIMITS },
		policySeeds: structuredClone(pair.policySeeds),
		teams: {
			A: { seed: [...pair.teams.A.seed], packedTeamSha256: 'a'.repeat(64) },
			B: { seed: [...pair.teams.B.seed], packedTeamSha256: 'b'.repeat(64) },
		},
		runs: pair.runs.map((run, index) => {
			const runOutcome = outcomes?.[index] || outcome('tie');
			const ids = policyIdsByRun?.[index] || {
				p1: run.policySeats.p1 === 'policyA' ? 'legal-random-v1' : 'move-first-v1',
				p2: run.policySeats.p2 === 'policyA' ? 'legal-random-v1' : 'move-first-v1',
			};
			return {
				runId: run.runId,
				seats: structuredClone(run.seats),
				policySeats: structuredClone(run.policySeats),
				seeds: structuredClone(run.seeds),
				policyIdsBySeat: ids,
				status: 'completed',
				resultRecord: battleRecord(run.seeds, runOutcome),
			};
		}),
	};
}

function config(matchups, changes = {}) {
	return {
		rootSeed: [...ROOT_SEED],
		pairCount: 3,
		policyA: 'legal-random-v1',
		policyB: 'move-first-v1',
		limits: { ...LIMITS },
		matchups,
		...changes,
	};
}

function clone(value) {
	return structuredClone(value);
}

function reverseObjectKeys(value) {
	if (Array.isArray(value)) return value.map(reverseObjectKeys);
	if (value && typeof value === 'object') {
		return Object.fromEntries(Object.entries(value).reverse().map(([key, item]) => [key, reverseObjectKeys(item)]));
	}
	return value;
}

function assertInvalid(run, label = 'invalid report input') {
	assert.throws(run, error => error instanceof Error && error.message === INVALID_MESSAGE, label);
}

test('controlled W/L/T fixtures tally by seat and role with independent known counts', () => {
	const matchups = [
		controlledMatchup(0, undefined, [outcome('win', 'p1'), outcome('win', 'p2')]),
		controlledMatchup(1, undefined, [outcome('tie'), outcome('tie')]),
		controlledMatchup(2, undefined, [outcome('win', 'p2'), outcome('win', 'p1')]),
	];
	const report = summarizeBaselineMatchups(config(matchups));
	assert.equal(report.schemaVersion, 1);
	assert.equal(report.kind, 'gen5-named-baseline-outcome-report');
	assert.deepEqual(Object.keys(report).sort(), ['configuration', 'coverage', 'kind', 'outcomes', 'schemaVersion'].sort());
	assert.deepEqual(report.configuration, {
		formatId: 'gen5randombattle',
		upstream: { repository: referenceLock.repository, commit: referenceLock.commit },
		rootSeed: ROOT_SEED,
		pairCount: 3,
		policiesByRole: { policyA: 'legal-random-v1', policyB: 'move-first-v1' },
		limits: LIMITS,
	});
	assert.deepEqual(report.coverage, {
		configuredPairs: 3, recordedPairs: 3, unrecordedPairs: 0, recordedBattles: 6, unrecordedRunSlots: 0,
	});
	assert.deepEqual(report.outcomes, {
		completedBattles: 6,
		bySeat: {
			p1: { wins: 2, losses: 2, ties: 2 },
			p2: { wins: 2, losses: 2, ties: 2 },
		},
		byPolicyRole: {
			policyA: {
				policyId: 'legal-random-v1',
				overall: { wins: 2, losses: 2, ties: 2 },
				bySeat: { p1: { wins: 1, losses: 1, ties: 1 }, p2: { wins: 1, losses: 1, ties: 1 } },
			},
			policyB: {
				policyId: 'move-first-v1',
				overall: { wins: 2, losses: 2, ties: 2 },
				bySeat: { p1: { wins: 1, losses: 1, ties: 1 }, p2: { wins: 1, losses: 1, ties: 1 } },
			},
		},
		overall: { winsBySeat: { p1: 2, p2: 2 }, ties: 2 },
	});
	assert.equal(JSON.stringify(report).includes('producerOwned'), false);
});

test('empty, short, and full prefixes describe recorded artifacts without execution status', () => {
	const matchups = [
		controlledMatchup(0),
		controlledMatchup(1),
		controlledMatchup(2),
	];
	const empty = summarizeBaselineMatchups(config([]));
	assert.deepEqual(empty.coverage, {
		configuredPairs: 3, recordedPairs: 0, unrecordedPairs: 3, recordedBattles: 0, unrecordedRunSlots: 6,
	});
	assert.deepEqual(empty.outcomes.bySeat, { p1: { wins: 0, losses: 0, ties: 0 }, p2: { wins: 0, losses: 0, ties: 0 } });
	assert.equal(JSON.stringify(empty).includes('unattempted'), false);
	assert.equal(JSON.stringify(empty).includes('failed'), false);
	const short = summarizeBaselineMatchups(config(matchups.slice(0, 1)));
	assert.deepEqual(short.coverage, {
		configuredPairs: 3, recordedPairs: 1, unrecordedPairs: 2, recordedBattles: 2, unrecordedRunSlots: 4,
	});
	const full = summarizeBaselineMatchups(config(matchups));
	assert.deepEqual(full.coverage, {
		configuredPairs: 3, recordedPairs: 3, unrecordedPairs: 0, recordedBattles: 6, unrecordedRunSlots: 0,
	});
});

test('same-ID self-play keeps policy roles separate and assigns each seat from the paired role', () => {
	const canonical = generatePair(ROOT_SEED, 0);
	const selfPlayOutcomes = canonical.runs.map(run => outcome(
		'win', run.policySeats.p1 === 'policyA' ? 'p1' : 'p2',
	));
	const matchup = controlledMatchup(0, [
		{ p1: 'legal-random-v1', p2: 'legal-random-v1' },
		{ p1: 'legal-random-v1', p2: 'legal-random-v1' },
	], selfPlayOutcomes, { policyA: 'legal-random-v1', policyB: 'legal-random-v1' });
	const report = summarizeBaselineMatchups(config([matchup], { pairCount: 1, policyB: 'legal-random-v1' }));
	assert.equal(report.outcomes.byPolicyRole.policyA.policyId, 'legal-random-v1');
	assert.equal(report.outcomes.byPolicyRole.policyB.policyId, 'legal-random-v1');
	assert.deepEqual(report.outcomes.byPolicyRole.policyA.overall, { wins: 2, losses: 0, ties: 0 });
	assert.deepEqual(report.outcomes.byPolicyRole.policyB.overall, { wins: 0, losses: 2, ties: 0 });
	assert.deepEqual(report.outcomes.byPolicyRole.policyA.bySeat, {
		p1: { wins: 1, losses: 0, ties: 0 }, p2: { wins: 1, losses: 0, ties: 0 },
	});
	assert.deepEqual(report.outcomes.byPolicyRole.policyB.bySeat, {
		p1: { wins: 0, losses: 1, ties: 0 }, p2: { wins: 0, losses: 1, ties: 0 },
	});
});

test('rejects malformed config, non-prefix artifacts, mixed config, and hostile nested data safely', () => {
	const good = controlledMatchup(0);
	const valid = config([good], { pairCount: 1 });
	const invalids = [
		{ ...valid, surprise: true },
		{ ...valid, rootSeed: [1, 2, 3] },
		{ ...valid, pairCount: 0 },
		{ ...valid, pairCount: 1.5 },
		{ ...valid, policyA: 'custom-v1' },
		{ ...valid, limits: { ...LIMITS, extra: 1 } },
		{ ...valid, limits: { timeoutMs: LIMITS.timeoutMs } },
		{ ...valid, matchups: [good, good] },
		{ ...valid, matchups: [controlledMatchup(1)] },
		{ ...valid, matchups: [Object.assign(clone(good), { pairId: 'wrong' })] },
		{ ...valid, matchups: [Object.assign(clone(good), { policySeeds: { ...good.policySeeds, policyA: [1, 2, 3, 4] } })] },
		{ ...valid, matchups: [Object.assign(clone(good), { teams: { ...good.teams, A: { ...good.teams.A, packedTeamSha256: 'bad' } } })] },
		{ ...valid, matchups: [Object.assign(clone(good), { limits: { ...LIMITS, maxPackets: LIMITS.maxPackets + 1 } })] },
		{ ...valid, matchups: [Object.assign(clone(good), { runs: [...good.runs].reverse() })] },
		{ ...valid, matchups: [Object.assign(clone(good), { runs: good.runs.slice(0, 1) })] },
		{ ...valid, matchups: [Object.assign(clone(good), { extra: true })] },
		{ ...valid, matchups: [Object.assign(clone(good), { kind: 'gen5-paired-matchup' })] },
	];
	for (const invalid of invalids) assertInvalid(() => summarizeBaselineMatchups(invalid));
	const mutations = [
		['wrong upstream', record => { record.upstream.commit = 'wrong'; }],
		['wrong format', record => { record.formatId = 'gen9randombattle'; }],
		['wrong root seed', record => { record.rootSeed[0]++; }],
		['wrong team seed', record => { record.teams.A.seed[0]++; }],
		['wrong run seat', record => { record.runs[0].seats.p1Team = 'B'; }],
		['wrong policy seat', record => { record.runs[0].policySeats.p1 = 'policyB'; }],
		['wrong run policy ID', record => { record.runs[0].policyIdsBySeat.p1 = 'move-first-v1'; }],
		['wrong run seed', record => { record.runs[0].seeds.policyP1[0]++; }],
		['wrong result pin', record => { record.runs[0].resultRecord.upstreamCommit = 'wrong'; }],
		['wrong nested status', record => { record.runs[0].resultRecord.status = 'failed'; }],
		['wrong nested limits', record => { record.runs[0].resultRecord.limits.timeoutMs++; }],
		['non-null failure', record => { record.runs[0].resultRecord.failure = { code: 'foreign' }; }],
		['non-null cleanup failure', record => { record.runs[0].resultRecord.cleanupFailure = { code: 'foreign' }; }],
		['invalid tie winner', record => { record.runs[0].resultRecord.outcome = outcome('tie', 'p1'); }],
		['invalid turn count', record => { record.runs[0].resultRecord.outcome.turns = -1; }],
		['unknown nested key', record => { record.runs[0].resultRecord.extra = 'untrusted'; }],
	];
	for (const [label, mutate] of mutations) {
		const artifact = clone(good);
		mutate(artifact);
		assertInvalid(
			() => summarizeBaselineMatchups({ ...valid, matchups: [artifact] }),
			label,
		);
	}
	const pairOne = controlledMatchup(1);
	assertInvalid(() => summarizeBaselineMatchups(config([good, pairOne, pairOne])));
	const failedOutcome = clone(good);
	failedOutcome.runs[0].resultRecord.outcome = { result: 'win', winner: 'secret', turns: 1 };
	assertInvalid(() => summarizeBaselineMatchups({ ...valid, matchups: [failedOutcome] }));
	const badElapsed = clone(good);
	badElapsed.runs[0].resultRecord.elapsedMs = Infinity;
	assertInvalid(() => summarizeBaselineMatchups({ ...valid, matchups: [badElapsed] }));
	const badCounters = clone(good);
	badCounters.runs[0].resultRecord.counters.bad = Number.NaN;
	assertInvalid(() => summarizeBaselineMatchups({ ...valid, matchups: [badCounters] }));
	const accessor = { ...valid };
	let topGetterReads = 0;
	Object.defineProperty(accessor, 'policyA', { enumerable: true, get() { topGetterReads++; return 'legal-random-v1'; } });
	assertInvalid(() => summarizeBaselineMatchups(accessor));
	assert.equal(topGetterReads, 0);
	const nestedAccessor = clone(good);
	let nestedGetterReads = 0;
	Object.defineProperty(nestedAccessor.teams.A, 'packedTeamSha256', { enumerable: true, get() { nestedGetterReads++; return 'a'.repeat(64); } });
	assertInvalid(() => summarizeBaselineMatchups({ ...valid, matchups: [nestedAccessor] }));
	assert.equal(nestedGetterReads, 0);
	const upstreamAccessor = clone(good);
	let upstreamGetterReads = 0;
	Object.defineProperty(upstreamAccessor.upstream, 'commit', { enumerable: true, get() { upstreamGetterReads++; return referenceLock.commit; } });
	assertInvalid(() => summarizeBaselineMatchups({ ...valid, matchups: [upstreamAccessor] }));
	assert.equal(upstreamGetterReads, 0);
	const sparse = [];
	sparse.length = 1;
	assertInvalid(() => summarizeBaselineMatchups({ ...valid, matchups: sparse }));
	const cyclic = clone(good);
	cyclic.runs[0].resultRecord.counters.self = cyclic.runs[0].resultRecord.counters;
	assertInvalid(() => summarizeBaselineMatchups({ ...valid, matchups: [cyclic] }));
	let foreignPropertyReads = 0;
	const foreignError = new Error('private foreign reflection detail');
	const foreignFailure = new Proxy({}, {
		ownKeys() { throw foreignError; },
		get() { foreignPropertyReads++; throw new Error('must not inspect foreign exception properties'); },
	});
	assertInvalid(() => summarizeBaselineMatchups({ ...valid, matchups: [foreignFailure] }));
	assert.equal(foreignPropertyReads, 0);
});

test('copies input data and detached report state without mutating source artifacts', () => {
	const artifact = controlledMatchup(0, undefined, [outcome('win', 'p1'), outcome('tie')]);
	const input = config([artifact], { pairCount: 1 });
	const before = JSON.stringify(input);
	const report = summarizeBaselineMatchups(input);
	assert.equal(JSON.stringify(input), before);
	report.configuration.rootSeed[0] = 999;
	report.outcomes.bySeat.p1.wins = 999;
	assert.deepEqual(input.rootSeed, ROOT_SEED);
	assert.deepEqual(input.matchups[0].runs[0].resultRecord.outcome, outcome('win', 'p1'));

	const producerReport = summarizeBaselineMatchups(config([artifact], { pairCount: 1 }));
	input.rootSeed[0] = 55;
	input.matchups[0].runs[0].resultRecord.outcome.winner = 'p2';
	assert.deepEqual(producerReport.configuration.rootSeed, ROOT_SEED);
	assert.equal(producerReport.outcomes.bySeat.p1.wins, 1);
});

test('treats record key order as irrelevant while preserving canonical array/run order', () => {
	const canonical = config([controlledMatchup(0)], { pairCount: 1 });
	const report = summarizeBaselineMatchups(JSON.parse(JSON.stringify(canonical)));
	const reversedKeys = JSON.parse(JSON.stringify(reverseObjectKeys(canonical)));
	assert.deepEqual(summarizeBaselineMatchups(reversedKeys), report);
	assertInvalid(() => summarizeBaselineMatchups({ ...canonical, matchups: [canonical.matchups[0], canonical.matchups[0]] }));
});

test('actual pinned JSON-roundtripped named matchup aggregates, while legacy reporting stays distinct', { timeout: 120_000 }, async () => {
	oracle.assertReady();
	const ps = require('../reference/vendor/pokemon-showdown/dist/sim');
	const artifact = await runBaselineMatchup({
		ps,
		rootSeed: [1, 2, 3, 4],
		pairIndex: 0,
		policyA: 'legal-random-v1',
		policyB: 'move-first-v1',
	});
	const roundTripped = JSON.parse(JSON.stringify(artifact));
	const report = summarizeBaselineMatchups({
		rootSeed: [1, 2, 3, 4],
		pairCount: 2,
		policyA: 'legal-random-v1',
		policyB: 'move-first-v1',
		limits: { ...LIMITS },
		matchups: [roundTripped],
	});
	const expected = { p1: { wins: 0, losses: 0, ties: 0 }, p2: { wins: 0, losses: 0, ties: 0 } };
	for (const run of artifact.runs) {
		if (run.resultRecord.outcome.result === 'tie') {
			expected.p1.ties++;
			expected.p2.ties++;
		} else {
			expected[run.resultRecord.outcome.winner].wins++;
			expected[run.resultRecord.outcome.winner === 'p1' ? 'p2' : 'p1'].losses++;
		}
	}
	assert.deepEqual(report.outcomes.bySeat, expected);
	assert.equal(report.coverage.recordedPairs, 1);
	assert.equal(report.coverage.unrecordedPairs, 1);
	const legacyManifest = generateManifest([1, 2, 3, 4], 1);
	assert.throws(() => summarizeResults({ manifest: legacyManifest, pairResults: [roundTripped] }));
	assertInvalid(() => summarizeBaselineMatchups({
		rootSeed: [1, 2, 3, 4], pairCount: 1,
		policyA: 'legal-random-v1', policyB: 'move-first-v1', limits: { ...LIMITS },
		matchups: [{ kind: 'gen5-paired-matchup' }],
	}));
});
