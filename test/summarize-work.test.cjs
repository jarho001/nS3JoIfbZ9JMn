'use strict';

const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const oracle = require('../reference/oracle.cjs');
const { generateManifest } = require('../environment/paired-seeds.cjs');
const { runPair } = require('../environment/run-pair.cjs');
const { summarizeWork } = require('../environment/summarize-work.cjs');

const ROOT = path.resolve(__dirname, '..');
const CLI = path.join(ROOT, 'environment/summarize-work.cjs');
const ROOT_SEED = [123, 456, 789, 1011];
const PLAYER_PATHS = [
	['policyInvocations'],
	['choiceSubmissions'],
	['choiceResolutions', 'accepted'],
	['choiceResolutions', 'rejected'],
	['choiceResolutions', 'pending'],
	['retryRequests'],
	['retrySubmissions'],
	['retryLimitExhaustions'],
	['actions', 'selected', 'move'],
	['actions', 'selected', 'switch'],
	['actions', 'submitted', 'move'],
	['actions', 'submitted', 'switch'],
];

function playerCounter(p1, p2) {
	return { total: p1 + p2, byPlayer: { p1, p2 } };
}

function makeCounters(n, { objectPackets = n + 10, publicEvents = n + 100 } = {}) {
	const p1 = {
		policyInvocations: n + 7,
		choiceSubmissions: 2 * n,
		choiceResolutions: { accepted: n, rejected: n, pending: 0 },
		retryRequests: n + 3,
		retrySubmissions: n + 2,
		retryLimitExhaustions: n + 1,
		actions: {
			selected: { move: n + 2, switch: n },
			submitted: { move: n + 1, switch: n - 1 },
		},
	};
	const p2 = {
		policyInvocations: n + 9,
		choiceSubmissions: 2 * n + 2,
		choiceResolutions: { accepted: n + 1, rejected: n, pending: 1 },
		retryRequests: n + 4,
		retrySubmissions: n + 3,
		retryLimitExhaustions: n + 2,
		actions: {
			selected: { move: n + 3, switch: n + 1 },
			submitted: { move: n + 2, switch: n },
		},
	};
	return {
		objectPackets,
		policyInvocations: playerCounter(p1.policyInvocations, p2.policyInvocations),
		choiceSubmissions: playerCounter(p1.choiceSubmissions, p2.choiceSubmissions),
		choiceResolutions: {
			accepted: playerCounter(p1.choiceResolutions.accepted, p2.choiceResolutions.accepted),
			rejected: playerCounter(p1.choiceResolutions.rejected, p2.choiceResolutions.rejected),
			pending: playerCounter(p1.choiceResolutions.pending, p2.choiceResolutions.pending),
		},
		retryRequests: playerCounter(p1.retryRequests, p2.retryRequests),
		retrySubmissions: playerCounter(p1.retrySubmissions, p2.retrySubmissions),
		retryLimitExhaustions: playerCounter(p1.retryLimitExhaustions, p2.retryLimitExhaustions),
		actions: {
			selected: {
				move: playerCounter(p1.actions.selected.move, p2.actions.selected.move),
				switch: playerCounter(p1.actions.selected.switch, p2.actions.selected.switch),
			},
			submitted: {
				move: playerCounter(p1.actions.submitted.move, p2.actions.submitted.move),
				switch: playerCounter(p1.actions.submitted.switch, p2.actions.submitted.switch),
			},
		},
		publicEvents,
		latestPublicTurn: 500 + n,
		terminalMetadataTurns: 700 + n,
		unknownCounter: 'PRIVATE_COUNTER_EXTRA',
	};
}

function nestedResult(manifest, expectedRun, status, counters) {
	return {
		schemaVersion: 1,
		kind: 'battle-run-result',
		upstreamCommit: manifest.upstream.commit,
		formatId: manifest.formatId,
		seeds: structuredClone(expectedRun.seeds),
		teamSource: 'prepared',
		status,
		outcome: status === 'completed' ? { result: 'win', winner: 'p1', turns: 3 } : null,
		failure: status === 'failed' ? { category: 'packet_limit', code: 'packet_limit_exhausted' } : null,
		counters,
		elapsedMs: 1.25,
		limits: { maxPackets: 20_000 },
		privateTranscript: 'DO_NOT_EMIT_TRANSCRIPT',
	};
}

function pairRecord(manifest, pairIndex, statuses, counters) {
	const expectedPair = manifest.pairs[pairIndex];
	const allNotAttempted = statuses.every(status => status === 'not_attempted');
	const completed = statuses.every(status => status === 'completed');
	return {
		schemaVersion: 1,
		kind: 'gen5-paired-battle-result',
		status: completed ? 'completed' : 'failed',
		upstream: structuredClone(manifest.upstream),
		formatId: manifest.formatId,
		rootSeed: [...manifest.rootSeed],
		pairIndex,
		pairId: expectedPair.pairId,
		teams: allNotAttempted
			? { A: { seed: [...expectedPair.teams.A.seed] }, B: null }
			: {
				A: { seed: [...expectedPair.teams.A.seed], packedTeamSha256: 'a'.repeat(64) },
				B: { seed: [...expectedPair.teams.B.seed], packedTeamSha256: 'b'.repeat(64) },
			},
		runs: expectedPair.runs.map((run, index) => ({
			runId: run.runId,
			status: statuses[index],
			seats: structuredClone(run.seats),
			policySeats: structuredClone(run.policySeats),
			seeds: structuredClone(run.seeds),
			resultRecord: statuses[index] === 'not_attempted' || counters[index] === null
				? null
				: nestedResult(manifest, run, statuses[index], structuredClone(counters[index])),
		})),
		failure: completed ? null : {
			stage: allNotAttempted ? 'preparation' : 'battle',
			category: allNotAttempted ? 'initialization' : 'packet_limit',
			code: 'controlled_failure',
		},
		elapsedScope: 'paired_runner',
		elapsedMs: 5,
		unknownPairField: 'PRIVATE_PAIR_EXTRA',
	};
}

function emptyPairResult(manifest, pairIndex) {
	return pairRecord(manifest, pairIndex, ['not_attempted', 'not_attempted'], [null, null]);
}

function metric([total, p1, p2, policyA, policyB]) {
	return { total, bySeat: { p1, p2 }, byPolicy: { policyA, policyB } };
}

function expectedTotals(objectPackets, publicEvents, values) {
	return {
		global: { objectPackets, publicEvents },
		playerCounters: {
			policyInvocations: metric(values.policyInvocations),
			choiceSubmissions: metric(values.choiceSubmissions),
			choiceResolutions: {
				accepted: metric(values.accepted),
				rejected: metric(values.rejected),
				pending: metric(values.pending),
			},
			retryRequests: metric(values.retryRequests),
			retrySubmissions: metric(values.retrySubmissions),
			retryLimitExhaustions: metric(values.retryLimitExhaustions),
			actions: {
				selected: { move: metric(values.selectedMove), switch: metric(values.selectedSwitch) },
				submitted: { move: metric(values.submittedMove), switch: metric(values.submittedSwitch) },
			},
		},
	};
}

function expectedCohort(runCount, withCounters, missingCounters, observedTotals) {
	return {
		runCount,
		counterRecords: { withCounters, missingCounters },
		counterCoverage: runCount === 0
			? 'no_recorded_runs'
			: missingCounters === 0 ? 'all_recorded_runs_have_counters' : 'some_recorded_runs_missing_counters',
		observedTotals,
	};
}

const EXPECTED_COMPLETED = expectedTotals(36, 306, {
	policyInvocations: [60, 27, 33, 29, 31],
	choiceSubmissions: [30, 12, 18, 14, 16],
	accepted: [15, 6, 9, 7, 8],
	rejected: [12, 6, 6, 6, 6],
	pending: [3, 0, 3, 1, 2],
	retryRequests: [33, 15, 18, 16, 17],
	retrySubmissions: [27, 12, 15, 13, 14],
	retryLimitExhaustions: [21, 9, 12, 10, 11],
	selectedMove: [27, 12, 15, 13, 14],
	selectedSwitch: [15, 6, 9, 7, 8],
	submittedMove: [21, 9, 12, 10, 11],
	submittedSwitch: [9, 3, 6, 4, 5],
});

const EXPECTED_FAILED = expectedTotals(14, 104, {
	policyInvocations: [24, 11, 13, 13, 11],
	choiceSubmissions: [18, 8, 10, 10, 8],
	accepted: [9, 4, 5, 5, 4],
	rejected: [8, 4, 4, 4, 4],
	pending: [1, 0, 1, 1, 0],
	retryRequests: [15, 7, 8, 8, 7],
	retrySubmissions: [13, 6, 7, 7, 6],
	retryLimitExhaustions: [11, 5, 6, 6, 5],
	selectedMove: [13, 6, 7, 7, 6],
	selectedSwitch: [9, 4, 5, 5, 4],
	submittedMove: [11, 5, 6, 6, 5],
	submittedSwitch: [7, 3, 4, 4, 3],
});

const EXPECTED_ALL = expectedTotals(50, 410, {
	policyInvocations: [84, 38, 46, 42, 42],
	choiceSubmissions: [48, 20, 28, 24, 24],
	accepted: [24, 10, 14, 12, 12],
	rejected: [20, 10, 10, 10, 10],
	pending: [4, 0, 4, 2, 2],
	retryRequests: [48, 22, 26, 24, 24],
	retrySubmissions: [40, 18, 22, 20, 20],
	retryLimitExhaustions: [32, 14, 18, 16, 16],
	selectedMove: [40, 18, 22, 20, 20],
	selectedSwitch: [24, 10, 14, 12, 12],
	submittedMove: [32, 14, 18, 16, 16],
	submittedSwitch: [16, 6, 10, 8, 8],
});

function setSeatCounter(counters, pathParts, player, value) {
	let counter = counters;
	for (const part of pathParts) counter = counter[part];
	counter.byPlayer[player] = value;
	counter.total = counter.byPlayer.p1 + counter.byPlayer.p2;
}

function getSeatCounter(counters, pathParts, player) {
	let counter = counters;
	for (const part of pathParts) counter = counter[part];
	return counter.byPlayer[player];
}

function cli(tempDir, args) {
	return spawnSync(process.execPath, ['-r', path.join(tempDir, 'block-oracle.cjs'), CLI, ...args], {
		cwd: ROOT,
		encoding: 'utf8',
	});
}

function createNoOraclePreload(tempDir) {
	const preloadPath = path.join(tempDir, 'block-oracle.cjs');
	fs.writeFileSync(preloadPath, [
		"const Module = require('node:module');",
		'const originalLoad = Module._load;',
		'Module._load = function(request, parent, isMain) {',
		"  if (typeof request === 'string' && request.includes('oracle.cjs')) throw new Error('oracle bootstrap is forbidden');",
		'  return originalLoad.call(this, request, parent, isMain);',
		'};',
	].join('\n'));
	return preloadPath;
}

function explicitTotals(runs) {
	const totals = {
		global: { objectPackets: 0, publicEvents: 0 },
		playerCounters: expectedTotals(0, 0, Object.fromEntries([
			['policyInvocations', [0, 0, 0, 0, 0]],
			['choiceSubmissions', [0, 0, 0, 0, 0]],
			['accepted', [0, 0, 0, 0, 0]],
			['rejected', [0, 0, 0, 0, 0]],
			['pending', [0, 0, 0, 0, 0]],
			['retryRequests', [0, 0, 0, 0, 0]],
			['retrySubmissions', [0, 0, 0, 0, 0]],
			['retryLimitExhaustions', [0, 0, 0, 0, 0]],
			['selectedMove', [0, 0, 0, 0, 0]],
			['selectedSwitch', [0, 0, 0, 0, 0]],
			['submittedMove', [0, 0, 0, 0, 0]],
			['submittedSwitch', [0, 0, 0, 0, 0]],
		].map(([key, value]) => [key, value]))).playerCounters,
	};
	const names = [
		'policyInvocations', 'choiceSubmissions', 'choiceResolutions.accepted',
		'choiceResolutions.rejected', 'choiceResolutions.pending', 'retryRequests',
		'retrySubmissions', 'retryLimitExhaustions', 'actions.selected.move',
		'actions.selected.switch', 'actions.submitted.move', 'actions.submitted.switch',
	];
	for (const run of runs) {
		const counters = run.resultRecord.counters;
		totals.global.objectPackets += counters.objectPackets;
		totals.global.publicEvents += counters.publicEvents;
		for (let index = 0; index < PLAYER_PATHS.length; index++) {
			const pathParts = PLAYER_PATHS[index];
			const name = names[index];
			const targetPath = name.split('.');
			let target = totals.playerCounters;
			let source = counters;
			for (const part of targetPath) target = target[part];
			for (const part of pathParts) source = source[part];
			for (const player of ['p1', 'p2']) {
				const count = source.byPlayer[player];
				target.total += player === 'p1' ? count : 0;
				target.bySeat[player] += count;
				target.byPolicy[run.policySeats[player]] += count;
			}
			target.total += source.byPlayer.p2;
		}
	}
	return totals;
}

test('exact work totals cover completed, failed partial, and all recorded swapped-seat runs', () => {
	const manifest = generateManifest(ROOT_SEED, 2);
	const pairResults = [
		pairRecord(manifest, 0, ['completed', 'completed'], [makeCounters(1), makeCounters(2)]),
		pairRecord(manifest, 1, ['completed', 'failed'], [makeCounters(3), makeCounters(4)]),
	];
	const before = JSON.stringify({ manifest, pairResults });
	assert.ok(makeCounters(1).policyInvocations.byPlayer.p1 > makeCounters(1).actions.selected.move.byPlayer.p1);
	assert.ok(makeCounters(1).actions.selected.move.byPlayer.p1 > makeCounters(1).actions.submitted.move.byPlayer.p1);
	const report = summarizeWork({ manifest, pairResults });
	assert.deepEqual(report, {
		schemaVersion: 1,
		kind: 'gen5-paired-work-accounting',
		manifest: {
			formatId: manifest.formatId,
			upstream: structuredClone(manifest.upstream),
			rootSeed: [...manifest.rootSeed],
			pairCount: manifest.pairCount,
		},
		coverage: {
			pairPrefix: 'manifest_prefix_complete',
			pairs: { recorded: 2, unrecorded: 0, completed: 1, failed: 1 },
			runSlots: { completed: 3, failed: 1, notAttempted: 0, unrecorded: 0 },
		},
		cohorts: {
			completed: expectedCohort(3, 3, 0, EXPECTED_COMPLETED),
			failedRecorded: expectedCohort(1, 1, 0, EXPECTED_FAILED),
			allRecorded: expectedCohort(4, 4, 0, EXPECTED_ALL),
		},
	});
	assert.equal(JSON.stringify({ manifest, pairResults }), before);
	const serialized = JSON.stringify(report);
	for (const secret of ['PRIVATE_COUNTER_EXTRA', 'DO_NOT_EMIT_TRANSCRIPT', 'PRIVATE_PAIR_EXTRA', 'elapsedMs', 'latestPublicTurn', 'terminalMetadataTurns']) {
		assert.equal(serialized.includes(secret), false, secret);
	}
});

test('empty and preparation-only prefixes have honest zero-work coverage', () => {
	const manifest = generateManifest(ROOT_SEED, 2);
	const empty = summarizeWork({ manifest, pairResults: [] });
	assert.equal(empty.coverage.pairPrefix, 'recorded_prefix');
	assert.deepEqual(empty.coverage.runSlots, { completed: 0, failed: 0, notAttempted: 0, unrecorded: 4 });
	for (const cohort of Object.values(empty.cohorts)) {
		assert.equal(cohort.runCount, 0);
		assert.deepEqual(cohort.counterRecords, { withCounters: 0, missingCounters: 0 });
		assert.equal(cohort.counterCoverage, 'no_recorded_runs');
		assert.deepEqual(cohort.observedTotals, expectedTotals(0, 0, Object.fromEntries([
			'policyInvocations', 'choiceSubmissions', 'accepted', 'rejected', 'pending',
			'retryRequests', 'retrySubmissions', 'retryLimitExhaustions', 'selectedMove',
			'selectedSwitch', 'submittedMove', 'submittedSwitch',
		].map(key => [key, [0, 0, 0, 0, 0]]))));
	}
	const prepared = summarizeWork({ manifest, pairResults: [emptyPairResult(manifest, 0)] });
	assert.deepEqual(prepared.coverage.runSlots, { completed: 0, failed: 0, notAttempted: 2, unrecorded: 2 });
	assert.equal(prepared.coverage.pairs.failed, 1);
	for (const cohort of Object.values(prepared.cohorts)) assert.equal(cohort.runCount, 0);

	const recordedPrefix = pairRecord(manifest, 0, ['completed', 'completed'], [makeCounters(1), makeCounters(2)]);
	const prefixReport = summarizeWork({ manifest, pairResults: [recordedPrefix] });
	assert.equal(prefixReport.coverage.pairPrefix, 'recorded_prefix');
	assert.deepEqual(prefixReport.coverage.pairs, { recorded: 1, unrecorded: 1, completed: 1, failed: 0 });
	assert.deepEqual(prefixReport.coverage.runSlots, { completed: 2, failed: 0, notAttempted: 0, unrecorded: 2 });
	assert.equal(prefixReport.cohorts.allRecorded.runCount, 2);
	assert.equal(prefixReport.cohorts.allRecorded.observedTotals.global.objectPackets, 23);
});

test('failed run with a null result record is counted as missing counters, not zero work', () => {
	const manifest = generateManifest(ROOT_SEED, 1);
	const failedWithoutResult = pairRecord(manifest, 0, ['failed', 'not_attempted'], [null, null]);
	const report = summarizeWork({ manifest, pairResults: [failedWithoutResult] });
	assert.equal(report.coverage.runSlots.failed, 1);
	for (const key of ['failedRecorded', 'allRecorded']) {
		const cohort = report.cohorts[key];
		assert.equal(cohort.runCount, 1);
		assert.deepEqual(cohort.counterRecords, { withCounters: 0, missingCounters: 1 });
		assert.equal(cohort.counterCoverage, 'some_recorded_runs_missing_counters');
		assert.deepEqual(cohort.observedTotals.global, { objectPackets: 0, publicEvents: 0 });
		assert.equal(cohort.observedTotals.playerCounters.policyInvocations.total, 0);
	}
});

test('present result records require the complete additive counter shape and safe numeric values', () => {
	const manifest = generateManifest(ROOT_SEED, 1);
	const cases = [
		['negative global count', counters => { counters.objectPackets = -1; }],
		['fractional global count', counters => { counters.publicEvents = 0.5; }],
		['infinite global count', counters => { counters.objectPackets = Infinity; }],
		['NaN global count', counters => { counters.objectPackets = Number.NaN; }],
		['unsafe player total', counters => { counters.policyInvocations.total = Number.MAX_SAFE_INTEGER + 1; }],
		['negative by-player count', counters => { counters.choiceSubmissions.byPlayer.p1 = -1; }],
		['missing global count', counters => { delete counters.publicEvents; }],
		['missing player total', counters => { delete counters.retryRequests.total; }],
		['missing seat value', counters => { delete counters.retrySubmissions.byPlayer.p2; }],
		['missing nested additive counter', counters => { delete counters.choiceResolutions.pending; }],
		['missing action counter', counters => { delete counters.actions.submitted.switch; }],
		['per-player total mismatch', counters => { counters.choiceSubmissions.byPlayer.p1++; }],
	];
	for (const [name, mutate] of cases) {
		const counters = makeCounters(1);
		mutate(counters);
		const record = pairRecord(manifest, 0, ['completed', 'completed'], [counters, makeCounters(2)]);
		assert.throws(() => summarizeWork({ manifest, pairResults: [record] }), error => (
			error.message === 'Invalid input.' && error.code === 'INVALID_INPUT'
		), name);
	}
});

test('per-player resolutions, submitted actions, and selection inequalities are enforced', () => {
	const manifest = generateManifest(ROOT_SEED, 1);
	const cases = [
		['resolution reconciliation', counters => {
			setSeatCounter(counters, ['choiceResolutions', 'accepted'], 'p1', getSeatCounter(counters, ['choiceResolutions', 'accepted'], 'p1') + 1);
		}],
		['submitted action reconciliation', counters => {
			setSeatCounter(counters, ['actions', 'submitted', 'switch'], 'p1', getSeatCounter(counters, ['actions', 'submitted', 'switch'], 'p1') + 1);
		}],
		['submitted move exceeds selected move', counters => {
			setSeatCounter(counters, ['actions', 'submitted', 'move'], 'p1', 4);
			setSeatCounter(counters, ['choiceSubmissions'], 'p1', 4);
			setSeatCounter(counters, ['choiceResolutions', 'accepted'], 'p1', 3);
		}],
		['submitted switch exceeds selected switch', counters => {
			setSeatCounter(counters, ['actions', 'submitted', 'switch'], 'p1', 2);
			setSeatCounter(counters, ['choiceSubmissions'], 'p1', 4);
			setSeatCounter(counters, ['choiceResolutions', 'accepted'], 'p1', 3);
		}],
		['selected actions exceed invocations', counters => {
			setSeatCounter(counters, ['actions', 'selected', 'move'], 'p1', 9);
		}],
	];
	for (const [name, mutate] of cases) {
		const counters = makeCounters(1);
		mutate(counters);
		const record = pairRecord(manifest, 0, ['completed', 'completed'], [counters, makeCounters(2)]);
		assert.throws(() => summarizeWork({ manifest, pairResults: [record] }), error => (
			error.message === 'Invalid input.' && error.code === 'INVALID_INPUT'
		), name);
	}
});

test('rejects per-record, cross-record, and allRecorded aggregate safe-integer overflow', () => {
	const manifest = generateManifest(ROOT_SEED, 2);
	const malformed = makeCounters(1);
	malformed.policyInvocations.byPlayer.p1 = Number.MAX_SAFE_INTEGER;
	malformed.policyInvocations.byPlayer.p2 = 1;
	malformed.policyInvocations.total = Number.MAX_SAFE_INTEGER;
	assert.throws(() => summarizeWork({
		manifest,
		pairResults: [pairRecord(manifest, 0, ['completed', 'completed'], [malformed, makeCounters(1)])],
	}));

	const tooMuchInCompleted = [makeCounters(1, { objectPackets: Number.MAX_SAFE_INTEGER }), makeCounters(2, { objectPackets: 1 })];
	assert.throws(() => summarizeWork({
		manifest,
		pairResults: [pairRecord(manifest, 0, ['completed', 'completed'], tooMuchInCompleted)],
	}));

	const completeOnly = pairRecord(manifest, 0, ['completed', 'completed'], [
		makeCounters(1, { objectPackets: Number.MAX_SAFE_INTEGER }),
		makeCounters(2, { objectPackets: 0 }),
	]);
	assert.equal(summarizeWork({ manifest, pairResults: [completeOnly] }).cohorts.completed.observedTotals.global.objectPackets, Number.MAX_SAFE_INTEGER);
	const failedOnly = pairRecord(manifest, 0, ['failed', 'not_attempted'], [makeCounters(1, { objectPackets: 1 }), null]);
	assert.equal(summarizeWork({ manifest, pairResults: [failedOnly] }).cohorts.failedRecorded.observedTotals.global.objectPackets, 1);

	const separateCohorts = pairRecord(manifest, 0, ['completed', 'failed'], [
		makeCounters(1, { objectPackets: Number.MAX_SAFE_INTEGER }),
		makeCounters(2, { objectPackets: 1 }),
	]);
	assert.throws(() => summarizeWork({ manifest, pairResults: [separateCohorts] }), error => (
		error.message === 'Invalid input.' && error.code === 'INVALID_INPUT'
	));
});

test('throwing getters produce a fixed safe error and JSON inputs are not modified', () => {
	const manifest = generateManifest(ROOT_SEED, 1);
	const normal = pairRecord(manifest, 0, ['completed', 'completed'], [makeCounters(1), makeCounters(2)]);
	const before = JSON.stringify({ manifest, pairResults: [normal] });
	const report = summarizeWork({ manifest, pairResults: [normal] });
	assert.equal(JSON.stringify({ manifest, pairResults: [normal] }), before);
	assert.equal(report.kind, 'gen5-paired-work-accounting');

	const badRecord = pairRecord(manifest, 0, ['completed', 'completed'], [makeCounters(1), makeCounters(2)]);
	let getterCalled = false;
	Object.defineProperty(badRecord.runs[0].resultRecord.counters, 'publicEvents', {
		enumerable: true,
		get() { getterCalled = true; throw new Error('PRIVATE_GETTER_DETAIL'); },
	});
	assert.throws(() => summarizeWork({ manifest, pairResults: [badRecord] }), error => (
		error.message === 'Invalid input.' && error.code === 'INVALID_INPUT'
	));
	assert.equal(getterCalled, false);
	const customPrototype = Object.assign(Object.create({ foreign: true }), { ...manifest });
	assert.throws(() => summarizeWork({ manifest: customPrototype, pairResults: [] }), error => (
		error.message === 'Invalid input.' && error.code === 'INVALID_INPUT'
	));
});

test('offline CLI reports aggregates, keeps artifacts unchanged, and emits safe errors without oracle bootstrap', t => {
	const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'paired-work-summary-'));
	t.after(() => fs.rmSync(tempDir, { recursive: true, force: true }));
	createNoOraclePreload(tempDir);
	const manifest = generateManifest(ROOT_SEED, 1);
	const record = pairRecord(manifest, 0, ['completed', 'completed'], [makeCounters(1), makeCounters(2)]);
	const manifestPath = path.join(tempDir, 'manifest.json');
	const resultsPath = path.join(tempDir, 'results.jsonl');
	const manifestBytes = `${JSON.stringify(manifest, null, 2)}\n`;
	const resultsBytes = `${JSON.stringify(record)}\n`;
	fs.writeFileSync(manifestPath, manifestBytes);
	fs.writeFileSync(resultsPath, resultsBytes);
	const success = cli(tempDir, ['--manifest', manifestPath, '--results', resultsPath]);
	assert.equal(success.status, 0, success.stderr);
	assert.equal(JSON.parse(success.stdout).kind, 'gen5-paired-work-accounting');
	assert.equal(fs.readFileSync(manifestPath, 'utf8'), manifestBytes);
	assert.equal(fs.readFileSync(resultsPath, 'utf8'), resultsBytes);

	for (const badJournal of ['{', JSON.stringify(record), `${JSON.stringify(record)}\n\n`]) {
		fs.writeFileSync(resultsPath, badJournal);
		const failure = cli(tempDir, ['--manifest', manifestPath, '--results', resultsPath]);
		assert.equal(failure.status, 1);
		assert.equal(failure.stdout, '');
		assert.doesNotThrow(() => JSON.parse(failure.stderr));
		assert.equal(failure.stderr.includes(manifestPath), false);
		assert.equal(failure.stderr.includes('PRIVATE'), false);
	}
	fs.writeFileSync(resultsPath, resultsBytes);
	fs.writeFileSync(manifestPath, '{');
	const malformedManifest = cli(tempDir, ['--manifest', manifestPath, '--results', resultsPath]);
	assert.equal(malformedManifest.status, 1);
	assert.equal(malformedManifest.stderr.includes('SyntaxError'), false);
	const missingFile = cli(tempDir, ['--manifest', path.join(tempDir, 'missing.json'), '--results', resultsPath]);
	assert.equal(missingFile.status, 1);
	assert.equal(missingFile.stderr.includes('ENOENT'), false);
	for (const args of [
		[], ['--manifest', manifestPath], ['--manifest', manifestPath, '--manifest', manifestPath, '--results', resultsPath],
		['--unknown', 'x', '--results', resultsPath],
	]) {
		const failure = cli(tempDir, args);
		assert.equal(failure.status, 1);
		assert.equal(failure.stderr.includes('oracle bootstrap is forbidden'), false);
	}
	const help = cli(tempDir, ['--help']);
	assert.equal(help.status, 0);
	assert.match(help.stdout, /Usage: node environment\/summarize-work\.cjs/);
});

test('pinned runPair success and genuine packet-limit failure preserve actual counters', { timeout: 180_000 }, async () => {
	oracle.assertReady();
	const ps = require('../reference/vendor/pokemon-showdown/dist/sim');
	const manifest = generateManifest(ROOT_SEED, 2);
	const successful = await runPair({ ps, rootSeed: ROOT_SEED, pairIndex: 0 });
	let failed;
	try {
		await runPair({ ps, rootSeed: ROOT_SEED, pairIndex: 1, limits: { maxPackets: 1 } });
	} catch (error) {
		failed = error.pairResult;
	}
	assert.equal(successful.status, 'completed');
	assert.equal(failed?.status, 'failed');
	assert.equal(failed.runs[0].status, 'failed');
	assert.equal(failed.runs[1].status, 'not_attempted');
	assert.ok(failed.runs[0].resultRecord.counters.objectPackets > 0);

	const successReport = summarizeWork({ manifest: generateManifest(ROOT_SEED, 1), pairResults: [successful] });
	const successRuns = successful.runs.filter(run => run.resultRecord !== null);
	assert.deepEqual(successReport.cohorts.completed.observedTotals, explicitTotals(successRuns));
	const failureReport = summarizeWork({ manifest, pairResults: [successful, failed] });
	assert.deepEqual(failureReport.cohorts.failedRecorded.observedTotals, explicitTotals([failed.runs[0]]));
	assert.equal(failureReport.cohorts.failedRecorded.counterRecords.withCounters, 1);
	assert.equal(failureReport.cohorts.failedRecorded.counterRecords.missingCounters, 0);
});
