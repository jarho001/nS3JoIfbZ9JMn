'use strict';

const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const oracle = require('../reference/oracle.cjs');
const referenceLock = require('../reference-lock.json');
const { generateManifest } = require('../environment/paired-seeds.cjs');
const { runPair } = require('../environment/run-pair.cjs');
const { parseJournal, summarizeResults } = require('../environment/summarize-results.cjs');

const ROOT = path.resolve(__dirname, '..');
const CLI = path.join(ROOT, 'environment/summarize-results.cjs');
const ROOT_SEED = [123, 456, 789, 1011];

function makeNested(expectedRun, outcome, status = 'completed') {
	return {
		schemaVersion: 1,
		kind: 'battle-run-result',
		upstreamCommit: referenceLock.commit,
		formatId: 'gen5randombattle',
		seeds: structuredClone(expectedRun.seeds),
		teamSource: 'prepared',
		status,
		outcome: status === 'completed' ? outcome : null,
		failure: status === 'failed' ? { category: 'packet_limit', code: 'packet_limit_exhausted' } : null,
	};
}

function completedPair(manifest, index, outcomes) {
	const expected = manifest.pairs[index];
	return {
		schemaVersion: 1,
		kind: 'gen5-paired-battle-result',
		status: 'completed',
		upstream: structuredClone(manifest.upstream),
		formatId: manifest.formatId,
		rootSeed: [...manifest.rootSeed],
		pairIndex: index,
		pairId: expected.pairId,
		teams: {
			A: { seed: [...expected.teams.A.seed], packedTeamSha256: 'a'.repeat(64) },
			B: { seed: [...expected.teams.B.seed], packedTeamSha256: 'b'.repeat(64) },
		},
		runs: expected.runs.map((run, runIndex) => ({
			runId: run.runId,
			status: 'completed',
			seats: structuredClone(run.seats),
			policySeats: structuredClone(run.policySeats),
			seeds: structuredClone(run.seeds),
			resultRecord: makeNested(run, outcomes[runIndex]),
		})),
		failure: null,
		elapsedScope: 'paired_runner',
		elapsedMs: 10,
	};
}

function failedPair(manifest, index, stage) {
	const expected = manifest.pairs[index];
	const record = completedPair(manifest, index, [
		{ result: 'win', winner: 'p1', turns: 8 },
		{ result: 'win', winner: 'p2', turns: 9 },
	]);
	record.status = 'failed';
	record.failure = { stage, category: stage === 'preparation' ? 'initialization' : 'packet_limit', code: 'controlled_failure' };
	if (stage === 'preparation') {
		record.teams = { A: { seed: [...expected.teams.A.seed] }, B: null };
		record.runs = expected.runs.map(run => ({
			runId: run.runId, status: 'not_attempted', seats: structuredClone(run.seats),
			policySeats: structuredClone(run.policySeats), seeds: structuredClone(run.seeds), resultRecord: null,
		}));
	} else if (stage === 'first_battle') {
		record.runs[0].status = 'failed';
		record.runs[0].resultRecord = makeNested(expected.runs[0], null, 'failed');
		record.runs[1].status = 'not_attempted';
		record.runs[1].resultRecord = null;
	} else {
		record.runs[1].status = 'failed';
		record.runs[1].resultRecord = makeNested(expected.runs[1], null, 'failed');
	}
	return record;
}

function cli(args) {
	return spawnSync(process.execPath, [CLI, ...args], { cwd: ROOT, encoding: 'utf8' });
}

test('summarizeResults counts wins, losses, and ties by seat and policy identity', () => {
	const manifest = generateManifest(ROOT_SEED, 3);
	const pairResults = [
		completedPair(manifest, 0, [
			{ result: 'win', winner: 'p1', turns: 4 },
			{ result: 'win', winner: 'p2', turns: 5 },
		]),
		completedPair(manifest, 1, [
			{ result: 'win', winner: 'p2', turns: 6 },
			{ result: 'tie', winner: null, turns: 7 },
		]),
		completedPair(manifest, 2, [
			{ result: 'win', winner: 'p2', turns: 8 },
			{ result: 'win', winner: 'p1', turns: 9 },
		]),
	];
	const beforeManifest = JSON.stringify(manifest);
	const beforeResults = JSON.stringify(pairResults);
	const report = summarizeResults({ manifest, pairResults });
	assert.deepEqual(report, {
		schemaVersion: 1,
		kind: 'gen5-paired-outcome-accounting',
		manifest: {
			formatId: 'gen5randombattle',
			upstream: { repository: referenceLock.repository, commit: referenceLock.commit },
			rootSeed: ROOT_SEED,
			pairCount: 3,
		},
		coverage: 'manifest_prefix_complete',
		pairs: { recorded: 3, unrecorded: 0, completed: 3, failed: 0 },
		runSlots: { completed: 6, failed: 0, notAttempted: 0, unrecorded: 0 },
		outcomes: {
			completedBattles: 6,
			bySeat: {
				p1: { wins: 2, losses: 3, ties: 1 },
				p2: { wins: 3, losses: 2, ties: 1 },
			},
			byPolicy: {
				policyA: {
					overall: { wins: 2, losses: 3, ties: 1 },
					bySeat: { p1: { wins: 1, losses: 2, ties: 0 }, p2: { wins: 1, losses: 1, ties: 1 } },
				},
				policyB: {
					overall: { wins: 3, losses: 2, ties: 1 },
					bySeat: { p1: { wins: 1, losses: 1, ties: 1 }, p2: { wins: 2, losses: 1, ties: 0 } },
				},
			},
			overall: { winsBySeat: { p1: 2, p2: 3 }, ties: 1 },
		},
	});
	assert.equal(JSON.stringify(manifest), beforeManifest);
	assert.equal(JSON.stringify(pairResults), beforeResults);
	assert.equal(JSON.stringify(report).includes('packedTeamSha256'), false);
	assert.equal(JSON.stringify(report).includes('elapsedMs'), false);
	assert.equal(JSON.stringify(report).includes('counters'), false);
	assert.equal(JSON.stringify(report).includes('private'), false);
});

test('empty prefix reports all pair and run slots as unrecorded', () => {
	const manifest = generateManifest(ROOT_SEED, 2);
	const report = summarizeResults({ manifest, pairResults: [] });
	assert.equal(report.coverage, 'recorded_prefix');
	assert.deepEqual(report.pairs, { recorded: 0, unrecorded: 2, completed: 0, failed: 0 });
	assert.deepEqual(report.runSlots, { completed: 0, failed: 0, notAttempted: 0, unrecorded: 4 });
	assert.equal(report.outcomes.completedBattles, 0);
});

test('failed preparation and first or second battle preserve run-slot topology', () => {
	const manifest = generateManifest(ROOT_SEED, 3);
	for (const [record, expectedSlots] of [
		[failedPair(manifest, 0, 'preparation'), { completed: 0, failed: 0, notAttempted: 2, unrecorded: 4 }],
		[failedPair(manifest, 0, 'first_battle'), { completed: 0, failed: 1, notAttempted: 1, unrecorded: 4 }],
		[failedPair(manifest, 0, 'second_battle'), { completed: 1, failed: 1, notAttempted: 0, unrecorded: 4 }],
	]) {
		const report = summarizeResults({ manifest, pairResults: [record] });
		assert.deepEqual(report.pairs, { recorded: 1, unrecorded: 2, completed: 0, failed: 1 });
		assert.deepEqual(report.runSlots, expectedSlots);
		assert.equal(report.outcomes.completedBattles, expectedSlots.completed);
	}
});

test('rejects manifest tampering, non-prefix ordering, forged identity and inconsistent topology', () => {
	const manifest = generateManifest(ROOT_SEED, 3);
	const full = [
		completedPair(manifest, 0, [{ result: 'win', winner: 'p1', turns: 1 }, { result: 'win', winner: 'p2', turns: 2 }]),
		completedPair(manifest, 1, [{ result: 'win', winner: 'p1', turns: 3 }, { result: 'win', winner: 'p2', turns: 4 }]),
		completedPair(manifest, 2, [{ result: 'win', winner: 'p1', turns: 5 }, { result: 'win', winner: 'p2', turns: 6 }]),
	];
	const alteredManifest = structuredClone(manifest);
	alteredManifest.pairs[0].runs[0].seeds.policyP1[0]++;
	assert.throws(() => summarizeResults({ manifest: alteredManifest, pairResults: [] }));
	for (const records of [[full[1]], [full[0], full[2]], [full[0], full[0]], [full[0], failedPair(manifest, 1, 'preparation'), full[2]]]) {
		assert.throws(() => summarizeResults({ manifest, pairResults: records }));
	}
	const mutations = [
		record => { record.schemaVersion = 2; },
		record => { record.pairId = 'foreign'; },
		record => { record.upstream.commit = 'foreign'; },
		record => { record.formatId = 'foreign'; },
		record => { record.rootSeed[0]++; },
		record => { record.pairIndex++; },
		record => { record.status = 'failed'; },
		record => { record.failure = { stage: 'battle', category: 'runtime', code: 'contradiction' }; },
		record => { record.runs[0].runId = 'foreign'; },
		record => { record.runs[1].seats.p1Team = 'A'; },
		record => { record.runs[1].policySeats.p1 = 'policyA'; },
		record => { record.runs[0].seeds.battle[0]++; },
		record => { record.teams.A.seed[0]++; },
		record => { record.runs[0].status = 'not_attempted'; },
		record => { record.runs[0].status = 'failed'; },
		record => { record.runs[0].resultRecord.seeds.policyP1[0]++; },
		record => { record.runs[0].resultRecord.schemaVersion = 2; },
		record => { record.runs[0].resultRecord.upstreamCommit = 'foreign'; },
		record => { record.runs[0].resultRecord.formatId = 'foreign'; },
		record => { record.status = 'unknown'; },
		record => { record.runs[0].status = 'unknown'; },
		record => { record.runs[0].resultRecord.status = 'failed'; },
		record => { record.runs[0].resultRecord.outcome.winner = 'p9'; },
		record => { record.runs[0].resultRecord.outcome.turns = -1; },
		record => { record.runs[0].resultRecord.teamSource = 'generated'; },
	];
	for (const mutate of mutations) {
		const record = structuredClone(full[0]);
		mutate(record);
		assert.throws(() => summarizeResults({ manifest, pairResults: [record] }));
	}
	const malformedTie = structuredClone(full[0]);
	malformedTie.runs[0].resultRecord.outcome = { result: 'tie', winner: 'p1', turns: 1 };
	assert.throws(() => summarizeResults({ manifest, pairResults: [malformedTie] }));
	const foreignResult = structuredClone(full[0]);
	foreignResult.runs[0].resultRecord.kind = 'foreign-battle-result';
	assert.throws(() => summarizeResults({ manifest, pairResults: [foreignResult] }));
});

test('turns throwing property getters into a generic input error', () => {
	const manifest = generateManifest(ROOT_SEED, 1);
	const poisonedManifest = {};
	Object.defineProperty(poisonedManifest, 'rootSeed', { get() { throw new Error('INJECTED_GETTER_DETAIL'); } });
	assert.throws(() => summarizeResults({ manifest: poisonedManifest, pairResults: [] }), error => (
		error.message === 'Invalid input.' && error.code === 'INVALID_INPUT'
	));
	const poisonedRecord = {};
	Object.defineProperty(poisonedRecord, 'schemaVersion', { get() { throw new Error('INJECTED_GETTER_DETAIL'); } });
	assert.throws(() => summarizeResults({ manifest, pairResults: [poisonedRecord] }), error => (
		error.message === 'Invalid input.' && error.code === 'INVALID_INPUT'
	));
});

test('does not expose extra input fields or accept records after a failed pair', () => {
	const manifest = generateManifest(ROOT_SEED, 2);
	const failure = failedPair(manifest, 0, 'preparation');
	failure.foreignPrivateData = 'DO_NOT_EMIT';
	const report = summarizeResults({ manifest, pairResults: [failure] });
	assert.equal(JSON.stringify(report).includes('DO_NOT_EMIT'), false);
	const following = completedPair(manifest, 1, [
		{ result: 'win', winner: 'p1', turns: 1 }, { result: 'win', winner: 'p2', turns: 2 },
	]);
	assert.throws(() => summarizeResults({ manifest, pairResults: [failure, following] }));
});

test('parseJournal accepts an empty file and complete newline-delimited records only', () => {
	assert.deepEqual(parseJournal(''), []);
	assert.deepEqual(parseJournal('{"a":1}\n{"b":2}\n'), [{ a: 1 }, { b: 2 }]);
	for (const text of ['{', '{"a":1}', '{"a":1}\n\n', '\n']) assert.throws(() => parseJournal(text));
});

test('offline CLI writes a safe report and rejects malformed or inaccessible input', t => {
	const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'paired-summary-'));
	t.after(() => fs.rmSync(temp, { recursive: true, force: true }));
	const manifest = generateManifest(ROOT_SEED, 1);
	const pair = completedPair(manifest, 0, [
		{ result: 'win', winner: 'p1', turns: 1 }, { result: 'tie', winner: null, turns: 2 },
	]);
	const manifestPath = path.join(temp, 'manifest.json');
	const journalPath = path.join(temp, 'results.jsonl');
	const manifestBytes = `${JSON.stringify(manifest, null, 2)}\n`;
	const journalBytes = `${JSON.stringify(pair)}\n`;
	fs.writeFileSync(manifestPath, manifestBytes);
	fs.writeFileSync(journalPath, journalBytes);
	const success = cli(['--manifest', manifestPath, '--results', journalPath]);
	assert.equal(success.status, 0, success.stderr);
	assert.equal(JSON.parse(success.stdout).outcomes.completedBattles, 2);
	assert.equal(fs.readFileSync(manifestPath, 'utf8'), manifestBytes);
	assert.equal(fs.readFileSync(journalPath, 'utf8'), journalBytes);
	for (const badJournal of ['{', `${JSON.stringify(pair)}`, `${JSON.stringify(pair)}\n\n`]) {
		fs.writeFileSync(journalPath, badJournal);
		const failure = cli(['--manifest', manifestPath, '--results', journalPath]);
		assert.notEqual(failure.status, 0);
		assert.equal(failure.stdout, '');
		assert.equal(failure.stderr.includes(manifestPath), false);
		assert.equal(failure.stderr.includes('DO_NOT_EMIT'), false);
		assert.doesNotThrow(() => JSON.parse(failure.stderr));
	}
	fs.writeFileSync(journalPath, journalBytes);
	fs.writeFileSync(manifestPath, '{');
	const badManifest = cli(['--manifest', manifestPath, '--results', journalPath]);
	assert.notEqual(badManifest.status, 0);
	assert.equal(badManifest.stderr.includes(manifestPath), false);
	const missingFile = cli(['--manifest', path.join(temp, 'missing.json'), '--results', journalPath]);
	assert.notEqual(missingFile.status, 0);
	assert.equal(missingFile.stderr.includes('ENOENT'), false);
	const badArgs = cli(['--manifest', manifestPath, '--manifest', manifestPath, '--results', journalPath]);
	assert.notEqual(badArgs.status, 0);
});

test('pinned runPair success and real packet-limit partial failure are accounted offline', async t => {
	oracle.assertReady();
	const ps = require('../reference/vendor/pokemon-showdown/dist/sim');
	const manifest = generateManifest(ROOT_SEED, 2);
	const successful = await runPair({ ps, rootSeed: ROOT_SEED, pairIndex: 0 });
	let failure;
	try {
		await runPair({ ps, rootSeed: ROOT_SEED, pairIndex: 1, limits: { maxPackets: 1 } });
	} catch (error) {
		failure = error.pairResult;
	}
	assert.ok(failure);
	assert.equal(failure.status, 'failed');
	assert.equal(failure.runs[0].status, 'failed');
	assert.equal(failure.runs[1].status, 'not_attempted');
	const report = summarizeResults({ manifest, pairResults: [successful, failure] });
	assert.equal(report.coverage, 'manifest_prefix_complete');
	assert.deepEqual(report.pairs, { recorded: 2, unrecorded: 0, completed: 1, failed: 1 });
	assert.deepEqual(report.runSlots, { completed: 2, failed: 1, notAttempted: 1, unrecorded: 0 });
	assert.equal(report.outcomes.completedBattles, 2);
	assert.deepEqual(report.outcomes.overall, {
		winsBySeat: { p1: 1, p2: 1 },
		ties: 0,
	});

	const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'paired-summary-offline-'));
	t.after(() => fs.rmSync(temp, { recursive: true, force: true }));
	const manifestPath = path.join(temp, 'manifest.json');
	const journalPath = path.join(temp, 'results.jsonl');
	const manifestBytes = `${JSON.stringify(manifest)}\n`;
	const journalBytes = `${JSON.stringify(successful)}\n${JSON.stringify(failure)}\n`;
	fs.writeFileSync(manifestPath, manifestBytes);
	fs.writeFileSync(journalPath, journalBytes);
	const result = cli(['--manifest', manifestPath, '--results', journalPath]);
	assert.equal(result.status, 0, result.stderr);
	assert.deepEqual(JSON.parse(result.stdout), report);
	assert.equal(fs.readFileSync(manifestPath, 'utf8'), manifestBytes);
	assert.equal(fs.readFileSync(journalPath, 'utf8'), journalBytes);
});

test('pinned initialization failure may have a failed result before prepared team source is recorded', async () => {
	oracle.assertReady();
	const pinned = require('../reference/vendor/pokemon-showdown/dist/sim');
	let pairRecord;
	try {
		await runPair({ ps: { Teams: pinned.Teams, Dex: pinned.Dex }, rootSeed: ROOT_SEED, pairIndex: 0 });
	} catch (error) {
		pairRecord = error.pairResult;
	}
	assert.ok(pairRecord);
	assert.equal(pairRecord.runs[0].status, 'failed');
	assert.equal(pairRecord.runs[0].resultRecord.status, 'failed');
	assert.equal(Object.hasOwn(pairRecord.runs[0].resultRecord, 'teamSource'), false);
	const report = summarizeResults({ manifest: generateManifest(ROOT_SEED, 1), pairResults: [pairRecord] });
	assert.deepEqual(report.runSlots, { completed: 0, failed: 1, notAttempted: 1, unrecorded: 0 });
});
