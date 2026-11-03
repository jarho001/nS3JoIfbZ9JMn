#!/usr/bin/env node
'use strict';

const fs = require('node:fs');
const { generateManifest } = require('./paired-seeds.cjs');

const REPORT_KIND = 'gen5-paired-outcome-accounting';
const PAIR_KIND = 'gen5-paired-battle-result';
const BATTLE_KIND = 'battle-run-result';
const USAGE = 'Usage: node environment/summarize-results.cjs --manifest PATH --results PATH';

function invalid() {
	const error = new Error('Invalid input.');
	error.code = 'INVALID_INPUT';
	return error;
}

function isPlainObject(value) {
	if (value === null || typeof value !== 'object' || Array.isArray(value)) return false;
	const prototype = Object.getPrototypeOf(value);
	return prototype === Object.prototype || prototype === null;
}

function canonicalJson(value, ancestors = new Set()) {
	if (value === null || typeof value === 'string' || typeof value === 'boolean') return JSON.stringify(value);
	if (typeof value === 'number') {
		if (!Number.isFinite(value)) throw invalid();
		return JSON.stringify(value);
	}
	if (typeof value !== 'object') throw invalid();
	if (ancestors.has(value)) throw invalid();
	ancestors.add(value);
	let encoded;
	if (Array.isArray(value)) {
		encoded = `[${value.map(item => canonicalJson(item, ancestors)).join(',')}]`;
	} else {
		if (!isPlainObject(value)) throw invalid();
		const keys = Object.keys(value).sort();
		encoded = `{${keys.map(key => `${JSON.stringify(key)}:${canonicalJson(value[key], ancestors)}`).join(',')}}`;
	}
	ancestors.delete(value);
	return encoded;
}

function sameJson(left, right) {
	return canonicalJson(left) === canonicalJson(right);
}

function requireObject(value) {
	if (!isPlainObject(value)) throw invalid();
	return value;
}

function requireOutcome(value) {
	requireObject(value);
	if (!Number.isSafeInteger(value.turns) || value.turns < 0) throw invalid();
	if (value.result === 'win' && ['p1', 'p2'].includes(value.winner)) return;
	if (value.result === 'tie' && value.winner === null) return;
	throw invalid();
}

function requireNestedResult(result, expectedRun, expectedManifest, status) {
	if (result === null) {
		if (status === 'failed') return;
		throw invalid();
	}
	requireObject(result);
	if (result.schemaVersion !== 1 || result.kind !== BATTLE_KIND
		|| result.upstreamCommit !== expectedManifest.upstream.commit
		|| result.formatId !== expectedManifest.formatId
		|| !sameJson(result.seeds, expectedRun.seeds)) throw invalid();
	if (status === 'completed') {
		if (result.status !== 'completed' || result.failure !== null || result.teamSource !== 'prepared') throw invalid();
		requireOutcome(result.outcome);
		return;
	}
	if (status === 'failed') {
		if (result.status !== 'failed' || result.outcome !== null
			|| (result.teamSource !== undefined && result.teamSource !== 'prepared')) throw invalid();
		requireObject(result.failure);
		if (typeof result.failure.category !== 'string' || typeof result.failure.code !== 'string') throw invalid();
		return;
	}
	throw invalid();
}

function validatePairRecord(record, expectedPair, expectedManifest) {
	requireObject(record);
	if (record.schemaVersion !== 1 || record.kind !== PAIR_KIND
		|| record.formatId !== expectedManifest.formatId
		|| record.upstream?.repository !== expectedManifest.upstream.repository
		|| record.upstream?.commit !== expectedManifest.upstream.commit
		|| !sameJson(record.rootSeed, expectedManifest.rootSeed)
		|| record.pairIndex !== expectedPair.pairIndex
		|| record.pairId !== expectedPair.pairId) throw invalid();
	if (!Array.isArray(record.runs) || record.runs.length !== expectedPair.runs.length) throw invalid();
	requireObject(record.teams);
	for (const teamId of ['A', 'B']) {
		const team = record.teams[teamId];
		if (team === null) continue;
		requireObject(team);
		if (!sameJson(team.seed, expectedPair.teams[teamId].seed)) throw invalid();
	}

	const statuses = [];
	for (let index = 0; index < expectedPair.runs.length; index++) {
		const actual = record.runs[index];
		const expected = expectedPair.runs[index];
		requireObject(actual);
		if (actual.runId !== expected.runId || !sameJson(actual.seats, expected.seats)
			|| !sameJson(actual.policySeats, expected.policySeats)
			|| !sameJson(actual.seeds, expected.seeds)) throw invalid();
		if (!['completed', 'failed', 'not_attempted'].includes(actual.status)) throw invalid();
		if (actual.status === 'not_attempted') {
			if (actual.resultRecord !== null) throw invalid();
		} else {
			requireNestedResult(actual.resultRecord, expected, expectedManifest, actual.status);
		}
		statuses.push(actual.status);
	}

	if (record.status === 'completed') {
		if (statuses.some(status => status !== 'completed') || record.failure !== null) throw invalid();
		if (record.teams.A === null || record.teams.B === null) throw invalid();
		return;
	}
	if (record.status !== 'failed') throw invalid();
	requireObject(record.failure);
	if (typeof record.failure.stage !== 'string' || typeof record.failure.category !== 'string'
		|| typeof record.failure.code !== 'string') throw invalid();
	const validFailureTopology = (statuses[0] === 'not_attempted' && statuses[1] === 'not_attempted')
		|| (statuses[0] === 'failed' && statuses[1] === 'not_attempted')
		|| (statuses[0] === 'completed' && statuses[1] === 'failed');
	if (!validFailureTopology) throw invalid();
	if (statuses[0] === 'not_attempted' && record.teams.A === null && record.teams.B !== null) throw invalid();
	if (statuses[0] !== 'not_attempted' && (record.teams.A === null || record.teams.B === null)) throw invalid();
}

function emptyTally() {
	return { wins: 0, losses: 0, ties: 0 };
}

function addOutcome(tally, outcome, seat) {
	if (outcome.result === 'tie') {
		tally.ties++;
	} else if (outcome.winner === seat) {
		tally.wins++;
	} else {
		tally.losses++;
	}
}

function summarizeResults(options) {
	try {
		requireObject(options);
		const manifest = options.manifest;
		const pairResults = options.pairResults;
		requireObject(manifest);
		if (!Array.isArray(pairResults)) throw invalid();
		let expectedManifest;
		try {
			expectedManifest = generateManifest(manifest.rootSeed, manifest.pairCount);
		} catch {
			throw invalid();
		}
		if (!sameJson(manifest, expectedManifest) || pairResults.length > expectedManifest.pairCount) throw invalid();

		const pairs = { completed: 0, failed: 0 };
		const runSlots = { completed: 0, failed: 0, notAttempted: 0, unrecorded: 0 };
		const bySeat = { p1: emptyTally(), p2: emptyTally() };
		const byPolicy = {
			policyA: { overall: emptyTally(), bySeat: { p1: emptyTally(), p2: emptyTally() } },
			policyB: { overall: emptyTally(), bySeat: { p1: emptyTally(), p2: emptyTally() } },
		};
		const overall = { winsBySeat: { p1: 0, p2: 0 }, ties: 0 };

		for (let index = 0; index < pairResults.length; index++) {
			const record = pairResults[index];
			const pair = expectedManifest.pairs[index];
			validatePairRecord(record, pair, expectedManifest);
			if (pairs.failed > 0) throw invalid();
			pairs[record.status]++;
			for (let runIndex = 0; runIndex < pair.runs.length; runIndex++) {
				const runRecord = record.runs[runIndex];
				runSlots[runRecord.status === 'not_attempted' ? 'notAttempted' : runRecord.status]++;
				if (runRecord.status !== 'completed') continue;
				const outcome = runRecord.resultRecord.outcome;
				const policyAtP1 = runRecord.policySeats.p1;
				const policyAtP2 = runRecord.policySeats.p2;
				addOutcome(bySeat.p1, outcome, 'p1');
				addOutcome(bySeat.p2, outcome, 'p2');
				addOutcome(byPolicy[policyAtP1].bySeat.p1, outcome, 'p1');
				addOutcome(byPolicy[policyAtP1].overall, outcome, 'p1');
				addOutcome(byPolicy[policyAtP2].bySeat.p2, outcome, 'p2');
				addOutcome(byPolicy[policyAtP2].overall, outcome, 'p2');
				if (outcome.result === 'tie') overall.ties++;
				else overall.winsBySeat[outcome.winner]++;
			}
		}
		const unrecordedPairs = expectedManifest.pairCount - pairResults.length;
		runSlots.unrecorded = unrecordedPairs * 2;
		return {
			schemaVersion: 1,
			kind: REPORT_KIND,
			manifest: {
				formatId: expectedManifest.formatId,
				upstream: { ...expectedManifest.upstream },
				rootSeed: [...expectedManifest.rootSeed],
				pairCount: expectedManifest.pairCount,
			},
			coverage: pairResults.length === expectedManifest.pairCount ? 'manifest_prefix_complete' : 'recorded_prefix',
			pairs: { recorded: pairResults.length, unrecorded: unrecordedPairs, ...pairs },
			runSlots,
			outcomes: {
				completedBattles: runSlots.completed,
				bySeat,
				byPolicy,
				overall,
			},
		};
	} catch {
		throw invalid();
	}
}

function parseJournal(text) {
	if (typeof text !== 'string') throw invalid();
	if (text === '') return [];
	if (!text.endsWith('\n')) throw invalid();
	const lines = text.slice(0, -1).split('\n');
	if (lines.some(line => line.trim() === '')) throw invalid();
	try {
		return lines.map(line => JSON.parse(line));
	} catch {
		throw invalid();
	}
}

function parseArgs(args) {
	let manifestPath;
	let resultsPath;
	for (let index = 0; index < args.length; index++) {
		const flag = args[index];
		if (!['--manifest', '--results'].includes(flag) || index + 1 >= args.length) throw invalid();
		const value = args[++index];
		if (typeof value !== 'string' || value.length === 0 || value.startsWith('--')) throw invalid();
		if (flag === '--manifest') {
			if (manifestPath !== undefined) throw invalid();
			manifestPath = value;
		} else {
			if (resultsPath !== undefined) throw invalid();
			resultsPath = value;
		}
	}
	if (manifestPath === undefined || resultsPath === undefined) throw invalid();
	return { manifestPath, resultsPath };
}

function main(args = process.argv.slice(2), io = { stdout: process.stdout, stderr: process.stderr }) {
	try {
		const { manifestPath, resultsPath } = parseArgs(args);
		const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
		const pairResults = parseJournal(fs.readFileSync(resultsPath, 'utf8'));
		const report = summarizeResults({ manifest, pairResults });
		io.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
		return 0;
	} catch {
		io.stderr.write(`${JSON.stringify({ error: { code: 'invalid_input' }, usage: USAGE })}\n`);
		return 1;
	}
}

if (require.main === module) process.exitCode = main();

module.exports = { main, parseArgs, parseJournal, summarizeResults };
