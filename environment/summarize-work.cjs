'use strict';

const fs = require('node:fs');
const { parseJournal, summarizeResults } = require('./summarize-results.cjs');

const REPORT_KIND = 'gen5-paired-work-accounting';
const USAGE = 'Usage: node environment/summarize-work.cjs --manifest PATH --results PATH';
const PLAYER_COUNTER_PATHS = [
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

function copyJsonData(value, ancestors = new Set()) {
	if (value === null || typeof value === 'string' || typeof value === 'boolean') return value;
	if (typeof value === 'number') {
		if (!Number.isFinite(value)) throw invalid();
		return value;
	}
	if (typeof value !== 'object' || ancestors.has(value)) throw invalid();
	ancestors.add(value);
	let copy;
	if (Array.isArray(value)) {
		if (Object.getPrototypeOf(value) !== Array.prototype) throw invalid();
		const keys = Reflect.ownKeys(value);
		const descriptors = Object.getOwnPropertyDescriptors(value);
		const length = descriptors.length?.value;
		if (!Number.isSafeInteger(length) || length < 0 || keys.length !== length + 1) throw invalid();
		copy = new Array(length);
		for (let index = 0; index < length; index++) {
			const descriptor = descriptors[index];
			if (!descriptor || !Object.hasOwn(descriptor, 'value') || !descriptor.enumerable) throw invalid();
			copy[index] = copyJsonData(descriptor.value, ancestors);
		}
	} else {
		const prototype = Object.getPrototypeOf(value);
		if (prototype !== Object.prototype && prototype !== null) throw invalid();
		copy = prototype === null ? Object.create(null) : {};
		const keys = Reflect.ownKeys(value);
		const descriptors = Object.getOwnPropertyDescriptors(value);
		for (const key of keys) {
			if (typeof key !== 'string') throw invalid();
			const descriptor = descriptors[key];
			if (!descriptor || !Object.hasOwn(descriptor, 'value') || !descriptor.enumerable) throw invalid();
			Object.defineProperty(copy, key, {
				value: copyJsonData(descriptor.value, ancestors),
				enumerable: true,
				writable: true,
				configurable: true,
			});
		}
	}
	ancestors.delete(value);
	return copy;
}

function requireObject(value) {
	if (!isPlainObject(value)) throw invalid();
	return value;
}

function requireCounter(value) {
	if (!Number.isSafeInteger(value) || value < 0) throw invalid();
	return value;
}

function checkedAdd(left, right) {
	const sum = left + right;
	if (!Number.isSafeInteger(sum) || sum < 0) throw invalid();
	return sum;
}

function getPath(value, path) {
	let current = value;
	for (const key of path) current = current?.[key];
	return current;
}

function emptyMetric() {
	return { total: 0, bySeat: { p1: 0, p2: 0 }, byPolicy: { policyA: 0, policyB: 0 } };
}

function emptyTotals() {
	const playerCounters = {
		policyInvocations: emptyMetric(),
		choiceSubmissions: emptyMetric(),
		choiceResolutions: { accepted: emptyMetric(), rejected: emptyMetric(), pending: emptyMetric() },
		retryRequests: emptyMetric(),
		retrySubmissions: emptyMetric(),
		retryLimitExhaustions: emptyMetric(),
		actions: {
			selected: { move: emptyMetric(), switch: emptyMetric() },
			submitted: { move: emptyMetric(), switch: emptyMetric() },
		},
	};
	return { global: { objectPackets: 0, publicEvents: 0 }, playerCounters };
}

function validatePlayerCounter(value) {
	requireObject(value);
	const total = requireCounter(value.total);
	requireObject(value.byPlayer);
	const p1 = requireCounter(value.byPlayer.p1);
	const p2 = requireCounter(value.byPlayer.p2);
	if (checkedAdd(p1, p2) !== total) throw invalid();
	return { total, byPlayer: { p1, p2 } };
}

function validateCounters(resultRecord) {
	const counters = requireObject(resultRecord.counters);
	const global = {
		objectPackets: requireCounter(counters.objectPackets),
		publicEvents: requireCounter(counters.publicEvents),
	};
	const playerCounters = {};
	for (const path of PLAYER_COUNTER_PATHS) {
		const name = path.join('.');
		playerCounters[name] = validatePlayerCounter(getPath(counters, path));
	}

	for (const player of ['p1', 'p2']) {
		const read = path => playerCounters[path.join('.')].byPlayer[player];
		const submissions = read(['choiceSubmissions']);
		const resolutions = checkedAdd(
			checkedAdd(read(['choiceResolutions', 'accepted']), read(['choiceResolutions', 'rejected'])),
			read(['choiceResolutions', 'pending']),
		);
		if (resolutions !== submissions) throw invalid();

		const submittedActions = checkedAdd(
			read(['actions', 'submitted', 'move']),
			read(['actions', 'submitted', 'switch']),
		);
		if (submittedActions !== submissions) throw invalid();
		if (read(['actions', 'submitted', 'move']) > read(['actions', 'selected', 'move'])
			|| read(['actions', 'submitted', 'switch']) > read(['actions', 'selected', 'switch'])) throw invalid();
		const selectedActions = checkedAdd(
			read(['actions', 'selected', 'move']),
			read(['actions', 'selected', 'switch']),
		);
		if (selectedActions > read(['policyInvocations'])) throw invalid();
	}
	return { global, playerCounters };
}

function addCounter(totals, recordCounters, policySeats) {
	for (const key of ['objectPackets', 'publicEvents']) {
		totals.global[key] = checkedAdd(totals.global[key], recordCounters.global[key]);
	}
	for (const path of PLAYER_COUNTER_PATHS) {
		const name = path.join('.');
		const target = getPath(totals.playerCounters, path);
		const counter = recordCounters.playerCounters[name];
		target.total = checkedAdd(target.total, counter.total);
		for (const player of ['p1', 'p2']) {
			target.bySeat[player] = checkedAdd(target.bySeat[player], counter.byPlayer[player]);
			const policy = policySeats[player];
			if (!['policyA', 'policyB'].includes(policy)) throw invalid();
			target.byPolicy[policy] = checkedAdd(target.byPolicy[policy], counter.byPlayer[player]);
		}
	}
}

function newCohort() {
	return {
		runCount: 0,
		counterRecords: { withCounters: 0, missingCounters: 0 },
		counterCoverage: 'no_recorded_runs',
		observedTotals: emptyTotals(),
	};
}

function addRunToCohort(cohort, runRecord) {
	cohort.runCount = checkedAdd(cohort.runCount, 1);
	if (runRecord.resultRecord === null) {
		cohort.counterRecords.missingCounters = checkedAdd(cohort.counterRecords.missingCounters, 1);
		cohort.counterCoverage = 'some_recorded_runs_missing_counters';
		return;
	}
	const counters = validateCounters(runRecord.resultRecord);
	cohort.counterRecords.withCounters = checkedAdd(cohort.counterRecords.withCounters, 1);
	if (cohort.counterRecords.missingCounters === 0) cohort.counterCoverage = 'all_recorded_runs_have_counters';
	addCounter(cohort.observedTotals, counters, runRecord.policySeats);
}

function summarizeWork(options) {
	try {
		const snapshot = copyJsonData(options);
		requireObject(snapshot);
		const { manifest, pairResults } = snapshot;
		const outcomeReport = summarizeResults({ manifest, pairResults });
		const cohorts = {
			completed: newCohort(),
			failedRecorded: newCohort(),
			allRecorded: newCohort(),
		};
		for (const pair of pairResults) {
			for (const runRecord of pair.runs) {
				if (runRecord.status === 'not_attempted') continue;
				const cohort = runRecord.status === 'completed' ? cohorts.completed : cohorts.failedRecorded;
				addRunToCohort(cohort, runRecord);
				addRunToCohort(cohorts.allRecorded, runRecord);
			}
		}
		return {
			schemaVersion: 1,
			kind: REPORT_KIND,
			manifest: outcomeReport.manifest,
			coverage: {
				pairPrefix: outcomeReport.coverage,
				pairs: outcomeReport.pairs,
				runSlots: outcomeReport.runSlots,
			},
			cohorts,
		};
	} catch {
		throw invalid();
	}
}

function parseArgs(args) {
	if (!Array.isArray(args) || args.some(argument => typeof argument !== 'string')) throw invalid();
	if (args.length === 1 && ['--help', 'help'].includes(args[0])) return { help: true };
	let manifestPath;
	let resultsPath;
	for (let index = 0; index < args.length; index++) {
		const argument = args[index];
		const separator = argument.indexOf('=');
		const flag = separator < 0 ? argument : argument.slice(0, separator);
		if (!['--manifest', '--results'].includes(flag)) throw invalid();
		const value = separator < 0 ? args[++index] : argument.slice(separator + 1);
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
		const parsed = parseArgs(args);
		if (parsed.help) {
			io.stdout.write(`${USAGE}\n`);
			return 0;
		}
		const manifest = JSON.parse(fs.readFileSync(parsed.manifestPath, 'utf8'));
		const pairResults = parseJournal(fs.readFileSync(parsed.resultsPath, 'utf8'));
		const report = summarizeWork({ manifest, pairResults });
		io.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
		return 0;
	} catch {
		try {
			io.stderr.write(`${JSON.stringify({ error: { code: 'invalid_input' }, usage: USAGE })}\n`);
		} catch {}
		return 1;
	}
}

if (require.main === module) process.exitCode = main();

module.exports = { main, parseArgs, summarizeWork };
