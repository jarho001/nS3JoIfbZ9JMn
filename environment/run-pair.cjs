#!/usr/bin/env node
'use strict';

const fs = require('node:fs');
const path = require('node:path');
const { performance } = require('node:perf_hooks');
const referenceLock = require('../reference-lock.json');
const { generatePair, parseRootSeed, validateSeed, MAX_PAIRS } = require('./paired-seeds.cjs');
const { prepareTeam } = require('./prepared-teams.cjs');
const { runBattle } = require('./battle-runner.cjs');

const FORMAT_ID = 'gen5randombattle';
const PAIR_ERRORS = new WeakSet();

function createPairRecord() {
	return {
		schemaVersion: 1,
		kind: 'gen5-paired-battle-result',
		status: 'failed',
		upstream: { repository: referenceLock.repository, commit: referenceLock.commit },
		formatId: FORMAT_ID,
		rootSeed: null,
		pairIndex: null,
		pairId: null,
		teams: { A: null, B: null },
		runs: [
			{ runId: null, status: 'not_attempted', seats: null, policySeats: null, seeds: null, resultRecord: null },
			{ runId: null, status: 'not_attempted', seats: null, policySeats: null, seeds: null, resultRecord: null },
		],
		failure: null,
		elapsedScope: 'paired_runner',
		elapsedMs: 0,
	};
}

function pairError(record, stage, code, cause = undefined, details = {}) {
	record.status = 'failed';
	record.failure = {
		stage,
		category: details.category || (stage === 'battle' ? 'runtime' : 'initialization'),
		code,
		...(details.runId ? { runId: details.runId } : {}),
	};
	const error = new Error(`Paired battle failed (${code}).`, { cause });
	error.name = 'PairRunnerError';
	error.pairResult = record;
	PAIR_ERRORS.add(error);
	return error;
}

function ownsPairError(error) {
	return Boolean(error && typeof error === 'object' && PAIR_ERRORS.has(error));
}

function copyRun(run) {
	return {
		runId: run.runId,
		status: 'not_attempted',
		seats: { ...run.seats },
		policySeats: { ...run.policySeats },
		seeds: structuredClone(run.seeds),
		resultRecord: null,
	};
}

function writeElapsed(record, started) {
	const elapsedMs = performance.now() - started;
	record.elapsedMs = Number.isFinite(elapsedMs) && elapsedMs >= 0 ? elapsedMs : 0;
}

async function runPair(options = {}) {
	const started = performance.now();
	const record = createPairRecord();
	try {
		if (options === null || typeof options !== 'object' || Array.isArray(options)) {
			throw pairError(record, 'configuration', 'invalid_configuration');
		}
		let supplied;
		try {
			const rootSeed = options.rootSeed;
			const pairIndex = options.pairIndex;
			const ps = options.ps;
			const inputLimits = options.limits;
			const streamFactory = options.streamFactory;
			supplied = {
				rootSeed,
				pairIndex,
				ps,
				limits: inputLimits === undefined ? undefined : { ...inputLimits },
				streamFactory,
			};
		} catch {
			throw pairError(record, 'configuration', 'invalid_configuration');
		}
		try {
			record.rootSeed = validateSeed(supplied.rootSeed);
		} catch {
			throw pairError(record, 'configuration', 'invalid_root_seed');
		}
		if (!Number.isSafeInteger(supplied.pairIndex) || supplied.pairIndex < 0 || supplied.pairIndex >= MAX_PAIRS) {
			throw pairError(record, 'configuration', 'invalid_pair_index');
		}
		record.pairIndex = supplied.pairIndex;

		let pair;
		try {
			pair = generatePair(record.rootSeed, record.pairIndex);
		} catch (error) {
			throw pairError(record, 'configuration', 'invalid_pair_selection', error);
		}
		record.pairId = pair.pairId;
		const runResults = pair.runs.map(copyRun);
		record.runs = runResults;

		let teamA;
		let teamB;
		try {
			teamA = prepareTeam(supplied.ps, pair.teams.A.seed);
			record.teams.A = { seed: [...teamA.seed], packedTeamSha256: teamA.packedTeamSha256 };
			teamB = prepareTeam(supplied.ps, pair.teams.B.seed);
			record.teams.B = { seed: [...teamB.seed], packedTeamSha256: teamB.packedTeamSha256 };
		} catch (error) {
			throw pairError(record, 'preparation', 'team_preparation_failed', error);
		}
		for (let index = 0; index < pair.runs.length; index++) {
			const config = pair.runs[index];
			const runRecord = record.runs[index];
			const handles = {
				p1: config.seats.p1Team === 'A' ? teamA : teamB,
				p2: config.seats.p2Team === 'A' ? teamA : teamB,
			};
			const battleOptions = {
				ps: supplied.ps,
				seeds: config.seeds,
				preparedTeams: handles,
				...(supplied.limits === undefined ? {} : { limits: supplied.limits }),
				...(supplied.streamFactory === undefined ? {} : { streamFactory: supplied.streamFactory }),
			};
			let result;
			try {
				result = await runBattle(battleOptions);
			} catch (error) {
				runRecord.status = 'failed';
				if (error && typeof error === 'object' && error.resultRecord && typeof error.resultRecord === 'object') {
					runRecord.resultRecord = structuredClone(error.resultRecord);
				}
				const failure = runRecord.resultRecord?.failure;
				const code = typeof failure?.code === 'string' ? failure.code : 'battle_failed';
				record.status = 'failed';
				throw pairError(record, 'battle', code, error, {
					category: typeof failure?.category === 'string' ? failure.category : 'runtime',
					runId: runRecord.runId,
				});
			}
			runRecord.status = 'completed';
			runRecord.resultRecord = structuredClone(result.resultRecord);
		}
		record.status = 'completed';
		return record;
	} catch (error) {
		if (ownsPairError(error) && error.pairResult === record) throw error;
		throw pairError(record, 'initialization', 'paired_run_initialization_failed', error);
	} finally {
		writeElapsed(record, started);
	}
}

function parseArgs(args) {
	if (!Array.isArray(args) || args.some(argument => typeof argument !== 'string')) {
		throw new TypeError('arguments must be strings.');
	}
	if (args.length === 1 && (args[0] === '--help' || args[0] === 'help')) return { help: true };
	const values = {};
	const names = new Set(['--root-seed', '--pair-index', '--result']);
	for (let index = 0; index < args.length; index++) {
		const argument = args[index];
		const separator = argument.indexOf('=');
		const name = separator < 0 ? argument : argument.slice(0, separator);
		if (!names.has(name)) throw new TypeError(`unknown argument: ${argument}`);
		if (Object.hasOwn(values, name)) throw new TypeError(`duplicate argument: ${name}`);
		const value = separator >= 0 ? argument.slice(separator + 1) : args[++index];
		if (typeof value !== 'string' || value.length === 0 || value.startsWith('--')) {
			throw new TypeError(`${name} requires a value.`);
		}
		values[name] = value;
	}
	if (!Object.hasOwn(values, '--root-seed')) throw new TypeError('required argument missing: --root-seed');
	if (!Object.hasOwn(values, '--pair-index')) throw new TypeError('required argument missing: --pair-index');
	if (!/^(?:0|[1-9]\d*)$/.test(values['--pair-index'])) throw new TypeError('--pair-index must be a zero-based decimal integer.');
	const pairIndex = Number(values['--pair-index']);
	if (!Number.isSafeInteger(pairIndex) || pairIndex >= MAX_PAIRS) {
		throw new RangeError(`--pair-index must be in the range 0..${MAX_PAIRS - 1}.`);
	}
	return {
		rootSeed: parseRootSeed(values['--root-seed']),
		pairIndex,
		resultPath: values['--result'] || null,
	};
}

function findResultPath(args) {
	for (let index = 0; index < args.length; index++) {
		const argument = args[index];
		const separator = argument.indexOf('=');
		if ((separator < 0 ? argument : argument.slice(0, separator)) !== '--result') continue;
		const value = separator < 0 ? args[index + 1] : argument.slice(separator + 1);
		return typeof value === 'string' && value && !value.startsWith('--') ? value : null;
	}
	return null;
}

function failedCliRecord(code, options = undefined) {
	const record = createPairRecord();
	if (options?.rootSeed) {
		try {
			record.rootSeed = Array.isArray(options.rootSeed)
				? validateSeed(options.rootSeed)
				: parseRootSeed(options.rootSeed);
		} catch {}
	}
	if (Number.isSafeInteger(options?.pairIndex) && options.pairIndex >= 0 && options.pairIndex < MAX_PAIRS) {
		record.pairIndex = options.pairIndex;
	}
	record.failure = { stage: 'initialization', category: 'initialization', code };
	return record;
}

async function main(args = process.argv.slice(2)) {
	if (args.length === 1 && (args[0] === '--help' || args[0] === 'help')) {
		process.stdout.write('Usage: node environment/run-pair.cjs --root-seed 1,2,3,4 --pair-index 0 [--result pair.json]\n');
		return;
	}
	const cliStarted = performance.now();
	let resultPath = findResultPath(args);
	let options;
	let pairResult;
	let pairRunStarted = null;
	try {
		options = parseArgs(args);
		resultPath = options.resultPath || resultPath;
		const oracle = require('../reference/oracle.cjs');
		oracle.assertReady();
		const ps = require('../reference/vendor/pokemon-showdown/dist/sim');
		pairRunStarted = performance.now();
		pairResult = await runPair({ ps, rootSeed: options.rootSeed, pairIndex: options.pairIndex });
	} catch (error) {
		if (pairRunStarted !== null && ownsPairError(error)) {
			pairResult = error.pairResult;
		} else if (pairRunStarted !== null) {
			pairResult = failedCliRecord('paired_run_failed', options);
			pairResult.failure.stage = 'paired_runner';
			pairResult.failure.category = 'runtime';
			pairResult.elapsedScope = 'paired_runner';
			const elapsedMs = performance.now() - pairRunStarted;
			pairResult.elapsedMs = Number.isFinite(elapsedMs) && elapsedMs >= 0 ? elapsedMs : 0;
		} else {
			pairResult = failedCliRecord(options ? 'cli_initialization_failed' : 'invalid_cli_configuration', options);
			pairResult.elapsedScope = 'cli_initialization';
			const elapsedMs = performance.now() - cliStarted;
			pairResult.elapsedMs = Number.isFinite(elapsedMs) && elapsedMs >= 0 ? elapsedMs : 0;
		}
		if (resultPath) {
			try {
				fs.writeFileSync(path.resolve(resultPath), `${JSON.stringify(pairResult, null, 2)}\n`);
			} catch {
				console.error('Pair result write failed (result_write_failed).');
				process.exitCode = 1;
				return;
			}
		}
		console.error(`Paired battle failed (${pairResult.failure?.code || 'paired_run_failed'}).`);
		process.exitCode = 1;
		return;
	}
	if (resultPath) {
		try {
			fs.writeFileSync(path.resolve(resultPath), `${JSON.stringify(pairResult, null, 2)}\n`);
		} catch {
			console.error('Pair result write failed (result_write_failed).');
			process.exitCode = 1;
			return;
		}
	}
	process.stdout.write(`${JSON.stringify(pairResult, null, 2)}\n`);
}

if (require.main === module) {
	main().catch(() => {
		console.error('Paired battle failed (cli_initialization_failed).');
		process.exitCode = 1;
	});
}

module.exports = { createPairRecord, main, parseArgs, runPair };
