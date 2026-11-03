#!/usr/bin/env node
'use strict';

const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const { performance } = require('node:perf_hooks');
const referenceLock = require('../reference-lock.json');
const {
	DERIVATION_DOMAIN,
	DERIVATION_ID,
	MAX_PAIRS,
	generateManifest,
	parseRootSeed,
	validatePairCount,
	validateSeed,
} = require('./paired-seeds.cjs');
const { DEFAULT_LIMITS } = require('./battle-runner.cjs');
const { ownsPairError, runPair } = require('./run-pair.cjs');

const FORMAT_ID = 'gen5randombattle';
const PAIR_ID_DOMAIN = 'gen5-paired-seeds-v1:pair-id';
const ERROR_INFO = new WeakMap();
const SAFE_CODE = /^[a-z0-9_]{1,96}$/;
const OPTION_KEYS = new Set(['ps', 'rootSeed', 'pairCount', 'outputDir', 'limits', 'streamFactory']);
const PAIR_FAILURE_CATEGORIES = new Set([
	'initialization', 'runtime', 'timeout', 'packet_limit', 'decision_limit', 'retry_limit', 'policy', 'protocol',
]);

function ownedError(code, summary = null) {
	const message = code === 'summary_write_failed'
		? 'Manifest run failed (summary_write_failed); summary.json could not be confirmed complete.'
		: `Manifest run failed (${code}).`;
	const error = new Error(message);
	error.name = 'ManifestRunnerError';
	if (summary) Object.defineProperty(error, 'summary', { value: summary, enumerable: true });
	ERROR_INFO.set(error, { code, summary });
	return error;
}

function safeCode(value, fallback) {
	return typeof value === 'string' && SAFE_CODE.test(value) ? value : fallback;
}

function copyDataObject(value, label) {
	if (value === null || typeof value !== 'object' || Array.isArray(value)) throw new TypeError(`${label} must be an object.`);
	const prototype = Object.getPrototypeOf(value);
	if (prototype !== Object.prototype && prototype !== null) throw new TypeError(`${label} must be a plain object.`);
	const descriptors = Object.getOwnPropertyDescriptors(value);
	const copy = {};
	for (const [key, descriptor] of Object.entries(descriptors)) {
		if (!descriptor.enumerable) continue;
		if (!Object.hasOwn(descriptor, 'value')) throw new TypeError(`${label} cannot contain accessors.`);
		copy[key] = descriptor.value;
	}
	return copy;
}

function snapshotOptions(options) {
	if (options === null || typeof options !== 'object' || Array.isArray(options)) throw new TypeError('options must be an object.');
	const prototype = Object.getPrototypeOf(options);
	if (prototype !== Object.prototype && prototype !== null) throw new TypeError('options must be a plain object.');
	const descriptors = Object.getOwnPropertyDescriptors(options);
	for (const key of Object.keys(descriptors)) {
		if (!OPTION_KEYS.has(key)) throw new TypeError('unknown option.');
		if (!Object.hasOwn(descriptors[key], 'value')) throw new TypeError('options cannot contain accessors.');
	}
	const read = key => {
		const descriptor = descriptors[key];
		return descriptor && Object.hasOwn(descriptor, 'value') ? descriptor.value : undefined;
	};
	const rootSeed = validateSeed(read('rootSeed'));
	const pairCount = validatePairCount(read('pairCount'));
	const outputDir = read('outputDir');
	if (typeof outputDir !== 'string' || outputDir.length === 0) throw new TypeError('outputDir must be a non-empty path.');
	const absoluteOutputDir = path.resolve(outputDir);
	const ps = read('ps');
	if (!ps || typeof ps !== 'object' || typeof ps.BattleStream !== 'function'
		|| typeof ps.Teams?.generate !== 'function' || typeof ps.Teams?.pack !== 'function'
		|| typeof ps.Dex?.formats?.get !== 'function' || ps.Dex.formats.get(FORMAT_ID)?.exists !== true) {
		throw new TypeError('a compatible Showdown API is required.');
	}
	const inputLimits = read('limits');
	const limits = inputLimits === undefined ? undefined : copyDataObject(inputLimits, 'limits');
	if (limits !== undefined) {
		const bounds = { ...DEFAULT_LIMITS, ...limits };
		if (Object.values(bounds).some(value => !Number.isFinite(value) || value <= 0)) {
			throw new TypeError('limits must be positive finite numbers.');
		}
	}
	const streamFactory = read('streamFactory');
	if (streamFactory !== undefined && typeof streamFactory !== 'function') throw new TypeError('streamFactory must be a function.');
	return { ps, rootSeed, pairCount, absoluteOutputDir, limits, streamFactory };
}

function elapsedMs(started) {
	const elapsed = performance.now() - started;
	return Number.isFinite(elapsed) && elapsed >= 0 ? elapsed : 0;
}

function ioFailure(stage, pairIndex = undefined) {
	return {
		stage,
		category: 'io',
		code: `${stage}_failed`,
		...(Number.isSafeInteger(pairIndex) ? { pairIndex } : {}),
	};
}

function summaryFor({ config, manifest, manifestSha256, counts, failure, started }) {
	return {
		schemaVersion: 1,
		kind: 'gen5-paired-manifest-run-summary',
		status: failure ? 'failed' : 'completed',
		upstream: { repository: referenceLock.repository, commit: referenceLock.commit },
		formatId: FORMAT_ID,
		derivation: { id: DERIVATION_ID, domain: DERIVATION_DOMAIN, pairIdDomain: PAIR_ID_DOMAIN },
		rootSeed: [...config.rootSeed],
		pairCount: config.pairCount,
		manifestSha256,
		counts: {
			totalPairs: config.pairCount,
			attemptedPairs: counts.attemptedPairs,
			completedPairs: counts.completedPairs,
			failedPairs: counts.failedPairs,
			unattemptedPairs: config.pairCount - counts.attemptedPairs,
			recordedPairs: counts.recordedPairs,
		},
		failure: failure ? { ...failure } : null,
		elapsedScope: 'manifest_runner',
		elapsedMs: elapsedMs(started),
		elapsedBoundary: 'sampled before final summary JSON serialization and write',
	};
}

function pairFailure(record, pairIndex) {
	const source = record?.failure;
	const category = PAIR_FAILURE_CATEGORIES.has(source?.category) ? source.category : 'runtime';
	return {
		stage: 'pair_execution',
		category,
		code: safeCode(source?.code, 'pair_execution_failed'),
		pairIndex,
	};
}

async function runManifest(options = {}) {
	const started = performance.now();
	let config;
	try {
		config = snapshotOptions(options);
	} catch {
		throw ownedError('invalid_configuration');
	}

	let manifest;
	let manifestText;
	let manifestSha256;
	try {
		manifest = generateManifest(config.rootSeed, config.pairCount);
		manifestText = `${JSON.stringify(manifest, null, 2)}\n`;
		manifestSha256 = crypto.createHash('sha256').update(manifestText, 'utf8').digest('hex');
	} catch {
		throw ownedError('manifest_initialization_failed');
	}

	const counts = { attemptedPairs: 0, completedPairs: 0, failedPairs: 0, recordedPairs: 0 };
	let directoryOwned = false;
	let failure = null;
	try {
		fs.mkdirSync(config.absoluteOutputDir);
		directoryOwned = true;
	} catch {
		failure = ioFailure('output_directory_create');
	}

	if (!failure) {
		try {
			fs.writeFileSync(path.join(config.absoluteOutputDir, 'manifest.json'), manifestText, { flag: 'wx' });
			fs.writeFileSync(path.join(config.absoluteOutputDir, 'results.jsonl'), '', { flag: 'wx' });
		} catch {
			failure = ioFailure('artifact_initialize');
		}
	}

	if (!failure) {
		for (const pair of manifest.pairs) {
			const pairOptions = {
				ps: config.ps,
				rootSeed: [...config.rootSeed],
				pairIndex: pair.pairIndex,
				...(config.limits === undefined ? {} : { limits: { ...config.limits } }),
				...(config.streamFactory === undefined ? {} : { streamFactory: config.streamFactory }),
			};
			counts.attemptedPairs++;
			let pairResult;
			let currentPairFailure = null;
			try {
				pairResult = await runPair(pairOptions);
			} catch (error) {
				if (ownsPairError(error)) {
					pairResult = error.pairResult;
					counts.failedPairs++;
					currentPairFailure = pairFailure(pairResult, pair.pairIndex);
				} else {
					counts.failedPairs++;
					currentPairFailure = {
						stage: 'pair_execution', category: 'runtime', code: 'pair_execution_failed', pairIndex: pair.pairIndex,
					};
				}
			}

			if (pairResult && pairResult.status === 'completed') {
				counts.completedPairs++;
			} else if (pairResult && !currentPairFailure) {
				counts.failedPairs++;
				currentPairFailure = {
					stage: 'pair_execution', category: 'runtime', code: 'invalid_pair_result', pairIndex: pair.pairIndex,
				};
			}

			if (pairResult) {
				let line;
				try {
					line = `${JSON.stringify(pairResult)}\n`;
					fs.appendFileSync(path.join(config.absoluteOutputDir, 'results.jsonl'), line, { flag: 'a' });
					counts.recordedPairs++;
				} catch {
					failure = ioFailure('result_append', pair.pairIndex);
					break;
				}
			}

			if (currentPairFailure) {
				failure = currentPairFailure;
				break;
			}
		}
	}

	let summary = summaryFor({ config, manifest, manifestSha256, counts, failure, started });
	if (directoryOwned) {
		try {
			fs.writeFileSync(path.join(config.absoluteOutputDir, 'summary.json'), `${JSON.stringify(summary, null, 2)}\n`, { flag: 'wx' });
		} catch {
			failure = failure ? { ...failure, summaryWriteFailure: 'summary_write_failed' } : ioFailure('summary_write');
			summary = summaryFor({ config, manifest, manifestSha256, counts, failure, started });
		}
	}

	if (failure) {
		const code = failure.summaryWriteFailure ? 'summary_write_failed' : safeCode(failure.code, 'manifest_run_failed');
		throw ownedError(code, summary);
	}
	return summary;
}

function parseArgs(args) {
	if (!Array.isArray(args) || args.some(argument => typeof argument !== 'string')) throw new TypeError('arguments must be strings.');
	if (args.length === 1 && ['--help', 'help'].includes(args[0])) return { help: true };
	const names = new Set(['--root-seed', '--pairs', '--output-dir']);
	const values = {};
	for (let index = 0; index < args.length; index++) {
		const argument = args[index];
		const separator = argument.indexOf('=');
		const name = separator < 0 ? argument : argument.slice(0, separator);
		if (!names.has(name)) throw new TypeError('unknown argument.');
		if (Object.hasOwn(values, name)) throw new TypeError('duplicate argument.');
		const value = separator < 0 ? args[++index] : argument.slice(separator + 1);
		if (typeof value !== 'string' || value.length === 0 || value.startsWith('--')) throw new TypeError('option requires a value.');
		values[name] = value;
	}
	for (const name of names) if (!Object.hasOwn(values, name)) throw new TypeError('required argument missing.');
	if (!/^[1-9]\d*$/.test(values['--pairs'])) throw new TypeError('pairs must be a positive decimal integer.');
	const pairCount = Number(values['--pairs']);
	validatePairCount(pairCount);
	return {
		rootSeed: parseRootSeed(values['--root-seed']),
		pairCount,
		outputDir: values['--output-dir'],
	};
}

async function main(args = process.argv.slice(2)) {
	if (Array.isArray(args) && args.length === 1 && ['--help', 'help'].includes(args[0])) {
		process.stdout.write('Usage: node environment/run-manifest.cjs --root-seed 1,2,3,4 --pairs N --output-dir NEW_DIR\n');
		return;
	}
	let options;
	try {
		options = parseArgs(args);
	} catch {
		console.error('Manifest runner failed (invalid_cli_configuration).');
		process.exitCode = 1;
		return;
	}

	let ps;
	try {
		const oracle = require('../reference/oracle.cjs');
		oracle.assertReady();
		ps = require('../reference/vendor/pokemon-showdown/dist/sim');
	} catch {
		console.error('Manifest runner failed (pinned_reference_unavailable).');
		process.exitCode = 1;
		return;
	}

	try {
		const summary = await runManifest({ ...options, ps });
		process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
		process.exitCode = 0;
	} catch (error) {
		const info = ERROR_INFO.get(error);
		const code = safeCode(info?.code, 'manifest_run_failed');
		if (code === 'summary_write_failed') {
			console.error('Manifest runner failed (summary_write_failed); summary.json could not be confirmed complete.');
		} else {
			console.error(`Manifest runner failed (${code}).`);
		}
		process.exitCode = 1;
	}
}

if (require.main === module) {
	main().catch(() => {
		console.error('Manifest runner failed (cli_internal_failure).');
		process.exitCode = 1;
	});
}

module.exports = { main, parseArgs, runManifest };
