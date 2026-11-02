#!/usr/bin/env node
'use strict';

const crypto = require('node:crypto');
const referenceLock = require('../reference-lock.json');

const FORMAT_ID = 'gen5randombattle';
const DERIVATION_ID = 'sha256-json-domain-v1';
const DERIVATION_DOMAIN = 'gen5-paired-seeds-v1';
const PAIR_ID_DOMAIN = 'gen5-paired-seeds-v1:pair-id';
const STREAMS = ['teamA', 'teamB', 'battle', 'policyA', 'policyB'];
const MAX_PAIRS = 100_000;

function validateSeed(seed, label = 'root seed') {
	if (!Array.isArray(seed) || seed.length !== 4) {
		throw new TypeError(`${label} must contain exactly four integer limbs in the range 0..65535.`);
	}
	for (let index = 0; index < 4; index++) {
		const limb = seed[index];
		if (!Number.isInteger(limb) || limb < 0 || limb > 65_535) {
			throw new TypeError(`${label} must contain exactly four integer limbs in the range 0..65535.`);
		}
	}
	return [...seed];
}

function validatePairCount(pairCount) {
	if (!Number.isSafeInteger(pairCount) || pairCount < 1 || pairCount > MAX_PAIRS) {
		throw new RangeError(`pair count must be an integer in the range 1..${MAX_PAIRS}.`);
	}
	return pairCount;
}

function sha256Tuple(tuple) {
	return crypto.createHash('sha256').update(JSON.stringify(tuple), 'utf8').digest();
}

function deriveSeed(rootSeed, pairIndex, streamName) {
	const digest = sha256Tuple([DERIVATION_DOMAIN, rootSeed, pairIndex, streamName]);
	return Array.from({ length: 4 }, (_, index) => digest.readUInt16BE(index * 2));
}

function derivePairId(rootSeed, pairIndex) {
	const digest = sha256Tuple([PAIR_ID_DOMAIN, rootSeed, pairIndex]).toString('hex');
	return `pair-${String(pairIndex + 1).padStart(6, '0')}-${digest.slice(0, 16)}`;
}

function makeRun(pairId, suffix, p1TeamIdentity, p2TeamIdentity, p1TeamSeed, p2TeamSeed, battle, p1Policy, p2Policy, p1PolicyIdentity, p2PolicyIdentity) {
	return {
		runId: `${pairId}-${suffix}`,
		seats: { p1Team: p1TeamIdentity, p2Team: p2TeamIdentity },
		policySeats: { p1: p1PolicyIdentity, p2: p2PolicyIdentity },
		seeds: {
			teamP1: [...p1TeamSeed],
			teamP2: [...p2TeamSeed],
			battle: [...battle],
			policyP1: [...p1Policy],
			policyP2: [...p2Policy],
		},
	};
}

function makePair(rootSeed, pairIndex) {
	const teams = {
		A: { seed: deriveSeed(rootSeed, pairIndex, 'teamA') },
		B: { seed: deriveSeed(rootSeed, pairIndex, 'teamB') },
	};
	const policies = {
		A: deriveSeed(rootSeed, pairIndex, 'policyA'),
		B: deriveSeed(rootSeed, pairIndex, 'policyB'),
	};
	const battle = deriveSeed(rootSeed, pairIndex, 'battle');
	const pairId = derivePairId(rootSeed, pairIndex);
	return {
		pairId,
		pairIndex,
		teams,
		policySeeds: { policyA: [...policies.A], policyB: [...policies.B] },
		runs: [
			makeRun(pairId, 'A-p1', 'A', 'B', teams.A.seed, teams.B.seed, battle, policies.A, policies.B, 'policyA', 'policyB'),
			makeRun(pairId, 'B-p1', 'B', 'A', teams.B.seed, teams.A.seed, battle, policies.B, policies.A, 'policyB', 'policyA'),
		],
	};
}

function generateManifest(rootSeed, pairCount) {
	const root = validateSeed(rootSeed);
	const count = validatePairCount(pairCount);
	return {
		schemaVersion: 1,
		kind: 'gen5-paired-seed-manifest',
		derivation: {
			id: DERIVATION_ID,
			domain: DERIVATION_DOMAIN,
			pairIdDomain: PAIR_ID_DOMAIN,
			encoding: 'UTF-8 JSON.stringify array [domain, rootSeed, pairIndex, streamName]',
			seedBytes: 'SHA-256 digest bytes 0..7 interpreted as four unsigned big-endian 16-bit limbs',
			streams: [...STREAMS],
			pairIndexing: 'zero-based; IDs and seeds depend on root and index, not requested pair count',
			pairId: 'first 16 lowercase hex characters of SHA-256 over UTF-8 JSON.stringify([pairIdDomain, rootSeed, pairIndex]), prefixed by pair- and a one-based six-digit index',
		},
		upstream: {
			repository: referenceLock.repository,
			commit: referenceLock.commit,
		},
		formatId: FORMAT_ID,
		rootSeed: root,
		pairCount: count,
		purpose: 'Trusted engineering seed manifest; contains reproducibility seeds and must not be used as policy input.',
		pairs: Array.from({ length: count }, (_, pairIndex) => makePair(root, pairIndex)),
	};
}

function parseRootSeed(value) {
	if (typeof value !== 'string' || !/^(?:0|[1-9]\d*)(?:,(?:0|[1-9]\d*)){3}$/.test(value)) {
		throw new TypeError('root seed must be four comma-separated decimal integers, for example 1,2,3,4.');
	}
	return validateSeed(value.split(',').map(Number));
}

function parseArgs(args) {
	if (!Array.isArray(args) || args.some(argument => typeof argument !== 'string')) {
		throw new TypeError('arguments must be strings.');
	}
	if (args.length === 1 && (args[0] === '--help' || args[0] === 'help')) return { help: true };
	const values = {};
	const names = new Set(['--root-seed', '--pairs']);
	for (let index = 0; index < args.length; index++) {
		const argument = args[index];
		const separator = argument.indexOf('=');
		const name = separator < 0 ? argument : argument.slice(0, separator);
		if (!names.has(name)) throw new TypeError(`unknown argument: ${argument}`);
		if (Object.hasOwn(values, name)) throw new TypeError(`duplicate argument: ${name}`);
		let value;
		if (separator >= 0) {
			value = argument.slice(separator + 1);
		} else {
			value = args[++index];
		}
		if (typeof value !== 'string' || value.length === 0 || value.startsWith('--')) {
			throw new TypeError(`${name} requires a value.`);
		}
		values[name] = value;
	}
	if (!Object.hasOwn(values, '--root-seed')) throw new TypeError('required argument missing: --root-seed');
	if (!Object.hasOwn(values, '--pairs')) throw new TypeError('required argument missing: --pairs');
	if (!/^[1-9]\d*$/.test(values['--pairs'])) throw new TypeError('--pairs must be a positive decimal integer.');
	const pairCount = Number(values['--pairs']);
	validatePairCount(pairCount);
	return { rootSeed: parseRootSeed(values['--root-seed']), pairCount };
}

function main(args = process.argv.slice(2)) {
	if (args.length === 1 && (args[0] === '--help' || args[0] === 'help')) {
		process.stdout.write('Usage: node environment/paired-seeds.cjs --root-seed 1,2,3,4 --pairs 10\n');
		return;
	}
	const options = parseArgs(args);
	process.stdout.write(`${JSON.stringify(generateManifest(options.rootSeed, options.pairCount), null, 2)}\n`);
}

if (require.main === module) {
	try {
		main();
	} catch (error) {
		console.error(`Paired seed manifest failed: ${error.message}`);
		process.exitCode = 1;
	}
}

module.exports = {
	DERIVATION_DOMAIN,
	DERIVATION_ID,
	MAX_PAIRS,
	generateManifest,
	main,
	parseArgs,
	parseRootSeed,
	validatePairCount,
	validateSeed,
};
