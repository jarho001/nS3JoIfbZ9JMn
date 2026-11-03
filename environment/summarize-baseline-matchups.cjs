'use strict';

const referenceLock = require('../reference-lock.json');
const { DEFAULT_LIMITS } = require('./battle-runner.cjs');
const { MAX_PAIRS, generatePair, validateSeed } = require('./paired-seeds.cjs');
const { POLICY_ID: MOVE_FIRST_ID } = require('./move-first-policy.cjs');

const KIND = 'gen5-named-baseline-outcome-report';
const MATCHUP_KIND = 'gen5-named-baseline-paired-matchup';
const FORMAT_ID = 'gen5randombattle';
const ERROR_MESSAGE = 'Invalid named baseline outcome inputs (named_baseline_report_invalid).';
const LIMIT_KEYS = Object.keys(DEFAULT_LIMITS);
const SEED_KEYS = ['teamP1', 'teamP2', 'battle', 'policyP1', 'policyP2'];
const POLICY_IDS = new Set(['legal-random-v1', MOVE_FIRST_ID]);
const CONFIG_KEYS = new Set(['rootSeed', 'pairCount', 'policyA', 'policyB', 'limits', 'matchups']);
const MATCHUP_KEYS = new Set([
	'schemaVersion', 'kind', 'status', 'formatId', 'upstream', 'rootSeed', 'pairIndex', 'pairId',
	'policiesByRole', 'limits', 'policySeeds', 'teams', 'runs',
]);
const RUN_KEYS = new Set(['runId', 'seats', 'policySeats', 'seeds', 'policyIdsBySeat', 'status', 'resultRecord']);
const RESULT_KEYS = new Set([
	'schemaVersion', 'kind', 'upstreamCommit', 'formatId', 'seeds', 'limits', 'status', 'outcome',
	'elapsedMs', 'elapsedScope', 'counters', 'failure', 'cleanupFailure', 'teamSource',
]);

function fail() {
	throw new Error(ERROR_MESSAGE);
}

function readRecord(value, allowedKeys, requiredKeys = []) {
	if (!value || typeof value !== 'object' || Array.isArray(value)) fail();
	const prototype = Object.getPrototypeOf(value);
	if (prototype !== Object.prototype && prototype !== null) fail();
	const result = Object.create(null);
	for (const key of Reflect.ownKeys(value)) {
		if (typeof key !== 'string' || !allowedKeys.has(key)) fail();
		const descriptor = Object.getOwnPropertyDescriptor(value, key);
		if (!descriptor || !descriptor.enumerable || !Object.hasOwn(descriptor, 'value')) fail();
		result[key] = descriptor.value;
	}
	for (const key of requiredKeys) {
		if (!Object.hasOwn(result, key)) fail();
	}
	return result;
}

function copyArray(value, transform = item => item) {
	if (!Array.isArray(value) || Object.getPrototypeOf(value) !== Array.prototype) fail();
	const length = Object.getOwnPropertyDescriptor(value, 'length');
	if (!length || !Number.isSafeInteger(length.value) || length.value < 0) fail();
	const keys = Reflect.ownKeys(value);
	if (keys.length !== length.value + 1 || !keys.includes('length')) fail();
	const result = [];
	for (let index = 0; index < length.value; index++) {
		const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
		if (!descriptor || !descriptor.enumerable || !Object.hasOwn(descriptor, 'value')) fail();
		result.push(transform(descriptor.value, index));
	}
	return result;
}

function copyJson(value, ancestors = new Set(), depth = 0) {
	if (depth > 16) fail();
	if (value === null || typeof value === 'string' || typeof value === 'boolean') return value;
	if (typeof value === 'number' && Number.isFinite(value)) return value;
	if (!value || typeof value !== 'object' || ancestors.has(value)) fail();
	ancestors.add(value);
	try {
		if (Array.isArray(value)) return copyArray(value, item => copyJson(item, ancestors, depth + 1));
		const source = readRecord(value, new Set(Reflect.ownKeys(value).filter(key => typeof key === 'string')));
		const result = Object.create(null);
		for (const key of Object.keys(source)) result[key] = copyJson(source[key], ancestors, depth + 1);
		return result;
	} finally {
		ancestors.delete(value);
	}
}

function canonicalJson(value) {
	if (Array.isArray(value)) return `[${value.map(item => canonicalJson(item)).join(',')}]`;
	if (value && typeof value === 'object') {
		const keys = Object.keys(value).sort();
		return `{${keys.map(key => `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(',')}}`;
	}
	return JSON.stringify(value);
}

function sameJson(left, right) {
	return canonicalJson(left) === canonicalJson(right);
}

function copySeed(value) {
	if (!Array.isArray(value)) fail();
	const result = copyArray(value, item => item);
	if (result.length !== 4) fail();
	try {
		return validateSeed(result);
	} catch {
		fail();
	}
}

function copySeedRecord(value) {
	const source = readRecord(value, new Set(SEED_KEYS), SEED_KEYS);
	return Object.fromEntries(SEED_KEYS.map(key => [key, copySeed(source[key])]));
}

function copyLimits(value, requireComplete) {
	const source = readRecord(value, new Set(LIMIT_KEYS), requireComplete ? LIMIT_KEYS : []);
	const result = {};
	for (const key of LIMIT_KEYS) {
		if (!Object.hasOwn(source, key)) {
			if (requireComplete) fail();
			continue;
		}
		const limit = source[key];
		if (typeof limit !== 'number' || !Number.isFinite(limit) || limit <= 0) fail();
		result[key] = limit;
	}
	return result;
}

function copyOutcome(value) {
	const source = readRecord(value, new Set(['result', 'winner', 'turns']), ['result', 'winner', 'turns']);
	if (!['win', 'tie'].includes(source.result) || !Number.isSafeInteger(source.turns) || source.turns < 0) fail();
	if (source.result === 'tie' ? source.winner !== null : !['p1', 'p2'].includes(source.winner)) fail();
	return { result: source.result, winner: source.winner, turns: source.turns };
}

function copyInput(config) {
	const source = readRecord(config, CONFIG_KEYS, [...CONFIG_KEYS]);
	const rootSeed = copySeed(source.rootSeed);
	if (!Number.isSafeInteger(source.pairCount) || source.pairCount < 1 || source.pairCount > MAX_PAIRS) fail();
	if (typeof source.policyA !== 'string' || !POLICY_IDS.has(source.policyA) ||
		typeof source.policyB !== 'string' || !POLICY_IDS.has(source.policyB)) fail();
	const limits = copyLimits(source.limits, true);
	if (!Array.isArray(source.matchups)) fail();
	const matchups = copyArray(source.matchups, item => copyJson(item));
	if (matchups.length > source.pairCount) fail();
	return {
		rootSeed,
		pairCount: source.pairCount,
		policyA: source.policyA,
		policyB: source.policyB,
		limits,
		matchups,
	};
}

function assertSame(actual, expected) {
	if (!sameJson(actual, expected)) fail();
}

function validateMatchup(value, config, pairIndex) {
	const source = readRecord(value, MATCHUP_KEYS, [...MATCHUP_KEYS]);
	if (source.schemaVersion !== 1 || source.kind !== MATCHUP_KIND || source.status !== 'completed' ||
		source.formatId !== FORMAT_ID || source.pairIndex !== pairIndex) fail();
	assertSame(readRecord(source.upstream, new Set(['repository', 'commit']), ['repository', 'commit']), {
		repository: referenceLock.repository,
		commit: referenceLock.commit,
	});
	assertSame(copySeed(source.rootSeed), config.rootSeed);
	const pair = generatePair(config.rootSeed, pairIndex);
	if (source.pairId !== pair.pairId) fail();
	assertSame(readRecord(source.policiesByRole, new Set(['policyA', 'policyB']), ['policyA', 'policyB']), {
		policyA: config.policyA,
		policyB: config.policyB,
	});
	assertSame(copyLimits(source.limits, true), config.limits);
	const policySeeds = readRecord(source.policySeeds, new Set(['policyA', 'policyB']), ['policyA', 'policyB']);
	assertSame({ policyA: copySeed(policySeeds.policyA), policyB: copySeed(policySeeds.policyB) }, pair.policySeeds);
	const teams = readRecord(source.teams, new Set(['A', 'B']), ['A', 'B']);
	for (const role of ['A', 'B']) {
		const team = readRecord(teams[role], new Set(['seed', 'packedTeamSha256']), ['seed', 'packedTeamSha256']);
		assertSame(copySeed(team.seed), pair.teams[role].seed);
		if (typeof team.packedTeamSha256 !== 'string' || !/^[0-9a-f]{64}$/.test(team.packedTeamSha256)) fail();
	}
	const runs = copyArray(source.runs, (run, index) => validateRun(run, config, pair, index));
	if (runs.length !== 2) fail();
	return { pairIndex, pairId: pair.pairId, runs };
}

function validateRun(value, config, pair, runIndex) {
	const source = readRecord(value, RUN_KEYS, [...RUN_KEYS]);
	const expected = pair.runs[runIndex];
	if (!expected || source.runId !== expected.runId || source.status !== 'completed') fail();
	assertSame(readRecord(source.seats, new Set(['p1Team', 'p2Team']), ['p1Team', 'p2Team']), expected.seats);
	assertSame(readRecord(source.policySeats, new Set(['p1', 'p2']), ['p1', 'p2']), expected.policySeats);
	assertSame(copySeedRecord(source.seeds), expected.seeds);
	assertSame(readRecord(source.policyIdsBySeat, new Set(['p1', 'p2']), ['p1', 'p2']), {
		p1: expected.policySeats.p1 === 'policyA' ? config.policyA : config.policyB,
		p2: expected.policySeats.p2 === 'policyA' ? config.policyA : config.policyB,
	});
	const result = readRecord(source.resultRecord, RESULT_KEYS, [...RESULT_KEYS]);
	if (result.schemaVersion !== 1 || result.kind !== 'battle-run-result' ||
		result.upstreamCommit !== referenceLock.commit || result.formatId !== FORMAT_ID ||
		result.status !== 'completed' || result.failure !== null || result.cleanupFailure !== null ||
		result.teamSource !== 'prepared' || result.elapsedScope !== 'battle_runner' ||
		typeof result.elapsedMs !== 'number' || !Number.isFinite(result.elapsedMs) || result.elapsedMs < 0) fail();
	assertSame(copySeedRecord(result.seeds), expected.seeds);
	assertSame(copyLimits(result.limits, true), config.limits);
	const outcome = copyOutcome(result.outcome);
	copyJson(result.counters);
	return { policySeats: { ...expected.policySeats }, outcome };
}

function newTally() {
	return { wins: 0, losses: 0, ties: 0 };
}

function addOutcome(tally, outcome, seat) {
	if (outcome.result === 'tie') tally.ties++;
	else if (outcome.winner === seat) tally.wins++;
	else tally.losses++;
}

function summarizeBaselineMatchups(options) {
	try {
		const config = copyInput(options);
		const policiesByRole = { policyA: config.policyA, policyB: config.policyB };
		const bySeat = { p1: newTally(), p2: newTally() };
		const byPolicyRole = {
			policyA: { policyId: config.policyA, overall: newTally(), bySeat: { p1: newTally(), p2: newTally() } },
			policyB: { policyId: config.policyB, overall: newTally(), bySeat: { p1: newTally(), p2: newTally() } },
		};
		const overall = { winsBySeat: { p1: 0, p2: 0 }, ties: 0 };
		for (let pairIndex = 0; pairIndex < config.matchups.length; pairIndex++) {
			const pair = validateMatchup(config.matchups[pairIndex], config, pairIndex);
			for (const run of pair.runs) {
				const { policySeats, outcome } = run;
				addOutcome(bySeat.p1, outcome, 'p1');
				addOutcome(bySeat.p2, outcome, 'p2');
				for (const seat of ['p1', 'p2']) addOutcome(byPolicyRole[policySeats[seat]].bySeat[seat], outcome, seat);
				for (const role of ['policyA', 'policyB']) {
					if (policySeats.p1 === role) addOutcome(byPolicyRole[role].overall, outcome, 'p1');
					else addOutcome(byPolicyRole[role].overall, outcome, 'p2');
				}
				if (outcome.result === 'tie') overall.ties++;
				else overall.winsBySeat[outcome.winner]++;
			}
		}
		const recordedPairs = config.matchups.length;
		const recordedBattles = recordedPairs * 2;
		return {
			schemaVersion: 1,
			kind: KIND,
			configuration: {
				formatId: FORMAT_ID,
				upstream: { repository: referenceLock.repository, commit: referenceLock.commit },
				rootSeed: [...config.rootSeed],
				pairCount: config.pairCount,
				policiesByRole,
				limits: { ...config.limits },
			},
			coverage: {
				configuredPairs: config.pairCount,
				recordedPairs,
				unrecordedPairs: config.pairCount - recordedPairs,
				recordedBattles,
				unrecordedRunSlots: (config.pairCount - recordedPairs) * 2,
			},
			outcomes: {
				completedBattles: recordedBattles,
				bySeat,
				byPolicyRole,
				overall,
			},
		};
	} catch {
		throw new Error(ERROR_MESSAGE);
	}
}

module.exports = { KIND, summarizeBaselineMatchups };
