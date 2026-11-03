'use strict';

const referenceLock = require('../reference-lock.json');
const { DEFAULT_LIMITS, runBattle } = require('./battle-runner.cjs');
const { MAX_PAIRS, generatePair, validateSeed } = require('./paired-seeds.cjs');
const { prepareTeam } = require('./prepared-teams.cjs');
const { randomPolicy } = require('./random-policy.cjs');
const { POLICY_ID: MOVE_FIRST_ID, moveFirstPolicy } = require('./move-first-policy.cjs');

const FORMAT_ID = 'gen5randombattle';
const KIND = 'gen5-named-baseline-paired-matchup';
const ERROR_MESSAGE = 'Baseline paired matchup failed (baseline_matchup_failed).';
const OPTION_KEYS = new Set(['rootSeed', 'pairIndex', 'policyA', 'policyB', 'ps', 'limits', 'streamFactory']);
const REQUIRED_OPTION_KEYS = ['rootSeed', 'pairIndex', 'policyA', 'policyB', 'ps'];
const SEED_KEYS = ['teamP1', 'teamP2', 'battle', 'policyP1', 'policyP2'];
const LIMIT_KEYS = Object.keys(DEFAULT_LIMITS);
const POLICY_ID_SET = new Set(['legal-random-v1', MOVE_FIRST_ID]);
const POLICY_IMPLEMENTATIONS = new Map([
	['legal-random-v1', randomPolicy],
	[MOVE_FIRST_ID, moveFirstPolicy],
]);
const RESULT_RECORD_KEYS = new Set([
	'schemaVersion', 'kind', 'upstreamCommit', 'formatId', 'seeds', 'limits', 'status', 'outcome',
	'elapsedMs', 'elapsedScope', 'counters', 'failure', 'cleanupFailure', 'teamSource',
]);
const SUMMARY_KEYS = new Set(['schemaVersion', 'upstreamCommit', 'formatId', 'seeds', 'outcome', 'packets', 'decisions', 'actions']);
const RESULT_KEYS = new Set(['summary', 'transcript', 'resultRecord']);

function fail() {
	throw new Error(ERROR_MESSAGE);
}

function readDataRecord(value, allowedKeys, requiredKeys = []) {
	if (!value || typeof value !== 'object' || Array.isArray(value)) fail();
	const prototype = Object.getPrototypeOf(value);
	if (prototype !== Object.prototype && prototype !== null) fail();
	const values = Object.create(null);
	for (const key of Reflect.ownKeys(value)) {
		if (typeof key !== 'string' || !allowedKeys.has(key)) fail();
		const descriptor = Object.getOwnPropertyDescriptor(value, key);
		if (!descriptor || !Object.hasOwn(descriptor, 'value') || !descriptor.enumerable) fail();
		values[key] = descriptor.value;
	}
	for (const key of requiredKeys) {
		if (!Object.hasOwn(values, key)) fail();
	}
	return values;
}

function copySeedArray(value) {
	if (!Array.isArray(value)) fail();
	const keys = Reflect.ownKeys(value);
	if (keys.length !== 5 || !keys.includes('length')) fail();
	const lengthDescriptor = Object.getOwnPropertyDescriptor(value, 'length');
	if (!lengthDescriptor || lengthDescriptor.value !== 4) fail();
	const copy = [];
	for (let index = 0; index < 4; index++) {
		const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
		if (!descriptor || !Object.hasOwn(descriptor, 'value') || !descriptor.enumerable ||
			!Number.isInteger(descriptor.value) || descriptor.value < 0 || descriptor.value > 65_535) fail();
		copy.push(descriptor.value);
	}
	return copy;
}

function copyRootSeed(value) {
	const seed = copySeedArray(value);
	return validateSeed(seed);
}

function snapshotLimits(value) {
	if (value === undefined) return {};
	const allowed = new Set(LIMIT_KEYS);
	const source = readDataRecord(value, allowed);
	const copy = {};
	for (const key of Object.keys(source)) {
		const limit = source[key];
		if (typeof limit !== 'number' || !Number.isFinite(limit) || limit <= 0) fail();
		copy[key] = limit;
	}
	return copy;
}

function snapshotOptions(options) {
	const supplied = readDataRecord(options, OPTION_KEYS, REQUIRED_OPTION_KEYS);
	const rootSeed = copyRootSeed(supplied.rootSeed);
	if (!Number.isSafeInteger(supplied.pairIndex) || supplied.pairIndex < 0 || supplied.pairIndex >= MAX_PAIRS) fail();
	if (typeof supplied.policyA !== 'string' || !POLICY_ID_SET.has(supplied.policyA) ||
		typeof supplied.policyB !== 'string' || !POLICY_ID_SET.has(supplied.policyB)) fail();
	if (!supplied.ps || typeof supplied.ps !== 'object' || Array.isArray(supplied.ps)) fail();
	const battleStream = supplied.ps.BattleStream;
	const teams = supplied.ps.Teams;
	const dex = supplied.ps.Dex;
	if (typeof battleStream !== 'function' || !teams || typeof teams !== 'object' ||
		typeof teams.generate !== 'function' || typeof teams.pack !== 'function' ||
		!dex?.formats || typeof dex.formats.get !== 'function' || dex.formats.get(FORMAT_ID)?.exists !== true) fail();
	const limits = snapshotLimits(supplied.limits);
	if (supplied.streamFactory !== undefined && typeof supplied.streamFactory !== 'function') fail();
	return {
		rootSeed,
		pairIndex: supplied.pairIndex,
		policyA: supplied.policyA,
		policyB: supplied.policyB,
		ps: supplied.ps,
		limits,
		streamFactory: supplied.streamFactory,
	};
}

function copySeedConfig(value) {
	const source = readDataRecord(value, new Set(SEED_KEYS), SEED_KEYS);
	return Object.fromEntries(SEED_KEYS.map(key => [key, copySeedArray(source[key])]));
}

function copyOutcome(value) {
	const source = readDataRecord(value, new Set(['result', 'winner', 'turns']), ['result', 'winner', 'turns']);
	if (!['win', 'tie'].includes(source.result) || !Number.isSafeInteger(source.turns) || source.turns < 0) fail();
	if (source.result === 'tie' ? source.winner !== null : !['p1', 'p2'].includes(source.winner)) fail();
	return { result: source.result, winner: source.winner, turns: source.turns };
}

function copyJsonData(value, depth = 0, ancestors = new Set()) {
	if (depth > 12) fail();
	if (value === null || typeof value === 'string' || typeof value === 'boolean') return value;
	if (typeof value === 'number' && Number.isFinite(value)) return value;
	if (!value || typeof value !== 'object' || ancestors.has(value)) fail();
	ancestors.add(value);
	try {
		if (Array.isArray(value)) {
			const keys = Reflect.ownKeys(value);
			const lengthDescriptor = Object.getOwnPropertyDescriptor(value, 'length');
			if (!lengthDescriptor || !Number.isSafeInteger(lengthDescriptor.value) ||
				lengthDescriptor.value < 0 || keys.length !== lengthDescriptor.value + 1 || !keys.includes('length')) fail();
			const output = [];
			for (let index = 0; index < lengthDescriptor.value; index++) {
				const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
				if (!descriptor || !Object.hasOwn(descriptor, 'value') || !descriptor.enumerable) fail();
				output.push(copyJsonData(descriptor.value, depth + 1, ancestors));
			}
			return output;
		}
		const prototype = Object.getPrototypeOf(value);
		if (prototype !== Object.prototype && prototype !== null) fail();
		const output = Object.create(null);
		for (const key of Reflect.ownKeys(value)) {
			if (typeof key !== 'string') fail();
			const descriptor = Object.getOwnPropertyDescriptor(value, key);
			if (!descriptor || !Object.hasOwn(descriptor, 'value') || !descriptor.enumerable) fail();
			output[key] = copyJsonData(descriptor.value, depth + 1, ancestors);
		}
		return output;
	} finally {
		ancestors.delete(value);
	}
}

function copyLimitsRecord(value) {
	const source = readDataRecord(value, new Set(LIMIT_KEYS), LIMIT_KEYS);
	const output = {};
	for (const key of LIMIT_KEYS) {
		if (typeof source[key] !== 'number' || !Number.isFinite(source[key]) || source[key] <= 0) fail();
		output[key] = source[key];
	}
	return output;
}

function sameJson(left, right) {
	if (left === right) return true;
	if (left === null || right === null || typeof left !== 'object' || typeof right !== 'object') return false;
	if (Array.isArray(left) || Array.isArray(right)) {
		return Array.isArray(left) && Array.isArray(right) && left.length === right.length && left.every((value, index) => sameJson(value, right[index]));
	}
	const leftKeys = Object.keys(left).sort();
	const rightKeys = Object.keys(right).sort();
	return leftKeys.length === rightKeys.length && leftKeys.every((key, index) =>
		key === rightKeys[index] && sameJson(left[key], right[key]));
}

function copyCompletedResult(result, expectedSeeds, expectedLimits) {
	const envelope = readDataRecord(result, RESULT_KEYS, ['summary', 'transcript', 'resultRecord']);
	const summary = readDataRecord(envelope.summary, SUMMARY_KEYS, [...SUMMARY_KEYS]);
	if (summary.schemaVersion !== 1 || summary.upstreamCommit !== referenceLock.commit || summary.formatId !== FORMAT_ID ||
		!Number.isSafeInteger(summary.packets) || summary.packets < 0 ||
		!Number.isSafeInteger(summary.decisions) || summary.decisions < 0) fail();
	const summarySeeds = copySeedConfig(summary.seeds);
	if (!sameJson(summarySeeds, expectedSeeds)) fail();
	const summaryOutcome = copyOutcome(summary.outcome);

	const source = readDataRecord(envelope.resultRecord, RESULT_RECORD_KEYS, [...RESULT_RECORD_KEYS]);
	if (source.schemaVersion !== 1 || source.kind !== 'battle-run-result' ||
		source.upstreamCommit !== referenceLock.commit || source.formatId !== FORMAT_ID || source.status !== 'completed' ||
		source.failure !== null || source.cleanupFailure !== null || source.teamSource !== 'prepared' ||
		source.elapsedScope !== 'battle_runner' || typeof source.elapsedMs !== 'number' ||
		!Number.isFinite(source.elapsedMs) || source.elapsedMs < 0) fail();
	const seeds = copySeedConfig(source.seeds);
	if (!sameJson(seeds, expectedSeeds)) fail();
	const limits = copyLimitsRecord(source.limits);
	if (!sameJson(limits, expectedLimits)) fail();
	const outcome = copyOutcome(source.outcome);
	if (!sameJson(outcome, summaryOutcome)) fail();
	const counters = copyJsonData(source.counters);
	return {
		schemaVersion: 1,
		kind: 'battle-run-result',
		upstreamCommit: referenceLock.commit,
		formatId: FORMAT_ID,
		seeds,
		limits,
		status: 'completed',
		outcome,
		elapsedMs: source.elapsedMs,
		elapsedScope: 'battle_runner',
		counters,
		failure: null,
		cleanupFailure: null,
		teamSource: 'prepared',
	};
}

function copyCanonicalRun(run, policyIdsByRole) {
	const roles = { p1: run.policySeats.p1, p2: run.policySeats.p2 };
	if (!['policyA', 'policyB'].includes(roles.p1) || !['policyA', 'policyB'].includes(roles.p2)) fail();
	const idsBySeat = {
		p1: policyIdsByRole[roles.p1],
		p2: policyIdsByRole[roles.p2],
	};
	const seeds = copySeedConfig(run.seeds);
	return {
		runId: run.runId,
		seats: { p1Team: run.seats.p1Team, p2Team: run.seats.p2Team },
		policySeats: roles,
		seeds,
		policyIdsBySeat: idsBySeat,
	};
}

function createFreshStreamFactory(streamFactory) {
	if (streamFactory === undefined) return undefined;
	const streams = new Set();
	return () => {
		const stream = streamFactory();
		if (!stream || (typeof stream !== 'object' && typeof stream !== 'function') || streams.has(stream)) fail();
		streams.add(stream);
		return stream;
	};
}

async function runBaselineMatchup(options = {}) {
	try {
		const config = snapshotOptions(options);
		const effectiveLimits = { ...DEFAULT_LIMITS, ...config.limits };
		for (const value of Object.values(effectiveLimits)) {
			if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) fail();
		}
		const pair = generatePair(config.rootSeed, config.pairIndex);
		const teamA = prepareTeam(config.ps, pair.teams.A.seed);
		const teamB = prepareTeam(config.ps, pair.teams.B.seed);
		const policyIdsByRole = { policyA: config.policyA, policyB: config.policyB };
		const freshStreamFactory = createFreshStreamFactory(config.streamFactory);
		const runs = [];

		for (const generatedRun of pair.runs) {
			const canonicalRun = copyCanonicalRun(generatedRun, policyIdsByRole);
			const preparedTeams = {
				p1: generatedRun.seats.p1Team === 'A' ? teamA : teamB,
				p2: generatedRun.seats.p2Team === 'A' ? teamA : teamB,
			};
			const policies = {
				p1: POLICY_IMPLEMENTATIONS.get(canonicalRun.policyIdsBySeat.p1),
				p2: POLICY_IMPLEMENTATIONS.get(canonicalRun.policyIdsBySeat.p2),
			};
			if (!policies.p1 || !policies.p2) fail();
			const result = await runBattle({
				ps: config.ps,
				seeds: { ...canonicalRun.seeds },
				preparedTeams,
				policies,
				limits: { ...config.limits },
				...(freshStreamFactory === undefined ? {} : { streamFactory: freshStreamFactory }),
			});
			const resultRecord = copyCompletedResult(result, canonicalRun.seeds, effectiveLimits);
			runs.push({ ...canonicalRun, status: 'completed', resultRecord });
		}

		return {
			schemaVersion: 1,
			kind: KIND,
			status: 'completed',
			formatId: FORMAT_ID,
			upstream: { repository: referenceLock.repository, commit: referenceLock.commit },
			rootSeed: [...config.rootSeed],
			pairIndex: config.pairIndex,
			pairId: pair.pairId,
			policiesByRole: { ...policyIdsByRole },
			limits: { ...effectiveLimits },
			policySeeds: {
				policyA: copySeedArray(pair.policySeeds.policyA),
				policyB: copySeedArray(pair.policySeeds.policyB),
			},
			teams: {
				A: { seed: copySeedArray(teamA.seed), packedTeamSha256: teamA.packedTeamSha256 },
				B: { seed: copySeedArray(teamB.seed), packedTeamSha256: teamB.packedTeamSha256 },
			},
			runs,
		};
	} catch {
		throw new Error(ERROR_MESSAGE);
	}
}

module.exports = { KIND, runBaselineMatchup };
