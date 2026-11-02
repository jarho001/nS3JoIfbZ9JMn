'use strict';

const fs = require('node:fs');
const path = require('node:path');
const { performance } = require('node:perf_hooks');
const { projectPackets } = require('./protocol.cjs');
const { candidateActions, createPolicyRng, deepFreeze, policyInput, randomPolicy } = require('./random-policy.cjs');

const FORMAT_ID = 'gen5randombattle';
const UPSTREAM_COMMIT = require('../reference-lock.json').commit;
const SEED_KEYS = ['teamP1', 'teamP2', 'battle', 'policyP1', 'policyP2'];
const LIMIT_KEYS = ['timeoutMs', 'maxPackets', 'maxDecisions', 'maxRetriesPerRequest'];
const BRANDED_FAILURES = new WeakSet();
const DEFAULT_SEEDS = Object.freeze({
	teamP1: [1, 2, 3, 4],
	teamP2: [5, 6, 7, 8],
	battle: [9, 10, 11, 12],
	policyP1: [13, 14, 15, 16],
	policyP2: [17, 18, 19, 20],
});
const DEFAULT_LIMITS = Object.freeze({
	timeoutMs: 60_000,
	maxPackets: 20_000,
	maxDecisions: 4_000,
	maxRetriesPerRequest: 5,
});
const PLAYERS = ['p1', 'p2'];

function safeRecordSeeds(seeds) {
	const source = seeds === undefined ? DEFAULT_SEEDS : seeds;
	return Object.fromEntries(SEED_KEYS.map(key => {
		const value = source && (typeof source === 'object' || typeof source === 'function') ? source[key] : undefined;
		if (!Array.isArray(value) || value.length !== 4 || value.some(limb => !Number.isInteger(limb) || limb < 0 || limb > 65535)) return [key, null];
		return [key, [...value]];
	}));
}

function safeRecordLimits(limits) {
	const source = limits === undefined || limits === null ? {} : limits;
	return Object.fromEntries(LIMIT_KEYS.map(key => {
		const value = source && (typeof source === 'object' || typeof source === 'function') && Object.hasOwn(source, key)
			? source[key]
			: DEFAULT_LIMITS[key];
		return [key, typeof value === 'number' && Number.isFinite(value) ? value : null];
	}));
}

function playerCounter() {
	return { total: 0, byPlayer: { p1: 0, p2: 0 } };
}

function resultCounterSet() {
	const actionCounts = () => ({ move: playerCounter(), switch: playerCounter() });
	return {
		objectPackets: 0,
		policyInvocations: playerCounter(),
		choiceSubmissions: playerCounter(),
		choiceResolutions: {
			accepted: playerCounter(), rejected: playerCounter(), pending: playerCounter(),
		},
		retryRequests: playerCounter(),
		retrySubmissions: playerCounter(),
		retryLimitExhaustions: playerCounter(),
		actions: { selected: actionCounts(), submitted: actionCounts() },
		publicEvents: 0,
		latestPublicTurn: null,
		terminalMetadataTurns: null,
	};
}

function createResultRecord(seeds, limits) {
	return {
		schemaVersion: 1,
		kind: 'battle-run-result',
		upstreamCommit: UPSTREAM_COMMIT,
		formatId: FORMAT_ID,
		seeds: safeRecordSeeds(seeds),
		limits: safeRecordLimits(limits),
		status: 'failed',
		outcome: null,
		elapsedMs: 0,
		elapsedScope: 'battle_runner',
		counters: resultCounterSet(),
		failure: null,
		cleanupFailure: null,
	};
}

function runnerFailure(category, code, message, cause = undefined) {
	const error = new Error(message, cause === undefined ? undefined : { cause });
	error.name = 'BattleRunnerError';
	error.category = category;
	error.code = code;
	BRANDED_FAILURES.add(error);
	return error;
}

function isRunnerFailure(error) {
	return error instanceof Error && BRANDED_FAILURES.has(error);
}

function incrementPlayer(counter, player) {
	counter.total++;
	counter.byPlayer[player]++;
}

function incrementAction(counter, action, player) {
	const kind = action.startsWith('move ') ? 'move' : 'switch';
	incrementPlayer(counter[kind], player);
}

function updateResolutionCounters(attempts, counters) {
	for (const attempt of attempts) {
		if (!attempt.sent) continue;
		const status = attempt.status === 'accepted' || attempt.status === 'rejected' ? attempt.status : 'pending';
		incrementPlayer(counters.choiceResolutions[status], attempt.player);
	}
}

function attachFailureRecord(error, resultRecord) {
	const attached = error instanceof Error ? error : new Error('Battle runner failed.');
	attached.resultRecord = resultRecord;
	return attached;
}

function assertSeed(seed, name) {
	if (!Array.isArray(seed) || seed.length !== 4 || seed.some(limb => !Number.isInteger(limb) || limb < 0 || limb > 65535)) {
		throw new Error(`${name} must contain four decimal Gen 5 seed limbs in the range 0..65535.`);
	}
}

function normalizedPacket(packet) {
	const normalized = packet.replace(/\r/g, '');
	const separator = normalized.indexOf('\n');
	if (separator < 1) throw new Error('Showdown emitted a malformed object packet.');
	const channel = normalized.slice(0, separator);
	if (channel === 'update') {
		const source = normalized.slice(separator + 1).split('\n');
		const lines = [];
		for (let index = 0; index < source.length; index++) {
			const line = source[index];
			if (/^\|split\|p[12]$/.test(line)) {
				const secret = source[index + 1] ?? '';
				const publicLine = source[index + 2] ?? '';
				lines.push(line, secret.startsWith('|t:|') ? '' : secret, publicLine.startsWith('|t:|') ? '' : publicLine);
				index += 2;
			} else if (!line.startsWith('|t:|')) {
				lines.push(line);
			}
		}
		return `update\n${lines.join('\n')}`;
	}
	return normalized;
}

function packetParts(packet) {
	const separator = packet.indexOf('\n');
	if (separator < 1) throw new Error('Showdown emitted a malformed object packet.');
	return [packet.slice(0, separator), packet.slice(separator + 1)];
}

function stableJson(value) {
	if (Array.isArray(value)) return `[${value.map(stableJson).join(',')}]`;
	if (value && typeof value === 'object') {
		return `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${stableJson(value[key])}`).join(',')}}`;
	}
	return JSON.stringify(value);
}

function requestKey(request) {
	return stableJson(request);
}

function canonicalSeedConfig(seeds) {
	const out = {};
	for (const key of Object.keys(DEFAULT_SEEDS)) {
		assertSeed(seeds[key], key);
		out[key] = [...seeds[key]];
	}
	return out;
}

function cloneAndFreeze(value) {
	return deepFreeze(structuredClone(value));
}

function markAttempt(attempt, status, policyByPlayer, detail = undefined) {
	attempt.status = status;
	if (detail !== undefined) attempt.detail = detail;
	policyByPlayer[attempt.player].history.push(cloneAndFreeze({
		type: 'ownChoiceResolution', requestKey: attempt.requestKey, episodeId: attempt.episodeId,
		command: attempt.command, status, detail,
	}));
}

function classifyRejection(message) {
	if (message.startsWith('|error|[Unavailable choice]')) return 'unavailable';
	if (message.startsWith('|error|[Invalid choice]')) return 'invalid';
	return null;
}

function progressEvidence(records) {
	return records.some(record => record.type === 'event' && /^\|(?:turn|move|switch|faint|win|tie)\|/.test(record.message));
}

function parseTerminal(endBody, publicTerminal) {
	let metadata;
	try {
		metadata = JSON.parse(endBody);
	} catch {
		throw new Error('Showdown end packet did not contain valid terminal metadata.');
	}
	if (!metadata || typeof metadata !== 'object' || Array.isArray(metadata)) throw new Error('Showdown end metadata is malformed.');
	if (typeof metadata.winner !== 'string' || !['', 'p1', 'p2'].includes(metadata.winner)) {
		throw new Error('Showdown end metadata has an invalid winner field.');
	}
	if (!Number.isInteger(metadata.turns) || metadata.turns < 0) throw new Error('Showdown end metadata has invalid turn count.');
	const winners = [...new Set(publicTerminal.winners || [])];
	if (winners.length > 1) throw new Error('Public terminal protocol contains conflicting winners.');
	if (publicTerminal.tie && winners.length) throw new Error('Public terminal protocol contains both a win and a tie.');
	let protocolOutcome;
	if (winners.length) {
		if (!PLAYERS.includes(winners[0])) throw new Error('Public winner is not a stable p1/p2 player name.');
		protocolOutcome = { result: 'win', winner: winners[0] };
	} else if (publicTerminal.tie) {
		protocolOutcome = { result: 'tie', winner: null };
	} else {
		throw new Error('Showdown ended without a public |win| or |tie| event.');
	}
	const metadataWinner = metadata.winner === '' ? null : metadata.winner;
	if (metadataWinner !== protocolOutcome.winner) {
		throw new Error(`Showdown terminal metadata contradicts public outcome (${metadataWinner} vs ${protocolOutcome.winner}).`);
	}
	for (const player of PLAYERS) {
		if (metadata[player] !== player) throw new Error(`Showdown end metadata has an unexpected ${player} identity.`);
	}
	return { ...protocolOutcome, turns: metadata.turns };
}

/**
 * Run one complete local Gen 5 battle. Policy callbacks receive only a frozen
 * copy of their own projected history, current own request, and own RNG.
 * Trusted packets, teams, stream, opponent view, and pending choices stay here.
 */
async function runBattle(options = {}) {
	const started = performance.now();
	const resultRecord = createResultRecord();
	let ps;
	let seeds = DEFAULT_SEEDS;
	let limits = {};
	let policies = {};
	let streamFactory;
	let seedConfig;
	let bounds;
	let stream;
	let output;
	let terminal;
	let endMetadata;
	let deadline;
	let phase = 'initialization';
	let primaryFailure;
	let cleanupFailure;
	const policyByPlayer = {};
	const attempts = [];
	const trustedRecords = [];
	const packets = [];
	const publicTerminal = {};
	let decisionCount = 0;
	let attemptNumber = 0;
	let packetCount = 0;
	let turnCount = 0;
	let latestPublicTurn = null;
	const actionCounts = { move: 0, switch: 0 };

	const fail = (category, code, message, cause = undefined) => {
		throw runnerFailure(category, code, message, cause);
	};
	const project = (packet, audience) => {
		try {
			return projectPackets([packet], audience);
		} catch (error) {
			fail('protocol', 'invalid_packet_projection', error.message, error);
		}
	};
	const timedNext = async () => {
		const remaining = deadline - performance.now();
		const timeoutMessage = `Gen 5 battle exceeded its ${bounds.timeoutMs} ms wall-clock limit (packets=${packetCount}, decisions=${decisionCount}, turns=${turnCount}).`;
		if (remaining <= 0) fail('timeout', 'wall_clock_timeout', timeoutMessage);
		let timer;
		try {
			return await Promise.race([
				output.next(),
				new Promise((_, reject) => {
					timer = setTimeout(() => reject(runnerFailure('timeout', 'wall_clock_timeout', timeoutMessage)), remaining);
				}),
			]);
		} finally {
			if (timer) clearTimeout(timer);
		}
	};
	const submit = async (player, request) => {
		const state = policyByPlayer[player];
		const key = requestKey(request);
		if (state.lastRequestKey === key && state.request?.submittedAttempt?.status === 'submitted') return;
		let validCandidates;
		try {
			validCandidates = candidateActions(request, [...state.rejectedCommands]);
		} catch (error) {
			fail('protocol', 'invalid_request', error.message, error);
		}
		if (!validCandidates.length) {
			state.request = { value: request, key, submittedAttempt: null };
			state.lastRequestKey = key;
			return;
		}
		if (decisionCount + 1 > bounds.maxDecisions) fail('decision_limit', 'decision_limit_exhausted', `Gen 5 battle exceeded ${bounds.maxDecisions} decisions.`);
		state.request = { value: request, key, submittedAttempt: null };
		state.lastRequestKey = key;
		if (!state.episodeStarted) {
			state.history.push(cloneAndFreeze({ type: 'requestEpisode', id: state.episodeId }));
			state.episodeStarted = true;
		}
		const ownInput = policyInput(state.history, request, state.random);
		decisionCount++;
		incrementPlayer(resultRecord.counters.policyInvocations, player);
		let action;
		try {
			action = state.choose(ownInput);
		} catch (error) {
			fail('policy', 'policy_invocation_failed', `Policy callback failed for ${player}.`, error);
		}
		if (action === null || action === undefined) fail('policy', 'invalid_choice', `Policy returned no choice for actionable ${player} request.`);
		if (typeof action !== 'string' || !validCandidates.includes(action)) fail('policy', 'invalid_choice', `Policy returned a choice outside ${player}'s current request.`);
		actionCounts[action.startsWith('move ') ? 'move' : 'switch']++;
		incrementAction(resultRecord.counters.actions.selected, action, player);
		if (process.env.POKEMON_RUNNER_DEBUG) console.error(`decision packet=${packetCount} ${player} ${action} decisions=${decisionCount} turns=${turnCount}`);
		state.history.push(cloneAndFreeze({ type: 'ownChoice', command: action, status: 'submitted', requestKey: key, episodeId: state.episodeId }));
		const attempt = {
			type: 'choice', attempt: ++attemptNumber, player, requestId: request.rqid ?? null,
			requestKey: key, command: action, status: 'submitted', sent: false, episodeId: state.episodeId,
		};
		attempts.push(attempt);
		trustedRecords.push({ type: 'choice', attempt: attempt.attempt, player, requestId: attempt.requestId, request, command: action, status: 'submitted' });
		state.request.submittedAttempt = attempt;
		try {
			await stream.write(`>${player} ${action}`);
		} catch (error) {
			fail('runtime', 'choice_write_failed', `Showdown choice submission failed for ${player}.`, error);
		}
		attempt.sent = true;
		incrementPlayer(resultRecord.counters.choiceSubmissions, player);
		incrementAction(resultRecord.counters.actions.submitted, action, player);
		if (state.retryPending) {
			incrementPlayer(resultRecord.counters.retrySubmissions, player);
			state.retryPending = false;
		}
	};

	try {
		if (options === null || (typeof options !== 'object' && typeof options !== 'function')) {
			fail('initialization', 'invalid_configuration', 'Battle runner options must be an object.');
		}
		ps = options.ps;
		seeds = options.seeds === undefined ? DEFAULT_SEEDS : options.seeds;
		limits = options.limits === undefined ? {} : options.limits;
		policies = options.policies === undefined ? {} : options.policies;
		streamFactory = options.streamFactory;
		resultRecord.seeds = safeRecordSeeds(seeds);
		resultRecord.limits = safeRecordLimits(limits);
		if (!ps?.BattleStream || !ps?.Teams || !ps?.Dex) fail('initialization', 'invalid_showdown_api', 'Pinned Showdown API is required.');
		try {
			seedConfig = canonicalSeedConfig(seeds);
		} catch (error) {
			fail('initialization', 'invalid_seed_config', error.message, error);
		}
		try {
			bounds = { ...DEFAULT_LIMITS, ...limits };
		} catch (error) {
			fail('initialization', 'invalid_limit_config', 'Configured battle limits must be a plain object.', error);
		}
		for (const [key, value] of Object.entries(bounds)) {
			if (!Number.isFinite(value) || value <= 0) fail('initialization', 'invalid_limit_config', `${key} must be a positive finite number.`);
		}
		if (!policies || (typeof policies !== 'object' && typeof policies !== 'function')) fail('initialization', 'invalid_policy_config', 'Policies must be provided as an object.');
		for (const player of PLAYERS) {
			if (policies[player] !== undefined && typeof policies[player] !== 'function') fail('policy', 'invalid_policy_config', `Policy for ${player} must be a function.`);
		}
		deadline = started + bounds.timeoutMs;
		let teams;
		try {
			teams = [
				ps.Teams.generate(FORMAT_ID, { seed: seedConfig.teamP1 }),
				ps.Teams.generate(FORMAT_ID, { seed: seedConfig.teamP2 }),
			];
		} catch (error) {
			fail('initialization', 'team_generation_failed', 'Showdown team generation failed.', error);
		}
		try {
			stream = streamFactory ? streamFactory() : new ps.BattleStream({ keepAlive: true });
		} catch (error) {
			fail('initialization', 'stream_initialization_failed', 'Could not initialize the Showdown battle stream.', error);
		}
		if (!stream || typeof stream.write !== 'function' || typeof stream.destroy !== 'function') fail('initialization', 'stream_initialization_failed', 'Showdown battle stream is incomplete.');
		for (let index = 0; index < PLAYERS.length; index++) {
			const player = PLAYERS[index];
			policyByPlayer[player] = {
				choose: policies[player] || randomPolicy,
				random: createPolicyRng(seedConfig[index === 0 ? 'policyP1' : 'policyP2']),
				history: [], request: null, lastRequestKey: null, episodeId: 1,
				episodeStarted: false, rejectedCommands: new Set(), retries: 0, retryPending: false,
			};
		}
		try {
			await stream.write(`>start ${JSON.stringify({ formatid: FORMAT_ID, seed: seedConfig.battle })}`);
			for (let index = 0; index < PLAYERS.length; index++) {
				const player = PLAYERS[index];
				await stream.write(`>player ${player} ${JSON.stringify({ name: player, team: ps.Teams.pack(teams[index]) })}`);
			}
			output = stream[Symbol.asyncIterator]();
		} catch (error) {
			fail('initialization', 'stream_start_failed', 'Showdown battle stream failed during initialization.', error);
		}
		if (!output || typeof output.next !== 'function') fail('initialization', 'stream_initialization_failed', 'Showdown battle stream has no packet iterator.');

		phase = 'runtime';
		while (!terminal) {
			if (packetCount + 1 > bounds.maxPackets) fail('packet_limit', 'packet_limit_exhausted', `Gen 5 battle exceeded ${bounds.maxPackets} object packets.`);
			let next;
			try {
				next = await timedNext();
			} catch (error) {
				if (isRunnerFailure(error)) throw error;
				fail('runtime', 'stream_read_failed', 'Showdown battle stream failed while reading a packet.', error);
			}
			if (next.done) fail('protocol', 'premature_stream_end', 'Showdown stream ended before terminal outcome metadata.');
			packetCount++;
			const rawPacket = next.value;
			if (typeof rawPacket !== 'string') fail('protocol', 'invalid_packet', 'Showdown emitted a non-string object packet.');
			let packet;
			let channel;
			let body;
			let originalPacket;
			let originalChannel;
			try {
				packet = normalizedPacket(rawPacket);
				[ channel, body ] = packetParts(packet);
				originalPacket = rawPacket.replace(/\r/g, '');
				[ originalChannel ] = packetParts(originalPacket);
			} catch (error) {
				fail('protocol', 'invalid_packet', error.message, error);
			}
			packets.push(packet);
			if (channel === 'update') {
				for (const line of body.split('\n')) {
					if (line.startsWith('|turn|')) turnCount = Number(line.slice('|turn|'.length)) || turnCount;
				}
			}
			if (channel === 'error') fail('protocol', 'reference_stream_error', 'Showdown emitted a reference stream error.');
			if (channel === 'end') {
				endMetadata = body;
				try {
					terminal = parseTerminal(body, publicTerminal);
				} catch (error) {
					fail('protocol', 'terminal_validation_failed', error.message, error);
				}
				resultRecord.counters.terminalMetadataTurns = terminal.turns;
				trustedRecords.push({ type: 'terminal', outcome: terminal });
				continue;
			}
			if (originalChannel !== 'update' && originalChannel !== 'sideupdate') fail('protocol', 'unsupported_channel', 'Showdown emitted an unsupported channel.');
			const visibleByPlayer = {};
			for (const player of PLAYERS) {
				const projected = project(originalPacket, player);
				visibleByPlayer[player] = projected.records;
				policyByPlayer[player].history.push(...projected.records.map(cloneAndFreeze));
			}
			const publicRecords = project(originalPacket, 'spectator').records;
			resultRecord.counters.publicEvents += publicRecords.filter(record => record.type === 'event').length;
			for (const record of publicRecords) {
				if (record.type !== 'event') continue;
				const turnMarker = /^\|turn\|(\d+)$/.exec(record.message);
				if (turnMarker) latestPublicTurn = Number(turnMarker[1]);
			}
			resultRecord.counters.latestPublicTurn = latestPublicTurn;
			if (progressEvidence(publicRecords)) {
				for (const attempt of attempts) {
					if (attempt.sent && attempt.status === 'submitted') {
						markAttempt(attempt, 'accepted', policyByPlayer);
						trustedRecords.push({ type: 'choiceResolution', attempt: attempt.attempt, status: 'accepted' });
					}
				}
				for (const player of PLAYERS) {
					const state = policyByPlayer[player];
					state.request = null;
					state.lastRequestKey = null;
					state.episodeId++;
					state.episodeStarted = false;
					state.rejectedCommands.clear();
					state.retries = 0;
					state.retryPending = false;
				}
			}
			for (const record of publicRecords) {
				if (record.type === 'event' && record.message.startsWith('|win|')) {
					publicTerminal.winners ||= [];
					publicTerminal.winners.push(record.message.slice('|win|'.length));
				}
				if (record.type === 'event' && record.message === '|tie|') publicTerminal.tie = true;
			}
			if (channel === 'sideupdate') {
				for (const player of PLAYERS) {
					for (const record of visibleByPlayer[player]) {
						if (record.type === 'privateEvent') {
							const rejection = classifyRejection(record.message);
							if (!rejection) fail('protocol', 'unexpected_private_event', `Unexpected private protocol event for ${player}.`);
						const attempt = [...attempts].reverse().find(item => item.player === player && item.sent && item.status === 'submitted');
						if (!attempt) fail('protocol', 'rejection_without_submission', `Showdown rejected a ${player} choice with no pending submission.`);
						markAttempt(attempt, 'rejected', policyByPlayer, rejection);
						policyByPlayer[player].rejectedCommands.add(attempt.command);
						trustedRecords.push({ type: 'choiceResolution', attempt: attempt.attempt, status: 'rejected', category: rejection });
						if (rejection === 'invalid') fail('policy', 'invalid_choice', `Random policy produced an invalid ${player} choice; harness failed closed.`);
						const state = policyByPlayer[player];
						state.request = null;
						incrementPlayer(resultRecord.counters.retryRequests, player);
						state.retryPending = true;
						if (state.retries + 1 > bounds.maxRetriesPerRequest) {
							incrementPlayer(resultRecord.counters.retryLimitExhaustions, player);
							fail('retry_limit', 'retry_limit_exhausted', `Showdown exceeded ${bounds.maxRetriesPerRequest} unavailable-choice retries for ${player}.`);
						}
						state.retries++;
						trustedRecords.push({ type: 'retry', player, reason: 'unavailable_choice', requestId: attempt.requestId });
					}
					if (record.type === 'request') {
						const request = structuredClone(record.request);
						trustedRecords.push({ type: 'request', player, request });
						const priorKey = policyByPlayer[player].lastRequestKey;
						if (priorKey === requestKey(request) && policyByPlayer[player].request?.submittedAttempt?.status === 'submitted') continue;
						await submit(player, request);
					}
				}
			}
			}
			trustedRecords.push({ type: 'packet', packet });
		}
	} catch (error) {
		primaryFailure = isRunnerFailure(error)
			? error
			: runnerFailure(phase === 'initialization' ? 'initialization' : 'runtime', phase === 'initialization' ? 'initialization_failed' : 'runtime_failed', error instanceof Error ? error.message : 'Battle runner failed.', error);
	} finally {
		if (stream) {
			try {
				await stream.destroy();
			} catch (error) {
				cleanupFailure = error;
				if (!primaryFailure) primaryFailure = runnerFailure('runtime', 'stream_cleanup_failed', 'Showdown battle stream cleanup failed.', error);
			}
		}
		resultRecord.counters.objectPackets = packetCount;
		resultRecord.counters.latestPublicTurn = latestPublicTurn;
		resultRecord.counters.terminalMetadataTurns = terminal?.turns ?? resultRecord.counters.terminalMetadataTurns;
		updateResolutionCounters(attempts, resultRecord.counters);
		resultRecord.elapsedMs = Math.max(0, performance.now() - started);
		if (!Number.isFinite(resultRecord.elapsedMs)) resultRecord.elapsedMs = 0;
		if (cleanupFailure) resultRecord.cleanupFailure = { category: 'runtime', code: 'stream_cleanup_failed' };
		if (primaryFailure) {
			resultRecord.status = 'failed';
			resultRecord.outcome = null;
			resultRecord.failure = {
				category: primaryFailure.category || (phase === 'initialization' ? 'initialization' : 'runtime'),
				code: primaryFailure.code || (phase === 'initialization' ? 'initialization_failed' : 'runtime_failed'),
			};
		}
	}

	if (primaryFailure) throw attachFailureRecord(primaryFailure, resultRecord);
	resultRecord.status = 'completed';
	resultRecord.outcome = terminal;
	return {
		summary: { schemaVersion: 1, upstreamCommit: UPSTREAM_COMMIT, formatId: FORMAT_ID, seeds: seedConfig, outcome: terminal, packets: packets.length, decisions: decisionCount, actions: { ...actionCounts } },
		transcript: {
			schemaVersion: 1,
			kind: 'trusted omniscient oracle transcript; includes private requests, split branches, and end metadata; not a policy observation',
			upstreamCommit: UPSTREAM_COMMIT,
			formatId: FORMAT_ID,
			seeds: seedConfig,
			records: trustedRecords,
			packets,
			endMetadata: JSON.parse(endMetadata),
		},
		resultRecord,
	};
}

function parseArgs(args, parseSeed) {
	const seeds = structuredClone(DEFAULT_SEEDS);
	let outputPath = null;
	let resultPath = null;
	const mapping = {
		'--team-seed-p1': 'teamP1', '--team-seed-p2': 'teamP2', '--battle-seed': 'battle',
		'--policy-seed-p1': 'policyP1', '--policy-seed-p2': 'policyP2',
	};
	try {
		for (let index = 0; index < args.length; index++) {
			const raw = args[index];
			const equal = raw.indexOf('=');
			const flag = equal < 0 ? raw : raw.slice(0, equal);
			const inline = equal < 0 ? undefined : raw.slice(equal + 1);
			if (inline?.includes('=')) throw new Error(`Invalid option: ${raw}`);
			if (flag === '--output' || flag === '--result') {
				const value = inline ?? args[++index];
				if (!value) throw new Error(`Missing value for ${flag}.`);
				if (flag === '--output') outputPath = value;
				else resultPath = value;
				continue;
			}
			if (!mapping[flag]) throw new Error(`Unknown option: ${flag}`);
			const value = inline ?? args[++index];
			if (!value) throw new Error(`Missing value for ${flag}.`);
			seeds[mapping[flag]] = parseSeed(value, flag);
		}
	} catch (error) {
		const resultRecord = createResultRecord(seeds);
		resultRecord.status = 'failed';
		resultRecord.failure = { category: 'initialization', code: 'invalid_cli_configuration' };
		const failure = runnerFailure('initialization', 'invalid_cli_configuration', error instanceof Error ? error.message : 'Invalid command-line configuration.', error);
		failure.resultRecord = resultRecord;
		throw failure;
	}
	return { seeds, outputPath, resultPath };
}

function findResultPath(args) {
	let resultPath = null;
	for (let index = 0; index < args.length; index++) {
		const raw = args[index];
		if (typeof raw !== 'string') continue;
		const equal = raw.indexOf('=');
		const flag = equal < 0 ? raw : raw.slice(0, equal);
		if (flag !== '--result') continue;
		resultPath = equal < 0 ? args[index + 1] ?? null : raw.slice(equal + 1);
		if (equal < 0 && resultPath !== null) index++;
	}
	return typeof resultPath === 'string' && resultPath ? resultPath : null;
}

async function main(args = process.argv.slice(2)) {
	if (args[0] === '--help' || args[0] === 'help') {
		console.log('Usage: node environment/run-random.cjs [--team-seed-p1 a,b,c,d --team-seed-p2 a,b,c,d --battle-seed a,b,c,d --policy-seed-p1 a,b,c,d --policy-seed-p2 a,b,c,d] [--output trusted.json] [--result result.json]');
		return;
	}
	const started = performance.now();
	let resultPath = findResultPath(args);
	let configuredSeeds = DEFAULT_SEEDS;
	let options;
	let result;
	let runnerInvoked = false;
	try {
		const oracle = require('../reference/oracle.cjs');
		options = parseArgs(args, oracle.parseSeed);
		configuredSeeds = options.seeds;
		resultPath = options.resultPath || resultPath;
		oracle.assertReady();
		const ps = require('../reference/vendor/pokemon-showdown/dist/sim');
		runnerInvoked = true;
		result = await runBattle({ ps, seeds: options.seeds });
	} catch (error) {
		const runnerRecord = isRunnerFailure(error) ? error.resultRecord : null;
		const resultRecord = runnerRecord || createResultRecord(configuredSeeds);
		resultRecord.status = 'failed';
		resultRecord.outcome = null;
		resultRecord.failure ||= {
			category: 'initialization',
			code: 'cli_initialization_failed',
		};
		if (!runnerInvoked) {
			resultRecord.elapsedScope = 'cli_initialization';
			resultRecord.elapsedMs = Math.max(0, performance.now() - started);
			if (!Number.isFinite(resultRecord.elapsedMs)) resultRecord.elapsedMs = 0;
		}
		if (resultPath) fs.writeFileSync(path.resolve(resultPath), `${JSON.stringify(resultRecord, null, 2)}\n`);
		throw attachFailureRecord(error, resultRecord);
	}
	if (resultPath) {
		const resultOutputPath = path.resolve(resultPath);
		fs.writeFileSync(resultOutputPath, `${JSON.stringify(result.resultRecord, null, 2)}\n`);
	}
	if (options.outputPath) {
		const outputPath = path.resolve(options.outputPath);
		fs.writeFileSync(outputPath, `${JSON.stringify(result.transcript, null, 2)}\n`);
	}
	process.stdout.write(`${JSON.stringify(result.summary, null, 2)}\n`);
}

if (require.main === module) {
	main().catch(error => {
		console.error(`Random battle runner failed: ${error.message}`);
		process.exitCode = 1;
	});
}

module.exports = { DEFAULT_LIMITS, DEFAULT_SEEDS, candidateActions, main, parseArgs, parseTerminal, runBattle };
