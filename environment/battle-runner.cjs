'use strict';

const fs = require('node:fs');
const path = require('node:path');
const { projectPackets } = require('./protocol.cjs');
const { candidateActions, createPolicyRng, deepFreeze, policyInput, randomPolicy } = require('./random-policy.cjs');

const FORMAT_ID = 'gen5randombattle';
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
async function runBattle({ ps, seeds = DEFAULT_SEEDS, limits = {}, policies = {}, streamFactory } = {}) {
	if (!ps?.BattleStream || !ps?.Teams || !ps?.Dex) throw new Error('Pinned Showdown API is required.');
	const seedConfig = canonicalSeedConfig(seeds);
	const bounds = { ...DEFAULT_LIMITS, ...limits };
	for (const [key, value] of Object.entries(bounds)) {
		if (!Number.isFinite(value) || value <= 0) throw new Error(`${key} must be a positive finite number.`);
	}
	const started = Date.now();
	const deadline = started + bounds.timeoutMs;
	const teams = [
		ps.Teams.generate(FORMAT_ID, { seed: seedConfig.teamP1 }),
		ps.Teams.generate(FORMAT_ID, { seed: seedConfig.teamP2 }),
	];
	const stream = streamFactory ? streamFactory() : new ps.BattleStream({ keepAlive: true });
	const policyByPlayer = Object.fromEntries(PLAYERS.map((player, index) => [
		player,
		{
			choose: policies[player] || randomPolicy,
			random: createPolicyRng(seedConfig[index === 0 ? 'policyP1' : 'policyP2']),
			history: [],
			request: null,
			lastRequestKey: null,
			episodeId: 1,
			episodeStarted: false,
			rejectedCommands: new Set(),
			retries: 0,
		},
	]));
	const attempts = [];
	const trustedRecords = [];
	const packets = [];
	const publicTerminal = {};
	let decisionCount = 0;
	let attemptNumber = 0;
	let packetCount = 0;
	let turnCount = 0;
	const actionCounts = { move: 0, switch: 0 };
	let output;
	let terminal;
	let endMetadata;

	const timedNext = async () => {
		const remaining = deadline - Date.now();
		if (remaining <= 0) throw new Error(`Gen 5 battle exceeded its ${bounds.timeoutMs} ms wall-clock limit (packets=${packetCount}, decisions=${decisionCount}, turns=${turnCount}).`);
		let timer;
		try {
			return await Promise.race([
				output.next(),
				new Promise((_, reject) => {
				timer = setTimeout(() => reject(new Error(`Gen 5 battle exceeded its ${bounds.timeoutMs} ms wall-clock limit (packets=${packetCount}, decisions=${decisionCount}, turns=${turnCount}).`)), remaining);
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
		const validCandidates = candidateActions(request, [...state.rejectedCommands]);
		if (!validCandidates.length) {
			state.request = { value: request, key, submittedAttempt: null };
			state.lastRequestKey = key;
			return;
		}
		if (++decisionCount > bounds.maxDecisions) throw new Error(`Gen 5 battle exceeded ${bounds.maxDecisions} decisions.`);
		state.request = { value: request, key, submittedAttempt: null };
		state.lastRequestKey = key;
		if (!state.episodeStarted) {
			state.history.push(cloneAndFreeze({ type: 'requestEpisode', id: state.episodeId }));
			state.episodeStarted = true;
		}
		const ownInput = policyInput(state.history, request, state.random);
		const action = state.choose(ownInput);
		if (action === null || action === undefined) throw new Error(`Policy returned no choice for actionable ${player} request.`);
		if (!validCandidates.includes(action)) throw new Error(`Policy returned a choice outside ${player}'s current request.`);
		actionCounts[action.startsWith('move ') ? 'move' : 'switch']++;
		if (process.env.POKEMON_RUNNER_DEBUG) console.error(`decision packet=${packetCount} ${player} ${action} decisions=${decisionCount} turns=${turnCount}`);
		state.history.push(cloneAndFreeze({ type: 'ownChoice', command: action, status: 'submitted', requestKey: key, episodeId: state.episodeId }));
		const attempt = {
			type: 'choice', attempt: ++attemptNumber, player, requestId: request.rqid ?? null,
			requestKey: key, command: action, status: 'submitted', episodeId: state.episodeId,
		};
		attempts.push(attempt);
		trustedRecords.push({ type: 'choice', attempt: attempt.attempt, player, requestId: attempt.requestId, request, command: action, status: 'submitted' });
		state.request.submittedAttempt = attempt;
		await stream.write(`>${player} ${action}`);
	};

	try {
		stream.write(`>start ${JSON.stringify({ formatid: FORMAT_ID, seed: seedConfig.battle })}`);
		for (let index = 0; index < PLAYERS.length; index++) {
			const player = PLAYERS[index];
			stream.write(`>player ${player} ${JSON.stringify({ name: player, team: ps.Teams.pack(teams[index]) })}`);
		}
		output = stream[Symbol.asyncIterator]();
		while (!terminal) {
			if (++packetCount > bounds.maxPackets) throw new Error(`Gen 5 battle exceeded ${bounds.maxPackets} object packets.`);
			const next = await timedNext();
			if (next.done) throw new Error('Showdown stream ended before terminal outcome metadata.');
			const rawPacket = next.value;
			if (typeof rawPacket !== 'string') throw new Error('Showdown emitted a non-string object packet.');
			const packet = normalizedPacket(rawPacket);
			packets.push(packet);
			const [channel, body] = packetParts(packet);
			const originalPacket = rawPacket.replace(/\r/g, '');
			const [originalChannel] = packetParts(originalPacket);
			if (channel === 'update') {
				for (const line of body.split('\n')) {
					if (line.startsWith('|turn|')) turnCount = Number(line.slice('|turn|'.length)) || turnCount;
				}
			}
			if (channel === 'error') throw new Error('Showdown emitted a reference stream error.');
			if (channel === 'end') {
				endMetadata = body;
				terminal = parseTerminal(body, publicTerminal);
				trustedRecords.push({ type: 'terminal', outcome: terminal });
				continue;
			}
			if (originalChannel !== 'update' && originalChannel !== 'sideupdate') {
				projectPackets([packet], 'spectator'); // unknown channels fail closed here
				throw new Error('Showdown emitted an unsupported channel.');
			}
			const visibleByPlayer = {};
			for (const player of PLAYERS) {
				const projected = projectPackets([originalPacket], player);
				visibleByPlayer[player] = projected.records;
				policyByPlayer[player].history.push(...projected.records.map(cloneAndFreeze));
			}
			const publicRecords = projectPackets([originalPacket], 'spectator').records;
			if (progressEvidence(publicRecords)) {
				for (const attempt of attempts) {
					if (attempt.status === 'submitted') {
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
						const category = classifyRejection(record.message);
						if (!category) throw new Error(`Unexpected private protocol event for ${player}.`);
						const attempt = [...attempts].reverse().find(item => item.player === player && item.status === 'submitted');
						if (!attempt) throw new Error(`Showdown rejected a ${player} choice with no pending submission.`);
							markAttempt(attempt, 'rejected', policyByPlayer, category);
							policyByPlayer[player].rejectedCommands.add(attempt.command);
						trustedRecords.push({ type: 'choiceResolution', attempt: attempt.attempt, status: 'rejected', category });
						if (category === 'invalid') throw new Error(`Random policy produced an invalid ${player} choice; harness failed closed: ${record.message.slice('|error|[Invalid choice]'.length).trim()}`);
						const state = policyByPlayer[player];
						state.request = null;
						if (++state.retries > bounds.maxRetriesPerRequest) throw new Error(`Showdown exceeded ${bounds.maxRetriesPerRequest} unavailable-choice retries for ${player}.`);
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
		return {
			summary: { schemaVersion: 1, upstreamCommit: require('../reference-lock.json').commit, formatId: FORMAT_ID, seeds: seedConfig, outcome: terminal, packets: packets.length, decisions: decisionCount, actions: { ...actionCounts } },
			transcript: {
				schemaVersion: 1,
				kind: 'trusted omniscient oracle transcript; includes private requests, split branches, and end metadata; not a policy observation',
				upstreamCommit: require('../reference-lock.json').commit,
				formatId: FORMAT_ID,
				seeds: seedConfig,
				records: trustedRecords,
				packets,
				endMetadata: JSON.parse(endMetadata),
			},
		};
	} finally {
		await stream.destroy();
	}
}

function parseArgs(args, parseSeed) {
	const seeds = structuredClone(DEFAULT_SEEDS);
	let outputPath = null;
	const mapping = {
		'--team-seed-p1': 'teamP1', '--team-seed-p2': 'teamP2', '--battle-seed': 'battle',
		'--policy-seed-p1': 'policyP1', '--policy-seed-p2': 'policyP2',
	};
	for (let index = 0; index < args.length; index++) {
		const raw = args[index];
		const equal = raw.indexOf('=');
		const flag = equal < 0 ? raw : raw.slice(0, equal);
		const inline = equal < 0 ? undefined : raw.slice(equal + 1);
		if (inline?.includes('=')) throw new Error(`Invalid option: ${raw}`);
		if (flag === '--output') {
			outputPath = inline ?? args[++index];
			if (!outputPath) throw new Error('Missing value for --output.');
			continue;
		}
		if (!mapping[flag]) throw new Error(`Unknown option: ${flag}`);
		const value = inline ?? args[++index];
		if (!value) throw new Error(`Missing value for ${flag}.`);
		seeds[mapping[flag]] = parseSeed(value, flag);
	}
	return { seeds, outputPath };
}

async function main(args = process.argv.slice(2)) {
	if (args[0] === '--help' || args[0] === 'help') {
		console.log('Usage: node environment/run-random.cjs [--team-seed-p1 a,b,c,d --team-seed-p2 a,b,c,d --battle-seed a,b,c,d --policy-seed-p1 a,b,c,d --policy-seed-p2 a,b,c,d] [--output trusted.json]');
		return;
	}
	const oracle = require('../reference/oracle.cjs');
	oracle.assertReady();
	const options = parseArgs(args, oracle.parseSeed);
	const ps = require('../reference/vendor/pokemon-showdown/dist/sim');
	const result = await runBattle({ ps, seeds: options.seeds });
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
