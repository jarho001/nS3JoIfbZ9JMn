#!/usr/bin/env node
'use strict';

const fs = require('node:fs');
const path = require('node:path');
const { projectPackets } = require('../environment/protocol.cjs');

const ROOT = path.resolve(__dirname, '..');
const LOCK = JSON.parse(fs.readFileSync(path.join(ROOT, 'reference-lock.json'), 'utf8'));
const FORMAT_ID = 'gen5randombattle';
const TRACE_KIND = 'trusted complete BattleStream input/output trace';
const SEED_NAMES = ['teamP1', 'teamP2', 'battle', 'policyP1', 'policyP2'];
const DEFAULT_LIMITS = Object.freeze({ timeoutMs: 60_000, maxEvents: 50_000 });

function fail(message) {
	throw new Error(message);
}

function cloneSeedConfig(seeds) {
	const result = {};
	for (const name of SEED_NAMES) {
		const seed = seeds?.[name];
		if (!Array.isArray(seed) || seed.length !== 4 || seed.some(limb => !Number.isInteger(limb) || limb < 0 || limb > 65535)) {
			fail(`Trace ${name} must contain four Gen 5 seed limbs in the range 0..65535.`);
		}
		result[name] = [...seed];
	}
	return result;
}

function packetParts(packet) {
	if (typeof packet !== 'string') fail('Trace packet must be a string.');
	const separator = packet.indexOf('\n');
	if (separator < 1) fail('Trace packet is malformed.');
	return [packet.slice(0, separator), packet.slice(separator + 1)];
}

/** Remove only documented clock lines, preserving split markers and both branch slots. */
function normalizePacket(packet) {
	const normalized = packet.replace(/\r/g, '');
	const [channel, body] = packetParts(normalized);
	if (channel !== 'update') return normalized;
	const source = body.split('\n');
	const output = [];
	for (let index = 0; index < source.length; index++) {
		const line = source[index];
		if (line.startsWith('|split')) {
			if (!/^\|split\|p[12]$/.test(line) || index + 2 >= source.length) fail('Malformed split in trace output.');
			const secret = source[index + 1];
			const publicLine = source[index + 2];
			if (secret.startsWith('|split') || publicLine.startsWith('|split')) fail('Nested split in trace output.');
			output.push(line, secret.startsWith('|t:|') ? '' : secret, publicLine.startsWith('|t:|') ? '' : publicLine);
			index += 2;
		} else if (!line.startsWith('|t:|')) {
			output.push(line);
		}
	}
	return `update\n${output.join('\n')}`;
}

function firstDifference(expected, actual, pathName = '$') {
	if (Object.is(expected, actual)) return null;
	if (!expected || !actual || typeof expected !== 'object' || typeof actual !== 'object') {
		return { path: pathName, expected, actual };
	}
	if (Array.isArray(expected) !== Array.isArray(actual)) return { path: pathName, expected, actual };
	const keys = [...new Set([...Object.keys(expected), ...Object.keys(actual)])].sort();
	for (const key of keys) {
		if (!Object.hasOwn(expected, key) || !Object.hasOwn(actual, key)) {
			return { path: `${pathName}.${key}`, expected: expected[key], actual: actual[key] };
		}
		const difference = firstDifference(expected[key], actual[key], `${pathName}.${key}`);
		if (difference) return difference;
	}
	return null;
}

function mismatchMessage(index, expectedPacket, actualPacket) {
	const [expectedChannel, expectedBody] = packetParts(expectedPacket);
	const [actualChannel, actualBody] = packetParts(actualPacket);
	if (expectedChannel !== actualChannel) {
		return `Trace mismatch at event ${index}: expected channel ${expectedChannel}, received ${actualChannel}.`;
	}
	if (expectedChannel === 'sideupdate') {
		const parseRequest = body => {
			const newline = body.indexOf('\n');
			const message = body.slice(newline + 1);
			if (!message.startsWith('|request|')) return null;
			try { return JSON.parse(message.slice('|request|'.length)); } catch { return null; }
		};
		const difference = firstDifference(parseRequest(expectedBody), parseRequest(actualBody), 'request');
		if (difference) return `Trace mismatch at event ${index}: ${difference.path} differs.`;
	}
	if (expectedChannel === 'end') {
		let expected;
		let actual;
		try { expected = JSON.parse(expectedBody); actual = JSON.parse(actualBody); } catch { /* report protocol line below */ }
		const difference = firstDifference(expected, actual, 'end');
		if (difference) return `Trace mismatch at event ${index}: ${difference.path} differs.`;
	}
	const expectedLines = expectedBody.split('\n');
	const actualLines = actualBody.split('\n');
	const lines = Math.max(expectedLines.length, actualLines.length);
	for (let line = 0; line < lines; line++) {
		if (expectedLines[line] !== actualLines[line]) return `Trace mismatch at event ${index}, channel ${expectedChannel}, line ${line + 1}.`;
	}
	return `Trace mismatch at event ${index}, channel ${expectedChannel}.`;
}

function validateInputSequence(events, seeds) {
	if (events.length < 3 || events[0]?.type !== 'input' || events[1]?.type !== 'input' || events[2]?.type !== 'input') {
		fail('Trace must record the start and both player teams before simulator output.');
	}
	let phase = 0;
	let choices = 0;
	const players = new Set();
	for (const [index, event] of events.entries()) {
		if (event.type !== 'input') continue;
		if (typeof event.command !== 'string' || event.command.includes('\n') || event.command.includes('\r')) {
			fail(`Trace input at event ${index} is malformed.`);
		}
		if (phase === 0) {
			const match = /^>start (.+)$/.exec(event.command);
			if (!match) fail(`Trace event ${index} must start the battle.`);
			let options;
			try { options = JSON.parse(match[1]); } catch { fail(`Trace start at event ${index} is invalid JSON.`); }
			if (!options || Array.isArray(options) || Object.keys(options).sort().join(',') !== 'formatid,seed' || options.formatid !== FORMAT_ID) {
				fail('Trace start command has unsupported options or format.');
			}
			if (JSON.stringify(options.seed) !== JSON.stringify(seeds.battle)) fail('Trace start seed does not match trace provenance.');
			phase = 1;
			continue;
		}
		if (phase === 1) {
			const match = /^>player (p[12]) (.+)$/.exec(event.command);
			if (!match) fail(`Trace event ${index} must submit a player team.`);
			const player = match[1];
			const expectedPlayer = players.size === 0 ? 'p1' : 'p2';
			if (player !== expectedPlayer || players.has(player)) fail('Trace player commands must submit p1 then p2 exactly once.');
			let options;
			try { options = JSON.parse(match[2]); } catch { fail(`Trace player command at event ${index} is invalid JSON.`); }
			if (!options || Array.isArray(options) || Object.keys(options).sort().join(',') !== 'name,team' || options.name !== player || typeof options.team !== 'string' || !options.team) {
				fail(`Trace ${player} command has unsupported player options.`);
			}
			players.add(player);
			if (players.size === 2) phase = 2;
			continue;
		}
		if (!/^>p[12] (?:move [1-4]|switch [1-6])$/.test(event.command)) {
			fail(`Trace choice at event ${index} is not an allowed move or switch command.`);
		}
		choices++;
	}
	if (phase !== 2 || players.size !== 2 || choices < 1) fail('Trace must include start, both teams, and at least one player choice.');
}

function validateOutcome(events, outcome) {
	const publicOutcome = { winners: [], tie: false };
	let endPacket = null;
	let endIndex = -1;
	for (const [index, event] of events.entries()) {
		if (event.type !== 'output') continue;
		const [channel] = packetParts(event.packet);
		if (channel === 'update') {
			const view = projectPackets([event.packet], 'spectator');
			for (const record of view.records) {
				if (record.type === 'event' && record.message.startsWith('|win|')) publicOutcome.winners.push(record.message.slice('|win|'.length));
				if (record.type === 'event' && record.message === '|tie|') publicOutcome.tie = true;
			}
		}
		if (channel === 'end') {
			if (endPacket) fail('Trace contains more than one end packet.');
			endPacket = event.packet;
			endIndex = index;
		}
	}
	if (!endPacket || endIndex !== events.length - 1 || events[events.length - 1].type !== 'output') fail('Trace is truncated or has events after terminal evidence.');
	const winners = [...new Set(publicOutcome.winners)];
	if (winners.length > 1 || (winners.length && publicOutcome.tie)) fail('Trace public terminal events contradict each other.');
	let metadata;
	try { metadata = JSON.parse(packetParts(endPacket)[1]); } catch { fail('Trace end metadata is invalid JSON.'); }
	if (!metadata || Array.isArray(metadata) || typeof metadata !== 'object') fail('Trace end metadata must be an object.');
	if (typeof metadata.winner !== 'string' || !['', 'p1', 'p2'].includes(metadata.winner)) fail('Trace end metadata winner is invalid.');
	if (!Number.isInteger(metadata.turns) || metadata.turns < 0) fail('Trace end metadata turns are invalid.');
	if (metadata.p1 !== 'p1' || metadata.p2 !== 'p2') fail('Trace end metadata player identities are invalid.');
	const expectedOutcome = winners.length ? { result: 'win', winner: winners[0] } : publicOutcome.tie ? { result: 'tie', winner: null } : null;
	if (!expectedOutcome) fail('Trace has no public win/tie terminal event.');
	const endWinner = metadata.winner === '' ? null : metadata.winner;
	if (endWinner !== expectedOutcome.winner) fail('Trace public terminal outcome contradicts trusted end metadata.');
	const verified = { ...expectedOutcome, turns: metadata.turns };
	if (JSON.stringify(outcome) !== JSON.stringify(verified)) fail('Trace outcome summary contradicts terminal evidence.');
}

function validateTrace(trace, expectedCommit = LOCK.commit, { maxEvents = DEFAULT_LIMITS.maxEvents } = {}) {
	if (!trace || typeof trace !== 'object' || Array.isArray(trace)) fail('Trace document must be an object.');
	if (trace.schemaVersion !== 1 || trace.kind !== TRACE_KIND) fail('Unsupported trace schema or kind.');
	if (trace.upstreamCommit !== expectedCommit) fail(`Trace Showdown SHA mismatch: expected ${expectedCommit}, received ${trace.upstreamCommit || '(missing)'}.`);
	if (trace.formatId !== FORMAT_ID) fail(`Trace format must be ${FORMAT_ID}.`);
	const seeds = cloneSeedConfig(trace.seeds);
	if (!Array.isArray(trace.events) || trace.events.length < 1 || trace.events.length > maxEvents) fail(`Trace event count must be between 1 and ${maxEvents}.`);
	for (const [index, event] of trace.events.entries()) {
		if (!event || typeof event !== 'object' || Array.isArray(event)) fail(`Trace event ${index} is malformed.`);
		if (event.type === 'input') {
			if (Object.keys(event).sort().join(',') !== 'command,type') fail(`Trace input ${index} has unsupported fields.`);
		} else if (event.type === 'output') {
			if (Object.keys(event).sort().join(',') !== 'packet,type') fail(`Trace output ${index} has unsupported fields.`);
			if (typeof event.packet !== 'string') fail(`Trace output ${index} packet is malformed.`);
			const [channel] = packetParts(event.packet);
			if (!['update', 'sideupdate', 'end'].includes(channel)) fail(`Trace output ${index} uses unsupported channel ${channel}.`);
			if (normalizePacket(event.packet) !== event.packet) fail(`Trace output ${index} is not clock-normalized.`);
			if (channel === 'update') {
				for (const audience of ['p1', 'p2', 'spectator']) projectPackets([event.packet], audience);
			} else if (channel === 'sideupdate') {
				for (const audience of ['p1', 'p2']) projectPackets([event.packet], audience);
			}
		} else {
			fail(`Trace event ${index} has an unknown type.`);
		}
	}
	validateInputSequence(trace.events, seeds);
	validateOutcome(trace.events, trace.outcome);
	return { seeds, outcome: trace.outcome, eventCount: trace.events.length };
}

function recordingStream(stream, events) {
	return {
		write(command) {
			events.push({ type: 'input', command });
			return stream.write(command);
		},
		[Symbol.asyncIterator]() {
			const output = stream[Symbol.asyncIterator]();
			return {
				async next() {
					const next = await output.next();
					if (!next.done) events.push({ type: 'output', packet: normalizePacket(next.value) });
					return next;
				},
			};
		},
		destroy() { return stream.destroy(); },
	};
}

async function captureTrace({ ps, seeds, limits } = {}) {
	if (!ps?.BattleStream || !ps?.Teams || !ps?.Dex) fail('Pinned Showdown API is required for trace capture.');
	const events = [];
	const runner = require('../environment/battle-runner.cjs');
	const result = await runner.runBattle({
		ps,
		seeds,
		limits,
		streamFactory: () => recordingStream(new ps.BattleStream({ keepAlive: true }), events),
	});
	const trace = {
		schemaVersion: 1,
		kind: TRACE_KIND,
		upstreamCommit: LOCK.commit,
		formatId: FORMAT_ID,
		seeds: result.summary.seeds,
		outcome: { result: result.summary.outcome.result, winner: result.summary.outcome.winner, turns: result.summary.outcome.turns },
		events,
	};
	validateTrace(trace);
	return { summary: { formatId: FORMAT_ID, upstreamCommit: LOCK.commit, seeds: trace.seeds, outcome: trace.outcome, events: events.length }, trace };
}

function writeWithDeadline(stream, command, deadline, timeoutMs) {
	const remaining = deadline - Date.now();
	if (remaining <= 0) return Promise.reject(new Error(`Trace replay exceeded ${timeoutMs} ms.`));
	let timer;
	return Promise.race([
		Promise.resolve(stream.write(command)),
		new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(`Trace replay exceeded ${timeoutMs} ms.`)), remaining); }),
	]).finally(() => { if (timer) clearTimeout(timer); });
}

function readWithDeadline(output, deadline, timeoutMs) {
	const remaining = deadline - Date.now();
	if (remaining <= 0) return Promise.reject(new Error(`Trace replay exceeded ${timeoutMs} ms.`));
	let timer;
	return Promise.race([
		output.next(),
		new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(`Trace replay exceeded ${timeoutMs} ms.`)), remaining); }),
	]).finally(() => { if (timer) clearTimeout(timer); });
}

function closeWithDeadline(stream, deadline, timeoutMs) {
	const remaining = deadline - Date.now();
	if (remaining <= 0) return Promise.reject(new Error(`Trace replay exceeded ${timeoutMs} ms.`));
	let timer;
	return Promise.race([
		Promise.resolve(stream.writeEnd()),
		new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(`Trace replay exceeded ${timeoutMs} ms.`)), remaining); }),
	]).finally(() => { if (timer) clearTimeout(timer); });
}

async function replayTrace(trace, { ps, expectedCommit = LOCK.commit, limits = {} } = {}) {
	if (!ps?.BattleStream) fail('Pinned BattleStream API is required for trace replay.');
	const bounds = { ...DEFAULT_LIMITS, ...limits };
	if (!Number.isFinite(bounds.timeoutMs) || bounds.timeoutMs <= 0 || !Number.isInteger(bounds.maxEvents) || bounds.maxEvents <= 0) fail('Replay limits must be positive finite values.');
	const verified = validateTrace(trace, expectedCommit, { maxEvents: bounds.maxEvents });
	const deadline = Date.now() + bounds.timeoutMs;
	const stream = new ps.BattleStream({ keepAlive: true });
	const output = stream[Symbol.asyncIterator]();
	let streamClosed = false;
	try {
		for (const [index, event] of trace.events.entries()) {
			if (Date.now() > deadline) fail(`Trace replay exceeded ${bounds.timeoutMs} ms.`);
			if (event.type === 'input') {
				await writeWithDeadline(stream, event.command, deadline, bounds.timeoutMs);
				continue;
			}
			const next = await readWithDeadline(output, deadline, bounds.timeoutMs);
			if (next.done) fail(`Trace mismatch at event ${index}: simulator ended before expected output.`);
			const actual = normalizePacket(next.value);
			if (actual !== event.packet) fail(mismatchMessage(index, event.packet, actual));
		}
		// The pinned BattleStream can push one last buffered update after end; explicitly
		// end the object stream only after consuming the recorded terminal packet.
		await closeWithDeadline(stream, deadline, bounds.timeoutMs);
		streamClosed = true;
		const trailing = await readWithDeadline(output, deadline, bounds.timeoutMs);
		if (!trailing.done) {
			const [channel] = packetParts(normalizePacket(trailing.value));
			fail(`Unexpected trailing simulator output after trace event ${trace.events.length - 1} (channel ${channel}).`);
		}
		return { schemaVersion: 1, upstreamCommit: expectedCommit, formatId: FORMAT_ID, outcome: verified.outcome, eventsReplayed: verified.eventCount };
	} finally {
		if (!streamClosed) await stream.destroy();
	}
}

async function main(args = process.argv.slice(2)) {
	const oracle = require('./oracle.cjs');
	oracle.assertReady();
	const ps = require('./vendor/pokemon-showdown/dist/sim');
	const [command, ...rest] = args;
	if (command === '--help' || command === 'help') {
		console.log('Usage: node reference/trace.cjs capture --output trace.json [team/battle/policy seed options] | replay trace.json');
		return;
	}
	if (command === 'capture') {
		const options = require('../environment/battle-runner.cjs').parseArgs(rest, oracle.parseSeed);
		if (!options.outputPath) fail('Capture requires --output trace.json.');
		const result = await captureTrace({ ps, seeds: options.seeds });
		fs.writeFileSync(path.resolve(options.outputPath), `${JSON.stringify(result.trace, null, 2)}\n`);
		process.stdout.write(`${JSON.stringify(result.summary, null, 2)}\n`);
		return;
	}
	if (command === 'replay') {
		if (rest.length !== 1) fail('Replay requires exactly one trace path.');
		const trace = JSON.parse(fs.readFileSync(path.resolve(rest[0]), 'utf8'));
		const result = await replayTrace(trace, { ps });
		process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
		return;
	}
	fail(`Unknown trace command: ${command || '(missing)'}`);
}

if (require.main === module) {
	main().catch(error => {
		console.error(`Showdown trace tool failed: ${error.message}`);
		process.exitCode = 1;
	});
}

module.exports = { DEFAULT_LIMITS, FORMAT_ID, TRACE_KIND, captureTrace, firstDifference, main, mismatchMessage, normalizePacket, replayTrace, validateTrace };
