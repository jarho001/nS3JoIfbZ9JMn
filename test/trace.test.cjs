'use strict';

const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const oracle = require('../reference/oracle.cjs');
oracle.assertReady();
const ps = require('../reference/vendor/pokemon-showdown/dist/sim');
const lock = require('../reference-lock.json');
const { DEFAULT_SEEDS } = require('../environment/battle-runner.cjs');
const { TRACE_KIND, captureTrace, normalizePacket, replayTrace, validateTrace } = require('../reference/trace.cjs');

const ROOT = path.resolve(__dirname, '..');

function tinyTrace() {
	const seeds = structuredClone(DEFAULT_SEEDS);
	const request = {
		rqid: 1,
		active: [{ moves: [{ move: 'Tackle', id: 'tackle', pp: 35, disabled: false }] }],
		side: { pokemon: [{ active: true, condition: '100/100' }] },
	};
	const end = JSON.stringify({ winner: 'p1', turns: 1, p1: 'p1', p2: 'p2' });
	const events = [
		{ type: 'input', command: `>start ${JSON.stringify({ formatid: 'gen5randombattle', seed: seeds.battle })}` },
		{ type: 'input', command: '>player p1 {"name":"p1","team":"packed-p1"}' },
		{ type: 'input', command: '>player p2 {"name":"p2","team":"packed-p2"}' },
		{ type: 'output', packet: 'update\n|gen|5\n|gametype|singles\n|turn|1' },
		{ type: 'output', packet: `sideupdate\np1\n|request|${JSON.stringify(request)}` },
		{ type: 'input', command: '>p1 move 1' },
		{ type: 'output', packet: 'update\n|move|p1a: Test|Tackle|p2a: Test\n|-damage|p2a: Test|50/100\n|win|p1' },
		{ type: 'output', packet: `end\n${end}` },
	];
	return { schemaVersion: 1, kind: TRACE_KIND, upstreamCommit: lock.commit, formatId: 'gen5randombattle', seeds, outcome: { result: 'win', winner: 'p1', turns: 1 }, events };
}

class ReplayFakeStream {
	constructor(outputs, { trailing = false, hang = false } = {}) {
		this.outputs = outputs;
		this.trailing = trailing;
		this.hang = hang;
		this.writes = [];
		this.destroyed = false;
		this.closed = false;
	}
	write(command) { this.writes.push(command); }
	writeEnd() { this.closed = true; return Promise.resolve(); }
	[Symbol.asyncIterator]() {
		let index = 0;
		return { next: async () => {
			if (this.hang) return new Promise(() => {});
			if (index < this.outputs.length) return { value: this.outputs[index++], done: false };
			if (this.trailing && index++ === this.outputs.length) return { value: 'update\n|turn|999', done: false };
			return { value: undefined, done: true };
		} };
	}
	async destroy() { this.destroyed = true; }
}

function fakePs(outputs, options = {}) {
	const stream = new ReplayFakeStream(outputs, options);
	return { ps: { BattleStream: class { constructor() { return stream; } } }, stream };
}

function traceOutputs(trace) {
	return trace.events.filter(event => event.type === 'output').map(event => event.packet);
}

test('trace clock normalization preserves complete split pairs and structural blanks', () => {
	const first = 'update\n|t:|123\n\n|split|p1\n|-damage|p1a: A|100/100\n|-damage|p1a: A|50/100\n|split|p2\n\n|-heal|p2a: B|80/100\n';
	const second = first.replace('|t:|123', '|t:|987654');
	assert.equal(normalizePacket(first), normalizePacket(second));
	assert.equal(normalizePacket(first), 'update\n\n|split|p1\n|-damage|p1a: A|100/100\n|-damage|p1a: A|50/100\n|split|p2\n\n|-heal|p2a: B|80/100\n');
});

test('trace validation rejects incompatible, unsafe, malformed, truncated and post-terminal data', () => {
	const trace = tinyTrace();
	assert.equal(validateTrace(trace).eventCount, 8);
	assert.throws(() => validateTrace({ ...trace, upstreamCommit: 'wrong' }), /SHA mismatch/);
	assert.throws(() => validateTrace({ ...trace, schemaVersion: 2 }), /Unsupported trace schema/);
	const injected = structuredClone(trace);
	injected.events[5].command = '>eval console.log(1)';
	assert.throws(() => validateTrace(injected), /allowed move or switch/);
	const outputBeforeTeams = structuredClone(trace);
	[outputBeforeTeams.events[0], outputBeforeTeams.events[3]] = [outputBeforeTeams.events[3], outputBeforeTeams.events[0]];
	assert.throws(() => validateTrace(outputBeforeTeams), /start and both player teams before simulator output/);
	const truncated = structuredClone(trace);
	truncated.events.pop();
	assert.throws(() => validateTrace(truncated), /truncated|terminal/i);
	const trailing = structuredClone(trace);
	trailing.events.push({ type: 'output', packet: 'update\n|turn|2' });
	assert.throws(() => validateTrace(trailing), /after terminal evidence/);
});

test('replay reports the first request, damage, or command mismatch and rejects trailing output', async () => {
	const trace = tinyTrace();
	const outputs = traceOutputs(trace);
	const { ps: fake } = fakePs(outputs);
	const changedRequest = structuredClone(trace);
	const requestRecord = changedRequest.events.find(event => event.type === 'output' && event.packet.startsWith('sideupdate\n'));
	const requestJson = JSON.parse(requestRecord.packet.split('\n').at(-1).slice('|request|'.length));
	requestJson.active[0].moves[0].pp++;
	requestRecord.packet = `sideupdate\np1\n|request|${JSON.stringify(requestJson)}`;
	const requestStream = fakePs(outputs);
	await assert.rejects(replayTrace(changedRequest, { ps: requestStream.ps }), /request\.active\.0\.moves\.0\.pp/);
	assert.equal(requestStream.stream.destroyed, true);

	const changedDamage = structuredClone(trace);
	const damageRecord = changedDamage.events.find(event => event.type === 'output' && event.packet.includes('|-damage|'));
	damageRecord.packet = damageRecord.packet.replace('50/100', '49/100');
	const damageStream = fakePs(outputs);
	await assert.rejects(replayTrace(changedDamage, { ps: damageStream.ps }), /channel update, line/);
	assert.equal(damageStream.stream.destroyed, true);

	const changedCommand = structuredClone(trace);
	changedCommand.events.find(event => event.type === 'input' && event.command.startsWith('>p1 ')).command = '>p1 move 2';
	const commandStream = new ReplayFakeStream([
		outputs[0], outputs[1], 'update\n|move|p1a: Test|Tackle|p2a: Test\n|-damage|p2a: Test|0 fnt\n|win|p2', outputs[3],
	]);
	const commandPs = { BattleStream: class { constructor() { return commandStream; } } };
	await assert.rejects(replayTrace(changedCommand, { ps: commandPs }), /Trace mismatch at event 6/);
	assert.equal(commandStream.destroyed, true);

	const extraStream = fakePs(outputs, { trailing: true });
	await assert.rejects(replayTrace(trace, { ps: extraStream.ps }), /Unexpected trailing simulator output/);
	assert.equal(extraStream.stream.closed, true);
});

test('replay has bounded reads, destroys stream on timeout, and needs no team generator or policy API', async () => {
	const trace = tinyTrace();
	const hanging = fakePs([], { hang: true });
	await assert.rejects(replayTrace(trace, { ps: hanging.ps, limits: { timeoutMs: 10 } }), /exceeded 10 ms/);
	assert.equal(hanging.stream.destroyed, true);

	const valid = fakePs(traceOutputs(trace));
	const replayOnlyApi = { BattleStream: valid.ps.BattleStream };
	const result = await replayTrace(trace, { ps: replayOnlyApi });
	assert.equal(result.eventsReplayed, trace.events.length);
	assert.equal(valid.stream.closed, true);
});

test('a complete seeded trusted trace replays in a fresh CLI process without policy/team generation', { timeout: 120_000 }, async () => {
	const captured = await captureTrace({ ps, seeds: DEFAULT_SEEDS });
	assert.equal(captured.trace.schemaVersion, 1);
	assert.equal(captured.trace.upstreamCommit, lock.commit);
	assert.equal(captured.trace.formatId, 'gen5randombattle');
	assert.ok(captured.trace.events.some(event => event.type === 'input' && event.command.startsWith('>player p1 ')));
	assert.ok(captured.trace.events.some(event => event.type === 'output' && event.packet.startsWith('end\n')));

	const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'gen5-battle-trace-'));
	try {
		const file = path.join(temp, 'trace.json');
		fs.writeFileSync(file, `${JSON.stringify(captured.trace)}\n`);
		const result = spawnSync(process.execPath, [path.join(ROOT, 'reference/trace.cjs'), 'replay', file], { cwd: ROOT, encoding: 'utf8', timeout: 60_000 });
		assert.equal(result.status, 0, result.stderr || result.stdout);
		const summary = JSON.parse(result.stdout);
		assert.deepEqual(summary.outcome, captured.trace.outcome);
		assert.equal(summary.eventsReplayed, captured.trace.events.length);
	} finally {
		fs.rmSync(temp, { recursive: true, force: true });
	}
});
