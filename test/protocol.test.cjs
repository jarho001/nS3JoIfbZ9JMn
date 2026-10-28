'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { projectPackets, ProtocolProjectionError } = require('../environment/protocol.cjs');
const fixturePath = path.join(__dirname, 'fixtures', 'protocol-transcript-v1.json');
const fixture = JSON.parse(fs.readFileSync(fixturePath, 'utf8'));
const audiences = ['p1', 'p2', 'spectator'];

test('hand-authored complete packets produce the exact p1, p2, and spectator views', () => {
	for (const audience of ['p1', 'p2', 'spectator']) {
		assert.deepEqual(projectPackets(fixture.packets, audience), fixture.expected[audience]);
	}
	for (const audience of ['p1', 'p2', 'spectator']) {
		const view = fixture.expected[audience];
		assert.ok(view.records.every(record => !record.message?.startsWith('|split|')));
		assert.ok(!JSON.stringify(view).includes('trusted-seed-sentinel'));
		assert.ok(!JSON.stringify(view).includes('complete-team-sentinel'));
	}
	assert.ok(!JSON.stringify(fixture.expected.p1).includes('hidden-move-sentinel'));
	assert.ok(!JSON.stringify(fixture.expected.p1).includes('private-roster-sentinel'));
	assert.equal(fixture.expected.spectator.records.filter(record => record.type === 'request').length, 0);
});

test('concealed opponent state and private request changes do not alter another recipient projection', () => {
	const baselineP1 = projectPackets(fixture.packets, 'p1');
	const baselineP2 = projectPackets(fixture.packets, 'p2');
	const baselineSpectator = projectPackets(fixture.packets, 'spectator');
	const opponentPrivate = structuredClone(fixture.packets);
	opponentPrivate[0] = opponentPrivate[0].replace('|detailschange|p2a: Opponent|Zoroark, L50', '|detailschange|p2a: Secret Alt|Mew, L50');
	opponentPrivate[4] = opponentPrivate[4].replace('hidden-move-sentinel', 'other-hidden-move').replace('private-roster-sentinel', 'other-private-roster');
	opponentPrivate[6] = 'end\n{"teams":["different-secret-team"],"seed":"different-secret-seed"}';
	assert.deepEqual(projectPackets(opponentPrivate, 'p1'), baselineP1);
	assert.deepEqual(projectPackets(opponentPrivate, 'spectator'), baselineSpectator);
	assert.notDeepEqual(projectPackets(opponentPrivate, 'p2'), baselineP2);
	const malformedOpponentRequest = structuredClone(opponentPrivate);
	malformedOpponentRequest[4] = 'sideupdate\np2\n|request|SECRET_SENTINEL invalid json';
	assert.deepEqual(projectPackets(malformedOpponentRequest, 'p1'), baselineP1);
	assert.deepEqual(projectPackets(malformedOpponentRequest, 'spectator'), baselineSpectator);
	assert.throws(() => projectPackets(malformedOpponentRequest, 'p2'), error => error.code === 'invalid_request_json');

	const hiddenExactHp = structuredClone(fixture.packets);
	hiddenExactHp[0] = hiddenExactHp[0].replace('|-damage|p1a: Active|60/120 brn', '|-damage|p1a: Active|61/122 brn');
	assert.notDeepEqual(projectPackets(hiddenExactHp, 'p1'), baselineP1);
	assert.deepEqual(projectPackets(hiddenExactHp, 'p2'), baselineP2);
	assert.deepEqual(projectPackets(hiddenExactHp, 'spectator'), baselineSpectator);

	const ownerPrivate = structuredClone(fixture.packets);
	ownerPrivate[1] = ownerPrivate[1].replace('|message|p1-private-event', '|message|changed-p1-private-event');
	ownerPrivate[3] = ownerPrivate[3].replace('"tackle"', '"changed-own-move"');
	assert.notDeepEqual(projectPackets(ownerPrivate, 'p1'), baselineP1);
	assert.deepEqual(projectPackets(ownerPrivate, 'p2'), baselineP2);
	assert.deepEqual(projectPackets(ownerPrivate, 'spectator'), baselineSpectator);
});

test('requests are fresh own-recipient records and no placeholder exposes hidden packet counts', () => {
	const hiddenOnly = ['sideupdate\np2\n|request|{"side":{"pokemon":[{"ident":"p2: hidden"}]}}'];
	assert.deepEqual(projectPackets(hiddenOnly, 'p1').records, []);
	assert.deepEqual(projectPackets(hiddenOnly, 'spectator').records, []);
	const input = ['sideupdate\np1\n|request|{"side":{"pokemon":[{"ident":"p1: own"}]}}'];
	const first = projectPackets(input, 'p1');
	first.records[0].request.side.pokemon[0].ident = 'mutated output';
	assert.deepEqual(projectPackets(input, 'p1').records[0].request.side.pokemon[0].ident, 'p1: own');
	assert.equal(input[0].includes('mutated output'), false);
	assert.deepEqual(projectPackets(['sideupdate\np1\n|t:|1700000002'], 'p1').records, []);
});

test('malformed, truncated, unsupported, and error packets fail closed without echoing payloads', () => {
	const cases = [
		[['update\n|split|p3\nSECRET_SENTINEL\nPUBLIC'], 'invalid_split_recipient'],
		[['update\n|split\nSECRET_SENTINEL\nPUBLIC'], 'invalid_split_recipient'],
		[['update\n|split|p1\nSECRET_SENTINEL'], 'truncated_split'],
		[['sideupdate\np1\n|split|p2\nSECRET_SENTINEL\nPUBLIC'], 'unsupported_sideupdate_split'],
		[['sideupdate\np1\n|request|{SECRET_SENTINEL'], 'invalid_request_json'],
		[['sideupdate\np3\nSECRET_SENTINEL'], 'unsupported_sideupdate_recipient'],
		[['update'], 'malformed_packet'],
		[['error\nSECRET_SENTINEL'], 'battle_error'],
		[['unexpected\nSECRET_SENTINEL'], 'unsupported_channel'],
	];
	for (const [packets, code] of cases) {
		assert.throws(() => projectPackets(packets, 'p1'), error => {
			assert.ok(error instanceof ProtocolProjectionError);
			assert.equal(error.code, code);
			assert.equal(error.message.includes('SECRET_SENTINEL'), false);
			return true;
		});
	}
	for (const packets of [
		['update\n|split|p1\n|split|p2\nSECRET_SENTINEL'],
		['update\n|split|p1\nSECRET_SENTINEL\n|split|p2'],
	]) {
		for (const audience of ['p1', 'p2', 'spectator']) {
			assert.throws(() => projectPackets(packets, audience), error => {
				assert.equal(error.code, 'nested_split');
				assert.equal(error.message.includes('SECRET_SENTINEL'), false);
				return true;
			});
		}
	}
	assert.throws(() => projectPackets([], 'p3'), /unsupported_audience/);
});

async function captureSeededInitialPackets(ps, timeoutMs = 10_000) {
	const seeds = { teamP1: [101, 102, 103, 104], teamP2: [201, 202, 203, 204], battle: [301, 302, 303, 304] };
	const teams = [
		ps.Teams.generate('gen5randombattle', { seed: seeds.teamP1 }),
		ps.Teams.generate('gen5randombattle', { seed: seeds.teamP2 }),
	];
	const stream = new ps.BattleStream({ keepAlive: true });
	const packets = [];
	const requestRecipients = new Set();
	const deadline = Date.now() + timeoutMs;
	try {
		stream.write(`>start ${JSON.stringify({ formatid: 'gen5randombattle', seed: seeds.battle })}`);
		for (let index = 0; index < 2; index++) {
			const player = `p${index + 1}`;
			stream.write(`>player ${player} ${JSON.stringify({ name: player, team: ps.Teams.pack(teams[index]) })}`);
		}
		const output = stream[Symbol.asyncIterator]();
		while (requestRecipients.size < 2) {
			const remaining = deadline - Date.now();
			if (remaining <= 0) throw new Error(`Seeded Showdown initialization timed out after ${timeoutMs} ms.`);
			let timer;
			let next;
			try {
				next = await Promise.race([
					output.next(),
					new Promise((_, reject) => {
						timer = setTimeout(() => reject(new Error(`Seeded Showdown initialization timed out after ${timeoutMs} ms.`)), remaining);
					}),
				]);
			} finally {
				if (timer) clearTimeout(timer);
			}
			assert.equal(next.done, false, 'seeded Showdown stream ended before both requests');
			packets.push(next.value);
			const separator = next.value.indexOf('\n');
			assert.ok(separator > 0, 'Showdown emitted a malformed packet');
			const channel = next.value.slice(0, separator);
			if (channel === 'error') throw new Error('Showdown emitted an error packet during seeded initialization.');
			if (channel === 'end') throw new Error('Showdown ended before both players received requests.');
			if (channel !== 'sideupdate') continue;
			const body = next.value.slice(separator + 1);
			const newline = body.indexOf('\n');
			const recipient = body.slice(0, newline);
			if (body.slice(newline + 1).startsWith('|request|')) requestRecipients.add(recipient);
		}
		return packets;
	} finally {
		await stream.destroy();
	}
}

function recordsFromShowdownChunks(chunks) {
	const records = [];
	for (const chunk of chunks) {
		if (!chunk) continue;
		if (chunk.startsWith('|request|')) {
			records.push({ type: 'request', request: JSON.parse(chunk.slice('|request|'.length)) });
			continue;
		}
		if (chunk.startsWith('|error|')) {
			records.push({ type: 'privateEvent', message: chunk });
			continue;
		}
		for (const line of chunk.split('\n')) {
			if (line && !line.startsWith('|t:|')) records.push({ type: 'event', message: line });
		}
	}
	return records;
}

async function showPlayerViews(ps, packets) {
	const source = {
		write() {},
		async *[Symbol.asyncIterator]() {
			yield* packets;
		},
	};
	const routed = ps.getPlayerStreams(source);
	const audiences = ['p1', 'p2', 'spectator'];
	const official = Object.fromEntries(await Promise.all(audiences.map(async audience => {
		const chunks = [];
		for await (const chunk of routed[audience]) chunks.push(chunk);
		return [audience, { schemaVersion: 1, audience, records: recordsFromShowdownChunks(chunks) }];
	})));
	return official;
}

test('pinned Showdown routing agrees with the hand-authored split and empty-branch fixture', async () => {
	const oracle = require('../reference/oracle.cjs');
	oracle.assertReady();
	const referencePath = path.join(__dirname, '..', 'reference', 'vendor', 'pokemon-showdown', 'dist', 'sim');
	const ps = require(referencePath);
	const official = await showPlayerViews(ps, fixture.packets);
	for (const audience of ['p1', 'p2', 'spectator']) {
		assert.deepEqual(projectPackets(fixture.packets, audience).records, official[audience].records);
	}
});

test('seeded Gen 5 packets agree with pinned Showdown routing', async () => {
	const oracle = require('../reference/oracle.cjs');
	oracle.assertReady();
	const referencePath = path.join(__dirname, '..', 'reference', 'vendor', 'pokemon-showdown', 'dist', 'sim');
	const ps = require(referencePath);
	const packets = await captureSeededInitialPackets(ps);
	const publicUpdates = packets.filter(packet => packet.startsWith('update\n')).flatMap(packet => packet.slice('update\n'.length).split('\n'));
	assert.ok(publicUpdates.includes('|gen|5'));
	assert.ok(publicUpdates.includes('|gametype|singles'));
	assert.ok(!publicUpdates.includes('|teampreview'));
	const official = await showPlayerViews(ps, packets);
	for (const audience of audiences) {
		assert.deepEqual(projectPackets(packets, audience), official[audience], `${audience} projection differs from pinned Showdown routing`);
	}
});

test('seeded packet capture fails closed on error/end and a bounded timeout, always destroying the stream', async () => {
	const makePs = packet => {
		let destroyed = false;
		return {
			ps: {
				BattleStream: class {
					write() {}
					[Symbol.asyncIterator]() { return { next: async () => packet }; }
					destroy() { destroyed = true; }
				},
				Teams: { generate: () => Array.from({ length: 6 }, (_, i) => ({ species: `Testmon${i}` })), pack: () => 'fixture' },
			},
			destroyed: () => destroyed,
		};
	};
	const error = makePs({ value: 'error\nPRIVATE_ERROR_SENTINEL', done: false });
	await assert.rejects(captureSeededInitialPackets(error.ps), /error packet/);
	assert.equal(error.destroyed(), true);
	const end = makePs({ value: 'end\nPRIVATE_END_SENTINEL', done: false });
	await assert.rejects(captureSeededInitialPackets(end.ps), /ended before both players/);
	assert.equal(end.destroyed(), true);
	const ended = makePs({ value: undefined, done: true });
	await assert.rejects(captureSeededInitialPackets(ended.ps), /stream ended before both requests/);
	assert.equal(ended.destroyed(), true);

	let resolvePending;
	let destroyed = false;
	const hanging = {
		BattleStream: class {
			write() {}
			[Symbol.asyncIterator]() { return { next: () => new Promise(resolve => { resolvePending = resolve; }) }; }
			destroy() { destroyed = true; resolvePending?.({ done: true }); }
		},
		Teams: { generate: () => Array.from({ length: 6 }, (_, i) => ({ species: `Testmon${i}` })), pack: () => 'fixture' },
	};
	await assert.rejects(captureSeededInitialPackets(hanging, 5), /timed out after 5 ms/);
	assert.equal(destroyed, true);
});
