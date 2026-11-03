'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const oracle = require('../reference/oracle.cjs');
oracle.assertReady();
const ps = require('../reference/vendor/pokemon-showdown/dist/sim');
const { DEFAULT_SEEDS, runBattle } = require('../environment/battle-runner.cjs');
const { actionSpace, decodeAction } = require('../environment/action-space.cjs');
const { candidateActions } = require('../environment/random-policy.cjs');

function roster(overrides = {}) {
	return Array.from({ length: 6 }, (_, index) => ({
		active: index === 0,
		condition: index === 2 ? '0 fnt' : '100/100',
		...overrides[index],
	}));
}

function moveRequest(moves, activeOverrides = {}, rosterOverrides = {}) {
	return {
		active: [{ moves, ...activeOverrides }],
		side: { pokemon: roster(rosterOverrides) },
	};
}

function assertRoundTrips(request, expected) {
	const space = actionSpace(request);
	assert.deepEqual(space.mask, expected.map(Boolean));
	for (let slot = 0; slot < space.mask.length; slot++) {
		assert.equal(space.mask[slot], space.commands[slot] !== null);
		if (space.mask[slot]) assert.equal(decodeAction(request, slot), space.commands[slot]);
	}
	return space;
}

function rejectedForCurrentEpisode(history) {
	const episode = [...history].reverse().find(record => record.type === 'requestEpisode')?.id;
	return history
		.filter(record => record.type === 'ownChoiceResolution' && record.status === 'rejected' && record.episodeId === episode)
		.map(record => record.command);
}

function adapterPolicy({ history, request, random }) {
	const rejected = rejectedForCurrentEpisode(history);
	const { mask } = actionSpace(request, rejected);
	const enabled = mask.flatMap((available, slot) => available ? [slot] : []);
	if (!enabled.length) return null;
	return decodeAction(request, enabled[Math.floor(random() * enabled.length)], rejected);
}

test('version-1 maps current move rows and own roster positions without leaking other fields', () => {
	const request = moveRequest([
		{ move: 'Tackle', id: 'tackle', pp: 35, disabled: false },
		{ move: 'Growl', id: 'growl', pp: 40, disabled: true },
		{ move: 'Recover', id: 'recover', pp: 0, disabled: false },
		{ move: 'Slam', id: 'slam', pp: 20, disabled: false },
	], { maybeTrapped: true, privateMarker: 'own-only' });
	const before = structuredClone(request);
	const space = assertRoundTrips(request, [true, false, false, true, false, true, false, true, true, true, false]);
	assert.equal(space.schemaVersion, 1);
	assert.equal(space.requestKind, 'move');
	assert.deepEqual(space.commands, ['move 1', null, null, 'move 4', null, 'switch 2', null, 'switch 4', 'switch 5', 'switch 6', null]);
	assert.equal(JSON.stringify(space).includes('own-only'), false);
	assert.deepEqual(request, before);
});

test('trapped, maybe-trapped, forced replacement, no-action replacement, and wait use request-only masks', () => {
	const moves = [{ move: 'Tackle', id: 'tackle', pp: 35, disabled: false }];
	assert.deepEqual(actionSpace(moveRequest(moves, { trapped: true })).mask,
		[true, false, false, false, false, false, false, false, false, false, false]);
	assert.deepEqual(actionSpace(moveRequest(moves, { maybeTrapped: true })).mask,
		[true, false, false, false, false, true, false, true, true, true, false]);
	const forced = { forceSwitch: [true], side: { pokemon: roster() } };
	const forcedSpace = assertRoundTrips(forced, [false, false, false, false, false, true, false, true, true, true, false]);
	assert.equal(forcedSpace.requestKind, 'forced_switch');
	const noReplacement = actionSpace({ forceSwitch: [false], side: { pokemon: [roster()[0]] } });
	assert.equal(noReplacement.requestKind, 'forced_switch');
	assert.deepEqual(noReplacement.mask, Array(11).fill(false));
	const wait = actionSpace({ wait: true, side: { pokemon: [roster()[0]] } });
	assert.equal(wait.requestKind, 'wait');
	assert.deepEqual(wait.mask, Array(11).fill(false));
	assert.throws(() => decodeAction({ wait: true, side: { pokemon: [roster()[0]] } }, 10), /not available/);
});

test('rejects incompatible wait, move, and forced-switch request fields', () => {
	const ordinaryMoves = [{ move: 'Tackle', id: 'tackle', pp: 35, disabled: false }];
	const ordinary = moveRequest(ordinaryMoves);
	const incompatible = [
		{ ...ordinary, forceSwitch: [true] },
		{ ...ordinary, forceSwitch: [false] },
		{ ...ordinary, wait: true },
		{ ...ordinary, wait: false },
		{ wait: true, forceSwitch: [false], side: { pokemon: roster() } },
		{ wait: true, forceSwitch: [true], side: { pokemon: roster() } },
		{ wait: true, active: [{ moves: ordinaryMoves }], side: { pokemon: roster() } },
		{ wait: false, side: { pokemon: roster() } },
	];
	for (const request of incompatible) {
		assert.throws(() => actionSpace(request), error => error.message === 'Malformed Gen 5 singles action request.');
	}
	assert.equal(actionSpace({ forceSwitch: [false], side: { pokemon: roster() } }).requestKind, 'forced_switch');
	assert.equal(actionSpace({ wait: true, side: { pokemon: roster() } }).requestKind, 'wait');
});

test('reduced rows, explicit Struggle, and current-episode exclusions decode to one command each', () => {
	const locked = moveRequest([{ move: 'Outrage', id: 'outrage' }], { trapped: true });
	assert.deepEqual(actionSpace(locked).mask, [true, false, false, false, false, false, false, false, false, false, false]);
	assert.equal(decodeAction(locked, 0), 'move 1');
	const recharge = moveRequest([{ move: 'Recharge', id: 'recharge' }], { trapped: true });
	assert.equal(decodeAction(recharge, 0), 'move 1');
	const struggle = moveRequest([{ move: 'Struggle', id: 'struggle', target: 'randomNormal', disabled: false }], { trapped: true });
	assert.deepEqual(actionSpace(struggle).mask, [false, false, false, false, false, false, false, false, false, false, true]);
	assert.equal(decodeAction(struggle, 10), 'move 1');
	assert.throws(() => actionSpace(struggle, ['move 1']), /Malformed Gen 5 singles action request/);

	const ordinary = moveRequest([
		{ move: 'Tackle', id: 'tackle', pp: 35, disabled: false },
		{ move: 'Growl', id: 'growl', pp: 40, disabled: false },
	]);
	const excluded = actionSpace(ordinary, ['move 1', 'switch 2']);
	assert.equal(excluded.mask[0], false);
	assert.equal(excluded.mask[5], false);
	assert.equal(excluded.mask[1], true);
	assert.throws(() => decodeAction(ordinary, 0, ['move 1']), /not available/);
	assert.throws(() => actionSpace(ordinary, ['move 8']), /rejected-command list/);
});

test('invalid dimensions and slots fail with bounded diagnostics and no input mutation', () => {
	const request = moveRequest([{ move: 'Tackle', id: 'tackle', pp: 35, disabled: false }]);
	for (const invalid of [-1, 11, 1.5, NaN, '0']) {
		assert.throws(() => decodeAction(request, invalid), /outside the version-1 singles space/);
	}
	assert.throws(() => actionSpace(moveRequest(Array.from({ length: 5 }, (_, i) => ({ move: `M${i}`, id: `m${i}`, pp: 1, disabled: false })))), /Malformed/);
	assert.throws(() => actionSpace({ wait: true, side: { pokemon: Array(7).fill({ active: false, condition: '100/100' }) } }), /Malformed/);
	assert.throws(() => actionSpace(moveRequest([{ move: 'Struggle', id: 'struggle', target: 'randomNormal', disabled: true }])), /Malformed/);
	assert.throws(() => actionSpace(moveRequest([{ move: 'Struggle', id: 'struggle', target: 'normal', disabled: false }])), /Malformed/);
	assert.throws(() => actionSpace(moveRequest([{ move: 'Tackle', id: 'struggle', pp: 1, disabled: false }])), /Malformed/);
	const repeatedStruggle = moveRequest([
		{ move: 'Struggle', id: 'struggle', target: 'randomNormal', disabled: false, pp: 1 },
		{ move: 'Struggle', id: 'struggle', target: 'randomNormal', disabled: false, pp: 1 },
	]);
	const repeatedSpace = actionSpace(repeatedStruggle);
	assert.deepEqual(repeatedSpace.mask, [true, true, false, false, false, true, false, true, true, true, false]);
	assert.deepEqual(repeatedSpace.commands.slice(0, 2), ['move 1', 'move 2']);
	assert.throws(() => actionSpace(moveRequest([])), /Malformed/);
	assert.throws(() => actionSpace({ get wait() { throw new Error('PRIVATE_REQUEST_TEXT'); } }), error => {
		assert.equal(error.message, 'Malformed Gen 5 singles action request.');
		assert.equal(error.message.includes('PRIVATE_REQUEST_TEXT'), false);
		return true;
	});
	const thrownProxy = new Proxy({}, { get() { throw new Error('THROWN_PROXY_TEXT'); } });
	const hostileInputs = [
		() => actionSpace({ get side() { throw thrownProxy; } }),
		() => actionSpace({ get active() { throw 'THROWN_PRIMITIVE_TEXT'; } }),
		() => actionSpace(request, new Proxy([], { get(target, key, receiver) {
			if (key === 'some') throw thrownProxy;
			return Reflect.get(target, key, receiver);
		} })),
	];
	for (const invoke of hostileInputs) {
		assert.throws(invoke, error => error.message === 'Malformed Gen 5 singles action request.');
	}
	assert.deepEqual(request.side.pokemon, roster());
});

async function openPinnedBattle() {
	const stream = new ps.BattleStream({ keepAlive: true });
	const output = stream[Symbol.asyncIterator]();
	const teams = [
		ps.Teams.generate('gen5randombattle', { seed: [...DEFAULT_SEEDS.teamP1] }),
		ps.Teams.generate('gen5randombattle', { seed: [...DEFAULT_SEEDS.teamP2] }),
	];
	stream.write(`>start ${JSON.stringify({ formatid: 'gen5randombattle', seed: [...DEFAULT_SEEDS.battle] })}`);
	for (const [index, player] of ['p1', 'p2'].entries()) {
		stream.write(`>player ${player} ${JSON.stringify({ name: player, team: ps.Teams.pack(teams[index]) })}`);
	}
	const requests = {};
	for (let count = 0; count < 20 && Object.keys(requests).length < 2; count++) {
		let timer;
		const next = await Promise.race([
			output.next(),
			new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('Pinned battle fixture timed out.')), 10_000); }),
		]).finally(() => clearTimeout(timer));
		if (next.done) throw new Error('Pinned battle ended before initial requests.');
		const [channel, ...bodyParts] = next.value.replace(/\r/g, '').split('\n');
		if (channel !== 'sideupdate') continue;
		const [player, ...messages] = bodyParts;
		const line = messages.find(message => message.startsWith('|request|'));
		if (line) requests[player] = JSON.parse(line.slice('|request|'.length));
	}
	assert.equal(Object.keys(requests).length, 2);
	return { stream, requests };
}

test('pinned Showdown move request generation supplies locked, recharge, and Struggle rows accepted by Battle choice handling', { timeout: 30_000 }, async () => {
	const { stream, requests } = await openPinnedBattle();
	try {
		const battle = stream.battle;
		const side = battle.sides[0];
		const pokemon = side.active[0];
		assert.equal(actionSpace(requests.p1).requestKind, 'move');

		assert.equal(pokemon.addVolatile('lockedmove', pokemon, battle.dex.moves.get('outrage')), true);
		let request = { active: [pokemon.getMoveRequestData()], side: side.getRequestData() };
		assert.deepEqual(request.active[0].moves, [{ move: 'Outrage', id: 'outrage' }]);
		assert.equal(side.choose(decodeAction(request, 0)), true);
		assert.equal(side.choice.actions[0].moveid, 'outrage');
		side.clearChoice();
		pokemon.removeVolatile('lockedmove');

		assert.equal(pokemon.addVolatile('mustrecharge', pokemon, battle.dex.moves.get('hyperbeam')), true);
		request = { active: [pokemon.getMoveRequestData()], side: side.getRequestData() };
		assert.deepEqual(request.active[0].moves, [{ move: 'Recharge', id: 'recharge' }]);
		assert.equal(side.choose(decodeAction(request, 0)), true);
		assert.equal(side.choice.actions[0].moveid, 'recharge');
		side.clearChoice();
		pokemon.removeVolatile('mustrecharge');

		const originalPp = pokemon.moveSlots.map(move => move.pp);
		for (const move of pokemon.moveSlots) move.pp = 0;
		request = { active: [pokemon.getMoveRequestData()], side: side.getRequestData() };
		assert.deepEqual(request.active[0].moves, [{ move: 'Struggle', id: 'struggle', target: 'randomNormal', disabled: false }]);
		assert.deepEqual(candidateActions(request), ['move 1']);
		const struggleSpace = actionSpace(request);
		assert.deepEqual(struggleSpace.mask, [false, false, false, false, false, false, false, false, false, false, true]);
		const struggleChoice = decodeAction(request, 10);
		assert.equal(struggleChoice, 'move 1');
		assert.equal(side.choose(struggleChoice), true);
		assert.equal(side.choice.actions[0].moveid, 'struggle');
		for (const [index, move] of pokemon.moveSlots.entries()) move.pp = originalPp[index];
	} finally {
		await stream.destroy();
	}
});

test('pinned complete battle accepts adapter-decoded commands for both seats', { timeout: 90_000 }, async () => {
	const seenKinds = new Set();
	const seenSeats = new Set();
	const policy = input => {
		seenKinds.add(actionSpace(input.request).requestKind);
		const owner = input.request.side.id;
		assert.ok(owner === 'p1' || owner === 'p2');
		seenSeats.add(owner);
		for (const record of input.history) {
			if (record.type === 'request') assert.equal(record.request.side.id, owner);
		}
		const space = actionSpace(input.request, rejectedForCurrentEpisode(input.history));
		const slots = space.mask.flatMap((enabled, slot) => enabled ? [slot] : []);
		if (!slots.length) return null;
		return decodeAction(input.request, slots[Math.floor(input.random() * slots.length)], rejectedForCurrentEpisode(input.history));
	};
	const result = await runBattle({ ps, policies: { p1: policy, p2: policy } });
	assert.equal(result.summary.upstreamCommit, require('../reference-lock.json').commit);
	assert.ok(result.summary.outcome.turns > 0);
	assert.ok(result.transcript.records.some(record => record.type === 'terminal'));
	assert.ok(result.transcript.records.some(record => record.type === 'request' && record.request.forceSwitch?.some(Boolean)));
	assert.ok(result.transcript.records.some(record => record.type === 'request' && record.request.wait === true));
	assert.deepEqual(seenSeats, new Set(['p1', 'p2']));
	assert.ok(seenKinds.has('move'));
	assert.ok(seenKinds.has('forced_switch'));
});
