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
const { DEFAULT_SEEDS, parseTerminal, runBattle } = require('../environment/battle-runner.cjs');
const { candidateActions, createPolicyRng, policyInput, randomPolicy } = require('../environment/random-policy.cjs');

const ROOT = path.resolve(__dirname, '..');

function moveRequest(rqid = undefined, override = {}) {
	return {
		...(rqid === undefined ? {} : { rqid }),
		active: [{ moves: [{ move: 'Tackle', id: 'tackle', pp: 35, disabled: false }] }],
		side: { pokemon: [
			{ active: true, condition: '100/100', details: 'Ownmon' },
			{ active: false, condition: '100/100', details: 'Benchmon' },
			{ active: false, condition: '0 fnt', details: 'Faintedmon' },
		] },
		...override,
	};
}

function waitRequest() {
	return { wait: true, side: { pokemon: [{ active: true, condition: '100/100', details: 'Ownmon' }] } };
}

function metadata(winner, extra = {}) {
	return JSON.stringify({ winner, turns: 2, p1: 'p1', p2: 'p2', score: winner === 'p1' ? [1, 0] : [0, 1], ...extra });
}

class FakeStream {
	constructor(packets) {
		this.packets = packets;
		this.writes = [];
		this.destroyed = false;
	}
	write(message) { this.writes.push(message); }
	[Symbol.asyncIterator]() {
		let index = 0;
		return { next: async () => index < this.packets.length ? { value: this.packets[index++], done: false } : { value: undefined, done: true } };
	}
	async destroy() { this.destroyed = true; }
}

function fakePs() {
	return {
		BattleStream: class {},
		Dex: {},
		Teams: {
			generate: () => Array.from({ length: 6 }, (_, index) => ({ species: `Teammon${index}` })),
			pack: () => 'fake-team',
		},
	};
}

function fixedTeamSet(species, ability, moves) {
	return {
		species, ability, moves, level: 80,
		evs: { hp: 252, atk: 4, def: 252, spa: 0, spd: 0, spe: 0 },
		ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 },
	};
}

const TRAPPING_TEAMS = [
	[
		fixedTeamSet('Swanna', 'Keen Eye', ['Scald', 'Roost', 'Brave Bird', 'Toxic']),
		fixedTeamSet('Ludicolo', 'Swift Swim', ['Surf', 'Giga Drain', 'Ice Beam', 'Rain Dance']),
		fixedTeamSet('Snorlax', 'Thick Fat', ['Body Slam', 'Rest', 'Sleep Talk', 'Earthquake']),
		fixedTeamSet('Golem', 'Sturdy', ['Earthquake', 'Rock Slide', 'Explosion', 'Stealth Rock']),
		fixedTeamSet('Arcanine', 'Intimidate', ['Flare Blitz', 'Extreme Speed', 'Crunch', 'Will-O-Wisp']),
		fixedTeamSet('Roserade', 'Natural Cure', ['Giga Drain', 'Sludge Bomb', 'Sleep Powder', 'Spikes']),
	],
	[
		fixedTeamSet('Wobbuffet', 'Shadow Tag', ['Counter', 'Mirror Coat', 'Encore', 'Destiny Bond']),
		fixedTeamSet('Gengar', 'Levitate', ['Shadow Ball', 'Focus Blast', 'Sludge Bomb', 'Thunderbolt']),
		fixedTeamSet('Machamp', 'No Guard', ['Dynamic Punch', 'Stone Edge', 'Payback', 'Bullet Punch']),
		fixedTeamSet('Jolteon', 'Volt Absorb', ['Thunderbolt', 'Shadow Ball', 'Signal Beam', 'Volt Switch']),
		fixedTeamSet('Gyarados', 'Intimidate', ['Waterfall', 'Earthquake', 'Ice Fang', 'Dragon Dance']),
		fixedTeamSet('Tyranitar', 'Sand Stream', ['Crunch', 'Stone Edge', 'Earthquake', 'Fire Blast']),
	],
];

function runFake(packets, options = {}) {
	const stream = new FakeStream(packets);
	const result = runBattle({ ps: fakePs(), streamFactory: () => stream, ...options });
	return result.then(value => ({ ...value, stream }));
}

test('candidate actions follow only request fields for moves, switches, forced switches, waits and trapping', () => {
	assert.deepEqual(candidateActions(moveRequest()), ['move 1', 'switch 2']);
	assert.deepEqual(candidateActions(moveRequest(undefined, { active: [{ trapped: true, moves: [{ move: 'Tackle', pp: 10, disabled: false }] }] })), ['move 1']);
	assert.deepEqual(candidateActions(moveRequest(undefined, { active: [{ maybeTrapped: true, moves: [{ move: 'Tackle', pp: 10, disabled: false }] }] })), ['move 1', 'switch 2']);
	const forced = {
		forceSwitch: [true],
		side: { pokemon: [
			{ active: true, condition: '0 fnt' },
			{ active: false, condition: '100/100' },
			{ active: false, condition: '0 fnt' },
		] },
	};
	assert.deepEqual(candidateActions(forced), ['switch 2']);
	assert.deepEqual(candidateActions(waitRequest()), []);
	assert.throws(() => candidateActions({ side: { pokemon: [{ active: true, condition: '100/100' }] } }), /singles move request/);
	assert.throws(() => candidateActions(moveRequest(undefined, { active: [{}] })), /move request: active move slots/);
	assert.throws(() => candidateActions(moveRequest(undefined, { side: { pokemon: [{ active: true }] } })), /roster entry is incomplete/);
	assert.throws(() => candidateActions(moveRequest(undefined, { active: [{ moves: [] }], side: { pokemon: [{ active: true, condition: '100/100' }] } })), /no legal move or switch/);
	assert.deepEqual(candidateActions(moveRequest(undefined, { active: [{ moves: [{ move: 'Recharge', pp: 1, disabled: false }] }], side: { pokemon: [{ active: true, condition: '100/100' }] } })), ['move 1']);
	assert.deepEqual(candidateActions(moveRequest(undefined, { active: [{ moves: [{ move: 'Outrage', id: 'outrage' }] }], side: { pokemon: [{ active: true, condition: '100/100' }] } })), ['move 1']);
	assert.deepEqual(candidateActions(moveRequest(undefined, { active: [{ moves: [{ move: 'Struggle', pp: 1, disabled: false }] }], side: { pokemon: [{ active: true, condition: '100/100' }] } })), ['move 1']);
	assert.deepEqual(candidateActions({ forceSwitch: [false], side: { pokemon: [{ active: true, condition: '100/100' }] } }), []);
	assert.throws(() => candidateActions({ forceSwitch: [true, false], side: { pokemon: [{ active: true, condition: '0 fnt' }, { active: false, condition: '100/100' }] } }), /singles forced-switch request/);
});

test('request episodes permit identical repeated requests after progress and mark submissions resolved', async () => {
	const repeated = moveRequest();
	const stream = new FakeStream([
		'update\n|gen|5\n|gametype|singles\n|turn|1',
		`sideupdate\np1\n|request|${JSON.stringify(repeated)}`,
		`sideupdate\np1\n|request|${JSON.stringify(repeated)}`,
		`sideupdate\np2\n|request|${JSON.stringify(waitRequest())}`,
		'update\n|move|p1a: Ownmon|Tackle|p2a: Other\n|turn|2',
		`sideupdate\np1\n|request|${JSON.stringify(repeated)}`,
		`sideupdate\np2\n|request|${JSON.stringify(waitRequest())}`,
		'update\n|win|p1',
		`end\n${metadata('p1')}`,
	]);
	const result = await runBattle({ ps: fakePs(), streamFactory: () => stream, policies: { p1: () => 'move 1' } });
	assert.deepEqual(stream.writes.filter(write => write.startsWith('>p1 ')), ['>p1 move 1', '>p1 move 1']);
	assert.equal(result.summary.decisions, 2);
	assert.ok(result.transcript.records.some(record => record.type === 'choiceResolution' && record.status === 'accepted'));
	assert.equal(stream.destroyed, true);
});

test('opponent-only retries cannot change later policy history or random actions', async () => {
	const observe = async withOpponentRetry => {
		const p1Inputs = [];
		const p1Actions = [];
		const p2 = moveRequest(31);
		const packets = [
			`sideupdate\np2\n|request|${JSON.stringify(p2)}`,
		];
		if (withOpponentRetry) {
			packets.push('sideupdate\np2\n|error|[Unavailable choice] hidden opponent retry');
			packets.push(`sideupdate\np2\n|request|${JSON.stringify({ ...p2, update: true })}`);
		}
		packets.push(
			'update\n|turn|1',
			`sideupdate\np1\n|request|${JSON.stringify(moveRequest(41))}`,
			`sideupdate\np2\n|request|${JSON.stringify(waitRequest())}`,
			'update\n|move|p1a: Ownmon|Tackle|p2a: Other\n|turn|2',
			`sideupdate\np1\n|request|${JSON.stringify(moveRequest(41))}`,
			`sideupdate\np2\n|request|${JSON.stringify(waitRequest())}`,
			'update\n|move|p1a: Ownmon|Tackle|p2a: Other\n|win|p1',
			`end\n${metadata('p1')}`,
		);
		const policy = input => {
			p1Inputs.push(JSON.parse(JSON.stringify({ history: input.history, request: input.request })));
			const action = randomPolicy(input);
			p1Actions.push(action);
			return action;
		};
		await runFake(packets, { policies: { p1: policy } });
		return { p1Inputs, p1Actions };
	};
	assert.deepEqual(await observe(false), await observe(true));
});

test('one-sided forced replacement submits while the other player waits', async () => {
	const forcedRequest = {
		forceSwitch: [true],
		side: { pokemon: [
			{ active: true, condition: '0 fnt' },
			{ active: false, condition: '100/100' },
		] },
	};
	const stream = new FakeStream([
		`sideupdate\np1\n|request|${JSON.stringify(forcedRequest)}`,
		`sideupdate\np2\n|request|${JSON.stringify(waitRequest())}`,
		'update\n|switch|p1a: Benchmon|Benchmon, L80|100/100\n|win|p1',
		`end\n${metadata('p1')}`,
	]);
	let p2Calls = 0;
	const result = await runBattle({ ps: fakePs(), streamFactory: () => stream, policies: { p1: input => candidateActions(input.request)[0], p2: () => { p2Calls++; return 'move 1'; } } });
	assert.deepEqual(stream.writes.filter(write => write.startsWith('>p1 ')), ['>p1 switch 2']);
	assert.equal(p2Calls, 0);
	assert.equal(result.summary.outcome.winner, 'p1');
});

test('no-choice forced-switch requests do not call a policy', async () => {
	const stream = new FakeStream([
		'sideupdate\np1\n|request|{"forceSwitch":[false],"side":{"pokemon":[{"active":true,"condition":"100/100"}]}}',
		`sideupdate\np2\n|request|${JSON.stringify(waitRequest())}`,
		'update\n|win|p1',
		`end\n${metadata('p1')}`,
	]);
	let calls = 0;
	await runBattle({ ps: fakePs(), streamFactory: () => stream, policies: { p1: () => { calls++; return 'move 1'; } } });
	assert.equal(calls, 0);
	assert.deepEqual(stream.writes.filter(write => /^>p[12] /.test(write)), []);
});

test('coordinator rejects malformed wait requests before treating them as no-choice', async () => {
	const malformed = [
		{ wait: 'yes', side: { pokemon: [{ active: true, condition: '100/100' }] } },
		{ wait: true, active: [{ moves: [] }], side: { pokemon: [{ active: true, condition: '100/100' }] } },
	];
	for (const request of malformed) {
		const stream = new FakeStream([`sideupdate\np1\n|request|${JSON.stringify(request)}`]);
		await assert.rejects(runBattle({ ps: fakePs(), streamFactory: () => stream }), /wait|Malformed/);
		assert.equal(stream.destroyed, true);
	}
});

test('policies receive only their own immutable projected history, request and RNG', async () => {
	const observe = async opponentPayload => {
		const p1Inputs = [];
		const packets = [
			'update\n|gen|5\n|gametype|singles\n|turn|1',
			`sideupdate\np1\n|request|${JSON.stringify(moveRequest(1, { side: { marker: 'OWN_REQUEST', pokemon: [{ active: true, condition: '100/100' }, { active: false, condition: '100/100' }] } }))}`,
			`sideupdate\np2\n|request|${JSON.stringify(moveRequest(1, { secret: opponentPayload, side: { marker: 'OPPONENT_PRIVATE_SENTINEL', pokemon: [{ active: true, condition: '100/100' }] } }))}`,
			'update\n|move|p1a: Ownmon|Tackle|p2a: Other\n|win|p1',
			`end\n${metadata('p1', { privateTestValue: opponentPayload })}`,
		];
		const policy = input => {
			assert.deepEqual(Object.keys(input).sort(), ['history', 'random', 'request']);
			assert.equal(Object.isFrozen(input), true);
			assert.equal(Object.isFrozen(input.history), true);
			assert.equal(Object.isFrozen(input.request), true);
			assert.throws(() => { input.request.marker = 'changed'; }, TypeError);
			p1Inputs.push(JSON.parse(JSON.stringify({ history: input.history, request: input.request })));
			return 'move 1';
		};
		const result = await runFake(packets, { policies: { p1: policy } });
		assert.equal(result.summary.outcome.winner, 'p1');
		return p1Inputs;
	};
	const first = await observe('SECRET_ROSTER_AND_MOVE_A');
	const changedOpponent = await observe('SECRET_ROSTER_AND_MOVE_B');
	assert.deepEqual(first, changedOpponent);
	assert.doesNotMatch(JSON.stringify(first), /OPPONENT_PRIVATE_SENTINEL|SECRET_ROSTER/);
	assert.match(JSON.stringify(first), /OWN_REQUEST/);
});

test('unavailable choice reissues use own retry evidence and bounded episode retries', async () => {
	const initial = moveRequest(9, { active: [{ moves: [
		{ move: 'Tackle', pp: 20, disabled: false }, { move: 'Growl', pp: 40, disabled: false },
	] }] });
	const updated = { ...initial, update: true, active: [{ trapped: true, moves: [{ move: 'Tackle', pp: 20, disabled: false }, { move: 'Growl', pp: 40, disabled: false }] }] };
	const p1Commands = [];
	const policy = input => {
		const episode = [...input.history].reverse().find(record => record.type === 'requestEpisode').id;
		const rejected = input.history.filter(record => record.type === 'ownChoiceResolution' && record.status === 'rejected' && record.episodeId === episode).map(record => record.command);
		const candidates = candidateActions(input.request, rejected);
		const action = candidates[0];
		p1Commands.push(action);
		return action;
	};
	const stream = new FakeStream([
		'update\n|gen|5\n|gametype|singles',
		`sideupdate\np1\n|request|${JSON.stringify(initial)}`,
		`sideupdate\np2\n|request|${JSON.stringify(waitRequest())}`,
		'sideupdate\np1\n|error|[Unavailable choice] request updated',
		`sideupdate\np1\n|request|${JSON.stringify(updated)}`,
		'update\n|move|p1a: Ownmon|Growl|p2a: Other\n|win|p1',
		`end\n${metadata('p1')}`,
	]);
	const result = await runBattle({ ps: fakePs(), streamFactory: () => stream, policies: { p1: policy }, limits: { maxRetriesPerRequest: 1 } });
	assert.deepEqual(p1Commands, ['move 1', 'move 2']);
	assert.deepEqual(stream.writes.filter(write => write.startsWith('>p1 ')), ['>p1 move 1', '>p1 move 2']);
	assert.ok(result.transcript.records.some(record => record.type === 'choiceResolution' && record.status === 'rejected'));
	assert.ok(result.transcript.records.some(record => record.type === 'choiceResolution' && record.status === 'accepted'));
	assert.equal(stream.destroyed, true);
});

test('unavailable-choice retry limit spans corrected request payloads and resets after progress', async () => {
	const initial = moveRequest(5, { active: [{ moves: [
		{ move: 'Tackle', pp: 20, disabled: false }, { move: 'Growl', pp: 40, disabled: false },
	] }] });
	const updated1 = { ...initial, update: true, active: [{ trapped: true, moves: [
		{ move: 'Tackle', pp: 20, disabled: false }, { move: 'Growl', pp: 40, disabled: false },
	] }] };
	const updated2 = { ...updated1, rqid: 6 };
	const stream = new FakeStream([
		`sideupdate\np1\n|request|${JSON.stringify(initial)}`,
		'sideupdate\np1\n|error|[Unavailable choice] first correction',
		`sideupdate\np1\n|request|${JSON.stringify(updated1)}`,
		'sideupdate\np1\n|error|[Unavailable choice] second correction',
		`sideupdate\np1\n|request|${JSON.stringify(updated2)}`,
	]);
	await assert.rejects(runBattle({ ps: fakePs(), streamFactory: () => stream, policies: { p1: input => candidateActions(input.request, input.history.filter(record => record.type === 'ownChoiceResolution' && record.status === 'rejected').map(record => record.command))[0], p2: () => 'move 1' }, limits: { maxRetriesPerRequest: 1 } }), /exceeded 1 unavailable-choice retries/);
	assert.equal(stream.destroyed, true);

	const history = [
		{ type: 'requestEpisode', id: 1 },
		{ type: 'ownChoiceResolution', episodeId: 1, status: 'rejected', command: 'move 1' },
		{ type: 'requestEpisode', id: 2 },
	];
	assert.equal(randomPolicy(policyInput(history, moveRequest(), () => 0)), 'move 1');
});

test('retry budget starts a fresh allowance only after visible battle progress', async () => {
	const initial = moveRequest(3, { active: [{ moves: [
		{ move: 'Tackle', pp: 20, disabled: false }, { move: 'Growl', pp: 40, disabled: false },
	] }] });
	const corrected = { ...initial, update: true };
	const stream = new FakeStream([
		`sideupdate\np1\n|request|${JSON.stringify(initial)}`,
		'sideupdate\np1\n|error|[Unavailable choice] first episode',
		`sideupdate\np1\n|request|${JSON.stringify(corrected)}`,
		'update\n|move|p1a: Ownmon|Growl|p2a: Other\n|turn|2',
		`sideupdate\np1\n|request|${JSON.stringify(initial)}`,
		'sideupdate\np1\n|error|[Unavailable choice] second episode',
		`sideupdate\np1\n|request|${JSON.stringify(corrected)}`,
		'update\n|move|p1a: Ownmon|Growl|p2a: Other\n|win|p1',
		`end\n${metadata('p1')}`,
	]);
	const result = await runBattle({ ps: fakePs(), streamFactory: () => stream, policies: { p1: input => randomPolicy({ ...input, random: () => 0 }) }, limits: { maxRetriesPerRequest: 1 } });
	assert.deepEqual(stream.writes.filter(write => write.startsWith('>p1 ')), ['>p1 move 1', '>p1 move 2', '>p1 move 1', '>p1 move 2']);
	assert.equal(result.summary.outcome.winner, 'p1');
});

test('pinned Gen 5 Showdown emits and recovers from a real unavailable trapping choice', { timeout: 90_000 }, async () => {
	const trappingPs = {
		BattleStream: ps.BattleStream,
		Dex: ps.Dex,
		Teams: {
			generate: (_format, { seed }) => structuredClone(TRAPPING_TEAMS[seed[0] === DEFAULT_SEEDS.teamP1[0] ? 0 : 1]),
			pack: team => ps.Teams.pack(team),
		},
	};
	const p1 = input => {
		const episode = [...input.history].reverse().find(record => record.type === 'requestEpisode')?.id;
		const hadUnavailable = input.history.some(record => record.type === 'ownChoiceResolution' && record.episodeId === episode && record.status === 'rejected');
		if (!hadUnavailable && input.request.active?.[0]?.maybeTrapped) return 'switch 2';
		return randomPolicy(input);
	};
	const result = await runBattle({ ps: trappingPs, policies: { p1 } });
	assert.equal(result.transcript.kind.startsWith('trusted omniscient'), true);
	assert.ok(result.transcript.records.some(record => record.type === 'choiceResolution' && record.status === 'rejected' && record.category === 'unavailable'));
	assert.ok(result.transcript.records.some(record => record.type === 'request' && record.player === 'p1' && record.request.update === true && record.request.active?.[0]?.trapped === true));
	assert.ok(result.transcript.records.some(record => record.type === 'terminal'));
});

test('reference errors, premature end and timeout fail without fabricated outcome and always close stream', async () => {
	for (const [packets, pattern] of [
		[['error\nprivate reference diagnostic'], /reference stream error/],
		[['update\n|gen|5'], /ended before terminal/],
		[['end\n{"winner":"p1","turns":2,"p1":"p1","p2":"p2"}'], /without a public/],
	]) {
		const stream = new FakeStream(packets);
		await assert.rejects(runBattle({ ps: fakePs(), streamFactory: () => stream }), pattern);
		assert.equal(stream.destroyed, true);
	}
	const never = {
		writes: [], destroyed: false,
		write(message) { this.writes.push(message); },
		[Symbol.asyncIterator]() { return { next: () => new Promise(() => {}) }; },
		async destroy() { this.destroyed = true; },
	};
	await assert.rejects(runBattle({ ps: fakePs(), streamFactory: () => never, limits: { timeoutMs: 10 } }), /wall-clock limit/);
	assert.equal(never.destroyed, true);
});

test('terminal evidence rejects conflicting public outcomes and metadata', () => {
	assert.throws(() => parseTerminal(metadata('p1'), { winners: ['p1', 'p2'] }), /conflicting winners/);
	assert.throws(() => parseTerminal(metadata(''), { winners: ['p1'], tie: true }), /both a win and a tie/);
	assert.throws(() => parseTerminal(metadata('p2'), { winners: ['p1'] }), /contradicts public outcome/);
	assert.deepEqual(parseTerminal(metadata(''), { tie: true }), { result: 'tie', winner: null, turns: 2 });
	assert.throws(() => parseTerminal('[]', { winners: ['p1'] }), /metadata is malformed/);
	assert.throws(() => parseTerminal(JSON.stringify({ winner: 'bad', turns: 2, p1: 'p1', p2: 'p2' }), { winners: ['p1'] }), /invalid winner/);
	assert.throws(() => parseTerminal(JSON.stringify({ winner: 'p1', turns: -1, p1: 'p1', p2: 'p2' }), { winners: ['p1'] }), /invalid turn count/);
});

test('random policy is seeded independently and uses its supplied policy RNG', () => {
	const request = moveRequest();
	const first = createPolicyRng([13, 14, 15, 16]);
	const second = createPolicyRng([13, 14, 15, 16]);
	const a = Array.from({ length: 20 }, () => randomPolicy(policyInput([], request, first)));
	const b = Array.from({ length: 20 }, () => randomPolicy(policyInput([], request, second)));
	assert.deepEqual(a, b);
	assert.ok(a.some(action => action.startsWith('switch ')));
	assert.ok(a.some(action => action.startsWith('move ')));
});

test('real seeded battles terminate, include forced replacements and reproduce across fresh processes', { timeout: 120_000 }, async () => {
	const first = await runBattle({ ps });
	assert.equal(first.summary.upstreamCommit, require('../reference-lock.json').commit);
	assert.ok(first.summary.outcome.turns > 0);
	assert.ok(first.summary.actions.move > 0 && first.summary.actions.switch > 0);
	assert.ok(first.transcript.records.some(record => record.type === 'request' && record.request.forceSwitch?.some(Boolean)));
	assert.ok(first.transcript.records.some(record => record.type === 'terminal' && record.outcome.winner === first.summary.outcome.winner));

	const extraSeeds = structuredClone(DEFAULT_SEEDS);
	extraSeeds.teamP1 = [101, 102, 103, 104];
	extraSeeds.teamP2 = [201, 202, 203, 204];
	extraSeeds.battle = [301, 302, 303, 304];
	extraSeeds.policyP1 = [401, 402, 403, 404];
	extraSeeds.policyP2 = [501, 502, 503, 504];
	const second = await runBattle({ ps, seeds: extraSeeds });
	assert.ok(second.summary.outcome.turns > 0);
	assert.ok(second.transcript.records.some(record => record.type === 'request' && record.request.forceSwitch?.some(Boolean)));

	const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'gen5-random-runner-'));
	try {
		const one = path.join(temp, 'one.json');
		const two = path.join(temp, 'two.json');
		for (const output of [one, two]) {
			const child = spawnSync(process.execPath, [path.join(ROOT, 'environment/run-random.cjs'), '--output', output], { cwd: ROOT, encoding: 'utf8', timeout: 60_000 });
			assert.equal(child.status, 0, child.stderr || child.stdout);
		}
		assert.deepEqual(JSON.parse(fs.readFileSync(one, 'utf8')), JSON.parse(fs.readFileSync(two, 'utf8')));
	} finally {
		fs.rmSync(temp, { recursive: true, force: true });
	}
});
