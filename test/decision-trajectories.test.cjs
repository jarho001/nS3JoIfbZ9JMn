'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const oracle = require('../reference/oracle.cjs');
oracle.assertReady();
const ps = require('../reference/vendor/pokemon-showdown/dist/sim');
const { actionSpace } = require('../environment/action-space.cjs');
const { collectDecisionTrajectories } = require('../environment/decision-trajectories.cjs');
const { DEFAULT_SEEDS, runBattle } = require('../environment/battle-runner.cjs');
const { prepareTeam } = require('../environment/prepared-teams.cjs');

function roster(id) {
	return Array.from({ length: 6 }, (_, index) => ({
		ident: `${id}: slot ${index + 1}`,
		active: index === 0,
		condition: index === 3 ? '0 fnt' : '100/100',
	}));
}

function moveRequest(id, { rqid = 0, moveCount = 2, marker } = {}) {
	const moves = [
		{ move: 'Tackle', id: 'tackle', pp: 35, disabled: false },
		{ move: 'Growl', id: 'growl', pp: 40, disabled: false },
		{ move: 'Recover', id: 'recover', pp: 10, disabled: false },
		{ move: 'Slam', id: 'slam', pp: 20, disabled: false },
	].slice(0, moveCount);
	return {
		rqid,
		active: [{ moves }],
		side: { id, pokemon: roster(id), ...(marker === undefined ? {} : { privateMarker: marker }) },
	};
}

function waitRequest(id) {
	return { wait: true, side: { id, pokemon: roster(id) } };
}

function forcedRequest(id) {
	const pokemon = roster(id);
	pokemon[0].condition = '0 fnt';
	return { forceSwitch: [true], side: { id, pokemon } };
}

function noActionForcedRequest(id) {
	return { forceSwitch: [false], side: { id, pokemon: roster(id) } };
}

function metadata(winner, { hiddenEnd = 'constant' } = {}) {
	return JSON.stringify({ winner: winner || '', turns: 2, p1: 'p1', p2: 'p2', hiddenEnd });
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
		Dex: { formats: { get(formatId) { return { id: formatId, exists: formatId === 'gen5randombattle' }; } } },
		Teams: {
			generate(formatId, { seed }) { return { formatId, seed: [...seed] }; },
			pack(team) { return `${team.formatId}:${team.seed.join(',')}`; },
		},
	};
}

function readyPackets(winner, middle = []) {
	return [
		'update\n|gen|5\n|gametype|singles\n|turn|1',
		...middle,
		`update\n|win|${winner || ''}`,
		`end\n${metadata(winner)}`,
	];
}

function mainScenario(winner = 'p1') {
	const p1 = moveRequest('p1', { rqid: 0 });
	const consecutive = moveRequest('p1', { rqid: 1 });
	const p2Wait = waitRequest('p2');
	const p2NoAction = noActionForcedRequest('p2');
	const forced = forcedRequest('p1');
	const p2 = moveRequest('p2', { rqid: 2 });
	return [
		'update\n|gen|5\n|gametype|singles\n|turn|1',
		`sideupdate\np1\n|request|${JSON.stringify(p1)}`,
		`sideupdate\np2\n|request|${JSON.stringify(p2Wait)}`,
		`sideupdate\np1\n|request|${JSON.stringify(consecutive)}`,
		`sideupdate\np1\n|request|${JSON.stringify(consecutive)}`,
		'sideupdate\np1\n|error|[Unavailable choice] The selected move cannot be used now.',
		`sideupdate\np1\n|request|${JSON.stringify(consecutive)}`,
		`sideupdate\np2\n|request|${JSON.stringify(p2Wait)}`,
		'update\n|move|p1a: Ownmon|Tackle|p2a: Othermon\n|turn|2',
		`sideupdate\np2\n|request|${JSON.stringify(p2Wait)}`,
		`sideupdate\np2\n|request|${JSON.stringify(p2NoAction)}`,
		`sideupdate\np1\n|request|${JSON.stringify(forced)}`,
		'update\n|switch|p1a: Ownmon|Pikachu, L100|100/100\n|turn|3',
		`sideupdate\np1\n|request|${JSON.stringify(p1)}`,
		`sideupdate\np2\n|request|${JSON.stringify(p2)}`,
		'update\n|move|p2a: Othermon|Growl|p1a: Ownmon\n|turn|4',
		`update\n|win|${winner}`,
		`end\n${metadata(winner)}`,
	];
}

function policyFor(p1Commands, p2Commands, callbackHistories = {}, callbackInputs = {}) {
	return {
		p1: input => {
			assert.deepEqual(Object.keys(input).sort(), ['history', 'random', 'request']);
			assert.equal(input.request.side.id, 'p1');
			callbackInputs.p1?.push(input);
			callbackHistories.p1?.push(structuredClone(input.history));
			const episode = [...input.history].reverse().find(record => record.type === 'requestEpisode')?.id;
			const rejected = input.history
				.filter(record => record.type === 'ownChoiceResolution' && record.status === 'rejected' && record.episodeId === episode)
				.map(record => record.command);
			const command = rejected.includes('move 1') ? 'move 2' : input.request.forceSwitch?.[0] ? 'switch 2' : 'move 1';
			p1Commands?.push(command);
			return command;
		},
		p2: input => {
			assert.deepEqual(Object.keys(input).sort(), ['history', 'random', 'request']);
			assert.equal(input.request.side.id, 'p2');
			callbackInputs.p2?.push(input);
			callbackHistories.p2?.push(structuredClone(input.history));
			const command = input.request.side.privateMarker === 'other-hidden' ? 'move 2' : 'move 1';
			p2Commands?.push(command);
			return command;
		},
	};
}

test('history deltas reconstruct each exact callback prefix across consecutive choices, wait, retry and later identical requests', async () => {
	const stream = new FakeStream(mainScenario());
	const p1Commands = [];
	const p2Commands = [];
	const callbackHistories = { p1: [], p2: [] };
	const callbackInputs = { p1: [], p2: [] };
	const result = await collectDecisionTrajectories({
		ps: fakePs(), streamFactory: () => stream, policies: policyFor(p1Commands, p2Commands, callbackHistories, callbackInputs),
	});

	assert.equal(stream.destroyed, true);
	assert.deepEqual(p1Commands, ['move 1', 'move 1', 'move 2', 'switch 2', 'move 1']);
	assert.deepEqual(p2Commands, ['move 1']);
	assert.deepEqual(Object.keys(result), ['schemaVersion', 'kind', 'actionSpaceSchemaVersion', 'formatId', 'upstream', 'outcome', 'players', 'targets']);
	assert.equal(result.schemaVersion, 1);
	assert.equal(result.actionSpaceSchemaVersion, 1);
	assert.equal(result.kind, 'gen5-player-decision-trajectories');
	assert.equal(result.formatId, 'gen5randombattle');
	assert.deepEqual(result.upstream, {
		repository: require('../reference-lock.json').repository,
		commit: require('../reference-lock.json').commit,
	});
	assert.deepEqual(result.outcome, { result: 'win', winner: 'p1' });
	assert.deepEqual(result.targets.terminalReward, { p1: 1, p2: -1 });
	assert.deepEqual(Object.keys(result.players.p1), ['player', 'steps']);
	assert.deepEqual(Object.keys(result.players.p2), ['player', 'steps']);
	assert.deepEqual(Object.keys(result.players), ['p1', 'p2']);
	assert.deepEqual(Object.keys(result.upstream).sort(), ['commit', 'repository']);
	assert.deepEqual(Object.keys(result.outcome).sort(), ['result', 'winner']);
	assert.deepEqual(Object.keys(result.targets), ['terminalReward']);
	assert.deepEqual(Object.keys(result.targets.terminalReward), ['p1', 'p2']);
	const roundTripped = JSON.parse(JSON.stringify(result));
	assert.deepEqual(Object.keys(roundTripped), Object.keys(result));
	assert.equal(roundTripped.schemaVersion, 1);
	assert.equal(roundTripped.kind, 'gen5-player-decision-trajectories');
	assert.deepEqual(roundTripped.targets.terminalReward, { p1: 1, p2: -1 });
	assert.equal(JSON.stringify(result).includes('trusted omniscient'), false);
	assert.equal(JSON.stringify(result).includes('endMetadata'), false);
	assert.equal(JSON.stringify(result).includes('seeds'), false);
	assert.equal(JSON.stringify(result).includes('packedTeam'), false);

	const p1 = result.players.p1.steps;
	assert.equal(p1.length, 5);
	assert.deepEqual(p1.map(step => step.decisionIndex), [0, 1, 2, 3, 4]);
	assert.deepEqual(p1.map(step => step.episodeId), [2, 2, 2, 3, 4]);
	assert.deepEqual(p1.map(step => step.requestKind), ['move', 'move', 'move', 'forced_switch', 'move']);
	assert.deepEqual(p1.map(step => step.action.command), p1Commands);
	assert.deepEqual(p1.map(step => step.action.slot), [0, 0, 1, 5, 0]);
	assert.equal(p1[0].observation.request.rqid, p1[4].observation.request.rqid, 'the same request payload remains a new decision in a later episode');
	assert.equal(p1[2].mask[0], false, 'the rejected action is excluded only in the current episode');
	assert.equal(p1[2].mask[1], true);
	assert.equal(p1.every(step => step.behaviorProbability === null), true);
	assert.equal(p1.some(step => Object.hasOwn(step, 'status')), false, 'a selected action is not labelled as submitted or accepted');

	const p2 = result.players.p2.steps;
	assert.equal(p2.length, 1, 'non-actionable wait requests are not policy decisions');
	assert.equal(p2[0].episodeId, 4);
	const reconstructed = [];
	for (const step of p2) {
		reconstructed.push(...step.observation.historyDelta);
		assert.equal(JSON.stringify(reconstructed).includes('wait'), true);
	}
	for (const trajectory of Object.values(result.players)) {
		for (const [index, step] of trajectory.steps.entries()) {
			assert.deepEqual(Object.keys(step).sort(), ['action', 'behaviorProbability', 'decisionIndex', 'episodeId', 'mask', 'observation', 'requestKind']);
			assert.deepEqual(Object.keys(step.observation).sort(), ['historyDelta', 'request']);
			assert.deepEqual(Object.keys(step.action).sort(), ['command', 'slot']);
			assert.equal(step.mask.length, 11);
			assert.equal(step.mask[step.action.slot], true);
			assert.equal(typeof step.action.command, 'string');
			assert.ok(step.observation.request);
			assert.ok(Array.isArray(step.observation.historyDelta));
			assert.equal(step.behaviorProbability, null);
			const reconstructed = trajectory.steps.slice(0, index + 1)
				.flatMap(prior => prior.observation.historyDelta);
			assert.deepEqual(reconstructed, callbackHistories[trajectory.player][index]);
		}
	}
	assert.equal(p2[0].observation.historyDelta.some(record => record.type === 'request' && record.request.forceSwitch?.[0] === false), true);
	for (const [player, trajectory] of Object.entries(roundTripped.players)) {
		let serializedHistory = [];
		for (const [index, step] of trajectory.steps.entries()) {
			assert.equal(step.mask.length, 11);
			assert.equal(step.mask[step.action.slot], true);
			assert.equal(step.action.command, (player === 'p1' ? p1Commands : p2Commands)[index]);
			assert.deepEqual(step.observation.request, JSON.parse(JSON.stringify(callbackInputs[player][index].request)));
			serializedHistory = serializedHistory.concat(step.observation.historyDelta);
			assert.deepEqual(serializedHistory, JSON.parse(JSON.stringify(callbackHistories[player][index])));
		}
	}

	const firstInput = callbackInputs.p1[0];
	const firstSample = p1[0];
	assert.notStrictEqual(firstSample.observation.request, firstInput.request);
	assert.notStrictEqual(firstSample.observation.request.side, firstInput.request.side);
	assert.notStrictEqual(firstSample.observation.historyDelta, firstInput.history);
	assert.notStrictEqual(firstSample.observation.historyDelta[0], firstInput.history[0]);
	const callbackRequestBeforeMutation = structuredClone(firstInput.request);
	const callbackHistoryBeforeMutation = structuredClone(firstInput.history);
	firstSample.observation.request.side.id = 'mutated-output';
	firstSample.observation.historyDelta.find(record => record.type === 'request').request.side.id = 'mutated-output';
	assert.deepEqual(firstInput.request, callbackRequestBeforeMutation);
	assert.deepEqual(firstInput.history, callbackHistoryBeforeMutation);
});

test('terminal rewards stay in fixed seat perspective for p1 win, p2 win, and a tie', async () => {
	for (const [winner, expected] of [
		['p1', { p1: 1, p2: -1 }],
		['p2', { p1: -1, p2: 1 }],
		['', { p1: 0, p2: 0 }],
	]) {
		const requests = [
			`sideupdate\np1\n|request|${JSON.stringify(moveRequest('p1'))}`,
			`sideupdate\np2\n|request|${JSON.stringify(moveRequest('p2'))}`,
		];
		const packets = [
			'update\n|gen|5\n|gametype|singles\n|turn|1',
			...requests,
			`update\n${winner ? `|win|${winner}` : '|tie|'}`,
			`end\n${metadata(winner)}`,
		];
		const result = await collectDecisionTrajectories({ ps: fakePs(), streamFactory: () => new FakeStream(packets) });
		assert.deepEqual(result.targets.terminalReward, expected);
		for (const player of ['p1', 'p2']) {
			assert.equal(Object.hasOwn(result.players[player].steps[0].observation, 'reward'), false);
			assert.equal(Object.hasOwn(result.players[player].steps[0].observation, 'outcome'), false);
		}
	}
});

test('opponent private requests, split secrets, pending choices and end metadata do not change the other seat trajectory', async () => {
	const makePackets = secret => {
		const p2 = moveRequest('p2', { rqid: secret === 'hidden-a' ? 1 : 2, moveCount: 2, marker: secret });
		return [
			'update\n|gen|5\n|gametype|singles\n|turn|1',
			`sideupdate\np2\n|request|${JSON.stringify(p2)}`,
			`update\n|split|p2\n|-boost|p2a: Othermon|atk|${secret}\n|-boost|p2a: Othermon|atk|public-same`,
			'sideupdate\np1\n|request|' + JSON.stringify(moveRequest('p1')),
			'update\n|win|p1',
			`end\n${metadata('p1', { hiddenEnd: secret })}`,
		];
	};
	const firstP2 = [];
	const secondP2 = [];
	const first = await collectDecisionTrajectories({
		ps: fakePs(), streamFactory: () => new FakeStream(makePackets('hidden-a')),
		policies: policyFor([], firstP2),
	});
	const second = await collectDecisionTrajectories({
		ps: fakePs(), streamFactory: () => new FakeStream(makePackets('other-hidden')),
		policies: policyFor([], secondP2),
	});
	assert.deepEqual(firstP2, ['move 1']);
	assert.deepEqual(secondP2, ['move 2']);
	assert.deepEqual(first.players.p1.steps, second.players.p1.steps);
	assert.equal(JSON.stringify(first.players.p1).includes('public-same'), true);
	assert.equal(JSON.stringify(first.players.p1).includes('hidden-a'), false);
	assert.equal(JSON.stringify(first.players.p1).includes('other-hidden'), false);
	assert.equal(JSON.stringify(first.players.p1).includes('hiddenEnd'), false);
	assert.deepEqual(first.targets.terminalReward, second.targets.terminalReward);
});

test('configuration, callback, choice, protocol and runner failures return only a fixed error without partial data', async () => {
	const packets = readyPackets('p1', [`sideupdate\np1\n|request|${JSON.stringify(moveRequest('p1'))}`]);
	const cases = [
		() => collectDecisionTrajectories({ get ps() { throw new Error('PRIVATE_CONFIG_SENTINEL'); } }),
		() => collectDecisionTrajectories({ ps: fakePs(), policies: { p1: null } }),
		() => collectDecisionTrajectories({ ps: fakePs(), policies: { p1: false } }),
		() => collectDecisionTrajectories({ ps: fakePs(), streamFactory: () => new FakeStream(packets), policies: { p1: () => { throw new Error('PRIVATE_POLICY_SENTINEL'); } } }),
		() => collectDecisionTrajectories({ ps: fakePs(), streamFactory: () => new FakeStream(packets), policies: { p1: () => 'move 4' } }),
		() => collectDecisionTrajectories({ ps: fakePs(), streamFactory: () => new FakeStream(['update\n|gen|5']) }),
	];
	for (const invoke of cases) {
		await assert.rejects(invoke(), error => {
			assert.equal(error.message, 'Decision trajectory collection failed.');
			assert.equal(Object.hasOwn(error, 'cause'), false);
			assert.equal(Object.hasOwn(error, 'resultRecord'), false);
			assert.equal(error.message.includes('SENTINEL'), false);
			return true;
		});
	}
});

test('guarded pinned completed battle reproduces the unwrapped selected command sequences', { timeout: 120_000 }, async () => {
	const seeds = structuredClone(DEFAULT_SEEDS);
	const baseline = await runBattle({ ps, seeds: structuredClone(seeds) });
	const preparedTeams = {
		p1: prepareTeam(ps, structuredClone(seeds.teamP1)),
		p2: prepareTeam(ps, structuredClone(seeds.teamP2)),
	};
	const collected = await collectDecisionTrajectories({ ps, seeds: structuredClone(seeds), preparedTeams });
	assert.deepEqual(collected.outcome, {
		result: baseline.summary.outcome.result,
		winner: baseline.summary.outcome.winner,
	});
	assert.deepEqual(collected.targets.terminalReward, {
		p1: baseline.summary.outcome.result === 'tie' ? 0 : baseline.summary.outcome.winner === 'p1' ? 1 : -1,
		p2: baseline.summary.outcome.result === 'tie' ? 0 : baseline.summary.outcome.winner === 'p2' ? 1 : -1,
	});
	for (const player of ['p1', 'p2']) {
		const transcriptCommands = baseline.transcript.records
			.filter(record => record.type === 'choice' && record.player === player)
			.map(record => record.command);
		assert.deepEqual(collected.players[player].steps.map(step => step.action.command), transcriptCommands);
		assert.ok(collected.players[player].steps.length > 0);
		for (const step of collected.players[player].steps) {
			assert.equal(step.observation.request.side.id, player);
			for (const record of step.observation.historyDelta) {
				if (record.type === 'request') assert.equal(record.request.side.id, player);
			}
			assert.equal(step.mask[step.action.slot], true);
		}
	}
});
