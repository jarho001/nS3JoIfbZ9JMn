'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const oracle = require('../reference/oracle.cjs');
oracle.assertReady();
const ps = require('../reference/vendor/pokemon-showdown/dist/sim');
const { actionSpace } = require('../environment/action-space.cjs');
const { collectDecisionTrajectories } = require('../environment/decision-trajectories.cjs');
const { generatePair } = require('../environment/paired-seeds.cjs');
const { prepareTeam } = require('../environment/prepared-teams.cjs');
const { randomPolicy } = require('../environment/random-policy.cjs');
const { POLICY_ID, moveFirstPolicy } = require('../environment/move-first-policy.cjs');

const FAILURE_MESSAGE = 'move-first-v1 policy could not select an available action.';

function roster(overrides = {}) {
	return Array.from({ length: 6 }, (_, index) => ({
		active: index === 0,
		condition: '100/100',
		...overrides[index],
	}));
}

function moveRequest(moves, activeOverrides = {}, rosterOverrides = {}) {
	return {
		active: [{ moves, ...activeOverrides }],
		side: { id: 'p1', pokemon: roster(rosterOverrides) },
	};
}

function move(name, id, { pp = 10, disabled = false } = {}) {
	return { move: name, id, pp, disabled };
}

function policyInput(request, random, history = [{ type: 'requestEpisode', id: 1 }]) {
	return { history, request, random };
}

test('exports the stable identity and uniformly prefers enabled move slots', () => {
	assert.equal(POLICY_ID, 'move-first-v1');
	assert.equal(typeof moveFirstPolicy, 'function');
	const request = moveRequest([move('Tackle', 'tackle'), move('Growl', 'growl')]);
	const before = structuredClone(request);
	const history = [{ type: 'requestEpisode', id: 1 }];
	let draws = 0;
	const choices = [
		moveFirstPolicy(policyInput(request, () => { draws++; return 0; }, history)),
		moveFirstPolicy(policyInput(request, () => { draws++; return 0.999999; }, history)),
	];
	assert.deepEqual(choices, ['move 1', 'move 2']);
	assert.equal(draws, 2);
	assert.deepEqual(request, before);
	assert.deepEqual(history, [{ type: 'requestEpisode', id: 1 }]);
});

test('uses the sole enabled move even when switches are available and draws once', () => {
	const request = moveRequest([
		move('Tackle', 'tackle', { disabled: true }),
		move('Growl', 'growl', { pp: 0 }),
		move('Recover', 'recover'),
	]);
	const space = actionSpace(request);
	assert.ok(space.mask.slice(4, 10).some(Boolean));
	let draws = 0;
	assert.equal(moveFirstPolicy(policyInput(request, () => { draws++; return 0.5; })), 'move 3');
	assert.equal(draws, 1);
});

test('supports the reduced locked row and explicit Struggle move slot', () => {
	const locked = moveRequest([{ move: 'Outrage', id: 'outrage' }]);
	let draws = 0;
	assert.equal(moveFirstPolicy(policyInput(locked, () => { draws++; return 0.7; })), 'move 1');
	assert.equal(draws, 1);
	const recharge = moveRequest([{ move: 'Recharge', id: 'recharge' }]);
	assert.equal(moveFirstPolicy(policyInput(recharge, () => { draws++; return 0.2; })), 'move 1');
	assert.equal(draws, 2);

	const struggle = moveRequest([{ move: 'Struggle', id: 'struggle', target: 'randomNormal', disabled: false }]);
	assert.equal(actionSpace(struggle).mask[10], true);
	assert.equal(moveFirstPolicy(policyInput(struggle, () => { draws++; return 0; })), 'move 1');
	assert.equal(draws, 3);
});

test('falls back uniformly to legal switches when moves are unavailable', () => {
	const request = moveRequest([
		move('Tackle', 'tackle', { disabled: true }),
		move('Growl', 'growl', { pp: 0 }),
	]);
	let draws = 0;
	assert.equal(moveFirstPolicy(policyInput(request, () => { draws++; return 0; })), 'switch 2');
	assert.equal(moveFirstPolicy(policyInput(request, () => { draws++; return 0.999999; })), 'switch 6');
	assert.equal(draws, 2);

	const forced = { forceSwitch: [true], side: { id: 'p1', pokemon: roster() } };
	assert.equal(moveFirstPolicy(policyInput(forced, () => { draws++; return 0.999999; })), 'switch 6');
	assert.equal(draws, 3);
});

test('respects fainted replacements, trapping, and disclosed maybeTrapped state', () => {
	const forced = { forceSwitch: [true], side: { id: 'p1', pokemon: roster({ 1: { condition: '0 fnt' } }) } };
	const forcedMask = actionSpace(forced).mask;
	assert.equal(forcedMask[4], false, 'the active Pokemon cannot be its own replacement');
	assert.equal(forcedMask[5], false, 'a fainted Pokemon cannot replace the active Pokemon');
	assert.equal(moveFirstPolicy(policyInput(forced, () => 0)), 'switch 3');

	const maybeTrapped = moveRequest([move('Tackle', 'tackle', { disabled: true })], { maybeTrapped: true });
	assert.equal(actionSpace(maybeTrapped).mask[5], true, 'maybeTrapped must not be resolved from concealed state');
	assert.equal(moveFirstPolicy(policyInput(maybeTrapped, () => 0)), 'switch 2');

	const trapped = moveRequest([move('Tackle', 'tackle')], { trapped: true });
	const trappedMask = actionSpace(trapped).mask;
	assert.equal(trappedMask.slice(4, 10).some(Boolean), false);
	assert.equal(moveFirstPolicy(policyInput(trapped, () => 0)), 'move 1');
});

test('keeps rejection exclusions inside the latest request episode', () => {
	const request = moveRequest([move('Tackle', 'tackle'), move('Growl', 'growl')]);
	const history = [
		{ type: 'requestEpisode', id: 1 },
		{ type: 'ownChoiceResolution', episodeId: 1, status: 'rejected', command: 'move 1' },
		{ type: 'requestEpisode', id: 2 },
		{ type: 'ownChoiceResolution', episodeId: 2, status: 'rejected', command: 'move 2' },
	];
	assert.equal(moveFirstPolicy(policyInput(request, () => 0, history)), 'move 1');

	const allMovesRejected = [
		{ type: 'requestEpisode', id: 3 },
		{ type: 'ownChoiceResolution', episodeId: 3, status: 'rejected', command: 'move 1' },
		{ type: 'ownChoiceResolution', episodeId: 3, status: 'rejected', command: 'move 2' },
	];
	assert.equal(moveFirstPolicy(policyInput(request, () => 0, allMovesRejected)), 'switch 2');
});

test('returns null for waits and no-action forced-switch requests without drawing', () => {
	let draws = 0;
	const random = () => { draws++; return 0; };
	assert.equal(moveFirstPolicy(policyInput({ wait: true, side: { id: 'p1', pokemon: roster() } }, random, [])), null);
	assert.equal(moveFirstPolicy(policyInput({ forceSwitch: [false], side: { id: 'p1', pokemon: roster() } }, random, [])), null);
	assert.equal(draws, 0);
});

test('rejects exhausted choices, malformed requests, and invalid draws with one fixed diagnostic', () => {
	const trapped = moveRequest([move('Tackle', 'tackle')], { trapped: true });
	const exhaustedHistory = [
		{ type: 'requestEpisode', id: 1 },
		{ type: 'ownChoiceResolution', episodeId: 1, status: 'rejected', command: 'move 1' },
	];
	const assertSafeFailure = call => assert.throws(call, error => {
		assert.equal(error.message, FAILURE_MESSAGE);
		assert.equal(Object.hasOwn(error, 'cause'), false);
		return true;
	});
	assertSafeFailure(() => moveFirstPolicy(policyInput(trapped, () => 0, exhaustedHistory)));
	assertSafeFailure(() => moveFirstPolicy(policyInput(moveRequest([]), () => 0)));
	assertSafeFailure(() => moveFirstPolicy({ ...policyInput(moveRequest([]), () => 0), secret: 'PRIVATE_REQUEST_SENTINEL' }));
	assertSafeFailure(() => moveFirstPolicy(policyInput(moveRequest([move('Tackle', 'tackle')]), () => { throw new Error('PRIVATE_RNG_SENTINEL'); })));
	assertSafeFailure(() => moveFirstPolicy(policyInput(moveRequest([move('Tackle', 'tackle')]), () => 0, [])));

	for (const invalidDraw of [-0.01, 1, NaN, Infinity, '0']) {
		let draws = 0;
		assertSafeFailure(() => moveFirstPolicy(policyInput(moveRequest([move('Tackle', 'tackle')]), () => {
			draws++;
			return invalidDraw;
		})));
		assert.equal(draws, 1);
	}
});

test('same request and supplied RNG sequence reproduce a stateless choice', () => {
	const request = moveRequest([move('Tackle', 'tackle'), move('Growl', 'growl')]);
	const history = [{ type: 'requestEpisode', id: 1 }];
	const run = () => {
		let state = 0x12345678;
		const random = () => {
			state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
			return state / 0x100000000;
		};
		return moveFirstPolicy(policyInput(structuredClone(request), random, structuredClone(history)));
	};
	assert.equal(run(), run());
});

function assertMoveFirstTrajectory(steps) {
	assert.ok(steps.length > 0);
	for (const step of steps) {
		assert.equal(step.mask.length, 11);
		assert.equal(step.mask[step.action.slot], true);
		const hasMove = step.mask.slice(0, 4).some(Boolean) || step.mask[10] === true;
		if (hasMove) assert.ok(step.action.slot <= 3 || step.action.slot === 10);
		else assert.ok(step.action.slot >= 4 && step.action.slot <= 9);
		assert.match(step.action.command, hasMove ? /^move [1-4]$/ : /^switch [1-6]$/);
	}
}

test('pinned paired battles keep move-first identity and its policy seed across seats', { timeout: 120_000 }, async () => {
	const pair = generatePair([31030, 43029, 1, 0], 0);
	const teamHandles = {
		A: prepareTeam(ps, pair.teams.A.seed),
		B: prepareTeam(ps, pair.teams.B.seed),
	};
	const cleanupRuns = [];
	assert.deepEqual(pair.runs.map(run => run.policySeats.p1), ['policyA', 'policyB']);
	assert.deepEqual(pair.runs.map(run => run.seats.p1Team), ['A', 'B']);

	for (const run of pair.runs) {
		assert.deepEqual(run.seeds.policyP1, pair.policySeeds[run.policySeats.p1]);
		assert.deepEqual(run.seeds.policyP2, pair.policySeeds[run.policySeats.p2]);
		assert.deepEqual(run.seeds.teamP1, pair.teams[run.seats.p1Team].seed);
		assert.deepEqual(run.seeds.teamP2, pair.teams[run.seats.p2Team].seed);
		const preparedTeams = {
			p1: teamHandles[run.seats.p1Team],
			p2: teamHandles[run.seats.p2Team],
		};
		const policies = Object.fromEntries(['p1', 'p2'].map(player => [
			player,
			run.policySeats[player] === 'policyA' ? moveFirstPolicy : randomPolicy,
		]));
		assert.equal(policies[run.policySeats.p1 === 'policyA' ? 'p1' : 'p2'], moveFirstPolicy);
		assert.equal(policies[run.policySeats.p1 === 'policyB' ? 'p1' : 'p2'], randomPolicy);

		const collected = await collectDecisionTrajectories({
			ps,
			seeds: run.seeds,
			preparedTeams,
			policies,
			streamFactory: () => {
				const stream = new ps.BattleStream({ keepAlive: true });
				return {
					write(message) { return stream.write(message); },
					[Symbol.asyncIterator]() { return stream[Symbol.asyncIterator](); },
					async destroy() {
						cleanupRuns.push(run.runId);
						return stream.destroy();
					},
				};
			},
		});

		assert.equal(collected.schemaVersion, 1);
		assert.ok(['win', 'tie'].includes(collected.outcome.result));
		for (const player of ['p1', 'p2']) {
			const expectedReward = collected.outcome.result === 'tie' ? 0 : collected.outcome.winner === player ? 1 : -1;
			assert.equal(collected.targets.terminalReward[player], expectedReward);
			assert.ok(collected.players[player].steps.length > 0);
		}
		const baselinePlayer = run.policySeats.p1 === 'policyA' ? 'p1' : 'p2';
		assertMoveFirstTrajectory(collected.players[baselinePlayer].steps);
	}
	assert.deepEqual(cleanupRuns, pair.runs.map(run => run.runId));
});
