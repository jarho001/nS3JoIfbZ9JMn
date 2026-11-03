'use strict';

const referenceLock = require('../reference-lock.json');
const { actionSpace, decodeAction } = require('./action-space.cjs');
const { randomPolicy } = require('./random-policy.cjs');
const { runBattle } = require('./battle-runner.cjs');

const PLAYERS = ['p1', 'p2'];
const PLAYER_KEYS = new Set(['ps', 'seeds', 'limits', 'preparedTeams', 'streamFactory', 'policies']);
const KIND = 'gen5-player-decision-trajectories';
const FORMAT_ID = 'gen5randombattle';

function fail() {
	throw new Error('Decision trajectory collection failed.');
}

function currentEpisodeInfo(history) {
	let episodeId;
	for (let index = history.length - 1; index >= 0; index--) {
		const record = history[index];
		if (record?.type === 'requestEpisode') {
			episodeId = record.id;
			break;
		}
	}
	if (!Number.isSafeInteger(episodeId) || episodeId < 1) fail();
	return {
		episodeId,
		rejectedCommands: history
			.filter(record => record?.type === 'ownChoiceResolution' && record.status === 'rejected' && record.episodeId === episodeId)
			.map(record => record.command),
	};
}

function rewardFor(result, player) {
	if (!result || !['win', 'tie'].includes(result.result)) fail();
	if (result.result === 'tie') {
		if (result.winner !== null) fail();
		return 0;
	}
	if (!PLAYERS.includes(result.winner)) fail();
	return result.winner === player ? 1 : -1;
}

function createRecorder(player, choose, trajectories, historyLengths) {
	return input => {
		try {
			if (!input || typeof input !== 'object' || !Array.isArray(input.history) || !input.request || typeof input.request !== 'object') fail();
			const history = input.history;
			const priorLength = historyLengths[player];
			if (!Number.isSafeInteger(priorLength) || history.length < priorLength) fail();
			const { episodeId: currentEpisode, rejectedCommands: rejected } = currentEpisodeInfo(history);
			const space = actionSpace(input.request, rejected);
			if (!space.mask.some(Boolean)) fail();

			// Snapshot only the new own-view records. Earlier knowledge stays in the
			// preceding deltas rather than being copied into every sample.
			const historyDelta = structuredClone(history.slice(priorLength));
			const request = structuredClone(input.request);
			historyLengths[player] = history.length;
			const command = choose(input);
			if (typeof command !== 'string') fail();
			const slot = space.commands.indexOf(command);
			if (slot < 0 || !space.mask[slot] || decodeAction(input.request, slot, rejected) !== command) fail();

			const trajectory = trajectories[player];
			trajectory.steps.push({
				decisionIndex: trajectory.steps.length,
				episodeId: currentEpisode,
				requestKind: space.requestKind,
				observation: { historyDelta, request },
				mask: [...space.mask],
				action: { slot, command },
				behaviorProbability: null,
			});
			return command;
		} catch {
			fail();
		}
	};
}

/**
 * Collect bounded per-decision own-view deltas and terminal targets from one
 * completed runBattle call. The combined result is post-game private training
 * data and must not be sent back to either policy as an observation.
 */
async function collectDecisionTrajectories(options = {}) {
	try {
		if (!options || (typeof options !== 'object' && typeof options !== 'function') || Array.isArray(options)) fail();
		const copied = {};
		for (const key of PLAYER_KEYS) copied[key] = options[key];
		const suppliedPolicies = copied.policies;
		if (suppliedPolicies !== undefined && (!suppliedPolicies || (typeof suppliedPolicies !== 'object' && typeof suppliedPolicies !== 'function'))) fail();

		const trajectories = Object.fromEntries(PLAYERS.map(player => [player, { steps: [] }]));
		const historyLengths = { p1: 0, p2: 0 };
		const policies = {};
		for (const player of PLAYERS) {
			const choose = suppliedPolicies?.[player];
			if (choose !== undefined && typeof choose !== 'function') fail();
			policies[player] = createRecorder(player, choose === undefined ? randomPolicy : choose, trajectories, historyLengths);
		}

		const result = await runBattle({ ...copied, policies });
		if (result?.resultRecord?.schemaVersion !== 1 || result.resultRecord.kind !== 'battle-run-result' || result.resultRecord.status !== 'completed') fail();
		if (result.summary.upstreamCommit !== referenceLock.commit || result.summary.formatId !== FORMAT_ID) fail();
		const outcome = result?.summary?.outcome;
		if (!outcome || !['win', 'tie'].includes(outcome.result) || !Number.isSafeInteger(outcome.turns) || outcome.turns < 0) fail();
		if (outcome.result === 'tie' ? outcome.winner !== null : !PLAYERS.includes(outcome.winner)) fail();
		return {
			schemaVersion: 1,
			kind: KIND,
			actionSpaceSchemaVersion: 1,
			formatId: FORMAT_ID,
			upstream: { repository: referenceLock.repository, commit: referenceLock.commit },
			outcome: { result: outcome.result, winner: outcome.winner },
			players: Object.fromEntries(PLAYERS.map(player => [player, {
				player,
				steps: trajectories[player].steps,
			}])),
			targets: { terminalReward: Object.fromEntries(PLAYERS.map(player => [player, rewardFor(outcome, player)])) },
		};
	} catch {
		fail();
	}
}

module.exports = { collectDecisionTrajectories };
