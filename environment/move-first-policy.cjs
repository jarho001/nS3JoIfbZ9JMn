'use strict';

const { ACTION_COUNT, actionSpace, decodeAction } = require('./action-space.cjs');

const POLICY_ID = 'move-first-v1';
const ERROR_MESSAGE = 'move-first-v1 policy could not select an available action.';
const MOVE_SLOTS = [0, 1, 2, 3, 10];

function currentEpisodeRejections(history) {
	let episodeIndex = -1;
	let episodeId;
	for (let index = history.length - 1; index >= 0; index--) {
		const record = history[index];
		if (record?.type === 'requestEpisode') {
			episodeIndex = index;
			episodeId = record.id;
			break;
		}
	}
	if (episodeIndex < 0 || !Number.isSafeInteger(episodeId) || episodeId < 1) throw new Error();
	return history.slice(episodeIndex + 1)
		.filter(record => record?.type === 'ownChoiceResolution' && record.status === 'rejected' && record.episodeId === episodeId)
		.map(record => record.command);
}

/**
 * Uniformly choose a currently available move, including the explicit Struggle
 * slot, before considering any switch. If no move is available, choose a switch.
 * The callback uses only the current request, its current request-episode
 * rejections, and the supplied random function. It keeps no state between calls.
 */
function moveFirstPolicy(input) {
	try {
		if (!input || typeof input !== 'object' || Array.isArray(input) ||
			!Array.isArray(input.history) || !input.request || typeof input.request !== 'object' ||
			Array.isArray(input.request) || typeof input.random !== 'function') {
			throw new Error();
		}

		const request = input.request;
		const initialSpace = actionSpace(request);
		if (initialSpace.requestKind === 'wait' ||
			(initialSpace.requestKind === 'forced_switch' && request.forceSwitch[0] === false)) {
			return null;
		}

		const rejected = currentEpisodeRejections(input.history);
		const space = actionSpace(request, rejected);
		if (!Array.isArray(space.mask) || space.mask.length !== ACTION_COUNT ||
			!Array.isArray(space.commands) || space.commands.length !== ACTION_COUNT) {
			throw new Error();
		}

		const moves = MOVE_SLOTS.filter(slot => space.mask[slot] === true);
		const switches = [];
		for (let slot = 4; slot <= 9; slot++) {
			if (space.mask[slot] === true) switches.push(slot);
		}
		const candidates = moves.length ? moves : switches;
		if (candidates.length === 0 || candidates.some(slot => typeof space.commands[slot] !== 'string')) {
			throw new Error();
		}

		const draw = input.random();
		if (typeof draw !== 'number' || !Number.isFinite(draw) || draw < 0 || draw >= 1) throw new Error();
		const slot = candidates[Math.floor(draw * candidates.length)];
		const command = decodeAction(request, slot, rejected);
		if (command !== space.commands[slot]) throw new Error();
		return command;
	} catch {
		throw new Error(ERROR_MESSAGE);
	}
}

module.exports = { POLICY_ID, moveFirstPolicy };
