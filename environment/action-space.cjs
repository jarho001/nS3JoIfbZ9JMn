'use strict';

const { candidateActions } = require('./random-policy.cjs');

const ACTION_COUNT = 11;
const REJECTED_COMMAND = /^(?:move [1-4]|switch [1-6])$/;
const SAFE_ERRORS = new WeakSet();

function fail(message = 'Malformed Gen 5 singles action request.') {
	const error = new Error(message);
	SAFE_ERRORS.add(error);
	throw error;
}

function requestKind(request) {
	if (request.wait === true) return 'wait';
	if (Object.hasOwn(request, 'forceSwitch')) return 'forced_switch';
	return 'move';
}

function slotForCommand(command, request) {
	if (command.startsWith('switch ')) return 3 + Number(command.slice('switch '.length));
	const moveIndex = Number(command.slice('move '.length)) - 1;
	const moves = request.active[0].moves;
	if (moves.length === 1 && moves[0].id === 'struggle' && moves[0].move === 'Struggle') return 10;
	return moveIndex;
}

/**
 * Build the version-1 Gen 5 singles action mask for one current Showdown request.
 * Rejected commands belong to the caller's current request episode. The returned
 * command table is a snapshot and grants no authority after the request changes.
 */
function buildActionSpace(request, rejectedCommands) {
	if (!Array.isArray(rejectedCommands) || rejectedCommands.some(command => typeof command !== 'string' || !REJECTED_COMMAND.test(command))) {
		fail('Malformed rejected-command list.');
	}
	if (!request || typeof request !== 'object' || Array.isArray(request)) fail();
	const hasWait = Object.hasOwn(request, 'wait');
	const hasActive = Object.hasOwn(request, 'active');
	const hasForcedSwitch = Object.hasOwn(request, 'forceSwitch');
	if (hasWait && (request.wait !== true || hasActive || hasForcedSwitch)) fail();
	if (hasActive && hasForcedSwitch) fail();
	if (request.side && Array.isArray(request.side.pokemon) && request.side.pokemon.length > 6) fail();
	if (Array.isArray(request.active) && request.active[0] && Array.isArray(request.active[0].moves)) {
		const moves = request.active[0].moves;
		if (moves.length < 1 || moves.length > 4) fail();
		for (const move of moves) {
			if (!move || typeof move !== 'object') fail();
			if ((move.id === 'struggle') !== (move.move === 'Struggle')) fail();
		}
	}

	let candidates;
	try {
		candidates = candidateActions(request, rejectedCommands);
	} catch {
		fail();
	}

	const mask = Array(ACTION_COUNT).fill(false);
	const commands = Array(ACTION_COUNT).fill(null);
	for (const command of candidates) {
		const slot = slotForCommand(command, request);
		if (!Number.isInteger(slot) || slot < 0 || slot >= ACTION_COUNT || mask[slot]) fail();
		mask[slot] = true;
		commands[slot] = command;
	}

	return {
		schemaVersion: 1,
		requestKind: requestKind(request),
		mask,
		commands,
	};
}

function actionSpace(request, rejectedCommands = []) {
	try {
		return buildActionSpace(request, rejectedCommands);
	} catch (error) {
		if (((typeof error === 'object' && error !== null) || typeof error === 'function') && SAFE_ERRORS.has(error)) throw error;
		fail();
	}
}

/** Decode an enabled slot against the current request and current episode exclusions. */
function decodeAction(request, slot, rejectedCommands = []) {
	if (!Number.isInteger(slot) || slot < 0 || slot >= ACTION_COUNT) fail('Action slot is outside the version-1 singles space.');
	const space = actionSpace(request, rejectedCommands);
	if (!space.mask[slot]) fail('Action slot is not available for this request.');
	return space.commands[slot];
}

module.exports = { ACTION_COUNT, actionSpace, decodeAction };
