'use strict';

function clone(value) {
	return value === undefined ? undefined : structuredClone(value);
}

function deepFreeze(value) {
	if (!value || typeof value !== 'object' || Object.isFrozen(value)) return value;
	Object.freeze(value);
	for (const entry of Object.values(value)) deepFreeze(entry);
	return value;
}

function createPolicyRng(seed) {
	let state = 0x9e3779b9;
	for (const limb of seed) {
		state = Math.imul(state ^ limb, 0x85ebca6b) >>> 0;
		state ^= state >>> 13;
	}
	if (!state) state = 0x6d2b79f5;
	return function random() {
		state = (state + 0x6d2b79f5) >>> 0;
		let value = state;
		value = Math.imul(value ^ value >>> 15, value | 1);
		value ^= value + Math.imul(value ^ value >>> 7, value | 61);
		return ((value ^ value >>> 14) >>> 0) / 0x100000000;
	};
}

function isFainted(pokemon) {
	if (!pokemon || /(?:^|\s)fnt(?:$|\s)/i.test(pokemon.condition || '')) return true;
	const match = /^(\d+)\/(\d+)/.exec(pokemon.condition || '');
	return Boolean(match && Number(match[1]) === 0);
}

function validateRoster(request) {
	const roster = request.side?.pokemon;
	if (!Array.isArray(roster) || roster.length < 1) throw new Error('Malformed choice request: own roster is missing.');
	for (const pokemon of roster) {
		if (!pokemon || typeof pokemon !== 'object' || typeof pokemon.active !== 'boolean' || typeof pokemon.condition !== 'string' || !pokemon.condition) {
			throw new Error('Malformed choice request: own roster entry is incomplete.');
		}
		if (pokemon.condition !== 'fnt' && !/^\d+(?:\/\d+)?(?:\s+[a-z]+)*$/i.test(pokemon.condition)) {
			throw new Error('Malformed choice request: own roster condition is invalid.');
		}
	}
	return roster;
}

function candidateActions(request, excluded = []) {
	if (!request || typeof request !== 'object' || Array.isArray(request)) throw new Error('Malformed choice request.');
	if (request.wait !== undefined && typeof request.wait !== 'boolean') throw new Error('Malformed choice request: wait must be boolean.');
	const sidePokemon = validateRoster(request);
	if (request.wait) {
		if (Object.hasOwn(request, 'active') || Object.hasOwn(request, 'forceSwitch')) throw new Error('Malformed wait request: it also contains an actionable request.');
		return [];
	}
	const candidates = [];
	const isActive = pokemon => pokemon?.active === true;
	const availableSwitches = () => (sidePokemon || [])
		.flatMap((pokemon, index) => !isActive(pokemon) && !isFainted(pokemon) ? [`switch ${index + 1}`] : []);

	if (Object.hasOwn(request, 'forceSwitch')) {
		if (!Array.isArray(request.forceSwitch) || request.forceSwitch.length !== 1 || typeof request.forceSwitch[0] !== 'boolean') {
			throw new Error('Malformed Gen 5 singles forced-switch request.');
		}
		if (request.forceSwitch.some(Boolean)) candidates.push(...availableSwitches());
		if (request.forceSwitch.some(Boolean) && sidePokemon.filter(pokemon => pokemon.active).length !== 1) {
			throw new Error('Malformed forced-switch request: expected one active singles slot.');
		}
		if (request.forceSwitch.some(Boolean) && !candidates.length) throw new Error('Forced switch request has no available own replacement.');
	} else {
		if (!Array.isArray(request.active) || request.active.length !== 1 || !request.active[0] || typeof request.active[0] !== 'object') {
			throw new Error('Expected a Gen 5 singles move request with one active slot.');
		}
		const active = request.active[0];
		if (sidePokemon.filter(pokemon => pokemon.active).length !== 1) throw new Error('Malformed move request: own roster must identify one active singles slot.');
		if (!Array.isArray(active.moves)) throw new Error('Malformed move request: active move slots are missing.');
		for (const [index, move] of active.moves.entries()) {
			// Showdown reduces locked/recharging move requests to a single {move, id} row.
			if (active.moves.length === 1 && move && typeof move.move === 'string' && typeof move.id === 'string' &&
				move.disabled === undefined && move.pp === undefined) {
				candidates.push('move 1');
				continue;
			}
			if (!move || typeof move !== 'object' || typeof move.disabled !== 'boolean' || !Number.isFinite(move.pp) || move.pp < 0) {
				throw new Error('Malformed move request: move slot is incomplete.');
			}
			if (move.disabled || move.pp <= 0) continue;
			candidates.push(`move ${index + 1}`);
		}
		if (active.trapped !== undefined && typeof active.trapped !== 'boolean') throw new Error('Malformed move request: trapped must be boolean.');
		if (active.maybeTrapped !== undefined && typeof active.maybeTrapped !== 'boolean') throw new Error('Malformed move request: maybeTrapped must be boolean.');
		if (active?.trapped !== true) candidates.push(...availableSwitches());
	}
	const excludedSet = new Set(excluded);
	const available = candidates.filter(action => !excludedSet.has(action));
	if (!available.length && !(Array.isArray(request.forceSwitch) && !request.forceSwitch.some(Boolean))) {
		throw new Error('Choice request has no legal move or switch candidate.');
	}
	return available;
}

function randomPolicy({ history, request, random }) {
	if (!history || !Array.isArray(history) || typeof random !== 'function') throw new Error('Malformed policy input.');
	const episode = [...history].reverse().find(record => record.type === 'requestEpisode')?.id;
	const rejected = history.filter(record => record.type === 'ownChoiceResolution' && record.status === 'rejected' && record.episodeId === episode).map(record => record.command);
	const candidates = candidateActions(request, rejected);
	if (!candidates.length) return null;
	return candidates[Math.floor(random() * candidates.length)];
}

function policyInput(history, request, random) {
	return deepFreeze({ history: Object.freeze(history.slice()), request: clone(request), random });
}

module.exports = { candidateActions, createPolicyRng, deepFreeze, policyInput, randomPolicy };
