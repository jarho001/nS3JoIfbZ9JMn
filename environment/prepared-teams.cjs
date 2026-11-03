'use strict';

const crypto = require('node:crypto');

const FORMAT_ID = 'gen5randombattle';
const UPSTREAM_COMMIT = require('../reference-lock.json').commit;
const HANDLES = new WeakMap();

function copySeed(seed) {
	if (!Array.isArray(seed) || seed.length !== 4) {
		throw new TypeError('Team seed must contain exactly four integer limbs in the range 0..65535.');
	}
	const copy = [];
	for (let index = 0; index < 4; index++) {
		const limb = seed[index];
		if (!Number.isInteger(limb) || limb < 0 || limb > 65_535) {
			throw new TypeError('Team seed must contain exactly four integer limbs in the range 0..65535.');
		}
		copy.push(limb);
	}
	return copy;
}

function prepareTeam(ps, seed) {
	const teamSeed = copySeed(seed);
	if (!ps?.Teams || typeof ps.Teams.generate !== 'function' || typeof ps.Teams.pack !== 'function') {
		throw new TypeError('Showdown Teams.generate and Teams.pack are required to prepare a team.');
	}
	if (!ps?.Dex?.formats || typeof ps.Dex.formats.get !== 'function' || ps.Dex.formats.get(FORMAT_ID)?.exists !== true) {
		throw new TypeError(`Pinned Showdown format ${FORMAT_ID} is required to prepare a team.`);
	}
	const team = ps.Teams.generate(FORMAT_ID, { seed: [...teamSeed] });
	const packedTeam = ps.Teams.pack(team);
	if (typeof packedTeam !== 'string' || packedTeam.length === 0) {
		throw new TypeError('Showdown produced an invalid packed team.');
	}
	const publicSeed = Object.freeze([...teamSeed]);
	const handle = Object.freeze({
		schemaVersion: 1,
		kind: 'prepared-showdown-team',
		formatId: FORMAT_ID,
		upstreamCommit: UPSTREAM_COMMIT,
		seed: publicSeed,
		packedTeamSha256: crypto.createHash('sha256').update(packedTeam, 'utf8').digest('hex'),
	});
	HANDLES.set(handle, Object.freeze({
		ps,
		formatId: FORMAT_ID,
		upstreamCommit: UPSTREAM_COMMIT,
		seed: publicSeed,
		packedTeam,
	}));
	return handle;
}

function resolvePreparedTeams(ps, preparedTeams, expectedSeeds, formatId, upstreamCommit) {
	if (!preparedTeams || typeof preparedTeams !== 'object' || Array.isArray(preparedTeams)) {
		throw new TypeError('Prepared teams must supply both p1 and p2 handles.');
	}
	const keys = Object.keys(preparedTeams).sort();
	if (keys.length !== 2 || keys[0] !== 'p1' || keys[1] !== 'p2') {
		throw new TypeError('Prepared teams must supply exactly p1 and p2 handles.');
	}
	const packed = {};
	for (const player of ['p1', 'p2']) {
		const data = HANDLES.get(preparedTeams[player]);
		const seedKey = player === 'p1' ? 'teamP1' : 'teamP2';
		const expected = expectedSeeds[seedKey];
		if (!data || data.ps !== ps || data.formatId !== formatId || data.upstreamCommit !== upstreamCommit ||
			data.seed.length !== expected.length || data.seed.some((limb, index) => limb !== expected[index])) {
			throw new TypeError(`Prepared ${player} team is not authentic for this Showdown API, format, revision, and player seed.`);
		}
		packed[player] = data.packedTeam;
	}
	return Object.freeze(packed);
}

module.exports = { prepareTeam, resolvePreparedTeams };
