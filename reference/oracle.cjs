#!/usr/bin/env node
'use strict';

const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const { assertReferenceIdentity, inspectCheckout } = require('./setup.cjs');

const root = path.resolve(__dirname, '..');
const lock = JSON.parse(fs.readFileSync(path.join(root, 'reference-lock.json'), 'utf8'));
const checkout = path.join(__dirname, 'vendor', 'pokemon-showdown');
const apiPath = path.join(checkout, 'dist', 'sim');
const DEFAULT_SEEDS = Object.freeze({
	teamP1: [1, 2, 3, 4],
	teamP2: [5, 6, 7, 8],
	battle: [9, 10, 11, 12],
});

function git(args) {
	return execFileSync('git', ['-c', `safe.directory=${checkout}`, ...args], {
		cwd: checkout,
		encoding: 'utf8',
		stdio: ['ignore', 'pipe', 'pipe'],
	}).trim();
}

function assertReady() {
	let actual;
	try {
		actual = inspectCheckout(checkout);
	} catch (error) {
		throw new Error(`Pinned Showdown is unavailable at ${checkout}. Run "node reference/setup.cjs" first. (${error.message})`);
	}
	assertReferenceIdentity(lock, actual);
	if (!fs.existsSync(path.join(apiPath, 'index.js'))) {
		throw new Error('Pinned Showdown is not built. Run "node reference/setup.cjs" first.');
	}
}

function loadShowdown() {
	assertReady();
	return require(apiPath);
}

function modChain(dex, modName) {
	const chain = [];
	let current = modName;
	while (current && !chain.includes(current) && chain.length < 16) {
		chain.push(current);
		current = dex.mod(current).parentMod;
	}
	return chain;
}

function formatManifest(ps) {
	const format = ps.Dex.formats.get('gen5randombattle');
	if (!format.exists) throw new Error('Pinned Showdown does not define gen5randombattle.');
	const ruleTable = ps.Dex.formats.getRuleTable(format);
	const effectiveRules = [...ruleTable.entries()].map(([id, source]) => ({ id, source }));
	const effectiveValues = [...ruleTable.valueRules.entries()].map(([id, value]) => ({ id, value }));
	return {
		schemaVersion: 1,
		formatId: format.id,
		name: format.name,
		generation: ps.Dex.mod(format.mod).gen,
		mod: format.mod,
		modInheritance: modChain(ps.Dex, format.mod),
		gameType: format.gameType,
		teamGenerator: format.team,
		declaredRules: [...format.ruleset],
		declaredBanlist: [...format.banlist],
		declaredUnbanlist: [...format.unbanlist],
		effectiveRules,
		effectiveValueRules: effectiveValues,
		effectiveBans: effectiveRules.filter(rule => rule.id.startsWith('-')),
		teamPreview: ruleTable.has('teampreview'),
		ruleTableNote: 'Effective rule table entries and values from the pinned Showdown revision; this is metadata, not an export of executable mechanics.',
		upstreamCommit: lock.commit,
	};
}

function parseSeed(value, name) {
	const parts = value.split(',');
	if (parts.length !== 4 || parts.some(part => !/^\d{1,5}$/.test(part))) {
		throw new Error(`${name} must be four comma-separated decimal Gen 5 seed limbs in the range 0..65535.`);
	}
	const seed = parts.map(Number);
	if (seed.some(number => number > 65535)) {
		throw new Error(`${name} Gen 5 seed limbs must each be in the range 0..65535.`);
	}
	return seed;
}

function parseSmokeArgs(args) {
	const seeds = { teamP1: [...DEFAULT_SEEDS.teamP1], teamP2: [...DEFAULT_SEEDS.teamP2], battle: [...DEFAULT_SEEDS.battle] };
	const names = {
		'--team-seed-p1': 'teamP1',
		'--team-seed-p2': 'teamP2',
		'--battle-seed': 'battle',
	};
	for (let i = 0; i < args.length; i++) {
		const equalIndex = args[i].indexOf('=');
		const flag = equalIndex < 0 ? args[i] : args[i].slice(0, equalIndex);
		const inlineValue = equalIndex < 0 ? undefined : args[i].slice(equalIndex + 1);
		if (inlineValue?.includes('=')) throw new Error(`Invalid seed option: ${args[i]}`);
		if (!names[flag]) throw new Error(`Unknown smoke option: ${flag}`);
		const value = inlineValue ?? args[++i];
		if (value === undefined || value === '') throw new Error(`Missing seed value for ${flag}.`);
		seeds[names[flag]] = parseSeed(value, flag);
	}
	return seeds;
}

function canonicalize(value) {
	if (Array.isArray(value)) return value.map(canonicalize);
	if (value && typeof value === 'object') {
		return Object.fromEntries(Object.keys(value).sort().map(key => [key, canonicalize(value[key])]));
	}
	return value;
}

function makeTeam(ps, seed) {
	return ps.Teams.generate('gen5randombattle', { seed });
}

function validateInitialBattle(publicEvents, requests, teams, toID) {
	const getValue = (prefix, label) => {
		const line = publicEvents.find(event => event.startsWith(prefix));
		if (!line) throw new Error(`Initial Showdown protocol is missing ${label}.`);
		return line.slice(prefix.length);
	};
	const generation = Number(getValue('|gen|', 'generation'));
	const gameType = getValue('|gametype|', 'game type');
	const teamPreview = publicEvents.includes('|teampreview');
	if (generation !== 5) throw new Error(`Expected Gen 5 from Showdown protocol, received generation ${generation}.`);
	if (gameType !== 'singles') throw new Error(`Expected singles from Showdown protocol, received ${gameType}.`);
	if (teamPreview) throw new Error('Unexpected Team Preview event in gen5randombattle initialization.');
	const teamSizes = Object.fromEntries(publicEvents.flatMap(event => {
		const match = /^\|teamsize\|(p[12])\|(\d+)$/.exec(event);
		return match ? [[match[1], Number(match[2])]] : [];
	}));
	const facts = { generation, gameType, teamPreview };
	for (const [index, player] of ['p1', 'p2'].entries()) {
		if (teamSizes[player] !== 6 || teams[index].length !== 6) {
			throw new Error(`Expected six submitted and reported Pokemon for ${player}.`);
		}
		const request = requests[player];
		if (!request?.side?.pokemon || request.side.pokemon.length !== 6) {
			throw new Error(`Initial request for ${player} does not describe all six submitted Pokemon.`);
		}
		for (let slot = 0; slot < 6; slot++) {
			const reportedSpecies = request.side.pokemon[slot].details?.split(',')[0];
			if (!reportedSpecies || toID(reportedSpecies) !== toID(teams[index][slot].species)) {
				throw new Error(`Initial request for ${player} does not match submitted team slot ${slot + 1}.`);
			}
		}
	}
	return facts;
}

async function initialBattle(ps, teams, battleSeed, timeoutMs = 10_000) {
	const stream = new ps.BattleStream({ keepAlive: true });
	const requests = {};
	const rawUpdateEvents = [];
	let timer;
	try {
		stream.write(`>start ${JSON.stringify({ formatid: 'gen5randombattle', seed: battleSeed })}`);
		for (let i = 0; i < 2; i++) {
			const side = i === 0 ? 'p1' : 'p2';
			stream.write(`>player ${side} ${JSON.stringify({ name: side, team: ps.Teams.pack(teams[i]) })}`);
		}

		const output = stream[Symbol.asyncIterator]();
		while (Object.keys(requests).length < 2) {
			const pendingPacket = output.next();
			const packet = await Promise.race([
				pendingPacket,
				new Promise((_, reject) => {
					timer = setTimeout(() => reject(new Error(`Showdown initialization timed out after ${timeoutMs} ms.`)), timeoutMs);
				}),
			]);
			clearTimeout(timer);
			timer = undefined;
			if (packet.done) throw new Error('Showdown stream ended before both players received initial requests.');
			const [channel, ...bodyParts] = packet.value.replace(/\r/g, '').split('\n');
			const body = bodyParts.join('\n');
			if (channel === 'error') throw new Error(`Showdown stream error: ${body}`);
			if (channel === 'update') {
				for (const line of body.split('\n')) {
					if (line && !line.startsWith('|t:|')) rawUpdateEvents.push(line);
				}
			} else if (channel === 'sideupdate') {
				const [player, ...messages] = body.split('\n');
				const requestLine = messages.find(line => line.startsWith('|request|'));
				if (requestLine) requests[player] = JSON.parse(requestLine.slice('|request|'.length));
			}
		}
		const facts = validateInitialBattle(rawUpdateEvents, requests, teams, ps.toID);
		return { rawUpdateEvents, requests, facts };
	} finally {
		if (timer) clearTimeout(timer);
		await stream.destroy();
	}
}

async function smoke(ps, seeds) {
	const teams = [makeTeam(ps, seeds.teamP1), makeTeam(ps, seeds.teamP2)];
	for (const [index, team] of teams.entries()) {
		if (team.length !== 6) throw new Error(`Showdown generated ${team.length} Pokemon for ${index === 0 ? 'p1' : 'p2'}, expected six.`);
		for (const [pokemonIndex, pokemon] of team.entries()) {
			if (!pokemon.species || pokemon.moves.length !== 4) {
				throw new Error(`Generated team member ${index + 1}:${pokemonIndex + 1} lacks a species or four moves.`);
			}
		}
	}
	const battle = await initialBattle(ps, teams, seeds.battle);
	return canonicalize({
		schemaVersion: 1,
		purpose: 'Trusted Showdown oracle fixture. It contains both complete private teams and private player requests; it is not a policy observation or training example.',
		formatId: 'gen5randombattle',
		upstreamCommit: lock.commit,
		seeds,
		teams,
		initialBattle: {
			...battle.facts,
			trustedEventNote: 'rawUpdateEvents includes Showdown |split| markers and private SECRET lines. It is an omniscient oracle transcript, not a public player view.',
			rawUpdateEvents: battle.rawUpdateEvents,
			playerRequests: battle.requests,
		},
	});
}

function provenance() {
	const packageLockPath = path.join(checkout, 'package-lock.json');
	const hashFile = filename => crypto.createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
	const npmVersion = execFileSync('npm', ['--version'], { encoding: 'utf8' }).trim();
	const fixtureHashes = {};
	for (const [name, filename] of [['format', 'format-v1.json'], ['smoke', 'smoke-v1.json']]) {
		const file = path.join(__dirname, 'fixtures', filename);
		if (fs.existsSync(file)) fixtureHashes[name] = hashFile(file);
	}
	return {
		kind: 'build-provenance (machine/runtime metadata; excluded from semantic fixture equality)',
		upstreamRepository: lock.repository,
		upstreamCommit: lock.commit,
		upstreamPackageLockSha256: hashFile(packageLockPath),
		runtime: { node: process.version, npm: npmVersion },
		semanticFixtureSha256: fixtureHashes,
	};
}

async function main() {
	const [command, ...args] = process.argv.slice(2);
	if (!command || command === '--help' || command === 'help') {
		console.log('Commands: format | smoke [--team-seed-p1 a,b,c,d --team-seed-p2 a,b,c,d --battle-seed a,b,c,d] | provenance');
		return;
	}
	const ps = loadShowdown();
	let result;
	if (command === 'format') {
		if (args.length) throw new Error('format takes no options.');
		result = formatManifest(ps);
	} else if (command === 'smoke') {
		result = await smoke(ps, parseSmokeArgs(args));
	} else if (command === 'provenance') {
		if (args.length) throw new Error('provenance takes no options.');
		result = provenance();
	} else {
		throw new Error(`Unknown command: ${command}`);
	}
	process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
}

if (require.main === module) {
	main().catch(error => {
		console.error(`Showdown oracle failed: ${error.message}`);
		process.exitCode = 1;
	});
}

module.exports = {
	DEFAULT_SEEDS,
	assertReady,
	assertReferenceIdentity,
	canonicalize,
	formatManifest,
	initialBattle,
	makeTeam,
	parseSeed,
	parseSmokeArgs,
	provenance,
	smoke,
	validateInitialBattle,
};
