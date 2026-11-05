#!/usr/bin/env node
'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');
const { assertReady, snapshot: baseSnapshot, withObservedBattleDraws } = require('./neutral-turn-oracle.cjs');

const ROOT = path.resolve(__dirname, '..');
const LOCK = require(path.join(ROOT, 'reference-lock.json'));
const API = path.join(__dirname, 'vendor', 'pokemon-showdown', 'dist', 'sim');
const FORMAT_ID = 'gen5randombattle';
const VARIANT = 'gen5-single-burn-choice-mew-1v1-v7';
const CASES = [
	{
		id: 'no-burn-control', seed: [1, 2, 3, 4],
		p1: { level: 50, speedEvs: 252 }, p2: { level: 50, speedEvs: 0 },
		burned: null, choices: { p1: 1, p2: 1 },
	},
	{
		id: 'p1-burned-mega-punch-critical-seed-21', seed: [21, 2, 3, 4],
		p1: { level: 50, speedEvs: 252 }, p2: { level: 50, speedEvs: 0 },
		burned: 'p1', choices: { p1: 2, p2: 1 },
	},
	{
		id: 'p2-burned-tackle-p2-faster', seed: [1, 2, 3, 4],
		p1: { level: 50, speedEvs: 0 }, p2: { level: 50, speedEvs: 252 },
		burned: 'p2', choices: { p1: 2, p2: 1 },
	},
	{
		id: 'p1-burned-mega-punch-miss-tick-seed-0', seed: [0, 2, 3, 4],
		p1: { level: 50, speedEvs: 252 }, p2: { level: 50, speedEvs: 0 },
		burned: 'p1', choices: { p1: 2, p2: 1 },
	},
	{
		id: 'burned-direct-attack-ko-skips-residual-seed-21', seed: [21, 2, 3, 4],
		p1: { level: 100, speedEvs: 252 }, p2: { level: 1, speedEvs: 0 },
		burned: 'p1', choices: { p1: 2, p2: 1 },
	},
	{
		id: 'burned-residual-ko-after-accuracy-miss-seed-0', seed: [0, 2, 3, 4],
		p1: { level: 50, speedEvs: 0, hp: 1 }, p2: { level: 50, speedEvs: 252 },
		burned: 'p1', choices: { p1: 1, p2: 2 },
	},
];

function assertMoveAndBurnFacts(ps) {
	assert.equal(LOCK.commit, '1242c1e2d9763ba5b85d33f13aecb89bae19ae3d');
	const dex = ps.Dex.mod('gen5');
	const tackle = dex.moves.get('tackle');
	const megaPunch = dex.moves.get('megapunch');
	assert.deepEqual(
		[tackle.id, tackle.category, tackle.basePower, tackle.accuracy, tackle.pp],
		['tackle', 'Physical', 50, 100, 35],
	);
	assert.deepEqual(
		[megaPunch.id, megaPunch.category, megaPunch.basePower, megaPunch.accuracy, megaPunch.pp, megaPunch.critRatio],
		['megapunch', 'Physical', 80, 85, 20, 1],
	);
	const burn = dex.conditions.get('brn');
	assert.equal(burn.id, 'brn');
	assert.equal(burn.name, 'brn');
	return {
		tackle: { id: tackle.id, category: tackle.category, basePower: tackle.basePower, accuracy: tackle.accuracy, basePp: tackle.pp, maxPp: 56 },
		megaPunch: { id: megaPunch.id, category: megaPunch.category, basePower: megaPunch.basePower, accuracy: megaPunch.accuracy, basePp: megaPunch.pp, maxPp: 32, criticalRatio: megaPunch.critRatio },
		burn: { id: burn.id, name: burn.name, effectType: burn.effectType, gen: burn.gen },
	};
}

function makeSet(descriptor) {
	const empty = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
	const perfect = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
	return {
		species: 'Mew', ability: 'Synchronize', item: '', level: descriptor.level,
		moves: ['Tackle', 'Mega Punch'], nature: 'Hardy', ivs: perfect,
		evs: { ...empty, spe: descriptor.speedEvs },
	};
}

function createBattle(ps, scenario) {
	const battle = new ps.Battle({
		formatid: FORMAT_ID,
		seed: [...scenario.seed],
		p1: { name: 'p1', team: [makeSet(scenario.p1)] },
		p2: { name: 'p2', team: [makeSet(scenario.p2)] },
	});
	assert.equal(battle.gen, 5);
	assert.equal(battle.gameType, 'singles');
	assert.equal(battle.ruleTable.has('hppercentagemod'), true);
	for (const side of [battle.p1, battle.p2]) {
		assert.equal(side.pokemon.length, 1);
		assert.deepEqual(side.active[0].moves, ['tackle', 'megapunch']);
		assert.equal(side.active[0].item, '');
		assert.equal(side.active[0].ability, 'synchronize');
		assert.deepEqual(side.active[0].moveSlots.map(slot => slot.id), ['tackle', 'megapunch']);
		assert.deepEqual(side.active[0].moveSlots.map(slot => slot.pp), [56, 32]);
	}
	return battle;
}

function snapshot(battle) {
	const result = baseSnapshot(battle);
	const pokemon = [battle.p1.pokemon[0], battle.p2.pokemon[0]];
	return {
		...result,
		pp: pokemon.map(mon => mon.moveSlots.map(slot => slot.pp)),
		statuses: pokemon.map(mon => mon.status),
		burned: pokemon.map(mon => mon.status === 'brn'),
	};
}

function projectRequest(side) {
	const request = side.activeRequest;
	if (!request) return null;
	assert.equal(request.active.length, 1);
	assert.equal(request.side.id, side.id);
	const own = request.side.pokemon.find(mon => mon.active);
	assert.ok(own);
	const slots = side.pokemon[0].moveSlots;
	const moves = request.active[0].moves;
	assert.equal(moves.length, slots.every(slot => slot.pp === 0) ? 1 : 2);
	if (moves.length === 1) {
		assert.equal(moves[0].id, 'struggle');
		return { phase: 'move', boundary: 'native-struggle-no-pp', hpCondition: own.condition, legalActions: ['move 1'], moves: [{ id: 'struggle', pp: moves[0].pp, disabled: false }] };
	}
	return {
		phase: 'move', boundary: 'accuracy-choice',
		hpCondition: own.condition,
		legalActions: moves.flatMap((move, index) => move.disabled ? [] : [`move ${index + 1}`]),
		moves: moves.map((move, index) => {
			assert.equal(move.id, ['tackle', 'megapunch'][index]);
			assert.equal(move.pp, slots[index].pp);
			assert.equal(move.disabled, slots[index].pp === 0);
			return { id: move.id, name: move.move, pp: move.pp, disabled: move.disabled };
		}),
	};
}

function sideFromIdent(ident) {
	if (typeof ident !== 'string') return null;
	if (ident.startsWith('p1')) return 'p1';
	if (ident.startsWith('p2')) return 'p2';
	return null;
}

function semanticEvent(line) {
	const fields = line.split('|');
	const cause = fields.find(field => field.startsWith('[from] '));
	switch (fields[1]) {
	case 'move': return { type: 'move', actor: sideFromIdent(fields[2]), move: fields[3].toLowerCase(), target: sideFromIdent(fields[4]) };
	case '-miss': return { type: 'miss', actor: sideFromIdent(fields[2]), target: sideFromIdent(fields[3]) };
	case '-crit': return { type: 'critical', target: sideFromIdent(fields[2]) };
	case '-damage': return { type: 'damage', target: sideFromIdent(fields[2]), condition: fields[3], ...(cause ? { from: cause.slice('[from] '.length) } : {}) };
	case '-status': return { type: 'status', target: sideFromIdent(fields[2]), status: fields[3] };
	case 'faint': return { type: 'faint', target: sideFromIdent(fields[2]) };
	case 'turn': return { type: 'turn', number: Number(fields[2]) };
	case 'win': return { type: 'winner', side: fields[2] };
	case 'tie': return { type: 'winner', side: 'tie' };
	default: return null;
	}
}

function projectEvents(lines, audience) {
	const projected = [];
	for (let index = 0; index < lines.length; index++) {
		let line = lines[index];
		if (line.startsWith('|split|')) {
			const owner = line.slice('|split|'.length);
			assert.ok(index + 2 < lines.length);
			line = audience === owner ? lines[index + 1] : lines[index + 2];
			index += 2;
		}
		if (line.startsWith('|t:|') || line === '|' || line === '|upkeep') continue;
		const event = semanticEvent(line);
		if (event) projected.push(event);
	}
	return projected;
}

function choiceToken(choice) {
	assert.ok(choice === 1 || choice === 2);
	return `move ${choice}`;
}

function setupScenario(ps, scenario, battle) {
	const beforeRng = [...baseSnapshot(battle).rng];
	const setupStart = battle.log.length;
	if (scenario.burned) {
		const mon = battle[scenario.burned].pokemon[0];
		assert.equal(mon.setStatus('brn', mon), true, `${scenario.id} trusted pre-existing burn installs`);
		assert.equal(mon.status, 'brn');
	}
	if (scenario.p1.hp !== undefined) {
		battle.p1.pokemon[0].sethp(scenario.p1.hp);
		assert.equal(battle.p1.pokemon[0].hp, scenario.p1.hp, `${scenario.id} trusted prior HP setup`);
	}
	battle.makeRequest('move');
	assert.deepEqual(baseSnapshot(battle).rng, beforeRng, `${scenario.id} trusted setup leaves native PRNG unchanged`);
	const setupLog = battle.log.slice(setupStart);
	const statusEvents = { p1: projectEvents(setupLog, 'p1'), p2: projectEvents(setupLog, 'p2') };
	const requests = { p1: projectRequest(battle.p1), p2: projectRequest(battle.p2) };
	const before = snapshot(battle);
	if (scenario.burned) {
		const sideIndex = scenario.burned === 'p1' ? 0 : 1;
		assert.equal(before.burned[sideIndex], true);
		assert.equal(before.burned[1 - sideIndex], false);
		if (before.hp[sideIndex] > 0) assert.match(requests[scenario.burned].hpCondition, /brn$/);
		for (const side of ['p1', 'p2']) {
			assert.ok(statusEvents[side].some(event => event.type === 'status' && event.target === scenario.burned && event.status === 'brn'));
		}
	}
	return {
		statusBoundary: 'trusted pre-existing public setStatus(brn) before makeRequest(move); this is not status-infliction or Synchronize-reflection parity',
		priorDamageBoundary: scenario.p1.hp === undefined ? null : 'trusted prior sethp(1) before makeRequest(move); no attack/residual/terminal state is forged',
		setupRngBefore: beforeRng,
		setupRngAfter: [...before.rng],
		statusEvents,
		requests,
		before,
	};
}

function captureCase(ps, scenario) {
	const battle = createBattle(ps, scenario);
	try {
		const setup = setupScenario(ps, scenario, battle);
		const turnBefore = snapshot(battle);
		const requests = { p1: projectRequest(battle.p1), p2: projectRequest(battle.p2) };
		const logStart = battle.log.length;
		const { draws } = withObservedBattleDraws(ps, battle, () => battle.makeChoices(
			choiceToken(scenario.choices.p1), choiceToken(scenario.choices.p2),
		));
		const logDelta = battle.log.slice(logStart);
		const events = { p1: projectEvents(logDelta, 'p1'), p2: projectEvents(logDelta, 'p2') };
		const after = snapshot(battle);
		const expectedActionOrder = events.p1.filter(event => event.type === 'move').map(event => event.actor);
		assert.ok(expectedActionOrder.length >= 1 && expectedActionOrder.length <= 2);
		assert.equal(draws.filter(draw => draw.context === 'accuracy').length, expectedActionOrder.length);
		const misses = events.p1.filter(event => event.type === 'miss').length;
		assert.equal(draws.filter(draw => draw.context === 'critical').length, expectedActionOrder.length - misses);
		assert.equal(draws.filter(draw => draw.context === 'damage').length, expectedActionOrder.length - misses);
		if (draws.length) {
			assert.deepEqual(draws[0].before, turnBefore.rng);
			assert.deepEqual(draws.at(-1).after, after.rng);
		}
		assert.equal(after.terminal, battle.ended);
		const residualEvents = events.p1.filter(event => event.type === 'damage' && event.from === 'brn');
		const residual = residualEvents.map(event => ({
			target: event.target,
			condition: event.condition,
			from: event.from,
			faintedImmediately: event.condition.startsWith('0 fnt'),
		}));
		const nextRequests = battle.ended ? { p1: null, p2: null } : { p1: projectRequest(battle.p1), p2: projectRequest(battle.p2) };
		return {
			id: scenario.id,
			seed: [...scenario.seed],
			sets: { p1: scenario.p1, p2: scenario.p2 },
			burnedSetup: scenario.burned,
			choices: scenario.choices,
			setup,
			turn: {
				before: turnBefore,
				requests,
				actionOrder: expectedActionOrder,
				draws,
				events,
				rawLog: logDelta,
				after,
				nextRequests,
				residual,
			},
		};
	} finally {
		battle.destroy();
	}
}

function generateFixture(ps) {
	const facts = assertMoveAndBurnFacts(ps);
	const cases = CASES.map(scenario => captureCase(ps, scenario));
	const miss = cases.find(candidate => candidate.id === 'p1-burned-mega-punch-miss-tick-seed-0');
	assert.ok(miss.turn.events.p1.some(event => event.type === 'miss' && event.actor === 'p1'));
	assert.ok(miss.turn.residual.some(event => event.target === 'p1'));
	const critical = cases.find(candidate => candidate.id === 'p1-burned-mega-punch-critical-seed-21');
	assert.ok(critical.turn.events.p1.some(event => event.type === 'critical' && event.target === 'p2'));
	const directKo = cases.find(candidate => candidate.id === 'burned-direct-attack-ko-skips-residual-seed-21');
	assert.equal(directKo.turn.after.terminal, true);
	assert.equal(directKo.turn.residual.length, 0);
	const residualKo = cases.find(candidate => candidate.id === 'burned-residual-ko-after-accuracy-miss-seed-0');
	assert.equal(residualKo.turn.after.terminal, true);
	assert.equal(residualKo.turn.residual.length, 1);
	assert.match(residualKo.turn.residual[0].condition, /^0 fnt/);
	assert.equal(residualKo.turn.after.statuses[0], 'brn', 'persistent burn status remains native-authoritative after one-Mew faint');
	return {
		schemaVersion: 1,
		kind: 'gen5-restricted-single-burn-choice-oracle',
		upstream: { repository: LOCK.repository, commit: LOCK.commit },
		formatId: FORMAT_ID,
		generation: 5,
		variant: VARIANT,
		moveAndStatusFacts: facts,
		initialization: 'Unmodified pinned gen5randombattle with directly supplied itemless Synchronize/Hardy/perfect-IV Mew sets and generator bypassed. Status and low-HP cases use trusted pre-turn native setup, followed by live makeRequest(move); setup changes the original PRNG zero times.',
		burnContract: 'Exactly one or zero pre-existing publicly known brn flags. Gen 5 ordinary physical Tackle/Mega Punch damage uses the native burn condition; once-per-turn residual is native max(1, floor(baseMaxHP/8)) damage only if attacks leave the battle live. Native status after faint and protocol 0 fnt are captured independently.',
		drawContract: 'The guarded PRNG observer returns original values unchanged. Every actual action retains native accuracy/critical/damage draws and four-limb pre/post seeds. Residual damage is identified by its native [from] brn cause and consumes no extra PRNG draw.',
		cases,
	};
}

function main() {
	assertReady();
	const ps = require(API);
	process.stdout.write(`${JSON.stringify(generateFixture(ps), null, 2)}\n`);
}

if (require.main === module) main();
module.exports = { generateFixture, main };
