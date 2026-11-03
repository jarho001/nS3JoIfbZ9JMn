'use strict';

const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const crypto = require('node:crypto');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const oracle = require('../reference/oracle.cjs');
oracle.assertReady();
const ps = require('../reference/vendor/pokemon-showdown/dist/sim');
const referenceLock = require('../reference-lock.json');
const { MAX_PAIRS, generateManifest, generatePair } = require('../environment/paired-seeds.cjs');
const { parseArgs, runPair } = require('../environment/run-pair.cjs');

const ROOT = path.resolve(__dirname, '..');
const CLI = path.join(ROOT, 'environment/run-pair.cjs');
const ROOT_SEED = [1, 2, 3, 4];

function request(rqid) {
	return {
		rqid,
		active: [{ moves: [{ move: 'Tackle', id: 'tackle', pp: 35, disabled: false }] }],
		side: { pokemon: [{ active: true, condition: '100/100', details: 'Ownmon' }] },
	};
}

function packets() {
	return [
		'update\n|turn|1',
		`sideupdate\np1\n|request|${JSON.stringify(request(1))}`,
		`sideupdate\np2\n|request|${JSON.stringify(request(1))}`,
		'update\n|move|p1a: Ownmon|Tackle|p2a: Other\n|win|p1',
		'end\n{"winner":"p1","turns":1,"p1":"p1","p2":"p2","score":[1,0]}',
	];
}

class FakeStream {
	constructor(data) { this.data = data; this.destroyCalls = 0; }
	write() {}
	[Symbol.asyncIterator]() {
		let index = 0;
		return { next: async () => index < this.data.length ? { value: this.data[index++], done: false } : { done: true } };
	}
	async destroy() { this.destroyCalls++; }
}

function fakePs({ generateError = undefined } = {}) {
	return {
		BattleStream: class {},
		Dex: { formats: { get: format => ({ exists: format === 'gen5randombattle' }) } },
		Teams: {
			generate(format, { seed }) {
				if (generateError) throw generateError;
				return { format, seed: [...seed] };
			},
			pack(team) { return `${team.format}:${team.seed.join(',')}`; },
		},
	};
}

async function captureFailure(promise) {
	try { await promise; } catch (error) { return error; }
	assert.fail('Expected paired-run failure.');
}

function recordingPinnedStreamFactory(writes, streams) {
	return () => {
		const stream = new ps.BattleStream({ keepAlive: true });
		streams.push(stream);
		return {
			write(command) {
				const match = /^>player (p[12]) (.+)$/.exec(command);
				if (match) writes.push({ player: match[1], packedTeam: JSON.parse(match[2]).team });
				return stream.write(command);
			},
			destroy() { return stream.destroy(); },
			[Symbol.asyncIterator]() { return stream[Symbol.asyncIterator](); },
		};
	};
}

test('generatePair validates bounded indices and exactly matches full-manifest entries', () => {
	for (const pairIndex of [0, 4, 19]) {
		const rootSeed = [0, 65_535, pairIndex + 1, 23];
		const copy = [...rootSeed];
		assert.deepEqual(generatePair(rootSeed, pairIndex), generateManifest(copy, pairIndex + 1).pairs[pairIndex]);
		rootSeed[0] = 1;
		assert.notDeepEqual(generatePair(rootSeed, pairIndex), generateManifest(copy, pairIndex + 1).pairs[pairIndex]);
	}
	const finalPair = generatePair(ROOT_SEED, MAX_PAIRS - 1);
	assert.equal(finalPair.pairIndex, MAX_PAIRS - 1);
	assert.match(finalPair.pairId, /^pair-100000-/);
	for (const seed of [new Array(4), [0, , 2, 3], [1, 2, 3], [0, 1, 2, 65_536]]) assert.throws(() => generatePair(seed, 0));
	for (const index of [-1, 1.5, Number.NaN, MAX_PAIRS]) assert.throws(() => generatePair(ROOT_SEED, index));
});

test('runPair preserves seed identities, prepared team hashes and both validated outcomes', async () => {
	const showdown = fakePs();
	const battleStreams = [];
	let generations = 0;
	let packs = 0;
	const originalGenerate = showdown.Teams.generate;
	const originalPack = showdown.Teams.pack;
	showdown.Teams.generate = (...args) => { generations++; return originalGenerate(...args); };
	showdown.Teams.pack = (...args) => { packs++; return originalPack(...args); };
	const streams = [];
	const result = await runPair({
		ps: showdown,
		rootSeed: ROOT_SEED,
		pairIndex: 2,
		streamFactory: () => { const stream = new FakeStream(packets()); streams.push(stream); return stream; },
	});
	const [pair] = generateManifest(ROOT_SEED, 3).pairs.slice(-1);
	assert.equal(result.status, 'completed');
	assert.equal(result.elapsedScope, 'paired_runner');
	assert.ok(Number.isFinite(result.elapsedMs) && result.elapsedMs >= 0);
	assert.equal(result.upstream.commit, referenceLock.commit);
	assert.equal(result.formatId, 'gen5randombattle');
	assert.equal(result.pairId, pair.pairId);
	assert.deepEqual(result.rootSeed, ROOT_SEED);
	assert.equal(result.runs.length, 2);
	assert.deepEqual(result.runs.map(run => run.status), ['completed', 'completed']);
	assert.deepEqual(result.runs.map(run => run.seeds), pair.runs.map(run => run.seeds));
	assert.deepEqual(result.runs.map(run => run.policySeats), pair.runs.map(run => run.policySeats));
	assert.deepEqual(result.runs.map(run => run.resultRecord.status), ['completed', 'completed']);
	assert.ok(result.runs.every(run => run.resultRecord.outcome.winner === 'p1'));
	assert.equal(generations, 2);
	assert.equal(packs, 2);
	assert.deepEqual(streams.map(stream => stream.destroyCalls), [1, 1]);
	const serialized = JSON.stringify(result);
	assert.equal(serialized.includes('fake-team'), false);
	assert.equal(serialized.includes('transcript'), false);
	assert.equal(serialized.includes('packets'), false);
});

test('pinned paired run generates and packs once, then writes each team in exchanged seats', { timeout: 180_000 }, async () => {
	const pair = generatePair(ROOT_SEED, 1);
	const writes = [];
	const streams = [];
	let generations = 0;
	let packs = 0;
	const originalGenerate = ps.Teams.generate;
	const originalPack = ps.Teams.pack;
	ps.Teams.generate = function (...args) { generations++; return originalGenerate.apply(this, args); };
	ps.Teams.pack = function (...args) { packs++; return originalPack.apply(this, args); };
	try {
		const result = await runPair({
			ps,
			rootSeed: ROOT_SEED,
			pairIndex: 1,
			streamFactory: recordingPinnedStreamFactory(writes, streams),
		});
		assert.equal(result.status, 'completed');
		assert.deepEqual(result.runs.map(run => run.status), ['completed', 'completed']);
		assert.equal(generations, 2);
		assert.equal(packs, 2);
		assert.deepEqual(streams.length, 2);
		assert.deepEqual(writes.map(write => write.player), ['p1', 'p2', 'p1', 'p2']);
		const hash = packed => crypto.createHash('sha256').update(packed).digest('hex');
		assert.equal(hash(writes[0].packedTeam), result.teams.A.packedTeamSha256);
		assert.equal(hash(writes[1].packedTeam), result.teams.B.packedTeamSha256);
		assert.equal(hash(writes[2].packedTeam), result.teams.B.packedTeamSha256);
		assert.equal(hash(writes[3].packedTeam), result.teams.A.packedTeamSha256);
		assert.deepEqual(result.runs.map(run => run.seeds), pair.runs.map(run => run.seeds));
		assert.deepEqual(result.runs.map(run => run.policySeats), pair.runs.map(run => run.policySeats));
		assert.ok(result.runs.every(run => ['p1', 'p2', null].includes(run.resultRecord.outcome.winner)));
		for (const write of writes) assert.equal(JSON.stringify(result).includes(write.packedTeam), false);
	} finally {
		ps.Teams.generate = originalGenerate;
		ps.Teams.pack = originalPack;
	}
});

test('preparation failures keep both runs unattempted and ignore foreign error metadata', async () => {
	const foreignError = new Error('DO NOT INCLUDE PREPARATION DETAILS');
	foreignError.pairResult = { sentinel: 'FOREIGN_PAIR_RESULT' };
	const error = await captureFailure(runPair({
		ps: fakePs({ generateError: foreignError }),
		rootSeed: ROOT_SEED,
		pairIndex: 0,
		streamFactory: () => assert.fail('battle must not start after preparation failure'),
	}));
	assert.equal(error.pairResult.failure.stage, 'preparation');
	assert.equal(error.pairResult.failure.code, 'team_preparation_failed');
	assert.deepEqual(error.pairResult.runs.map(run => run.status), ['not_attempted', 'not_attempted']);
	assert.equal(JSON.stringify(error.pairResult).includes('DO NOT INCLUDE'), false);
	assert.equal(JSON.stringify(error.pairResult).includes('FOREIGN_PAIR_RESULT'), false);
});

test('first-run failure is safe and leaves the swapped run unattempted', async () => {
	const error = new Error('FOREIGN STREAM ERROR DETAIL');
	error.pairResult = { sentinel: 'FOREIGN_RESULT' };
	let streamFactories = 0;
	const failure = await captureFailure(runPair({
		ps,
		rootSeed: ROOT_SEED,
		pairIndex: 0,
		streamFactory: () => { streamFactories++; throw error; },
	}));
	const result = failure.pairResult;
	assert.equal(streamFactories, 1);
	assert.deepEqual(result.runs.map(run => run.status), ['failed', 'not_attempted']);
	assert.equal(result.runs[0].resultRecord.failure.code, 'stream_initialization_failed');
	assert.equal(result.failure.runId, result.runs[0].runId);
	assert.equal(JSON.stringify(result).includes('FOREIGN STREAM ERROR DETAIL'), false);
	assert.equal(JSON.stringify(result).includes('FOREIGN_RESULT'), false);
});

test('second-run failure preserves the completed first result and does not launch another run', async () => {
	let streamFactories = 0;
	const fakeError = new Error('SECOND STREAM RAW DETAIL');
	const failure = await captureFailure(runPair({
		ps,
		rootSeed: ROOT_SEED,
		pairIndex: 0,
		streamFactory: () => {
			streamFactories++;
			if (streamFactories === 1) {
				const stream = new ps.BattleStream({ keepAlive: true });
				return {
					write: command => stream.write(command),
					destroy: () => stream.destroy(),
					[Symbol.asyncIterator]: () => stream[Symbol.asyncIterator](),
				};
			}
			throw fakeError;
		},
	}));
	const result = failure.pairResult;
	assert.equal(streamFactories, 2);
	assert.deepEqual(result.runs.map(run => run.status), ['completed', 'failed']);
	assert.equal(result.runs[0].resultRecord.status, 'completed');
	assert.equal(result.runs[1].resultRecord.failure.code, 'stream_initialization_failed');
	assert.equal(result.failure.runId, result.runs[1].runId);
	assert.equal(JSON.stringify(result).includes('SECOND STREAM RAW DETAIL'), false);
});

test('runPair rejects malformed inputs with safe pair records', async () => {
	for (const [rootSeed, pairIndex, code] of [
		[[1, 2, 3], 0, 'invalid_root_seed'],
		[[1, , 3, 4], 0, 'invalid_root_seed'],
		[ROOT_SEED, undefined, 'invalid_pair_index'],
		[ROOT_SEED, 100_000, 'invalid_pair_index'],
	]) {
		const error = await captureFailure(runPair({ ps, rootSeed, pairIndex }));
		assert.equal(error.pairResult.failure.code, code);
		assert.equal(error.pairResult.status, 'failed');
		assert.ok(Number.isFinite(error.pairResult.elapsedMs));
	}
});

test('runPair treats throwing option getters as configuration failures before preparation or battle', async () => {
	for (const optionName of ['limits', 'streamFactory']) {
		const injected = new Error('FOREIGN CONFIGURATION GETTER DETAIL');
		injected.resultRecord = {
			status: 'completed',
			failure: { category: 'INJECTED_CATEGORY', code: 'INJECTED_CODE' },
			foreignData: 'INJECTED_RECORD',
		};
		let generateCalls = 0;
		let packCalls = 0;
		const countingPs = fakePs();
		const originalGenerate = countingPs.Teams.generate;
		const originalPack = countingPs.Teams.pack;
		countingPs.Teams.generate = (...args) => { generateCalls++; return originalGenerate(...args); };
		countingPs.Teams.pack = (...args) => { packCalls++; return originalPack(...args); };
		let streamFactoryCalls = 0;
		const options = { ps: countingPs, rootSeed: ROOT_SEED, pairIndex: 0 };
		Object.defineProperty(options, optionName, {
			get() { throw injected; },
		});
		if (optionName === 'limits') {
			options.streamFactory = () => { streamFactoryCalls++; assert.fail('battle must not start after configuration failure'); };
		}
		const error = await captureFailure(runPair(options));
		const result = error.pairResult;
		assert.equal(result.failure.stage, 'configuration');
		assert.equal(result.failure.category, 'initialization');
		assert.equal(result.failure.code, 'invalid_configuration');
		assert.deepEqual(result.runs.map(run => run.status), ['not_attempted', 'not_attempted']);
		assert.deepEqual(result.runs.map(run => run.resultRecord), [null, null]);
		assert.equal(generateCalls, 0);
		assert.equal(packCalls, 0);
		assert.equal(streamFactoryCalls, 0);
		const serialized = JSON.stringify(result);
		for (const secret of ['FOREIGN CONFIGURATION GETTER DETAIL', 'INJECTED_CATEGORY', 'INJECTED_CODE', 'INJECTED_RECORD']) {
			assert.equal(serialized.includes(secret), false);
		}
	}
});

test('CLI ignores foreign pair records during oracle bootstrap and records CLI timing', async () => {
	const outputDirectory = fs.mkdtempSync(path.join(os.tmpdir(), 'pair-runner-bootstrap-'));
	const resultPath = path.join(outputDirectory, 'failure.json');
	const injected = new Error('FOREIGN BOOTSTRAP DETAILS');
	injected.pairResult = {
		kind: 'gen5-paired-battle-result', status: 'completed',
		failure: { code: 'INJECTED_PAIR_CODE' }, foreignData: 'INJECTED_PAIR_DATA',
	};
	const originalAssertReady = oracle.assertReady;
	const originalConsoleError = console.error;
	const originalExitCode = process.exitCode;
	const diagnostics = [];
	try {
		oracle.assertReady = () => { throw injected; };
		console.error = (...parts) => diagnostics.push(parts.join(' '));
		process.exitCode = undefined;
		await require('../environment/run-pair.cjs').main([
			'--root-seed=1,2,3,4', '--pair-index=0', `--result=${resultPath}`,
		]);
		const result = JSON.parse(fs.readFileSync(resultPath, 'utf8'));
		assert.equal(process.exitCode, 1);
		assert.equal(result.status, 'failed');
		assert.deepEqual(result.failure, { stage: 'initialization', category: 'initialization', code: 'cli_initialization_failed' });
		assert.equal(result.elapsedScope, 'cli_initialization');
		assert.ok(Number.isFinite(result.elapsedMs) && result.elapsedMs >= 0);
		assert.equal(JSON.stringify(result).includes('INJECTED_PAIR_CODE'), false);
		assert.equal(JSON.stringify(result).includes('INJECTED_PAIR_DATA'), false);
		assert.equal(diagnostics.some(line => line.includes('INJECTED_PAIR_CODE')), false);
	} finally {
		oracle.assertReady = originalAssertReady;
		console.error = originalConsoleError;
		process.exitCode = originalExitCode;
		fs.rmSync(outputDirectory, { recursive: true, force: true });
	}
});

test('CLI writes success and structured failure output, rejects implicit pair selection', { timeout: 180_000 }, () => {
	const outputDirectory = fs.mkdtempSync(path.join(os.tmpdir(), 'pair-runner-'));
	try {
		const resultPath = path.join(outputDirectory, 'success.json');
		const success = spawnSync(process.execPath, [CLI, '--root-seed=1,2,3,4', '--pair-index=0', `--result=${resultPath}`], {
			cwd: ROOT, encoding: 'utf8', timeout: 150_000,
		});
		assert.equal(success.status, 0, success.stderr);
		const stdoutResult = JSON.parse(success.stdout);
		const fileResult = JSON.parse(fs.readFileSync(resultPath, 'utf8'));
		assert.deepEqual(stdoutResult, fileResult);
		assert.deepEqual(stdoutResult.runs.map(run => run.status), ['completed', 'completed']);

		const failurePath = path.join(outputDirectory, 'failure.json');
		const failed = spawnSync(process.execPath, [CLI, '--root-seed=bad', '--pair-index=0', `--result=${failurePath}`], {
			cwd: ROOT, encoding: 'utf8', timeout: 15_000,
		});
		assert.equal(failed.status, 1);
		assert.match(failed.stderr, /invalid_cli_configuration/);
		const failedResult = JSON.parse(fs.readFileSync(failurePath, 'utf8'));
		assert.equal(failedResult.failure.code, 'invalid_cli_configuration');
		assert.equal(failedResult.elapsedScope, 'cli_initialization');
		assert.ok(Number.isFinite(failedResult.elapsedMs) && failedResult.elapsedMs >= 0);

		const missing = spawnSync(process.execPath, [CLI, '--root-seed=1,2,3,4'], { cwd: ROOT, encoding: 'utf8', timeout: 15_000 });
		assert.equal(missing.status, 1);
		assert.match(missing.stderr, /invalid_cli_configuration/);
	} finally {
		fs.rmSync(outputDirectory, { recursive: true, force: true });
	}
});
