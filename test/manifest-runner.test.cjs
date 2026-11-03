'use strict';

const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const crypto = require('node:crypto');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const oracle = require('../reference/oracle.cjs');
const referenceLock = require('../reference-lock.json');
const { generateManifest } = require('../environment/paired-seeds.cjs');
const { ownsPairError, runPair } = require('../environment/run-pair.cjs');
const { main, parseArgs, runManifest } = require('../environment/run-manifest.cjs');

const ROOT = path.resolve(__dirname, '..');
const CLI = path.join(ROOT, 'environment/run-manifest.cjs');
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

function fakePs({ failGenerateAt = Infinity } = {}) {
	const calls = { generate: 0, pack: 0 };
	const ps = {
		BattleStream: class {},
		Dex: { formats: { get: format => ({ exists: format === 'gen5randombattle' }) } },
		Teams: {
			generate(format, { seed }) {
				calls.generate++;
				if (calls.generate === failGenerateAt) throw new Error('FOREIGN TEAM ERROR');
				return { format, seed: [...seed] };
			},
			pack(team) { calls.pack++; return `${team.format}:${team.seed.join(',')}`; },
		},
	};
	return { ps, calls };
}

function optionsFor(outputDir, { ps, pairCount = 2, streamFactory, limits } = {}) {
	return {
		ps,
		rootSeed: ROOT_SEED,
		pairCount,
		outputDir,
		...(streamFactory === undefined ? {} : { streamFactory }),
		...(limits === undefined ? {} : { limits }),
	};
}

function resultLines(outputDir) {
	const text = fs.readFileSync(path.join(outputDir, 'results.jsonl'), 'utf8');
	return text.split('\n').filter(Boolean).map(line => JSON.parse(line));
}

async function captureFailure(promise) {
	try { await promise; } catch (error) { return error; }
	assert.fail('Expected manifest-runner failure.');
}

function newTempRoot(prefix) {
	return fs.mkdtempSync(path.join(os.tmpdir(), prefix));
}

test('parseArgs requires an explicit root, pair count, and output directory', () => {
	assert.deepEqual(parseArgs(['--root-seed=1,2,3,4', '--pairs=2', '--output-dir=out']), {
		rootSeed: ROOT_SEED, pairCount: 2, outputDir: 'out',
	});
	assert.deepEqual(parseArgs(['--help']), { help: true });
	for (const args of [[], ['--root-seed=1,2,3,4', '--pairs=2'], ['--root-seed=x', '--pairs=2', '--output-dir=x'], ['--root-seed=1,2,3,4', '--pairs=0', '--output-dir=x']]) {
		assert.throws(() => parseArgs(args));
	}
});

test('exported ownership predicate accepts only actual pair-runner errors', async () => {
	const { ps } = fakePs();
	const owned = await captureFailure(runPair({ ps, rootSeed: ROOT_SEED, pairIndex: -1 }));
	assert.equal(ownsPairError(owned), true);
	assert.equal(ownsPairError({ pairResult: { kind: 'gen5-paired-battle-result', status: 'completed' } }), false);
});

test('configuration and an existing destination are rejected before artifact or battle work', async () => {
	const root = newTempRoot('manifest-runner-validation-');
	const outputDir = path.join(root, 'new-output');
	const { ps, calls } = fakePs();
	const poison = new Error('FOREIGN CONFIGURATION DETAILS');
	poison.summary = { status: 'completed', foreignData: 'INJECTED_SUMMARY' };
	const invalidRoot = [1, 2, 3, 4];
	Object.defineProperty(invalidRoot, '0', { enumerable: true, get() { throw poison; } });
	try {
		const invalid = await captureFailure(runManifest({ ...optionsFor(outputDir, { ps }), rootSeed: invalidRoot }));
		assert.equal(invalid.message, 'Manifest run failed (invalid_configuration).');
		assert.equal(invalid.summary, undefined);
		assert.equal(fs.existsSync(outputDir), false);
		assert.equal(calls.generate, 0);

		const invalidLimits = await captureFailure(runManifest(optionsFor(outputDir, { ps, limits: { timeoutMs: 0 } })));
		assert.equal(invalidLimits.message, 'Manifest run failed (invalid_configuration).');
		assert.equal(fs.existsSync(outputDir), false);
		assert.equal(calls.generate, 0);

		fs.mkdirSync(outputDir);
		fs.writeFileSync(path.join(outputDir, 'keep.txt'), 'preserve me');
		let streams = 0;
		const existing = await captureFailure(runManifest(optionsFor(outputDir, {
			ps,
			streamFactory: () => { streams++; return new FakeStream(packets()); },
		})));
		assert.equal(existing.summary.status, 'failed');
		assert.equal(existing.summary.failure.stage, 'output_directory_create');
		assert.deepEqual(fs.readdirSync(outputDir), ['keep.txt']);
		assert.equal(fs.readFileSync(path.join(outputDir, 'keep.txt'), 'utf8'), 'preserve me');
		assert.equal(calls.generate, 0);
		assert.equal(streams, 0);
	} finally {
		fs.rmSync(root, { recursive: true, force: true });
	}
});

test('fake manifest execution writes each result before beginning the next pair', async () => {
	const root = newTempRoot('manifest-runner-order-');
	const outputDir = path.join(root, 'results');
	const { ps, calls } = fakePs();
	const streams = [];
	let factoryCalls = 0;
	let active = 0;
	const originalGenerate = ps.Teams.generate;
	ps.Teams.generate = function (...args) {
		if (calls.generate === 2) assert.equal(resultLines(outputDir).length, 1, 'pair zero journal line must precede pair one preparation');
		return originalGenerate.apply(this, args);
	};
	const summary = await runManifest(optionsFor(outputDir, {
		ps,
		streamFactory: () => {
			assert.equal(active, 0, 'the previous battle stream must be cleaned up first');
			if (factoryCalls === 2) assert.equal(resultLines(outputDir).length, 1, 'pair zero journal line must precede pair one');
			const stream = new FakeStream(packets());
			streams.push(stream);
			factoryCalls++;
			active++;
			return {
				write(command) { return stream.write(command); },
				async destroy() { try { return await stream.destroy(); } finally { active--; } },
				[Symbol.asyncIterator]() { return stream[Symbol.asyncIterator](); },
			};
		},
	}));
	try {
		const expectedManifest = generateManifest(ROOT_SEED, 2);
		const manifestText = fs.readFileSync(path.join(outputDir, 'manifest.json'), 'utf8');
		assert.equal(manifestText, `${JSON.stringify(expectedManifest, null, 2)}\n`);
		assert.deepEqual(JSON.parse(manifestText), expectedManifest);
		const records = resultLines(outputDir);
		assert.deepEqual(records.map(record => record.pairIndex), [0, 1]);
		assert.deepEqual(records.map(record => record.pairId), expectedManifest.pairs.map(pair => pair.pairId));
		assert.ok(records.every(record => record.status === 'completed'));
		assert.deepEqual(summary.counts, {
			totalPairs: 2, attemptedPairs: 2, completedPairs: 2, failedPairs: 0, unattemptedPairs: 0, recordedPairs: 2,
		});
		assert.equal(summary.status, 'completed');
		assert.equal(summary.elapsedScope, 'manifest_runner');
		assert.equal(summary.elapsedBoundary, 'sampled before final summary JSON serialization and write');
		assert.ok(Number.isFinite(summary.elapsedMs) && summary.elapsedMs >= 0);
		assert.equal(summary.upstream.commit, referenceLock.commit);
		assert.equal(summary.manifestSha256, crypto.createHash('sha256').update(fs.readFileSync(path.join(outputDir, 'manifest.json'))).digest('hex'));
		assert.deepEqual(JSON.parse(fs.readFileSync(path.join(outputDir, 'summary.json'), 'utf8')), summary);
		assert.equal(calls.generate, 4);
		assert.equal(calls.pack, 4);
		assert.equal(factoryCalls, 4);
		assert.equal(active, 0);
		assert.deepEqual(streams.map(stream => stream.destroyCalls), [1, 1, 1, 1]);
	} finally {
		fs.rmSync(root, { recursive: true, force: true });
	}
});

test('later pair failure journals the owned partial record and preserves earlier results', async () => {
	const root = newTempRoot('manifest-runner-later-failure-');
	const outputDir = path.join(root, 'results');
	const { ps, calls } = fakePs({ failGenerateAt: 3 });
	const streams = [];
	const error = await captureFailure(runManifest(optionsFor(outputDir, {
		ps, pairCount: 3,
		streamFactory: () => { const stream = new FakeStream(packets()); streams.push(stream); return stream; },
	})));
	try {
		const summary = JSON.parse(fs.readFileSync(path.join(outputDir, 'summary.json'), 'utf8'));
		const records = resultLines(outputDir);
		assert.deepEqual(records.map(record => record.pairIndex), [0, 1]);
		assert.deepEqual(records.map(record => record.status), ['completed', 'failed']);
		assert.equal(records[1].failure.stage, 'preparation');
		assert.deepEqual(summary.counts, {
			totalPairs: 3, attemptedPairs: 2, completedPairs: 1, failedPairs: 1, unattemptedPairs: 1, recordedPairs: 2,
		});
		assert.equal(summary.failure.stage, 'pair_execution');
		assert.equal(summary.failure.pairIndex, 1);
		assert.deepEqual(error.summary, summary);
		assert.equal(calls.generate, 3);
		assert.equal(streams.length, 2);
		assert.deepEqual(streams.map(stream => stream.destroyCalls), [1, 1]);
	} finally {
		fs.rmSync(root, { recursive: true, force: true });
	}
});

test('first pair partial battle failure is journaled and later pairs remain unattempted', async () => {
	const root = newTempRoot('manifest-runner-first-failure-');
	const outputDir = path.join(root, 'results');
	const { ps, calls } = fakePs();
	const streams = [];
	let streamIndex = 0;
	const error = await captureFailure(runManifest(optionsFor(outputDir, {
		ps, pairCount: 3,
		streamFactory: () => {
			const stream = new FakeStream(streamIndex++ === 0 ? packets() : []);
			streams.push(stream);
			return stream;
		},
	})));
	try {
		const summary = JSON.parse(fs.readFileSync(path.join(outputDir, 'summary.json'), 'utf8'));
		const records = resultLines(outputDir);
		assert.equal(records.length, 1);
		assert.equal(records[0].pairIndex, 0);
		assert.equal(records[0].runs[0].status, 'completed');
		assert.equal(records[0].runs[1].status, 'failed');
		assert.deepEqual(summary.counts, {
			totalPairs: 3, attemptedPairs: 1, completedPairs: 0, failedPairs: 1, unattemptedPairs: 2, recordedPairs: 1,
		});
		assert.equal(error.summary.failure.pairIndex, 0);
		assert.equal(calls.generate, 2);
		assert.equal(calls.pack, 2);
		assert.equal(streams.length, 2);
		assert.deepEqual(streams.map(stream => stream.destroyCalls), [1, 1]);
	} finally {
		fs.rmSync(root, { recursive: true, force: true });
	}
});

test('owned runner failure category is preserved in the manifest summary', async () => {
	const root = newTempRoot('manifest-runner-category-');
	const outputDir = path.join(root, 'results');
	const { ps } = fakePs();
	const error = await captureFailure(runManifest(optionsFor(outputDir, {
		ps, pairCount: 1, limits: { maxPackets: 1 },
		streamFactory: () => new FakeStream(packets()),
	})));
	try {
		const [record] = resultLines(outputDir);
		const summary = JSON.parse(fs.readFileSync(path.join(outputDir, 'summary.json'), 'utf8'));
		assert.equal(record.failure.category, 'packet_limit');
		assert.equal(record.failure.code, 'packet_limit_exhausted');
		assert.equal(summary.failure.category, record.failure.category);
		assert.equal(summary.failure.code, record.failure.code);
		assert.equal(summary.failure.pairIndex, record.pairIndex);
		assert.deepEqual(error.summary.failure, summary.failure);
	} finally {
		fs.rmSync(root, { recursive: true, force: true });
	}
});

test('owned retry exhaustion category is preserved in the manifest summary', async () => {
	const root = newTempRoot('manifest-runner-retry-limit-');
	const outputDir = path.join(root, 'results');
	const { ps } = fakePs();
	const initial = {
		rqid: 5,
		active: [{ moves: [
			{ move: 'Tackle', id: 'tackle', pp: 20, disabled: false },
			{ move: 'Growl', id: 'growl', pp: 40, disabled: false },
		] }],
		side: { pokemon: [{ active: true, condition: '100/100', details: 'Ownmon' }] },
	};
	const updated = { ...initial, update: true, rqid: 6 };
	const retryStream = new FakeStream([
		`sideupdate\np1\n|request|${JSON.stringify(initial)}`,
		'sideupdate\np1\n|error|[Unavailable choice] first correction',
		`sideupdate\np1\n|request|${JSON.stringify(updated)}`,
		'sideupdate\np1\n|error|[Unavailable choice] second correction',
	]);
	const error = await captureFailure(runManifest(optionsFor(outputDir, {
		ps, pairCount: 1, limits: { maxRetriesPerRequest: 1 }, streamFactory: () => retryStream,
	})));
	try {
		const [record] = resultLines(outputDir);
		const summary = JSON.parse(fs.readFileSync(path.join(outputDir, 'summary.json'), 'utf8'));
		assert.equal(record.failure.category, 'retry_limit');
		assert.equal(record.failure.code, 'retry_limit_exhausted');
		assert.equal(summary.failure.category, record.failure.category);
		assert.equal(summary.failure.code, record.failure.code);
		assert.equal(summary.failure.pairIndex, record.pairIndex);
		assert.equal(record.runs[1].status, 'not_attempted');
		assert.equal(retryStream.destroyCalls, 1);
		assert.deepEqual(error.summary.failure, summary.failure);
	} finally {
		fs.rmSync(root, { recursive: true, force: true });
	}
});

test('append failure distinguishes executed pairs from recorded lines and ignores foreign metadata', async () => {
	const root = newTempRoot('manifest-runner-append-failure-');
	const outputDir = path.join(root, 'results');
	const { ps } = fakePs();
	const originalAppend = fs.appendFileSync;
	let appends = 0;
	const injected = new Error('FOREIGN IO DETAILS');
	injected.pairResult = { status: 'completed', foreignData: 'INJECTED_PAIR_RESULT' };
	injected.summary = { status: 'completed', foreignData: 'INJECTED_SUMMARY' };
	fs.appendFileSync = function (file, ...args) {
		if (String(file).endsWith('results.jsonl') && ++appends === 2) throw injected;
		return originalAppend.call(this, file, ...args);
	};
	try {
		const error = await captureFailure(runManifest(optionsFor(outputDir, {
			ps, pairCount: 3,
			streamFactory: () => new FakeStream(packets()),
		})));
		const summary = JSON.parse(fs.readFileSync(path.join(outputDir, 'summary.json'), 'utf8'));
		assert.deepEqual(summary.counts, {
			totalPairs: 3, attemptedPairs: 2, completedPairs: 2, failedPairs: 0, unattemptedPairs: 1, recordedPairs: 1,
		});
		assert.equal(summary.failure.code, 'result_append_failed');
		assert.deepEqual(resultLines(outputDir).map(record => record.pairIndex), [0]);
		assert.deepEqual(error.summary, summary);
		assert.equal(JSON.stringify(error.summary).includes('INJECTED_'), false);
		assert.equal(error.message.includes('FOREIGN'), false);
	} finally {
		fs.appendFileSync = originalAppend;
		fs.rmSync(root, { recursive: true, force: true });
	}
});

test('summary-write failure is safe, nonzero, and does not claim a summary artifact', async () => {
	const root = newTempRoot('manifest-runner-summary-failure-');
	const outputDir = path.join(root, 'results');
	const { ps } = fakePs();
	const originalWrite = fs.writeFileSync;
	const injected = new Error('FOREIGN SUMMARY WRITE DETAILS');
	injected.summary = { status: 'completed', foreignData: 'INJECTED_SUMMARY' };
	fs.writeFileSync = function (file, ...args) {
		if (String(file).endsWith('summary.json')) throw injected;
		return originalWrite.call(this, file, ...args);
	};
	try {
		const error = await captureFailure(runManifest(optionsFor(outputDir, {
			ps,
			streamFactory: () => new FakeStream(packets()),
		})));
		assert.equal(error.message, 'Manifest run failed (summary_write_failed); summary.json could not be confirmed complete.');
		assert.equal(error.summary.status, 'failed');
		assert.equal(error.summary.failure.code, 'summary_write_failed');
		assert.equal(error.summary.counts.completedPairs, 2);
		assert.equal(error.summary.counts.recordedPairs, 2);
		assert.equal(fs.existsSync(path.join(outputDir, 'summary.json')), false);
		assert.equal(JSON.stringify(error.summary).includes('INJECTED_SUMMARY'), false);
	} finally {
		fs.writeFileSync = originalWrite;
		fs.rmSync(root, { recursive: true, force: true });
	}
});

test('CLI rejects parse and bootstrap spoofing before output creation', async () => {
	const root = newTempRoot('manifest-runner-cli-init-');
	const parsePath = path.join(root, 'parse-output');
	const bootstrapPath = path.join(root, 'bootstrap-output');
	const originalAssertReady = oracle.assertReady;
	const originalConsoleError = console.error;
	const originalExitCode = process.exitCode;
	const diagnostics = [];
	let bootstrapCalls = 0;
	try {
		oracle.assertReady = () => { bootstrapCalls++; };
		console.error = (...parts) => diagnostics.push(parts.join(' '));
		process.exitCode = undefined;
		await main(['--root-seed=1,2,3,4', '--pairs=1', `--output-dir=${parsePath}`, '--unknown']);
		assert.equal(process.exitCode, 1);
		assert.equal(bootstrapCalls, 0);
		assert.equal(fs.existsSync(parsePath), false);

		const injected = new Error('FOREIGN BOOTSTRAP DETAILS');
		injected.pairResult = { status: 'completed', foreignData: 'INJECTED_PAIR_RESULT' };
		injected.summary = { status: 'completed', foreignData: 'INJECTED_SUMMARY' };
		oracle.assertReady = () => { throw injected; };
		process.exitCode = undefined;
		await main(['--root-seed=1,2,3,4', '--pairs=1', `--output-dir=${bootstrapPath}`]);
		assert.equal(process.exitCode, 1);
		assert.equal(fs.existsSync(bootstrapPath), false);
		assert.ok(diagnostics.every(line => !line.includes('INJECTED_') && !line.includes('FOREIGN')));
		assert.match(diagnostics.at(-1), /pinned_reference_unavailable/);
	} finally {
		oracle.assertReady = originalAssertReady;
		console.error = originalConsoleError;
		process.exitCode = originalExitCode;
		fs.rmSync(root, { recursive: true, force: true });
	}
});

test('CLI writes a pinned success summary and reports an I/O spoof as safe nonzero failure', { timeout: 180_000 }, async () => {
	oracle.assertReady();
	const root = newTempRoot('manifest-runner-cli-');
	const successDir = path.join(root, 'success');
	const failedDir = path.join(root, 'io-failure');
	const success = spawnSync(process.execPath, [CLI, '--root-seed=1,2,3,4', '--pairs=1', `--output-dir=${successDir}`], {
		cwd: ROOT, encoding: 'utf8', timeout: 150_000,
	});
	try {
		assert.equal(success.status, 0, success.stderr);
		const stdoutSummary = JSON.parse(success.stdout);
		const fileSummary = JSON.parse(fs.readFileSync(path.join(successDir, 'summary.json'), 'utf8'));
		assert.deepEqual(stdoutSummary, fileSummary);
		assert.equal(stdoutSummary.status, 'completed');
		assert.deepEqual(stdoutSummary.counts, {
			totalPairs: 1, attemptedPairs: 1, completedPairs: 1, failedPairs: 0, unattemptedPairs: 0, recordedPairs: 1,
		});

		const originalAppend = fs.appendFileSync;
		const originalConsoleError = console.error;
		const originalExitCode = process.exitCode;
		const diagnostics = [];
		const injected = new Error('FOREIGN APPEND DETAILS');
		injected.summary = { status: 'completed', foreignData: 'INJECTED_SUMMARY' };
		fs.appendFileSync = function (file, ...args) {
			if (String(file).endsWith('results.jsonl')) throw injected;
			return originalAppend.call(this, file, ...args);
		};
		try {
			console.error = (...parts) => diagnostics.push(parts.join(' '));
			process.exitCode = undefined;
			await main(['--root-seed=1,2,3,4', '--pairs=1', `--output-dir=${failedDir}`]);
			assert.equal(process.exitCode, 1);
			assert.match(diagnostics.join('\n'), /result_append_failed/);
			assert.equal(diagnostics.join('\n').includes('INJECTED_SUMMARY'), false);
			const failureSummary = JSON.parse(fs.readFileSync(path.join(failedDir, 'summary.json'), 'utf8'));
			assert.equal(failureSummary.failure.code, 'result_append_failed');
			assert.equal(JSON.stringify(failureSummary).includes('INJECTED_SUMMARY'), false);
		} finally {
			fs.appendFileSync = originalAppend;
			console.error = originalConsoleError;
			process.exitCode = originalExitCode;
		}
	} finally {
		fs.rmSync(root, { recursive: true, force: true });
	}
});

test('pinned manifest execution exchanges packed teams and policy identities one pair at a time', { timeout: 180_000 }, async () => {
	oracle.assertReady();
	const ps = require('../reference/vendor/pokemon-showdown/dist/sim');
	const root = newTempRoot('manifest-runner-pinned-');
	const outputDir = path.join(root, 'results');
	const expectedManifest = generateManifest(ROOT_SEED, 2);
	const originalGenerate = ps.Teams.generate;
	const originalPack = ps.Teams.pack;
	let generations = 0;
	let packs = 0;
	let nextStream = 0;
	let activeStreams = 0;
	const streams = [];
	const writes = [];
	ps.Teams.generate = function (...args) { generations++; return originalGenerate.apply(this, args); };
	ps.Teams.pack = function (...args) { packs++; return originalPack.apply(this, args); };
	const streamFactory = () => {
		assert.equal(activeStreams, 0, 'previous battle stream was not cleaned up');
		const pairIndex = Math.floor(nextStream / 2);
		const runIndex = nextStream % 2;
		nextStream++;
		const stream = new ps.BattleStream({ keepAlive: true });
		streams.push(stream);
		activeStreams++;
		let destroyed = false;
		return {
			write(command) {
				const match = /^>player (p[12]) (.+)$/.exec(command);
				if (match) writes.push({ pairIndex, runIndex, player: match[1], packedTeam: JSON.parse(match[2]).team });
				return stream.write(command);
			},
			async destroy() {
				try { return await stream.destroy(); } finally {
					if (!destroyed) { destroyed = true; activeStreams--; }
				}
			},
			[Symbol.asyncIterator]() { return stream[Symbol.asyncIterator](); },
		};
	};
	try {
		const summary = await runManifest({ ps, rootSeed: ROOT_SEED, pairCount: 2, outputDir, streamFactory });
		const manifest = JSON.parse(fs.readFileSync(path.join(outputDir, 'manifest.json'), 'utf8'));
		const results = resultLines(outputDir);
		assert.deepEqual(manifest, expectedManifest);
		assert.deepEqual(results.map(result => result.pairIndex), [0, 1]);
		assert.deepEqual(results.map(result => result.pairId), expectedManifest.pairs.map(pair => pair.pairId));
		assert.ok(results.every(result => result.status === 'completed'));
		assert.ok(results.every(result => result.runs.every(run => run.status === 'completed' && run.resultRecord.status === 'completed')));
		assert.deepEqual(results.map(result => result.runs.map(run => run.seeds)), expectedManifest.pairs.map(pair => pair.runs.map(run => run.seeds)));
		assert.deepEqual(results.map(result => result.runs.map(run => run.policySeats)), expectedManifest.pairs.map(pair => pair.runs.map(run => run.policySeats)));
		assert.deepEqual(summary.counts, {
			totalPairs: 2, attemptedPairs: 2, completedPairs: 2, failedPairs: 0, unattemptedPairs: 0, recordedPairs: 2,
		});
		assert.equal(generations, 4);
		assert.equal(packs, 4);
		assert.equal(nextStream, 4);
		assert.equal(activeStreams, 0);
		assert.equal(streams.length, 4);
		assert.equal(writes.length, 8);
		for (let pairIndex = 0; pairIndex < 2; pairIndex++) {
			const pairWrites = writes.filter(write => write.pairIndex === pairIndex);
			const result = results[pairIndex];
			for (let runIndex = 0; runIndex < 2; runIndex++) {
				const runWrites = pairWrites.filter(write => write.runIndex === runIndex);
				const run = result.runs[runIndex];
				assert.deepEqual(runWrites.map(write => write.player), ['p1', 'p2']);
				assert.deepEqual(runWrites.map(write => crypto.createHash('sha256').update(write.packedTeam, 'utf8').digest('hex')), [
					result.teams[run.seats.p1Team].packedTeamSha256,
					result.teams[run.seats.p2Team].packedTeamSha256,
				]);
			}
		}
	} finally {
		ps.Teams.generate = originalGenerate;
		ps.Teams.pack = originalPack;
		fs.rmSync(root, { recursive: true, force: true });
	}
});
