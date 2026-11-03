'use strict';

const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const referenceLock = require('../reference-lock.json');
const { MAX_PAIRS, generatePair } = require('../environment/paired-seeds.cjs');
const { main, parseArgs } = require('../environment/run-baseline-matchup.cjs');

const ROOT = path.resolve(__dirname, '..');
const CLI = path.join(ROOT, 'environment/run-baseline-matchup.cjs');
const args = (output, rootSeed = '1,2,3,4') => [
	'--root-seed', rootSeed, '--pair-index=0', '--policy-a', 'legal-random-v1',
	'--policy-b=move-first-v1', '--output', output,
];

function streams() {
	const stdout = { value: '', write(chunk) { this.value += String(chunk); return true; } };
	const stderr = { value: '', write(chunk) { this.value += String(chunk); return true; } };
	return { stdout, stderr };
}

function temporaryDirectory() {
	return fs.mkdtempSync(path.join(os.tmpdir(), 'named-baseline-cli-'));
}

function fakeFileSystem(overrides = {}) {
	return Object.assign(Object.create(fs), overrides);
}

const fakeArtifact = {
	schemaVersion: 1,
	kind: 'gen5-named-baseline-paired-matchup',
	status: 'completed',
	pairId: 'pair-test',
};

test('parseArgs accepts explicit separated and equals forms and only sole help', () => {
	assert.deepEqual(parseArgs(args('result.json')), {
		rootSeed: [1, 2, 3, 4],
		pairIndex: 0,
		policyA: 'legal-random-v1',
		policyB: 'move-first-v1',
		outputPath: path.resolve('result.json'),
	});
	assert.deepEqual(parseArgs(['--help']), { help: true });
	for (const invalid of [
		[], ['--help', '--output', 'x'], args('x').slice(0, -2),
		[...args('x'), '--pair-index', '1'], [...args('x'), '--unknown', 'value'],
		['--root-seed', '1,2,3,4', '--pair-index', '+1', '--policy-a', 'legal-random-v1', '--policy-b', 'move-first-v1', '--output', 'x'],
		['--root-seed', '1,2,3,4', '--pair-index', String(MAX_PAIRS), '--policy-a', 'legal-random-v1', '--policy-b', 'move-first-v1', '--output', 'x'],
		['--root-seed', '1,2,3,4', '--pair-index', '0', '--policy-a', 'unknown', '--policy-b', 'move-first-v1', '--output', 'x'],
		['--root-seed', '1,2,3,4', '--pair-index', '0', '--policy-a', 'legal-random-v1', '--policy-b', 'move-first-v1', '--output', ''],
	]) assert.throws(() => parseArgs(invalid));
});

test('main validates before bootstrap and help does not inspect the reference', async () => {
	const help = streams();
	assert.equal(await main(['--help'], { ...help, bootstrap() { assert.fail('help bootstrapped'); } }), 0);
	assert.match(help.stdout.value, /Usage: node environment\/run-baseline-matchup\.cjs/);
	const invalid = streams();
	let bootstrapCalls = 0;
	assert.equal(await main(['--policy-a', 'unknown'], { ...invalid, bootstrap() { bootstrapCalls++; } }), 1);
	assert.equal(await main(args('unused.json', '1,2,3,65536'), { ...invalid, bootstrap() { bootstrapCalls++; } }), 1);
	assert.equal(bootstrapCalls, 0);
	assert.equal(invalid.stderr.value, 'Named baseline matchup failed (invalid_cli_configuration).\n'.repeat(2));
});

test('bootstrap and output reservation failures start no matchup and preserve existing paths', async () => {
	const directory = temporaryDirectory();
	try {
		const output = path.join(directory, 'result.json');
		fs.writeFileSync(output, 'keep this');
		const io = streams();
		let runCalls = 0;
		assert.equal(await main(args(output), {
			...io,
			bootstrap() { return {}; },
			runMatchup() { runCalls++; return fakeArtifact; },
		}), 1);
		assert.equal(runCalls, 0);
		assert.equal(fs.readFileSync(output, 'utf8'), 'keep this');
		assert.equal(io.stderr.value, 'Named baseline matchup failed (output_unavailable).\n');

		const noParent = path.join(directory, 'missing', 'out.json');
		const failedBootstrap = streams();
		assert.equal(await main(args(noParent), {
			...failedBootstrap,
			bootstrap() { throw new Error('private bootstrap detail'); },
			runMatchup() { runCalls++; return fakeArtifact; },
		}), 1);
		assert.equal(fs.existsSync(noParent), false);
		assert.equal(failedBootstrap.stderr.value, 'Named baseline matchup failed (pinned_reference_unavailable).\n');
		assert.equal(runCalls, 0);
		const noParentAfterBootstrap = streams();
		assert.equal(await main(args(noParent), {
			...noParentAfterBootstrap,
			bootstrap() { return {}; },
			runMatchup() { runCalls++; return fakeArtifact; },
		}), 1);
		assert.equal(fs.existsSync(noParent), false);
		assert.equal(noParentAfterBootstrap.stderr.value, 'Named baseline matchup failed (output_unavailable).\n');
		assert.equal(runCalls, 0);

		const target = path.join(directory, 'target');
		const link = path.join(directory, 'link.json');
		fs.writeFileSync(target, 'symlink target stays');
		fs.symlinkSync(target, link);
		const failedLink = streams();
		assert.equal(await main(args(link), {
			...failedLink, bootstrap: () => ({}), runMatchup() { runCalls++; },
		}), 1);
		assert.equal(fs.lstatSync(link).isSymbolicLink(), true);
		assert.equal(fs.readFileSync(target, 'utf8'), 'symlink target stays');
	} finally {
		fs.rmSync(directory, { recursive: true, force: true });
	}
});

test('a failed matchup emits no artifact and removes only its reserved file', async () => {
	const directory = temporaryDirectory();
	try {
		const output = path.join(directory, 'result.json');
		const io = streams();
		let seenOptions;
		assert.equal(await main(args(output), {
			...io,
			bootstrap: () => ({ pinned: true }),
			runMatchup: async received => {
				seenOptions = received;
				assert.equal(fs.existsSync(output), true);
				throw new Error('foreign exception details must not escape');
			},
		}), 1);
		assert.deepEqual(seenOptions, { ps: { pinned: true }, rootSeed: [1, 2, 3, 4], pairIndex: 0, policyA: 'legal-random-v1', policyB: 'move-first-v1' });
		assert.equal(fs.existsSync(output), false);
		assert.equal(io.stdout.value, '');
		assert.equal(io.stderr.value, 'Named baseline matchup failed (matchup_execution_failed).\n');
	} finally {
		fs.rmSync(directory, { recursive: true, force: true });
	}
});

test('a completed artifact is written, synced, closed, and only then reported', async () => {
	const directory = temporaryDirectory();
	try {
		const output = path.join(directory, 'result.json');
		const io = streams();
		let closed = false;
		const injectedFs = fakeFileSystem({
			closeSync(descriptor) { fs.closeSync(descriptor); closed = true; },
		});
		assert.equal(await main(args(output), {
			...io, fs: injectedFs, bootstrap: () => ({}), runMatchup: async () => fakeArtifact,
		}), 0);
		assert.equal(closed, true);
		assert.deepEqual(JSON.parse(fs.readFileSync(output, 'utf8')), fakeArtifact);
		assert.deepEqual(JSON.parse(io.stdout.value), {
			kind: fakeArtifact.kind, pairId: fakeArtifact.pairId, status: fakeArtifact.status, output,
		});
		assert.equal(io.stderr.value, '');
	} finally {
		fs.rmSync(directory, { recursive: true, force: true });
	}
});

for (const failedOperation of ['writeFileSync', 'fsyncSync', 'closeSync']) {
	test(`a ${failedOperation} failure reports failure and cleans its owned output reservation`, async () => {
		const directory = temporaryDirectory();
		try {
			const output = path.join(directory, 'result.json');
			const io = streams();
			let closeCalls = 0;
			const injectedFs = fakeFileSystem({
				[failedOperation](descriptor) {
					if (failedOperation === 'closeSync') {
						if (closeCalls++ === 0) throw new Error('foreign close detail');
						return fs.closeSync(descriptor);
					}
					throw new Error(`foreign ${failedOperation} detail`);
				},
			});
			assert.equal(await main(args(output), {
				...io, fs: injectedFs, bootstrap: () => ({}), runMatchup: async () => fakeArtifact,
			}), 1);
			assert.equal(fs.existsSync(output), false);
			assert.equal(io.stdout.value, '');
			assert.equal(io.stderr.value, 'Named baseline matchup failed (artifact_write_failed).\n');
		} finally {
			fs.rmSync(directory, { recursive: true, force: true });
		}
	});
}

test('path replacement during a run is detected without deleting the replacement', async () => {
	const directory = temporaryDirectory();
	try {
		const output = path.join(directory, 'result.json');
		const movedReservation = path.join(directory, 'moved-reservation');
		const io = streams();
		assert.equal(await main(args(output), {
			...io,
			bootstrap: () => ({}),
			async runMatchup() {
				fs.renameSync(output, movedReservation);
				fs.writeFileSync(output, 'replacement is unrelated');
				return fakeArtifact;
			},
		}), 1);
		assert.equal(fs.readFileSync(output, 'utf8'), 'replacement is unrelated');
		assert.equal(fs.readFileSync(movedReservation, 'utf8'), '');
		assert.equal(io.stdout.value, '');
		assert.equal(io.stderr.value, 'Named baseline matchup failed (artifact_write_failed).\n');
	} finally {
		fs.rmSync(directory, { recursive: true, force: true });
	}
});

test('the command writes a real guarded pinned mixed-policy matchup artifact', { timeout: 120_000 }, () => {
	const directory = temporaryDirectory();
	try {
		const output = path.join(directory, 'matchup.json');
		const child = spawnSync(process.execPath, [
			CLI,
			'--root-seed=1,2,3,4',
			'--pair-index', '0',
			'--policy-a', 'legal-random-v1',
			'--policy-b=move-first-v1',
			'--output', output,
		], { cwd: ROOT, encoding: 'utf8', timeout: 110_000 });
		assert.equal(child.error, undefined, child.error?.message);
		assert.equal(child.status, 0, child.stderr);
		const artifact = JSON.parse(fs.readFileSync(output, 'utf8'));
		const canonical = generatePair([1, 2, 3, 4], 0);
		assert.equal(artifact.kind, 'gen5-named-baseline-paired-matchup');
		assert.equal(artifact.status, 'completed');
		assert.equal(artifact.formatId, 'gen5randombattle');
		assert.deepEqual(artifact.upstream, {
			repository: referenceLock.repository,
			commit: referenceLock.commit,
		});
		assert.deepEqual(artifact.rootSeed, [1, 2, 3, 4]);
		assert.equal(artifact.pairIndex, 0);
		assert.equal(artifact.pairId, canonical.pairId);
		assert.deepEqual(artifact.policiesByRole, { policyA: 'legal-random-v1', policyB: 'move-first-v1' });
		assert.equal(artifact.runs.length, 2);
		assert.deepEqual(artifact.runs.map(({ runId, seats, policySeats, seeds }) => ({ runId, seats, policySeats, seeds })),
			canonical.runs.map(({ runId, seats, policySeats, seeds }) => ({ runId, seats, policySeats, seeds })));
		for (const run of artifact.runs) {
			assert.equal(run.status, 'completed');
			assert.equal(run.policyIdsBySeat.p1, run.policySeats.p1 === 'policyA' ? 'legal-random-v1' : 'move-first-v1');
			assert.equal(run.policyIdsBySeat.p2, run.policySeats.p2 === 'policyA' ? 'legal-random-v1' : 'move-first-v1');
			assert.equal(run.resultRecord.kind, 'battle-run-result');
			assert.equal(run.resultRecord.status, 'completed');
			assert.equal(run.resultRecord.teamSource, 'prepared');
			assert.deepEqual(run.resultRecord.seeds, run.seeds);
			assert.ok(run.resultRecord.outcome);
		}
		const report = JSON.parse(child.stdout);
		assert.deepEqual(report, {
			kind: artifact.kind, pairId: artifact.pairId, status: 'completed', output,
		});
		assert.equal(child.stderr, '');
	} finally {
		fs.rmSync(directory, { recursive: true, force: true });
	}
});
