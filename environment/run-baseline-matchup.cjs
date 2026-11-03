#!/usr/bin/env node
'use strict';

const fs = require('node:fs');
const path = require('node:path');
const { MAX_PAIRS, parseRootSeed } = require('./paired-seeds.cjs');
const { POLICY_ID: MOVE_FIRST_POLICY_ID } = require('./move-first-policy.cjs');
const { runBaselineMatchup } = require('./baseline-matchup.cjs');

const POLICY_IDS = new Set(['legal-random-v1', MOVE_FIRST_POLICY_ID]);
const ARGUMENT_NAMES = new Set(['--root-seed', '--pair-index', '--policy-a', '--policy-b', '--output']);
const HELP = 'Usage: node environment/run-baseline-matchup.cjs --root-seed 1,2,3,4 --pair-index 0 --policy-a legal-random-v1 --policy-b move-first-v1 --output NEW_FILE\n';

function parseArgs(args) {
	if (!Array.isArray(args) || args.some(argument => typeof argument !== 'string')) {
		throw new TypeError('arguments must be strings.');
	}
	if (args.length === 1 && ['--help', 'help'].includes(args[0])) return { help: true };

	const values = Object.create(null);
	for (let index = 0; index < args.length; index++) {
		const argument = args[index];
		const separator = argument.indexOf('=');
		const name = separator < 0 ? argument : argument.slice(0, separator);
		if (!ARGUMENT_NAMES.has(name)) throw new TypeError('unknown argument.');
		if (Object.hasOwn(values, name)) throw new TypeError('duplicate argument.');
		const value = separator < 0 ? args[++index] : argument.slice(separator + 1);
		if (typeof value !== 'string' || value.length === 0 || value.startsWith('--')) {
			throw new TypeError('argument requires a value.');
		}
		values[name] = value;
	}
	for (const name of ARGUMENT_NAMES) {
		if (!Object.hasOwn(values, name)) throw new TypeError('required argument missing.');
	}
	if (!/^(?:0|[1-9]\d*)$/.test(values['--pair-index'])) throw new TypeError('invalid pair index.');
	const pairIndex = Number(values['--pair-index']);
	if (!Number.isSafeInteger(pairIndex) || pairIndex >= MAX_PAIRS) throw new TypeError('invalid pair index.');
	if (!POLICY_IDS.has(values['--policy-a']) || !POLICY_IDS.has(values['--policy-b'])) {
		throw new TypeError('unsupported policy identifier.');
	}
	if (values['--output'].includes('\0')) throw new TypeError('invalid output path.');
	return {
		rootSeed: parseRootSeed(values['--root-seed']),
		pairIndex,
		policyA: values['--policy-a'],
		policyB: values['--policy-b'],
		outputPath: path.resolve(values['--output']),
	};
}

function sameFile(left, right) {
	return left.dev === right.dev && left.ino === right.ino && left.isFile() && right.isFile();
}

function targetIsReservation(fileSystem, outputPath, descriptor, expected) {
	try {
		return (descriptor === undefined || sameFile(fileSystem.fstatSync(descriptor), expected))
			&& sameFile(fileSystem.lstatSync(outputPath), expected);
	} catch {
		return false;
	}
}

function closeQuietly(fileSystem, descriptor) {
	if (descriptor === undefined) return;
	try { fileSystem.closeSync(descriptor); } catch {}
}

function removeOwnedReservation(fileSystem, outputPath, descriptor, expected) {
	if (!targetIsReservation(fileSystem, outputPath, descriptor, expected)) return false;
	try {
		fileSystem.unlinkSync(outputPath);
		return true;
	} catch {
		return false;
	}
}

async function execute(options, dependencies = {}) {
	const fileSystem = dependencies.fs || fs;
	const bootstrap = dependencies.bootstrap || (() => {
		const oracle = require('../reference/oracle.cjs');
		oracle.assertReady();
		return require('../reference/vendor/pokemon-showdown/dist/sim');
	});
	const runMatchup = dependencies.runMatchup || runBaselineMatchup;
	const stdout = dependencies.stdout || process.stdout;
	const stderr = dependencies.stderr || process.stderr;

	let ps;
	try {
		ps = bootstrap();
	} catch {
		stderr.write('Named baseline matchup failed (pinned_reference_unavailable).\n');
		return 1;
	}

	let descriptor;
	let expected;
	try {
		descriptor = fileSystem.openSync(options.outputPath, 'wx', 0o600);
		expected = fileSystem.fstatSync(descriptor);
		if (!expected.isFile()) throw new Error('not a regular file');
	} catch {
		closeQuietly(fileSystem, descriptor);
		if (expected) removeOwnedReservation(fileSystem, options.outputPath, undefined, expected);
		stderr.write('Named baseline matchup failed (output_unavailable).\n');
		return 1;
	}

	let artifact;
	try {
		artifact = await runMatchup({
			ps,
			rootSeed: [...options.rootSeed],
			pairIndex: options.pairIndex,
			policyA: options.policyA,
			policyB: options.policyB,
		});
	} catch {
		closeQuietly(fileSystem, descriptor);
		removeOwnedReservation(fileSystem, options.outputPath, undefined, expected);
		stderr.write('Named baseline matchup failed (matchup_execution_failed).\n');
		return 1;
	}

	try {
		const contents = `${JSON.stringify(artifact, null, 2)}\n`;
		if (!targetIsReservation(fileSystem, options.outputPath, descriptor, expected)) throw new Error('output path changed');
		fileSystem.writeFileSync(descriptor, contents, 'utf8');
		fileSystem.fsyncSync(descriptor);
		if (!targetIsReservation(fileSystem, options.outputPath, descriptor, expected)) throw new Error('output path changed');
		fileSystem.closeSync(descriptor);
		descriptor = undefined;
		if (!sameFile(fileSystem.lstatSync(options.outputPath), expected)) throw new Error('output path changed');
	} catch {
		closeQuietly(fileSystem, descriptor);
		removeOwnedReservation(fileSystem, options.outputPath, undefined, expected);
		stderr.write('Named baseline matchup failed (artifact_write_failed).\n');
		return 1;
	}

	stdout.write(`${JSON.stringify({ kind: artifact.kind, pairId: artifact.pairId, status: artifact.status, output: options.outputPath }, null, 2)}\n`);
	return 0;
}

async function main(args = process.argv.slice(2), dependencies = {}) {
	let options;
	try {
		options = parseArgs(args);
	} catch {
		(dependencies.stderr || process.stderr).write('Named baseline matchup failed (invalid_cli_configuration).\n');
		return 1;
	}
	if (options.help) {
		(dependencies.stdout || process.stdout).write(HELP);
		return 0;
	}
	return execute(options, dependencies);
}

if (require.main === module) {
	main().then(code => { process.exitCode = code; }, () => {
		console.error('Named baseline matchup failed (internal_error).');
		process.exitCode = 1;
	});
}

module.exports = { main, parseArgs };
