#!/usr/bin/env node
'use strict';

const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

const root = path.resolve(__dirname, '..');
const lock = JSON.parse(fs.readFileSync(path.join(root, 'reference-lock.json'), 'utf8'));
const checkout = path.join(__dirname, 'vendor', 'pokemon-showdown');

function run(command, args, cwd = root) {
	const safeArgs = command === 'git' ? ['-c', `safe.directory=${path.resolve(cwd)}`, ...args] : args;
	const result = spawnSync(command, safeArgs, { cwd, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
	if (result.error) throw result.error;
	if (result.status !== 0) {
		const details = (result.stderr || result.stdout || '').trim();
		throw new Error(`${command} ${args.join(' ')} failed${details ? `:\n${details}` : ''}`);
	}
	return result.stdout.trim();
}

function assertReferenceIdentity(expected, actual) {
	if (actual.origin !== expected.repository) {
		throw new Error(`Showdown origin mismatch: expected ${expected.repository}, found ${actual.origin || '(missing)'}.`);
	}
	if (actual.commit !== expected.commit) {
		throw new Error(`Showdown commit mismatch: expected ${expected.commit}, found ${actual.commit || '(missing)'}.`);
	}
	if (actual.dirty) {
		throw new Error(`Showdown checkout is modified (${actual.dirty.split('\n')[0]}). Restore or reclone it manually before continuing; setup will not overwrite local changes.`);
	}
}

function inspectCheckout(dir) {
	const origin = run('git', ['remote', 'get-url', 'origin'], dir);
	const commit = run('git', ['rev-parse', 'HEAD'], dir);
	const dirty = run('git', ['status', '--porcelain', '--untracked-files=normal'], dir);
	return { origin, commit, dirty };
}

function clonePinnedCheckout() {
	if (fs.existsSync(checkout)) {
		throw new Error(`Reference path already exists without a usable checkout: ${checkout}. Inspect or remove it manually; setup will not overwrite it.`);
	}
	const vendor = path.dirname(checkout);
	fs.mkdirSync(vendor, { recursive: true });
	const temporary = fs.mkdtempSync(path.join(vendor, '.pokemon-showdown-bootstrap-'));
	try {
		run('git', ['clone', '--filter=blob:none', '--no-checkout', '--', lock.repository, temporary]);
		run('git', ['fetch', '--depth', '1', 'origin', lock.commit], temporary);
		run('git', ['checkout', '--detach', lock.commit], temporary);
		assertReferenceIdentity(lock, inspectCheckout(temporary));
		if (fs.existsSync(checkout)) {
			throw new Error(`Reference path appeared during setup: ${checkout}. No files were replaced.`);
		}
		fs.renameSync(temporary, checkout);
	} catch (error) {
		fs.rmSync(temporary, { recursive: true, force: true });
		throw error;
	}
}

async function main() {
	if (!fs.existsSync(checkout)) clonePinnedCheckout();
	assertReferenceIdentity(lock, inspectCheckout(checkout));
	if (!fs.existsSync(path.join(checkout, 'package-lock.json'))) {
		throw new Error(`Pinned checkout has no package-lock.json: ${checkout}`);
	}
	run('npm', ['ci'], checkout);
	run('npm', ['run', 'build'], checkout);
	console.log(`Showdown ready at ${lock.commit}`);
	console.log(`Node ${process.version}; npm ${run('npm', ['--version'])}`);
}

if (require.main === module) {
	main().catch(error => {
		console.error(`Reference setup failed: ${error.message}`);
		process.exitCode = 1;
	});
}

module.exports = { assertReferenceIdentity, inspectCheckout };
