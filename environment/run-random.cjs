#!/usr/bin/env node
'use strict';

const { main } = require('./battle-runner.cjs');

main().catch(error => {
	console.error(`Random battle runner failed: ${error.message}`);
	process.exitCode = 1;
});
