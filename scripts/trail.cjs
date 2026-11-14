#!/usr/bin/env node
'use strict';
// Portable project-local CLI/MCP entrypoint; no global install or subprocess.
const path=require('node:path'),fs=require('node:fs'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'..'),tool=path.join(root,'tools/trail-code');
const entry=path.join(tool,'node_modules/trail-code/dist/bin/trail.js');
if(!fs.existsSync(entry)){
 console.error('Trail is not installed for this checkout. Run: node scripts/setup-trail.cjs');process.exit(1);
}
const expected=JSON.parse(fs.readFileSync(path.join(tool,'package.json'))).dependencies['trail-code'];
const actual=JSON.parse(fs.readFileSync(path.join(tool,'node_modules/trail-code/package.json'))).version;
if(actual!==expected){console.error(`Trail ${actual} differs from project pin ${expected}; run node scripts/setup-trail.cjs`);process.exit(1);}
process.chdir(root);process.env.DO_NOT_TRACK='1';process.argv[1]=entry;
import(pathToFileURL(entry).href).catch(error=>{console.error(error);process.exitCode=1;});
