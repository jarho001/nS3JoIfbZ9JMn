#!/usr/bin/env node
'use strict';
// Small real CLI/MCP preflight; no simulator execution or external service.
const {spawn,spawnSync}=require('node:child_process');
const path=require('node:path'),fs=require('node:fs'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'..'),launcher=path.join(root,'scripts/trail.cjs');
function cli(args){const r=spawnSync(process.execPath,[launcher,...args],{cwd:root,encoding:'utf8',timeout:30000,maxBuffer:2*1024*1024});if(r.error)throw r.error;assert.equal(r.status,0,r.stderr||r.stdout);return r.stdout;}
const pin=JSON.parse(fs.readFileSync(path.join(root,'tools/trail-code/package.json'))).dependencies['trail-code'];
assert.equal(cli(['--version']).trim(),pin);
const answer=JSON.parse(cli(['ask','absorbs','--json']));
assert.ok(answer.hits.some(h=>h.pointer.includes('simulator/status_immunity.py')),'CLI query must find the current absorption implementation');
console.log(`Trail ${pin}: CLI query found simulator/status_immunity.py`);
const child=spawn(process.execPath,[launcher,'mcp'],{cwd:root,stdio:['pipe','pipe','pipe']});
let buffer='',diagnostics='',finished=false;
const timeout=setTimeout(()=>finish(new Error('MCP preflight timed out')),30000);
function finish(error){if(finished)return;finished=true;clearTimeout(timeout);child.stdin.end();child.kill();if(error){console.error(error.message);if(diagnostics)console.error(diagnostics);process.exitCode=1;}else console.log('Trail MCP: initialize, tools/list and live code query passed');}
function send(id,method,params){child.stdin.write(JSON.stringify({jsonrpc:'2.0',id,method,params})+'\n');}
child.stderr.on('data',b=>{diagnostics+=b.toString();});
child.on('error',finish);child.on('exit',code=>{if(!finished)finish(new Error(`MCP exited early (${code})`));});
child.stdout.on('data',b=>{buffer+=b.toString();let end;while((end=buffer.indexOf('\n'))>=0){const line=buffer.slice(0,end);buffer=buffer.slice(end+1);if(!line.trim())continue;try{const r=JSON.parse(line);assert.ok(!r.error,JSON.stringify(r.error));if(r.id===1){send(undefined,'notifications/initialized',{});send(2,'tools/list',{});}else if(r.id===2){assert.ok(r.result.tools.some(t=>t.name==='trail_find_code'));send(3,'tools/call',{name:'trail_find_code',arguments:{query:'absorbs',limit:3}});}else if(r.id===3){assert.ok(!r.result.isError,JSON.stringify(r.result));assert.ok(JSON.stringify(r.result).includes('simulator/status_immunity.py'));finish();}}catch(error){finish(error);}}});
send(1,'initialize',{protocolVersion:'2024-11-05',capabilities:{},clientInfo:{name:'project-trail-preflight',version:'1'}});
