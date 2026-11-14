#!/usr/bin/env node
'use strict';
// Fresh checkout/cloud setup; locked install, deterministic map and live check.
const path=require('node:path'),{spawnSync}=require('node:child_process');
const root=path.resolve(__dirname,'..'),tool=path.join(root,'tools/trail-code');
const [major,minor]=process.versions.node.split('.').map(Number);
if(!(major>22||major===22&&minor>=3||major===20&&minor>=16)){
 console.error('Trail requires Node.js 22.3+ (or 20.16+) and npm.');process.exit(1);
}
function run(command,args,options={}){
 const r=spawnSync(command,args,{cwd:root,stdio:'inherit',...options});
 if(r.error){console.error(r.error.message);process.exit(1);}
 if(r.status!==0)process.exit(r.status||1);
}
// On Windows, npm.cmd needs the shell; every shell argument here is a fixed
// literal. The checkout path goes through cwd, never shell interpolation.
const windows=process.platform==='win32';
run(windows?'npm.cmd':'npm',['ci','--no-audit','--no-fund'],{
 cwd:tool,shell:windows,env:{...process.env,ONNXRUNTIME_NODE_INSTALL_CUDA:'skip'}
});
const launcher=path.join(root,'scripts/trail.cjs');
run(process.execPath,[launcher,'build',root,'--no-semantic']);
run(process.execPath,[launcher,'status',root]);
run(process.execPath,[path.join(root,'scripts/check-trail.cjs')]);
