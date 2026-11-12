'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),zlib=require('node:zlib'),{execFileSync}=require('node:child_process');
test('pinned Gen5 Substitute corpus preserves primary-hit distinctions and complete games',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'g4-substitute-'));
 try{
  const out=path.join(dir,'native.json');execFileSync('node',['reference/shared-engine-cases.cjs','--batch','substitute','--output',out]);
  const actual=JSON.parse(fs.readFileSync(out));assert.deepEqual(actual,JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/substitute.json.gz'))));
  assert.equal(actual.cases.length,51);
  const find=id=>actual.cases.find(c=>c.id==='G4-substitute-'+id);
  const messages=id=>find(id).steps[0].events.p1.map(e=>e.message);
  const pound=find('Pound').steps[0].after.sides;
  assert.equal(pound[0].pokemon[0].volatiles.substitute.hp,26);assert.deepEqual(pound[0].pokemon[0].phaseState.attackedBy,[]);assert.equal(pound[1].pokemon[0].phaseState.lastDamage,17);
  assert.ok(messages('weak').some(s=>s.endsWith('|move: Substitute|[weak]')));
  assert.ok(messages('ThunderWave').includes('|-fail|p2a: Mew'));
  assert.ok(messages('sound').includes('|-activate|p1a: Mew|Substitute|[damage]'));
  const broken=messages('break-multihit');assert.ok(broken.includes('|-end|p1a: Mew|Substitute'));assert.ok(broken.filter(s=>s.startsWith('|-damage|p1a: Mew|')).length>1);
  assert.ok(messages('rough-skin').every(s=>!s.includes('Rough Skin')&&!s.includes('Rocky Helmet')));
  assert.ok(messages('drain-ooze').some(s=>s.includes('Liquid Ooze')));
  for(const c of actual.cases){assert.ok(c.steps.every(s=>s.accepted.every(a=>a.accepted)),c.id);if(c.setup.fullBattle)assert.equal(c.steps.at(-1).after.terminal,true);}
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
