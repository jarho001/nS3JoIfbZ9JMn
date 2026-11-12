'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),zlib=require('node:zlib'),{execFileSync}=require('node:child_process');
test('pinned phazing covers synchronous entry, cancellation and faint ordering',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'g4-phaze-'));
 try{
  const out=path.join(dir,'native.json');execFileSync('node',['reference/shared-engine-cases.cjs','--batch','phazing','--output',out]);
  const x=JSON.parse(fs.readFileSync(out));assert.deepEqual(x,JSON.parse(zlib.gunzipSync(fs.readFileSync('test-logs/g4-cpu-conformance/phazing.json.gz'))));assert.equal(x.cases.length,36);assert.equal(x.cases.reduce((n,c)=>n+c.steps.length,0),46);
  const find=id=>x.cases.find(c=>c.id==='G4-phaze-'+id),lines=id=>find(id).steps.flatMap(s=>s.events.p1.map(e=>e.message));
  assert.equal(lines('tied-cancel').filter(s=>s.startsWith('|move|')).length,1);
  assert.ok(lines('source-contact-ko').some(s=>s.startsWith('|drag|')));
  const end=lines('source-orb-terminal');assert.ok(end.findIndex(s=>s.startsWith('|win|'))<end.findIndex(s=>s.startsWith('|drag|')));
  assert.ok(lines('soundproof-ghost').includes('|-immune|p2a: Gengar'));assert.ok(lines('roar-soundproof').some(s=>s.includes('ability: Soundproof')));
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
