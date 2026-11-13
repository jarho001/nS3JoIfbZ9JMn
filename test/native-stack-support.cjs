'use strict';
// Only the checkout prefix in diagnostic draw stacks is host-dependent.
function normalizeStacks(value){
 if(Array.isArray(value))return value.map(normalizeStacks);
 if(value&&typeof value==='object')return Object.fromEntries(Object.entries(value).map(([key,v])=>[key,key==='stack'?v.map(line=>line.replace(/\/[^()]*\/reference\//g,'reference/')):normalizeStacks(v)]));
 return value;
}
module.exports={normalizeStacks};
