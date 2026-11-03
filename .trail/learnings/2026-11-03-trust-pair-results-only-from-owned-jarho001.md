---
title: Trust pair results only from owned execution
author: Jarho001
date: 2026-11-03
branch: main
id: l-d998d91c
scope: repo
status: active
src: agent
---

## Decided
A CLI pair-result envelope is trusted only when it comes from this module's privately branded pair error and the pair-run stage has actually begun. In `environment/run-pair.cjs:209-266`, `main` now starts the pair timer immediately before `runPair`, accepts an owned `pairResult` only after that boundary, and records parse/reference bootstrap failures as measured `cli_initialization` time. The regression in `test/pair-runner.test.cjs:263-284` patches `oracle.assertReady` to throw an Error carrying `{kind:"gen5-paired-battle-result", status:"completed", failure:{code:"INJECTED_PAIR_CODE"}, foreignData:"INJECTED_PAIR_DATA"}`; `main(validArgsWithResultPath)` must return nonzero and write only a safe failed initialization record without the injected code/data.

## Ruled out
Checking only `kind`, `schemaVersion`, or other envelope-shaped fields is insufficient: a foreign exception can copy the expected discriminator and status while adding arbitrary data, forging apparent success. Branding alone also does not establish that execution reached the runner; stage gating is needed so even result-shaped metadata thrown during parsing or pinned-oracle bootstrap remains a CLI initialization failure. Raw foreign exceptions and their nested result objects are therefore not copied into output.

## Watch out
Keep the runner-start boundary adjacent to the `runPair` call and set it only after argument parsing, `oracle.assertReady()`, and pinned simulator loading. If bootstrap moves across that boundary, failure timing will be mislabeled and the CLI may accept metadata from the wrong stage. The result-file behavior is covered by the injected-bootstrap regression; a file-write failure still cannot guarantee a result file at the unwritable destination.
