# Gen 5 Random Battle research foundation

This repository is beginning a JAX self-play project against Pokémon Showdown's `[Gen 5] Random Battle` (`gen5randombattle`). The current foundation pins an upstream Showdown revision and provides a deterministic oracle fixture for its format rules, team generation, and first player requests.

## Set up the pinned reference

Use Node.js 22.18 or newer and npm. From the repository root, run:

```sh
node reference/setup.cjs
```

The command clones the revision recorded in [`reference-lock.json`](reference-lock.json) only when the ignored reference checkout is absent, verifies its origin, exact commit, and clean tracked/untracked status, then runs upstream `npm ci` and `npm run build`. It never resets, cleans, or updates an existing checkout. If the checkout has changed, inspect it and restore or remove it yourself before retrying.

## Inspect the reference outputs

```sh
node reference/oracle.cjs format
node reference/oracle.cjs smoke
node reference/oracle.cjs smoke --team-seed-p1=1,2,3,4 --team-seed-p2=5,6,7,8 --battle-seed=9,10,11,12
node reference/oracle.cjs provenance
node --test test/reference.test.cjs
node --test test/protocol.test.cjs
node --test test/*.test.cjs
```

The committed JSON files under `reference/fixtures/` are canonical semantic fixtures. The format export comes from Showdown's resolved Dex/rule table. The smoke artifact contains both complete generated teams, private initial requests, and raw omniscient update messages. Those updates can include `|split|` markers and private `SECRET` lines; this trusted oracle artifact is not a public player view or a policy training record. The harness validates Gen 5, singles, no Team Preview, team sizes, and that each player's request names match its submitted team. Wall-clock protocol lines are excluded. Provenance reports the upstream package lock hash and Node/npm versions separately from semantic fixture hashes.

## Protocol views

[`environment/protocol.cjs`](environment/protocol.cjs) exports `projectPackets(packets, audience)`. `packets` is an ordered array of complete Showdown object messages, each shaped `channel\nbody`; `audience` is exactly `p1`, `p2`, or `spectator`. It returns one object `{ schemaVersion, audience, records }` for that audience only. Records are ordered and typed: `{ type: "event", message }` for visible events, `{ type: "privateEvent", message }` for that player's private events, and `{ type: "request", request }` for that player's parsed request JSON. A hidden-only packet adds no placeholder record. For example:

```js
const { projectPackets } = require('./environment/protocol.cjs');
const p1View = projectPackets(completePackets, 'p1');
```

The projector routes `SECRET`/`PUBLIC` branches before any event becomes actor-visible; raw message bodies never enter a player view. It filters clock-only events, routes private requests/errors only to their owner, and drops trusted `end` metadata. Malformed, unsupported, and battle-error messages fail closed for the harness; their raw text is not an actor observation. It does not rebuild transport chunks or define model features. The hand-authored routing fixture is [`test/fixtures/protocol-transcript-v1.json`](test/fixtures/protocol-transcript-v1.json); tests also compare its split routing and live seeded battles against pinned Showdown.

## Random-policy baseline runner

[`environment/run-random.cjs`](environment/run-random.cjs) runs a complete seeded Gen 5 Random Battle against the pinned reference:

```sh
node environment/run-random.cjs --help
node environment/run-random.cjs \
  --team-seed-p1=1,2,3,4 --team-seed-p2=5,6,7,8 \
  --battle-seed=9,10,11,12 \
  --policy-seed-p1=13,14,15,16 --policy-seed-p2=17,18,19,20 \
  --output trusted-battle.json
node --test test/battle-runner.test.cjs
node --test test/*.test.cjs
```

All five seeds are separate four-limb Gen 5 seeds. The summary reports the pinned commit, seeds, result, turn count, packet/decision counts, and move/switch counts. `--output` writes a versioned trusted transcript containing complete private requests, split update branches, choice attempts, terminal outcome, and end metadata. Treat it as omniscient oracle data; the runner never passes that transcript to a policy.

The two random policies receive a frozen `{ history, request, random }` callback input. `history` contains only that side's `projectPackets` output and its own submitted-choice resolutions; `request` is the current request addressed to that side; `random` is that policy's independent seeded RNG. The runner submits each choice on its own request and relies on Showdown's choice commitment behavior to hold simultaneous choices. Candidate moves and switches come from the request payload. `trapped` is honored when disclosed; `maybeTrapped` is not resolved from simulator state. Wait requests create no action. Unavailable-choice errors and corrected requests stay inside a bounded decision episode.

This is a deliberately small baseline. It does not provide a learned policy, observation features, a full game-memory contract, or JAX mechanics.

## Trusted trace capture and replay

[`reference/trace.cjs`](reference/trace.cjs) captures and replays the full BattleStream input/output sequence:

```sh
node reference/trace.cjs capture --output /tmp/gen5-random-trace.json
node reference/trace.cjs capture --output /tmp/gen5-random-trace-seeded.json \
  --team-seed-p1=1,2,3,4 --team-seed-p2=5,6,7,8 \
  --battle-seed=9,10,11,12 \
  --policy-seed-p1=13,14,15,16 --policy-seed-p2=17,18,19,20
node reference/trace.cjs replay /tmp/gen5-random-trace.json
node --test test/trace.test.cjs
```

The version 1 trusted trace records the pinned SHA, format, all five seeds, complete ordered start/player/choice inputs, complete output packets, and verified outcome. Replay submits those recorded inputs directly to a fresh pinned simulator; it does not regenerate teams or call a policy. Only clock-only `|t:|` messages are normalized, with blank split branches retained in place. These files contain packed teams, private requests, split `SECRET` records, and trusted end metadata. Never expose a trace to an actor; use `projectPackets` to derive a single recipient's view.

## First JAX arithmetic slice

Task005 implements only the post-`ModifyBoost` non-HP stat-stage scaling substep from pinned Gen 5 `Pokemon.calculateStat`, before stat modifiers. The current pure int32 kernel accepts a nonnegative stored stat in `[0, 65535]` and a signed int32 stage; stages clamp to `[-6, 6]`. Task007 extends the original Task005 positive range because Gen 5 nature-product truncation can produce a valid stored zero. These are caller preconditions, not runtime checks. The scaler excludes base-stat derivation, natures, EVs, Wonder Room stat selection, event hooks, and later stat modifiers.

Create an isolated CPU environment without changing the existing GPU JAX installation:

```sh
python3 -m venv .venv-jax
. .venv-jax/bin/activate
python -m pip install -r requirements-jax-cpu.txt
node reference/stat-stage-oracle.cjs fixture
JAX_PLATFORMS=cpu python -m unittest discover -s test -p 'test_*.py' -v
node --test test/*.test.cjs
```

`requirements-jax-cpu.txt` pins JAX and JAXLIB to 0.12.0. The versioned oracle fixture is generated by the pinned Showdown runtime in a neutral Gen 5 custom game, calling `Pokemon.calculateStat` with an identity `ModifyBoost` event, no Wonder Room, and modifier 1. The differential test compares all 13 stages for stored stats 0 through 1024, plus the int32 stage extremes and stored-stat upper boundary. Tests cover eager/JIT/vmap/broadcast execution on CPU and GPU. The stat range `[0, 65535]` is a caller precondition, not a runtime check. This is a mechanics kernel parity slice, not a full stat calculator, battle simulator, or learner.

## Gen 5 stored-stat derivation

Task006 adds [`mechanics/stats.py`](mechanics/stats.py), which derives `hp, atk, def, spa, spd, spe` from base stats, level, IVs, EVs, and integer nature multipliers (`9` minus, `10` neutral, `11` plus). The final axis has six stats; level and the explicit `shedinja` flag broadcast over batch axes. Inputs must be int32 with base stats `[1,255]`, level `[1,100]`, IVs `[0,31]`, EVs `[0,255]`, and multipliers `[9,10,11]`; these value ranges are caller preconditions, not runtime checks. Shedinja's species override forces HP to 1. Base HP 1 by itself does not trigger that override.

The pinned `gen5randombattle` format does not enable `overflowstatmod`, so the kernel preserves Gen 5's unsigned low-16-bit truncation on nature products and does not apply the optional 595/728 caps. [`reference/stats-oracle.cjs`](reference/stats-oracle.cjs) constructs actual pinned Pokemon and records `maxhp`/`storedStats`; it also uses actual `Battle.statModify` calls for synthetic base-stat threshold cases. Regenerate the fixture and run the full Python suite with:

```sh
node reference/stats-oracle.cjs fixture
JAX_PLATFORMS=cpu python -m unittest discover -s test -p 'test_*.py' -v
node --test test/*.test.cjs
```

The fixture includes 125 real Pokemon across all 25 natures and levels `1, 17, 50, 99, 100`, plus nine synthetic cases across Adamant, Modest, and Hardy for pre-nature Attack 595, 596, and 609. At 595, `595*110=65450` stays below the 16-bit wrap; at 596, `596*110=65560` wraps to 24 and the stored Attack becomes 0. Tests compare eager, JIT, and JIT-vmap results on the pinned CPU environment and the available GPU environment. Task006 does not derive stat stages, model event hooks/forms, or validate competitive teams. Task007 extends stage scaling to accept zero, which is needed to compose the valid synthetic result.

## Attack stored-stat and stage composition

Task007's [`mechanics/attack.py`](mechanics/attack.py) composes the stored-stat and stage kernels for Attack only: derive the six Gen 5 stored stats, select Attack, and apply the requested stage. It matches `Pokemon.calculateStat('atk', stage, 1)` only with neutral `ModifyBoost`, no Wonder Room, and no stat modifier. It does not include abilities, items, status, or dynamic stat selection.

Generate the guarded reference fixture and run the same CPU/GPU Python test suites as Tasks005–006:

```sh
node reference/attack-oracle.cjs fixture
JAX_PLATFORMS=cpu python -m unittest discover -s test -p 'test_*.py' -v
node --test test/*.test.cjs
```

The compact fixture covers all 25 natures with initialized Pokemon and all 13 stages plus clamp inputs. Ordinary expected values come from the actual `Pokemon.calculateStat` call without overwriting `storedStats`. A separate synthetic case uses real `Battle.statModify` to derive zero at the Gen 5 nature-wrap boundary, then injects it only to check the real stage arithmetic. The fixture records the pin and source paths. On 2026-10-28, all 20 Python tests passed on both CPU and available GPU environments; all 34 Node tests passed. Task007 is accepted and complete.

## Damage randomizer arithmetic

Task008 implements the `Battle.randomizer(baseDamage)` arithmetic as an explicit-roll JAX kernel. It models the uint32-wrapped multiplication by `100-roll`, followed by floor division by 100. The input roll is supplied by the caller; this slice does not generate randomness or include the minimum-damage clamp, STAB, type effectiveness, or other modifiers. Its live oracle records battle seed `[101, 202, 303, 404]` and observes later `Battle.random(16)` results after Battle initialization; it does not label them as the PRNG's first draws.

The bounded oracle/test commands are:

```sh
node reference/damage-roll-oracle.cjs fixture
JAX_PLATFORMS=cpu python -m unittest discover -s test -p 'test_*.py' -v
node --test test/*.test.cjs
```

Task008 and Task009 are accepted and complete. Task008's 27-test Python suite passes on CPU and available GPU; Task009's 34-test Python suite passes on CPU and GPU. The existing Node suite passes 34/34. Both are arithmetic slices, not a complete damage calculator.

## Base-damage integer core

Task009 implements the nested integer core inside pinned `BattleActions.getDamage`, before `modifyDamage` and the later `+2`. Inputs are the effective level, base power, Attack, and Defense. It preserves uint32 product truncation and matches the zero-Defense result safely. The kernel does not derive stats or model events, critical hits, `+2`, the random roll, STAB, or type effectiveness.

```sh
node reference/base-damage-oracle.cjs fixture
JAX_PLATFORMS=cpu python -m unittest discover -s test -p 'test_*.py' -v
node --test test/*.test.cjs
```

Task009 is accepted and complete. Its full Python suite passes 34/34 on CPU and GPU, and the Node suite passes 34/34. Task010 composes the accepted core with Showdown's `+2`, the real neutral event's fixed-point identity behavior, and an explicit random-roll stage.

## Neutral damage after the random roll

Task010's pure JAX `neutral_damage_after_roll` returns the intermediate after `Battle.randomizer`, not final damage. It composes the base-damage core, the `+2` in `modifyDamage`, and `Battle.modify(value, 1)` as applied by the neutral `WeatherModifyDamage` event, then applies the supplied roll. That nominal identity event wraps to 20 bits because Showdown truncates `value * 4096` to uint32. This kernel omits the PRNG, full modifier engine, critical hits, STAB, type, burn, Gen 5 minimum-one check, and final 16-bit truncation.

```sh
node reference/neutral-damage-oracle.cjs fixture
JAX_PLATFORMS=cpu python -m unittest discover -s test -p 'test_*.py' -v
node --test test/*.test.cjs
```

Task010 is accepted and complete. Its guarded pinned oracle delegates to the real `modifyDamage`, neutral event, and `Battle.randomizer`, controlling only the explicit random draw. The 41-test Python suite passes on CPU and GPU; the Node suite passes 34/34. Task008 separately tests randomizer uint32-product wrapping; the neutral event bounds this composition's roll input to at most `2^20-1`.

## Fixed-point modifier arithmetic

Task011 adds `apply_fixedpoint_modifier(value, modifier)` for an already encoded Gen 5 modifier. Strict int32 inputs have caller-precondition ranges `value` `[0, 2147483647]` and `modifier` `[0, 65535]`. It follows `Battle.modify(value, modifier, 4096)`, including unsigned 32-bit product wrap and round-half-down behavior. It does not encode fractions or combine a chain of modifiers.

```sh
node reference/fixedpoint-oracle.cjs fixture
JAX_PLATFORMS=cpu python -m unittest discover -s test -p 'test_*.py' -v
node --test test/*.test.cjs
```

Task011 is accepted and complete. The 47-test Python suite passes on CPU and GPU; the Node suite passes 34/34. The same primitive implements Task010's neutral identity event, preserving its 20-bit wrap behavior. Task012 adds the bounded damage-finalization tail under explicit inputs.

## Gen 5 damage finalization

Task012 adds `finalize_damage(damage, modifier)` in `mechanics/finalize_damage.py`. It consumes damage after type effectiveness and burn, replaces zero with one, applies the already encoded final modifier through Task011, and truncates to 16 bits. The returned damage may be zero: Gen 5 does not repeat its minimum-one check after the final modifier or truncation. Both inputs are strict int32 and broadcast; caller-precondition ranges are damage `[0, 2147483647]` and modifier `[0, 65535]`.

The guarded oracle delegates to the actual pinned `getDamage`/`modifyDamage` tail. Eight ordinary cases use initialized Pokemon with no item or real Life Orb (observed modifier 5324). The 430 synthetic cases inject only an explicit randomizer output in a neutral path and register one real modifier event; they preserve real zero-check, event application, and final truncation. Tests also compose the primitive with Task010's existing neutral damage fixture. This slice does not implement STAB, effectiveness, burn, event dispatch, modifier chaining, bypassProtect, or a full damage calculator.

```sh
node reference/finalize-damage-oracle.cjs fixture > reference/fixtures/finalize-damage-v1.json
JAX_PLATFORMS=cpu .venv-jax/bin/python -m unittest discover -s test -p 'test_*.py' -v
node --test test/*.test.cjs
```

Task012 is accepted and complete. All 54 Python tests and 35 Node tests pass. GPU validation is unavailable in this resumed CPU-only environment; it has no CUDA backend. See `docs/TASKS.md` for current Node validation evidence and the Task013 proposal.

## Current task and next work

Tasks 001–012 are accepted and complete. Task 003's deterministic request-to-request random-policy baseline passed 15 targeted tests and contributed to the 29-test suite at that milestone; evidence includes the 102-turn default game, fresh-process transcript equality, and a real Shadow Tag unavailable-choice retry. Task 004's full suite passed 34/34, and a fresh CLI process replayed the complete 553-event, 102-turn trace with matching outcome. Task006's stored-stat derivation kernel passed 13 Python tests on CPU and GPU; the 34 Node tests pass. Task007's full Python suite passes 20/20 on both CPU and GPU, with 34/34 Node tests. Task008's full Python suite passes 27/27 on both CPU and GPU, with 34/34 Node tests. Task009's full Python suite passes 34/34 on CPU and GPU, with 34/34 Node tests. Task010's full Python suite passes 41/41 on CPU and GPU, with 34/34 Node tests. Task011's full Python suite passes 47/47 on CPU and GPU, with 34/34 Node tests. This work is a portion of G0/G1, not completion of either milestone in the larger [project plan](PROJECT_PLAN.md).

See [`docs/TASKS.md`](docs/TASKS.md) for acceptance state and handoff details.
