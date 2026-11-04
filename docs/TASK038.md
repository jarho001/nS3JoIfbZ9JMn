# Task038 — initial actor view for a request session

Status: **Accepted** — 2026-11-04. Owner `/root/task038`; primary `/root`.
Source base: `9bc3bf0594449709adc213409c0ff6d3896971ad`.

## Scope and acceptance

Expose a read-only per-player current request-session view before any submission, closing the initial-observation API gap without advancing battle state.

Exclusions: battle mechanics/RNG/choice semantics changes; new variants or networking; long integration/GPU/performance runs; milestone acceptance.

- Add a documented typed per-player request-view API taking RequestSession and static audience, available immediately after make_request_session. Include current player request, local episode and own commit status only; no opponent commitment/full state/RNG.
- Reuse the current request projection in project_submission_view while preserving its public return type and separate prior-event projection. Do not call a full transition merely to observe.
- Fast tests compare initial own request data to guarded native --sessions fixture, check opponent-commit view invariance, input immutability, invalid audiences/structures and projection-only eager/JIT/vmap consistency. Every command <=30s and total tests <=90s; do not run known-long session suite.
- Document initial-view use; preserve archives and review state. Record deferred long session regression with source blobs. Refresh/check Trail and freeze one concise evidence package for primary acceptance.

## Evidence and acceptance

Primary independently reviewed the complete projection/export/test/API diff and owner evidence. The initial SessionRequestView is available without a submission, exposes only the selected current request/local episode/own commitment, validates structure/audience, and uses an empty transition. Existing SessionView fields and separate prior events remain unchanged. Native initial request parity, private opponent-commit invariance, no mutation, invalid input and projection-only eager/JIT/vmap checks pass. No battle kernel, RNG or submit behavior changed. Accepted with optional long integration regression deferred; no required acceptance evidence is missing.

Executed: `timeout --signal=TERM --kill-after=5s 30s env JAX_PLATFORMS=cpu python -m unittest test.test_request_view -v` — 1/1 passed; test3.741s/command5.1s. Diff and Trail graph/current checks passed. Primary performs final staged learning/JSON/history/scope checks before delivery. No full transition or performance probe ran.

Executable Git blobs:

- `simulator/request_session.py`: `4a94e7a4eaec59d574dab5a5274a7875c4ba1e0e`
- `simulator/__init__.py`: `162da03469955fa7088fc034c44de39e21a4314e`
- `test/test_request_view.py`: `a6611019117f52f1306bd92ad85d1c5811d2e96b`

## Deferred user-run regression

Known-long optional regression, not run; not a required acceptance dependency:

```sh
JAX_PLATFORMS=cpu python -m unittest test.test_request_session test.test_neutral_tackle test.test_neutral_duel test.test_neutral_tackle_ties -v
```

Run against unchanged relevant source blobs above and dependencies from the source base. Count38/baseline30/next40; no GPU requirement/block or milestone acceptance.
