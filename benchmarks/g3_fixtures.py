"""Portable fixture and provenance handling; no simulator execution on import."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "test-logs/g3-testing-profiling/diagnostic/native-pack.json"
BASELINE = ROOT / "test-logs/g3-testing-profiling/diagnostic/results.json"
PROFILE = ROOT / "test-logs/g3-testing-profiling/diagnostic-profile-96/results.json"
CASE_IDS = ("F01-p1", "F02-p1", "F03-p1")
SOURCE_ROOTS = frozenset(("simulator", "mechanics", "benchmarks", "test", "reference", "rules"))


def source_path(recorded):
    """Resolve historical /workspace paths against this checkout, not another one."""
    path = Path(recorded)
    if path.is_absolute():
        try:
            path = path.relative_to(ROOT)
        except ValueError:
            positions = [k for k, part in enumerate(path.parts) if part in SOURCE_ROOTS]
            if not positions:
                raise ValueError(f"Unrecognized recorded source path: {recorded}")
            path = Path(*path.parts[positions[0]:])
    if ".." in path.parts or not path.parts or path.parts[0] not in SOURCE_ROOTS:
        raise ValueError(f"Source is outside the repository source roots: {recorded}")
    return ROOT / path


def source_identity(path):
    raw = Path(path).read_bytes()
    return {"raw_sha256": hashlib.sha256(raw).hexdigest(),
            "lf_sha256": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()}


def function_sha256(path, name):
    """Hash the exact named function with canonical line endings, not its CLI."""
    source = Path(path).read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest()


def verify_source(recorded, expected):
    """Accept only the recorded bytes or their exact LF/CRLF equivalent.

    Preserve the original capture hash and actual raw hash. No whitespace,
    comments, code or fixture contents are otherwise normalized or ignored.
    """
    path = source_path(recorded)
    raw = path.read_bytes()
    identity = source_identity(path)
    lf = raw.replace(b"\r\n", b"\n")
    candidates = {hashlib.sha256(lf).hexdigest(),
                  hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest()}
    if identity["raw_sha256"] == expected:
        match = "exact"
    elif expected in candidates:
        match = "line-endings-only"
    else:
        raise ValueError(f"Measured source changed: {path.relative_to(ROOT)}")
    return {"path": path.relative_to(ROOT).as_posix(), "recorded_sha256": expected,
            **identity, "match": match}


def frozen_pack(path=PACK):
    data = json.loads(Path(path).read_text())
    pin = json.loads((ROOT / "reference-lock.json").read_text())["commit"]
    if data.get("pin") != pin:
        raise ValueError("Frozen pack does not match the pinned reference")
    cases = {case["id"]: case for case in data["cases"]}
    chosen = [cases[name] for name in CASE_IDS]
    if any(not c["steps"] or not c["steps"][-1]["after"]["terminal"] for c in chosen):
        raise ValueError("Frozen diagnostic requires three complete native battles")
    return {**data, "cases": chosen}
