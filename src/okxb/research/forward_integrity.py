"""Forward-test integrity helpers (audit 2026-06 RV-1/RV-2/RV-3).

Shared by the pre-registration forward-shadow runners so the anti-overfitting machinery
is testable in the main suite (the runner scripts themselves do heavy imports):

  RV-1  pre-registration immutability — content-hash MANIFEST proves frozen artifacts
        were not changed after the freeze, even without git.
  RV-2  KILL stickiness — a DEAD marker so a killed candidate can never revive to PASS.
  RV-3  multiple-testing — Bonferroni significance threshold as a function of the true
        number of parallel candidates / screened hypotheses.
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Optional

from .labeling import _norm_ppf

DEFAULT_ARTIFACTS = ("model.pkl", "feature_list.json", "median.json", "meta.json")


# ----------------------------- RV-3: multiple testing -----------------------------

def bonferroni_t(n: int, alpha: float = 0.05) -> float:
    """One-sided Bonferroni significance z-threshold after correcting for n parallel hypotheses."""
    n = max(1, int(n))
    return float(_norm_ppf(1.0 - alpha / n))


# ----------------------------- RV-1: artifact immutability -----------------------------

def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_manifest(code_dir: Path, artifacts=DEFAULT_ARTIFACTS) -> dict:
    """Write a content-hash manifest of the frozen artifacts in code_dir. Returns the manifest."""
    code_dir = Path(code_dir)
    files = {n: sha256_file(code_dir / n) for n in artifacts if (code_dir / n).exists()}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    man = {"files": files, "manifest_sha256": digest, "written_at_ms": int(time.time() * 1000)}
    (code_dir / "MANIFEST.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    return man


def verify_manifest(code_dir: Path) -> Optional[str]:
    """None if artifacts match the manifest; "no-manifest" if absent; else a description of the
    mismatch (proof the frozen artifacts were changed after the freeze -> pre-registration void)."""
    code_dir = Path(code_dir)
    mf = code_dir / "MANIFEST.json"
    if not mf.exists():
        return "no-manifest"
    try:
        declared = json.loads(mf.read_text(encoding="utf-8")).get("files", {})
    except Exception as e:  # noqa: BLE001
        return f"manifest unreadable: {e!r}"
    cur = {n: sha256_file(code_dir / n) for n in declared if (code_dir / n).exists()}
    changed = [n for n, h in declared.items() if cur.get(n) != h]
    missing = [n for n in declared if not (code_dir / n).exists()]
    if changed or missing:
        return f"artifacts changed={changed} missing={missing} (modified after freeze -> pre-registration void!)"
    return None


# ----------------------------- RV-2: sticky KILL -----------------------------

def dead_path(frozen_dir: Path, code: str) -> Path:
    return Path(frozen_dir) / code / "DEAD.json"


def read_dead(frozen_dir: Path, code: str) -> Optional[dict]:
    p = dead_path(frozen_dir, code)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {"reason": "dead (marker unreadable)"}


def write_dead(frozen_dir: Path, code: str, reason: str, stats: Optional[dict] = None) -> None:
    """Persist a sticky DEAD marker. Once written, the candidate must never re-evaluate to PASS."""
    p = dead_path(frozen_dir, code)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(
        {"code": code, "reason": reason, "stats": stats or {}, "dead_at_ms": int(time.time() * 1000)},
        indent=2, ensure_ascii=False), encoding="utf-8")


def clear_dead(frozen_dir: Path, code: str) -> None:
    """Remove the DEAD marker (only on a fresh re-freeze = a brand-new registration)."""
    dead_path(frozen_dir, code).unlink(missing_ok=True)


# ----------------- RV-1: append-only hash-chained status log -----------------
# Each appended row carries prev_sha + row_sha = sha256(prev_sha | canonical(row)). Any
# retroactive edit/insert/delete of an earlier row breaks the chain -> tamper-evident forward log.

_HASH_COLS = ("prev_sha", "row_sha")
_GENESIS = "GENESIS"


def chain_hash(prev_sha: str, row: dict) -> str:
    """Deterministic hash of a row's data (hash columns excluded; None and '' treated equal,
    all values stringified so write-time python types and read-time CSV strings agree)."""
    payload = {k: ("" if row[k] is None else str(row[k]))
               for k in row if k not in _HASH_COLS}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256((str(prev_sha) + "|" + blob).encode("utf-8")).hexdigest()


def _last_row_sha(path: Path) -> str:
    last = None
    with open(path, newline="", encoding="utf-8") as f:
        for last in csv.DictReader(f):
            pass
    return (last or {}).get("row_sha") or _GENESIS


def append_rows_hashchain(path: Path, rows: list) -> None:
    """Append rows to a CSV as a tamper-evident hash chain. Archives the old file if the column
    set changed (one-time schema migration), then continues the chain from the last row_sha."""
    path = Path(path)
    rows = [dict(r) for r in rows]
    if not rows:
        return
    data_fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in data_fields and k not in _HASH_COLS:
                data_fields.append(k)
    fieldnames = data_fields + list(_HASH_COLS)
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            existing = next(csv.reader(f), [])
        if existing and set(existing) != set(fieldnames):
            path.replace(path.with_name(f"{path.stem}_archived_{int(time.time())}{path.suffix}"))
    prev = _last_row_sha(path) if path.exists() else _GENESIS
    write_header = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for r in rows:
            full = {k: r.get(k) for k in data_fields}
            row_sha = chain_hash(prev, full)
            full["prev_sha"], full["row_sha"] = prev, row_sha
            w.writerow(full)
            prev = row_sha


def verify_hashchain(path: Path) -> Optional[str]:
    """None if the chain is intact; else a description of the first broken/tampered row."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        if "row_sha" not in cols or "prev_sha" not in cols:
            return "no hash columns (legacy file, not chained)"
        data_fields = [c for c in cols if c not in _HASH_COLS]
        prev = _GENESIS
        for i, r in enumerate(reader):
            if (r.get("prev_sha") or _GENESIS) != prev:
                return f"row {i}: prev_sha broken (insert/delete/reorder)"
            full = {k: r.get(k, "") for k in data_fields}
            if r.get("row_sha") != chain_hash(prev, full):
                return f"row {i}: content tampered (row_sha mismatch)"
            prev = r.get("row_sha")
    return None
