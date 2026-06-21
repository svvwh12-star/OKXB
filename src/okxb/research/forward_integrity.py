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
