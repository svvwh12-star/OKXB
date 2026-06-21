"""Tests for forward-test integrity helpers (audit 2026-06 RV-1/RV-2/RV-3)."""
import math

from okxb.research import forward_integrity as fi


# ----------------- RV-3: Bonferroni threshold -----------------

def test_bonferroni_monotonic_and_known_values():
    assert abs(fi.bonferroni_t(1) - 1.6449) < 0.01          # one-sided 0.05
    assert abs(fi.bonferroni_t(3) - 2.1284) < 0.02          # the old hardcoded 2.13 == correcting for 3
    assert fi.bonferroni_t(3) < fi.bonferroni_t(7) < fi.bonferroni_t(20)
    # correcting for the true family (7) is materially stricter than the old 2.13
    assert fi.bonferroni_t(7) > 2.13


# ----------------- RV-1: artifact immutability manifest -----------------

def test_manifest_detects_tamper_and_missing(tmp_path):
    d = tmp_path / "A"
    d.mkdir()
    (d / "model.pkl").write_bytes(b"weights")
    (d / "meta.json").write_text('{"tau": 1.0}', encoding="utf-8")

    assert fi.verify_manifest(d) == "no-manifest"           # before writing one
    fi.write_manifest(d)
    assert fi.verify_manifest(d) is None                    # matches right after freeze

    (d / "meta.json").write_text('{"tau": 9.9}', encoding="utf-8")   # retune after freeze
    msg = fi.verify_manifest(d)
    assert msg and msg != "no-manifest" and "meta.json" in msg

    fi.write_manifest(d)                                    # re-freeze re-baselines
    assert fi.verify_manifest(d) is None
    (d / "model.pkl").unlink()                              # artifact deleted
    assert "model.pkl" in (fi.verify_manifest(d) or "")


# ----------------- RV-2: sticky KILL marker -----------------

def test_dead_marker_roundtrip(tmp_path):
    frozen = tmp_path / "frozen"
    (frozen / "C").mkdir(parents=True)
    assert fi.read_dead(frozen, "C") is None
    fi.write_dead(frozen, "C", "forward KILL: net10=-2.0", {"n_ts": 41})
    d = fi.read_dead(frozen, "C")
    assert d and d["reason"].startswith("forward KILL") and d["stats"]["n_ts"] == 41
    fi.clear_dead(frozen, "C")                              # only a fresh re-freeze clears it
    assert fi.read_dead(frozen, "C") is None


# ----------------- RV-1: append-only hash-chained status log -----------------

def test_hashchain_detects_history_tamper(tmp_path):
    csv_path = tmp_path / "forward_status.csv"
    fi.append_rows_hashchain(csv_path, [{"asof": "t0", "code": "A", "verdict": "PENDING", "n_ts": 10}])
    fi.append_rows_hashchain(csv_path, [{"asof": "t1", "code": "A", "verdict": "KILL", "n_ts": 41}])
    assert fi.verify_hashchain(csv_path) is None            # intact chain across two appends

    # retroactively rewrite an earlier verdict (the cherry-pick attack) -> chain must break
    txt = csv_path.read_text(encoding="utf-8").replace("PENDING", "PASS")
    csv_path.write_text(txt, encoding="utf-8")
    assert fi.verify_hashchain(csv_path) is not None


def test_hashchain_handles_heterogeneous_rows(tmp_path):
    csv_path = tmp_path / "s.csv"
    fi.append_rows_hashchain(csv_path, [{"code": "A", "verdict": "PENDING", "net10_bps": 1.5}])
    fi.append_rows_hashchain(csv_path, [{"code": "B", "verdict": "DEAD"}])   # missing net10_bps
    assert fi.verify_hashchain(csv_path) is None            # None-filled columns still verify
