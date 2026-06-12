"""Forward shadow test runner for the pre-registered 180min/6h/9h candidates.

Implements docs/FORWARD_SHADOW_PROTOCOL_h180_6h_9h.md exactly:
  freeze  -> train each candidate on the SAME 150d data, save model + selected features
             + median fill + scaler + the FROZEN absolute confidence threshold tau.
  evaluate-> reuse the IDENTICAL feature pipeline, score ONLY bars after the freeze cutoff
             that have already settled (ts <= now-H), using the frozen artifacts (NO refit,
             NO re-selection, NO re-bucketing), net of taker + funding-hold + x1.5 stress;
             check PASS/KILL.
  selftest-> assert the verdict logic.

Honest prior: 0 PASS (clean closure) is the expected, acceptable outcome. 9h is a
falsification target (train IC<0, AUC<0.5). Rules are frozen by the committed protocol.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]               # OKXB
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPTS))

import run_btc_enhanced_research as enh                  # noqa: E402  (reuse the exact pipeline)
from okxb.research import candle_data as cd              # noqa: E402
from okxb.research import candle_research as cr          # noqa: E402
from okxb.research import feature_lab as fl              # noqa: E402
from okxb.research import pro_model_workflow as pmw      # noqa: E402
from sklearn.preprocessing import StandardScaler         # noqa: E402

OUT = ROOT / "btc_single_asset_research"
FROZEN = OUT / "frozen"
FWD = OUT / "forward"
PROTOCOL = ROOT / "docs" / "FORWARD_SHADOW_PROTOCOL_h180_6h_9h.md"

# ----- pre-registered, frozen candidate set (must match the committed protocol) -----
# oos_ic_sign = the sign of the ORIGINAL OOS primary_ic (from enhanced_summary.csv), NOT the
# in-sample fit (which is always +). The forward "same-sign" check compares against THIS.
# C (9h) is pre-registered as -1: it is a falsification target (OOS IC was -0.0356).
CANDIDATES = {
    "A": {"H": 180, "model": "hist_gbm", "top_frac": 0.01, "oos_ic_sign": +1},
    "B": {"H": 360, "model": "lightgbm", "top_frac": 0.02, "oos_ic_sign": +1},
    "C": {"H": 540, "model": "mlp",      "top_frac": 0.05, "oos_ic_sign": -1},
}
BAR = "5m"
K_SEL = 30
PRESET = "deep"
TAKER_BPS = 10.0
STRESS_MULT = 1.5            # protocol: net judged after cost x1.5  -> taker 10 -> 15 bps
T_PASS = 2.13               # Bonferroni for 3 candidates (alpha=0.05)
N_MIN = 50
N_KILL = 30
WEEK_MS = 7 * 86_400_000


def log(m: str) -> None:
    print(m, flush=True)


def build_panel(H: int, *, days: int, source: str, force: bool, apply_funding: bool):
    """Reproduce the enhanced pipeline (build_augmented_bank + funding-hold + enrich) for one H."""
    btc = enh.load_candles("BTC-USDT-SWAP", BAR, days, source=source, force=force)
    spot = enh.load_candles("BTC-USDT", BAR, days, source=source, force=force)
    funding = cd.fetch_funding_series("BTC-USDT-SWAP", days + 5, log=log)
    daily_external, _ = enh.fetch_daily_external(days, force=force)
    short_external, _ = enh.fetch_short_intraday_external(force=force)
    perp = {"BTC-USDT-SWAP": btc}
    panel, bar_ms = pmw.build_augmented_bank(perp, {"BTC-USDT-SWAP": spot}, {"BTC-USDT-SWAP": funding}, BAR, H)
    if apply_funding:
        panel = enh.apply_funding_hold_cost(panel, funding, H)
    panel = enh.enrich_panel(panel, btc, BAR, daily_external, short_external, include_short=False)
    return panel, bar_ms


def freeze() -> None:
    FROZEN.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time() * 1000)
    for code, cfg in CANDIDATES.items():
        H, model_name, top_frac = cfg["H"], cfg["model"], cfg["top_frac"]
        # SAME 150d data + dist candles as the original research run (reproduce the candidate exactly)
        panel, bar_ms = build_panel(H, days=150, source="dist", force=False, apply_funding=(H >= 240))
        feats = pmw.feature_cols(panel)
        data = panel[["ts", "inst", "fwd", "y"] + feats].replace([np.inf, -np.inf], np.nan).dropna(subset=["fwd", "y"])
        sel, _ = fl.select_features(data, feats, k=K_SEL)
        med = data[sel].median()
        X = data[sel].fillna(med).values
        kind, factory = pmw._model_factories(preset=PRESET)[model_name]
        scaler = StandardScaler().fit(X) if kind.endswith("scale") else None
        Xf = scaler.transform(X) if scaler is not None else X
        target = data["y"].values.astype(int) if kind.startswith("clf") else data["fwd"].values.astype(float)
        model = factory().fit(Xf, target)
        train_score = pmw._score_from_model(kind, model, Xf)
        tau = float(np.nanquantile(np.abs(train_score), 1 - top_frac))      # FROZEN absolute threshold
        ic_sign = int(cfg["oos_ic_sign"])                                   # pre-registered OOS sign (NOT in-sample)
        insample_ic_sign = int(np.sign(cr._spearman(train_score, data["fwd"].values) or 0.0))
        d = FROZEN / code
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "model.pkl", "wb") as fh:
            pickle.dump({"model": model, "kind": kind, "scaler": scaler}, fh)
        (d / "feature_list.json").write_text(json.dumps(sel), encoding="utf-8")
        (d / "median.json").write_text(json.dumps(med.to_dict()), encoding="utf-8")
        (d / "meta.json").write_text(json.dumps({
            "code": code, "H": H, "model": model_name, "top_frac": top_frac, "bar_ms": bar_ms,
            "tau": tau, "ic_sign": ic_sign, "insample_ic_sign": insample_ic_sign, "n_sel": len(sel),
            "freeze_cutoff_ts": int(data["ts"].max()), "frozen_at_ms": stamp,
            "apply_funding_train": bool(H >= 240),
        }, indent=2), encoding="utf-8")
        log(f"[freeze {code}] H={H} {model_name} sel={len(sel)} tau={tau:.5f} "
            f"oos_ic_sign={ic_sign:+d} (in-sample {insample_ic_sign:+d}) cutoff={enh.ts_utc(int(data['ts'].max()))}")


def _net(port: np.ndarray, cost_bps: float):
    st = cr._stats_from_gross(port, cost_bps, 1) if len(port) >= 8 else None
    return (st["net_bps"], st["nw_t"]) if st else (np.nan, np.nan)


def forward_verdict(net15: float, t15: float, net10: float, n_ts: int,
                    fwd_ic_sign: int, train_ic_sign: int, weeks: float) -> str:
    """Pure PASS/KILL logic per protocol section 4/5."""
    if n_ts >= 8:
        if (net10 is not None and net10 <= 0) or (fwd_ic_sign != 0 and fwd_ic_sign != train_ic_sign):
            return "KILL"
    if weeks >= 6 and n_ts < N_KILL:
        return "KILL"
    if (net15 is not None and net15 > 0 and (t15 or 0) > T_PASS and n_ts >= N_MIN
            and fwd_ic_sign == train_ic_sign):
        return "PASS"
    return "PENDING"


def evaluate() -> None:
    FWD.mkdir(parents=True, exist_ok=True)
    now_ms = int(time.time() * 1000)
    rows = []
    for code, cfg in CANDIDATES.items():
        d = FROZEN / code
        if not (d / "meta.json").exists():
            log(f"[{code}] not frozen yet — run --mode freeze first")
            continue
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        H, bar_ms, tau = meta["H"], meta["bar_ms"], meta["tau"]
        cutoff, train_ic_sign = meta["freeze_cutoff_ts"], meta["ic_sign"]
        weeks = (now_ms - meta["frozen_at_ms"]) / WEEK_MS
        blob = pickle.load(open(d / "model.pkl", "rb"))
        model, kind, scaler = blob["model"], blob["kind"], blob["scaler"]
        sel = json.loads((d / "feature_list.json").read_text(encoding="utf-8"))
        med = pd.Series(json.loads((d / "median.json").read_text(encoding="utf-8")))

        # evaluate on the TRUE net target (taker + funding always folded in), only settled forward bars
        days = 150 + int((now_ms - cutoff) / 86_400_000) + 10
        panel, _ = build_panel(H, days=days, source="refresh", force=True, apply_funding=True)
        for c in sel:                                   # tolerate a forward source that briefly drops a column
            if c not in panel.columns:
                panel[c] = np.nan
        settle_hi = now_ms - H * 60_000
        sub = panel[(panel["ts"] > cutoff) & (panel["ts"] <= settle_hi)]
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=["fwd", "y"])
        row = {"code": code, "H": H, "weeks_since_freeze": round(weeks, 2), "n_signals": 0,
               "n_ts": 0, "verdict": "PENDING"}
        if len(sub):
            X = sub[sel].fillna(med).values
            Xf = scaler.transform(X) if scaler is not None else X
            score = pmw._score_from_model(kind, model, Xf)
            hi = sub.assign(score=score)
            hi = hi[np.abs(hi["score"].values) >= tau]                      # FROZEN tau, not a forward quantile
            h = max(1, round(H / (bar_ms / 60_000)))
            reb = set(cr._rebalance_ts(np.sort(hi["ts"].unique()), bar_ms, h).tolist())
            tr = hi[hi["ts"].isin(reb)]
            if len(tr):
                signed = np.sign(tr["score"].values) * tr["fwd"].values
                port = pd.Series(signed, index=tr["ts"].values).groupby(level=0).mean().values
                net10, t10 = _net(port, TAKER_BPS)
                net15, t15 = _net(port, TAKER_BPS * STRESS_MULT)
                fic = cr._spearman(tr["score"].values, tr["fwd"].values) if len(tr) > 5 else 0.0
                fic_sign = int(np.sign(fic or 0.0))
                verdict = forward_verdict(net15, t15, net10, len(port), fic_sign, train_ic_sign, weeks)
                row.update({"n_signals": len(tr), "n_ts": len(port),
                            "net10_bps": round(net10, 1), "t10": round(t10, 2),
                            "net15_bps": round(net15, 1), "t15": round(t15, 2),
                            "fwd_ic": round(float(fic), 4), "fwd_ic_sign": fic_sign,
                            "train_ic_sign": train_ic_sign, "verdict": verdict})
        rows.append(row)
        log(f"[{code}] H={H} weeks={weeks:.1f} n_ts={row.get('n_ts')} "
            f"net10={row.get('net10_bps','-')}(t{row.get('t10','-')}) "
            f"net15={row.get('net15_bps','-')}(t{row.get('t15','-')}) "
            f"fwd_ic={row.get('fwd_ic','-')} -> {row['verdict']}")
    df = pd.DataFrame(rows)
    stamp = enh.ts_utc(now_ms)
    df.insert(0, "asof_utc", stamp)
    hist = FWD / "forward_status.csv"
    df.to_csv(hist, mode="a", header=not hist.exists(), index=False)
    log(f"\nappended {len(rows)} rows to {hist}")
    log("decision: 0 PASS at window end -> clean closure; any PASS -> second independent forward + tiny live fill test.")


def selftest() -> None:
    assert forward_verdict(5.0, 2.5, 6.0, 60, 1, 1, 2.0) == "PASS"
    assert forward_verdict(5.0, 2.5, 6.0, 40, 1, 1, 2.0) == "PENDING"   # n<50
    assert forward_verdict(5.0, 1.8, 6.0, 60, 1, 1, 2.0) == "PENDING"   # t<2.13
    assert forward_verdict(5.0, 2.5, -1.0, 60, 1, 1, 2.0) == "KILL"     # net10<=0
    assert forward_verdict(5.0, 2.5, 6.0, 60, -1, 1, 2.0) == "KILL"     # IC flipped
    assert forward_verdict(5.0, 2.5, 6.0, 20, 1, 1, 6.5) == "KILL"      # 6wk and n<30
    log("selftest OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="Forward shadow test (pre-registered).")
    ap.add_argument("--mode", choices=["freeze", "evaluate", "selftest"], required=True)
    args = ap.parse_args()
    {"freeze": freeze, "evaluate": evaluate, "selftest": selftest}[args.mode]()


if __name__ == "__main__":
    main()
