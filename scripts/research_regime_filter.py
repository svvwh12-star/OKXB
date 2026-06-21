#!/usr/bin/env python
"""Regime-filter experiment: does conditioning a thin cross-sectional reversal on an
orthogonal daily regime (DVOL / VRP / funding extreme) clear cost in some regime?

Research-only, public data, no keys. Pre-registers regimes before evaluation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                       # noqa: E402
import pandas as pd                                      # noqa: E402

from okxb.research import candle_data as cd              # noqa: E402
from okxb.research import deribit_data as dd             # noqa: E402
from okxb.research import regime_filter as rf            # noqa: E402

DAILY = ROOT / "dist" / "daily"
REPORT = DAILY / "regime_filter_report.txt"
HYP = DAILY / "hypotheses.jsonl"


def main() -> None:
    majors = [f"{s}-USDT-SWAP" for s in ("BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX")]
    days = 90

    DAILY.mkdir(parents=True, exist_ok=True)
    with HYP.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"stage": "P1b-regime-filter", "base": "15m xs-reversal k2/h2",
                             "regimes": ["dvol_hi(above_median)", "vrp_pos(positive)", "fund_extreme(|z|>1)"],
                             "holdout_frac": 0.3,
                             "pass": "held-out net_taker>0 & NW-t>2 & beats unconditional"}) + "\n")

    dfs = cd.fetch_universe(majors, "15m", days, DAILY / "candles15m")
    if len(dfs) < 4:
        raise SystemExit("not enough intraday instruments")

    # --- BTC daily regime series ---
    btc_d = (cd.fetch_universe(["BTC-USDT-SWAP"], "1D", 365 * 3, DAILY / "candles")["BTC-USDT-SWAP"]
             .drop_duplicates("ts").sort_values("ts"))
    dvol = dd.fetch_dvol("BTC", 365 * 3).drop_duplicates("ts")   # df(ts, dvol)
    fund = cd.fetch_funding_series("BTC-USDT-SWAP", days).rename(columns={"funding": "value"})
    c = pd.Series(btc_d["c"].to_numpy(float), index=btc_d["ts"].to_numpy(np.int64))
    rv = (np.log(c).diff().rolling(10).std() * np.sqrt(365.0) * 100.0)
    dvser = dvol.set_index("ts")["dvol"].reindex(c.index, method="ffill")
    vrp_d = pd.DataFrame({"ts": c.index.values, "value": (dvser ** 2 - rv ** 2).values}).dropna()
    dvol_d = dvol.rename(columns={"dvol": "value"})

    panel = rf.build_reversal_panel(dfs, "15m", k_lookback=2, h_fwd=2)
    # RV-6/RV-8: 日频制度并到日内信号必须按"周期收盘后才可得"对齐 (DAY_MS), 否则跨日前视
    panel = rf.attach_regime(panel, dvol_d, "dvol_hi", publish_lag_ms=rf.DAY_MS, mode="above_median")
    panel = rf.attach_regime(panel, vrp_d, "vrp_pos", publish_lag_ms=rf.DAY_MS, mode="positive")
    panel = rf.attach_regime(panel, fund[["ts", "value"]], "fund_extreme", publish_lag_ms=rf.DAY_MS, mode="abs_extreme")

    rows = rf.eval_regimes(panel, ["dvol_hi", "vrp_pos", "fund_extreme"])
    ok, msg = rf.verdict(rows)

    L = ["=" * 80, "Regime-filter experiment (gate xs-reversal by orthogonal regime)", "=" * 80,
         f"insts={len(dfs)} bar=15m days={days} base=xs-reversal(k2,h2) holdout=30%", "",
         f"{'regime':>16} {'n':>8} {'net_maker':>10} {'t_m':>6} {'net_taker':>10} {'t_t':>6}"]
    for r in rows:
        if r.get("net_maker") is None:
            L.append(f"{r['regime']:>16} {r['n']:>8}   (n<min)")
            continue
        L.append(f"{r['regime']:>16} {r['n']:>8} {r['net_maker']:>+10.1f} {r['t_maker']:>+6.1f} "
                 f"{r['net_taker']:>+10.1f} {r['t_taker']:>+6.1f}")
    L += ["", "verdict: " + msg]
    report = "\n".join(L)
    REPORT.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
