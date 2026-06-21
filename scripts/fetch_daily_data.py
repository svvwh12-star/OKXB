#!/usr/bin/env python
"""Pull all v2 data sources for the universe and write a data-quality report.

Research-only, public data, no keys. Reports per-source coverage (span in days)
so data-availability limits (e.g. OKX OI history is recent-window only) are visible.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from okxb.research import candle_data as cd          # noqa: E402
from okxb.research import deribit_data as dd          # noqa: E402
from okxb.research import onchain_data as oc          # noqa: E402
from okxb.research import macro_data as md            # noqa: E402

import pandas as pd                                   # noqa: E402

DAILY = ROOT / "dist" / "daily"


def _span_days(df: "pd.DataFrame") -> float:
    if df is None or len(df) < 2:
        return 0.0
    return round((int(df["ts"].iloc[-1]) - int(df["ts"].iloc[0])) / 86_400_000, 1)


def main() -> None:
    insts = ["BTC-USDT-SWAP", "ETH-USDT-SWAP"] + [f"{s}-USDT-SWAP" for s in ("SOL", "BNB", "XRP")]
    days = 365 * 3
    L = ["data-quality report (P0 daily sources)", "=" * 56,
         f"universe: {len(insts)} perps | request window: {days} days", ""]

    perp = cd.fetch_universe(insts, "1D", days, DAILY / "candles")
    L.append(f"OKX 1D candles: {len(perp)}/{len(insts)} instruments")
    for inst in perp:
        oi = cd.fetch_oi_history(inst, "1D", days)
        L.append(f"  {inst}: candles_rows={len(perp[inst])} span={_span_days(perp[inst])}d | "
                 f"oi_rows={len(oi)} oi_span={_span_days(oi)}d")

    L.append("")
    for ccy in ("BTC", "ETH"):
        dv = dd.fetch_dvol(ccy, days)
        L.append(f"Deribit {ccy} DVOL: rows={len(dv)} span={_span_days(dv)}d")

    L.append("")
    for asset in ("btc", "eth"):
        ocd = oc.fetch_onchain(asset, None, days, DAILY / "onchain")
        L.append(f"on-chain {asset}: " + ", ".join(f"{k}={len(v)}({_span_days(v)}d)" for k, v in ocd.items()))

    L.append("")
    mac = md.fetch_macro(days, DAILY / "macro")
    L.append("macro (stooq): " + (", ".join(f"{k}={len(v)}({_span_days(v)}d)" for k, v in mac.items())
                                   or "NONE REACHABLE"))

    L += ["",
          "notes:",
          "  - OKX OI history is recent-window only (~100 daily points); build long OI",
          "    history forward via the live recorder (P4). Older dates remain NaN in the panel.",
          "  - Deribit per-strike skew/term/GEX is a current snapshot only; DVOL is the",
          "    historical options signal. Snapshot job accumulates the forward archive.",
          "  - on-chain (Coin Metrics community) & macro are daily-cadence -> >=1d horizons only,",
          "    consumed with a publish lag (point-in-time)."]

    report = "\n".join(L)
    DAILY.mkdir(parents=True, exist_ok=True)
    (DAILY / "data_quality_report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
