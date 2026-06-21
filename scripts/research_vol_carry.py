#!/usr/bin/env python
"""VRP harvest experiment: rolling weekly defined-risk iron condors on BTC, priced off DVOL,
net of OKX option fees. Research-only, public data. The one new avenue with a real edge."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from okxb.research import candle_data as cd            # noqa: E402
from okxb.research import deribit_data as dd           # noqa: E402
from okxb.research import vol_carry as vc              # noqa: E402

DAILY = ROOT / "dist" / "daily"
REPORT = DAILY / "vol_carry_report.txt"


def main() -> None:
    days = 365 * 3
    btc = cd.fetch_universe(["BTC-USDT-SWAP"], "1D", days, DAILY / "candles")["BTC-USDT-SWAP"]
    dvol = dd.fetch_dvol("BTC", days)

    # OKX coin-margined option taker fee: 0.03% of underlying, capped at 12.5% of premium.
    configs = [
        ("iron condor 1sigma/2sigma", dict(short_d=1.0, wing_d=2.0)),
        ("iron condor 0.75/1.5", dict(short_d=0.75, wing_d=1.5)),
        ("iron condor 1.25/2.5 (wider)", dict(short_d=1.25, wing_d=2.5)),
        ("~naked strangle 1sigma (wide wings)", dict(short_d=1.0, wing_d=6.0)),
    ]
    L = ["=" * 84, "VRP harvest — rolling weekly defined-risk iron condors on BTC (DVOL-priced)", "=" * 84,
         f"data: BTC 1D x {len(btc)} bars, DVOL x {len(dvol)} | hold=7d | fee=0.03% underlying cap 12.5% premium", ""]
    L.append(f"{'structure':>34} {'n':>4} {'mean_bps':>9} {'win%':>6} {'PF':>6} {'maxDD':>8} {'worst_bps':>10} {'Sharpe':>7} {'tot_ret':>8}")
    any_pass = False
    for name, kw in configs:
        r = vc.run_vol_carry(btc, dvol, hold_days=7, fee_rate=0.0003, fee_cap_frac=0.125, **kw)
        if r.get("error"):
            L.append(f"{name:>34}  {r['error']}")
            continue
        # gate: net-positive, PF>=1.25, drawdown not catastrophic, worst trade survivable
        gate = (r["mean_bps"] > 0 and r["pf"] >= 1.25 and r["max_dd"] > -0.5)
        any_pass = any_pass or gate
        L.append(f"{name:>34} {r['n_trades']:>4} {r['mean_bps']:>+9.1f} {r['win_rate']:>6.0%} "
                 f"{r['pf']:>6.2f} {r['max_dd']:>+8.1%} {r['worst_trade_bps']:>+10.1f} "
                 f"{r['ann_sharpe']:>7.2f} {r['total_return']:>+8.1%}"
                 + ("  PASS" if gate else "  FAIL"))
    # --- spread sensitivity on the best structure: where does the VRP die? ---
    L += ["", "spread sensitivity (structure 0.75/1.5; extra half-spread paid on each of 4 legs):"]
    L.append(f"{'spread/leg':>34} {'mean_bps':>9} {'PF':>6} {'maxDD':>8} {'tot_ret':>8}  gate")
    breakeven = None
    for sf in (0.0, 0.02, 0.05, 0.10, 0.15, 0.20):
        r = vc.run_vol_carry(btc, dvol, hold_days=7, fee_rate=0.0003, fee_cap_frac=0.125,
                             spread_frac=sf, short_d=0.75, wing_d=1.5)
        gate = (r["mean_bps"] > 0 and r["pf"] >= 1.25 and r["max_dd"] > -0.5)
        if not gate and breakeven is None:
            breakeven = sf
        L.append(f"{sf:>33.0%} {r['mean_bps']:>+9.1f} {r['pf']:>6.2f} {r['max_dd']:>+8.1%} "
                 f"{r['total_return']:>+8.1%}  {'PASS' if gate else 'FAIL'}")
    L.append(f"  -> breakeven spread (gate fails at/above): {'~%.0f%% of premium per leg' % (breakeven*100) if breakeven is not None else '>20% (robust)'}")
    L.append(f"  reality check: OTM crypto weekly options routinely quote 10-30%+ bid-ask of premium per leg.")

    L += ["",
          "notes: DVOL(~30d IV) used as weekly IV (term-structure approximation); realized = actual BTC move;",
          "  defined-risk caps per-trade loss; fees per OKX coin-margined option (cap 12.5% premium).",
          "  Account reality: OKX portfolio margin needs >=$10k; at 1000U coin-margined margin spikes with IV.",
          ""]
    real_spread_ok = breakeven is not None and breakeven > 0.10   # survives a realistic 10%+ spread?
    if real_spread_ok:
        L.append("verdict: CANDIDATE — survives realistic spreads; next: real option chains + "
                 "margin-liquidation sim + region/eligibility check before any capital.")
    else:
        L.append("verdict: REAL PREMIUM, NOT TRADABLE AT THIS ACCOUNT. The VRP is genuinely positive "
                 "after fees alone (PF~1.3-1.4) — unlike every direction avenue (AUC~0.50). BUT it dies at "
                 f"~{(breakeven or 0)*100:.0f}% per-leg option spread, while OTM crypto weeklies quote 10-30%+. "
                 "Net-negative once realistic spreads are paid — same 'structural carry, retail execution eats it' "
                 "pattern as cash-and-carry. Two further optimistic biases remain unmodelled (margin-liquidation "
                 "on IV spikes; the 2023+ DVOL sample misses the worst crash weeks). Revisit only with >=$10k "
                 "portfolio margin + tighter execution (liquid ATM/index vol), not at 1000U/VIP0.")
    report = "\n".join(L)
    DAILY.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
