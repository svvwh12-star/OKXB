#!/usr/bin/env python
"""Stage-1 daily orthogonal net-edge study. Research-only, public data, no keys.

Pre-registers the hypothesis grid (horizon, feature-groups, benchmark) to hypotheses.jsonl
BEFORE fitting, then runs the audited net-edge gate and writes a verdict report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from okxb.research import candle_data as cd            # noqa: E402
from okxb.research import deribit_data as dd            # noqa: E402
from okxb.research import onchain_data as oc            # noqa: E402
from okxb.research import daily_orthogonal as do        # noqa: E402

DAILY = ROOT / "dist" / "daily"
HYP = DAILY / "hypotheses.jsonl"
REPORT = DAILY / "daily_workflow_report.txt"


def main() -> None:
    insts = ["BTC-USDT-SWAP", "ETH-USDT-SWAP"] + [f"{s}-USDT-SWAP" for s in ("SOL", "BNB", "XRP")]
    days = 365 * 3
    horizons = (1440, 2880, 4320)  # 1d / 2d / 3d (candidate bands)

    DAILY.mkdir(parents=True, exist_ok=True)
    with HYP.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"stage": "P1-MVP", "horizons_min": list(horizons),
                             "feature_groups": ["price_vol", "basis", "funding", "dvol_vrp", "onchain", "cross_asset"],
                             "benchmark": "net-edge gate (maker4/taker10/stress15 + NW-t>2, >=2 buckets)",
                             "universe": insts}) + "\n")

    perp = cd.fetch_universe(insts, "1D", days, DAILY / "candles")
    spot = {f"{k}-SWAP": v for k, v in
            cd.fetch_universe([cd.perp_to_spot(i) for i in perp], "1D", days, DAILY / "candles").items()}
    funding = {i: cd.fetch_funding_series(i, days) for i in perp}
    dvol = {c: dd.fetch_dvol(c, days) for c in ("BTC", "ETH")}
    onchain = {a: oc.fetch_onchain(a, None, days, DAILY / "onchain") for a in ("btc", "eth")}

    res = do.run_daily(perp, spot_dfs=spot, funding_dfs=funding,
                       dvol_by_ccy=dvol, onchain_by_asset=onchain, horizons_min=horizons)

    lines = ["=" * 80, "Stage-1 daily orthogonal net-edge study (MVP)", "=" * 80,
             f"mode={res['mode']} horizons={res['horizons']} costs={res['costs']}", ""]
    any_trade = False
    for H in res["horizons"]:
        d = res["by_h"][H]; best = d.get("best"); pos = d.get("position", {})
        any_trade = any_trade or bool(pos.get("tradable"))
        lines.append(f"--- H={H}min  features={d['n_features']} ---")
        lines.append("often selected: " + ", ".join(f"{f}x{n}" for f, n in d["selected_freq"][:12]))
        for name, mm in d["metrics"].items():
            if mm.get("skip"):
                continue
            cells = [c for _, c in mm["curve"] if c is not None]
            bc = max(cells, key=lambda c: c.get("net_4", -1e9), default=None)
            if bc:
                lines.append(f"  {name:>12}: auc={mm['auc']:.3f} ic={mm.get('primary_ic', float('nan')):+.4f} "
                             f"best net4={bc['net_4']:+.1f}(t{bc['t_4']:+.1f}) net10={bc['net_10']:+.1f}(t{bc['t_10']:+.1f}) n={bc['n_ts']}")
        if best:
            lines.append(f"  best: {best['model']} gate={'PASS' if best['gate'] else 'FAIL'}")
        lines.append(f"  sizing: tradable={pos.get('tradable')} reason={pos.get('reason', '')}")
        lines.append("")
    lines.append("verdict: " + ("CANDIDATE — run P1b (DSR/PBO + held-out + CMOM + conformal) before believing."
                                if any_trade else
                                "NO TRADE — no daily-horizon orthogonal model passed the net-edge gate."))
    report = "\n".join(lines)
    REPORT.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
