#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run the professional 15/30min modeling workflow.

This is a research-only command. It uses public OKX market data and local caches,
does not place orders, and does not read API keys.

Example:
  python scripts/research_pro_model.py --n 25 --days 30 --horizons 15,30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from okxb.research import candle_data as cd  # noqa: E402
from okxb.research import pro_model_workflow as pmw  # noqa: E402

CANDLE_ROOT = ROOT / "dist" / "candles"
REPORT = ROOT / "dist" / "pro_model_workflow_report.txt"


def _resolve_insts(arg: str | None, n: int) -> list[str]:
    if not arg:
        return cd.top_crypto(n)
    out = []
    for tok in arg.split(","):
        tok = tok.strip().upper()
        if not tok:
            continue
        out.append(tok if tok.endswith("-SWAP") else f"{tok}-USDT-SWAP")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25, help="top crypto USDT swaps by 24h quote volume")
    ap.add_argument("--days", type=float, default=30, help="historical days")
    ap.add_argument("--bar", default="5m", choices=list(cd.BAR_MS), help="candle bar")
    ap.add_argument("--insts", default=None, help="comma-separated symbols, e.g. BTC,ETH,SOL")
    ap.add_argument("--horizons", default="15,30", help="comma-separated minutes, default 15,30")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=22, help="features selected per fold")
    ap.add_argument("--min-train", type=int, default=None, help="minimum training rows per fold")
    ap.add_argument("--preset", choices=["fast", "deep"], default="fast",
                    help="fast for interactive iteration; deep adds heavier tree ensembles")
    ap.add_argument("--cost-maker", type=float, default=4.0)
    ap.add_argument("--cost-taker", type=float, default=10.0)
    ap.add_argument("--cost-stress", type=float, default=15.0)
    ap.add_argument("--skip-spot", action="store_true", help="skip spot/basis features")
    ap.add_argument("--skip-funding", action="store_true", help="skip funding features")
    ap.add_argument("--force", action="store_true", help="ignore candle cache and refetch")
    args = ap.parse_args()

    horizons = tuple(int(x) for x in args.horizons.split(",") if x.strip())
    bar_min = cd.BAR_MS[args.bar] // 60_000
    horizons = tuple(h for h in horizons if h >= bar_min)
    if not horizons:
        raise SystemExit("No valid horizons after filtering by bar size.")

    insts = _resolve_insts(args.insts, args.n)
    print(f"instruments({len(insts)}): {', '.join(i.split('-')[0] for i in insts)}")
    print(f"fetch/reuse perp {args.bar} candles: {args.days:g} days")
    perp = cd.fetch_universe(insts, args.bar, args.days, CANDLE_ROOT, force=args.force)
    if len(perp) < 1:
        raise SystemExit("no usable perp instruments")

    spot = {}
    if not args.skip_spot:
        print("\nfetch/reuse spot candles for basis features")
        spot_raw = cd.fetch_universe([cd.perp_to_spot(i) for i in perp], args.bar, args.days, CANDLE_ROOT, force=args.force)
        spot = {f"{k}-SWAP": v for k, v in spot_raw.items()}

    funding = {}
    if not args.skip_funding:
        print("\nfetch public funding history")
        for inst in perp:
            f = cd.fetch_funding_series(inst, args.days)
            if len(f):
                funding[inst] = f
        print(f"funding available: {len(funding)}/{len(perp)}")

    spans = [(df["ts"].iloc[-1] - df["ts"].iloc[0]) / 86_400_000 for df in perp.values()]
    span_days = round(min(spans), 1) if spans else 0.0
    costs = pmw.WorkflowCosts(args.cost_maker, args.cost_taker, args.cost_stress)
    res = pmw.run(
        perp,
        spot_dfs=spot,
        funding_dfs=funding,
        bar=args.bar,
        horizons_min=horizons,
        n_folds=args.folds,
        k_sel=args.top_k,
        min_train=args.min_train,
        preset=args.preset,
        costs=costs,
    )
    report = pmw.format_report({"span_days": span_days}, res)
    print("\n" + report)
    report_path = REPORT
    if len(perp) == 1:
        only = next(iter(perp)).split("-")[0].lower()
        report_path = REPORT.with_name(f"pro_model_workflow_{only}_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"\nreport saved: {report_path}")


if __name__ == "__main__":
    main()
