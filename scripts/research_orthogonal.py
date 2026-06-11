#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""正交因子离线验证 (影子盘计划第一步): 资金费做方向 / 基差做方向 / 大盘 lead-lag。
用法: python scripts/research_orthogonal.py --n 25 --days 30
诚实: 结论以扣费后 OOS 组合 NW-t 为准, 单格过=多重检验噪声。
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from okxb.research import candle_data as cd            # noqa: E402
from okxb.research import orthogonal_research as orth   # noqa: E402

CANDLE_ROOT = ROOT / "dist" / "candles"
REPORT = ROOT / "dist" / "orthogonal_research_report.txt"


def _resolve(arg, n):
    if not arg:
        return cd.top_crypto(n)
    return [(t.strip().upper() if t.strip().upper().endswith("-SWAP") else f"{t.strip().upper()}-USDT-SWAP")
            for t in arg.split(",") if t.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--days", type=float, default=30)
    ap.add_argument("--bar", default="5m")
    ap.add_argument("--insts", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    insts = _resolve(args.insts, args.n)
    print(f"标的({len(insts)}): {', '.join(s.split('-')[0] for s in insts)}\n")
    print(f"拉永续 {args.bar} ..."); perp = cd.fetch_universe(insts, args.bar, args.days, CANDLE_ROOT, force=args.force)
    print(f"\n拉现货 {args.bar} (basis腿) ...")
    spot_raw = cd.fetch_universe([cd.perp_to_spot(i) for i in perp], args.bar, args.days, CANDLE_ROOT, force=args.force)
    spot = {f"{k}-SWAP": v for k, v in spot_raw.items()}
    print("\n拉资金费 ...")
    funding = {i: cd.fetch_funding_series(i, args.days) for i in perp}
    funding = {k: v for k, v in funding.items() if len(v) >= 10}
    print(f"perp={len(perp)} spot={len(spot)} funding={len(funding)}")

    spans = [(perp[i]['ts'].iloc[-1] - perp[i]['ts'].iloc[0]) / 86_400_000 for i in perp]
    span_days = round(min(spans), 1) if spans else 0
    res = orth.run(perp, spot, funding, bar=args.bar)
    rep = orth.format_report({"span_days": span_days}, res)
    print("\n" + rep)
    REPORT.parent.mkdir(parents=True, exist_ok=True); REPORT.write_text(rep, encoding="utf-8")
    print(f"\n报告已存: {REPORT}")


if __name__ == "__main__":
    main()
