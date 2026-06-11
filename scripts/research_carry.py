#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""结构性 carry 验证 (选项B, 只验证数据): 资金费套利 + perp-spot basis, 扣真实两腿费。

回答: OKX 单账户、扣费后, 资金费 carry 有没有可捕捉的正 spread?
(跨所价差不在范围: Binance 地区不可达 + 需第二账户跨境, 用户安全约束禁止VPN绕过。)

用法:
  python scripts/research_carry.py                 # 默认 30币 / 60天
  python scripts/research_carry.py --n 40 --days 90
  python scripts/research_carry.py --insts BTC,ETH,SOL
诚实: 结论以扣费后篮子回测 (年化净>0 ∧ NW-t>2 ∧ Sharpe>1) 为准; 必要非充分。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from okxb.research import candle_data as cd        # noqa: E402
from okxb.research import carry_research as car     # noqa: E402

CANDLE_ROOT = ROOT / "dist" / "candles"
REPORT = ROOT / "dist" / "carry_research_report.txt"


def _resolve(arg, n):
    if not arg:
        return cd.top_crypto(n)
    out = []
    for tok in arg.split(","):
        tok = tok.strip().upper()
        if tok:
            out.append(tok if tok.endswith("-SWAP") else f"{tok}-USDT-SWAP")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--days", type=float, default=60)
    ap.add_argument("--bar", default="5m")
    ap.add_argument("--insts", default=None)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--min-bps", type=float, default=0.0, help="最低过去均资金费(bps/8h)才harvest(默认0=收所有正费)")
    ap.add_argument("--holds", default="3,7,14,30", help="持有天数, 逗号分隔 (carry宜长持以摊薄固定费)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    holds = tuple(float(x) for x in args.holds.split(","))

    insts = _resolve(args.insts, args.n)
    print(f"标的({len(insts)}): {', '.join(s.split('-')[0] for s in insts)}\n")

    print(f"拉永续 {args.bar} K线 ...")
    perp = cd.fetch_universe(insts, args.bar, args.days, CANDLE_ROOT, force=args.force)
    spot_ids = [cd.perp_to_spot(i) for i in perp]
    print(f"\n拉现货 {args.bar} K线 (对冲腿) ...")
    spot_raw = cd.fetch_universe(spot_ids, args.bar, args.days, CANDLE_ROOT, force=args.force)
    # 把现货 key 映射回永续 key 以便对齐
    spot = {f"{k}-SWAP": v for k, v in spot_raw.items()}

    print("\n拉资金费率历史 ...")
    funding = {}
    for inst in perp:
        fs = cd.fetch_funding_series(inst, args.days)
        if len(fs) >= 10:
            funding[inst] = fs
    print(f"资金费可用 {len(funding)}/{len(perp)} 标的")

    common = [i for i in perp if i in spot and i in funding]
    print(f"三者齐全 {len(common)} 标的\n")
    if len(common) < 5:
        print("可用标的太少, 无法验证。")
        return

    spans = [(perp[i]['ts'].iloc[-1] - perp[i]['ts'].iloc[0]) / 86_400_000 for i in common]
    span_days = round(min(spans), 1)

    res = car.run(perp, spot, funding, holds_days=holds, top_k=args.top_k, min_per8h_bps=args.min_bps)
    report = car.format_report({"span_days": span_days}, res)
    print("\n" + report)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")
    print(f"\n报告已存: {REPORT}")


if __name__ == "__main__":
    main()
