#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分钟~小时级方向预测检验 (K线驱动) —— 选项(i) 的可执行入口。

回答: 用合适的分钟级特征, 能否预测未来 5/15/30/60/120/240min 的方向, 且扣费后净正?
这是秒级 tick (validate.py) 无法检验的问题。

用法:
  python scripts/research_candles.py                      # 默认 25币 / 30天 / 5m
  python scripts/research_candles.py --n 30 --days 45 --bar 5m
  python scripts/research_candles.py --insts BTC,ETH,SOL  # 指定币 (base 名或全 instId)
  python scripts/research_candles.py --force              # 忽略缓存重新拉取

输出: 终端报告 + dist/candles/ 缓存 + dist/candle_research_report.txt
诚实: 结论以【样本外组合、扣费后】为准; 必要非充分条件。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from okxb.research import candle_data as cd            # noqa: E402
from okxb.research import candle_research as cr        # noqa: E402

CANDLE_ROOT = ROOT / "dist" / "candles"
REPORT = ROOT / "dist" / "candle_research_report.txt"


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
    ap.add_argument("--n", type=int, default=25, help="加密永续个数 (按24h量, 默认25)")
    ap.add_argument("--days", type=float, default=30, help="回填天数 (默认30)")
    ap.add_argument("--bar", default="5m", choices=list(cr.BAR_MIN), help="K线周期 (默认5m)")
    ap.add_argument("--insts", default=None, help="逗号分隔的币 (覆盖--n)")
    ap.add_argument("--horizons", default=None, help="逗号分隔的预测分钟数, 如 5,15,30,60,120,240")
    ap.add_argument("--cost-taker", type=float, default=10.0)
    ap.add_argument("--cost-maker", type=float, default=4.0)
    ap.add_argument("--q", type=float, default=0.2, help="多空各取的分位 (默认前/后20%)")
    ap.add_argument("--no-funding", action="store_true", help="跳过资金费拉取(方向择时书将不扣资金费)")
    ap.add_argument("--force", action="store_true", help="忽略缓存重拉")
    args = ap.parse_args()

    horizons = (tuple(int(x) for x in args.horizons.split(","))
                if args.horizons else cr.DEFAULT_HORIZONS_MIN)
    # 预测窗必须 ≥ 1 根 K线
    bar_min = cr.BAR_MIN[args.bar]
    horizons = tuple(h for h in horizons if h >= bar_min)

    insts = _resolve_insts(args.insts, args.n)
    print(f"标的({len(insts)}): {', '.join(s.split('-')[0] for s in insts)}")
    print(f"拉取 {args.days} 天 {args.bar} K线 → {CANDLE_ROOT} (缓存{'强制重拉' if args.force else '复用'})\n")

    dfs = cd.fetch_universe(insts, args.bar, args.days, CANDLE_ROOT, force=args.force)
    if len(dfs) < 5:
        print(f"\n可用标的太少 ({len(dfs)}), 无法做横截面分析。检查网络/缓存。")
        return

    spans = [(df["ts"].iloc[-1] - df["ts"].iloc[0]) / 86_400_000 for df in dfs.values()]
    span_days = round(min(spans), 1)

    # 资金费 (供方向择时书扣减; 横截面多空多空相抵不需要)
    funding = {}
    if not args.no_funding:
        print("\n拉取资金费率历史 ...")
        for inst in dfs:
            funding[inst] = cd.fetch_funding_mean(inst, args.days)
        ok = sum(1 for v in funding.values() if v is not None)
        print(f"资金费可用 {ok}/{len(dfs)} 标的")

    res = cr.run(dfs, args.bar, horizons_min=horizons, funding=funding,
                 cost_taker_bps=args.cost_taker, cost_maker_bps=args.cost_maker, q=args.q)
    report = cr.format_report({"span_days": span_days}, res)
    print("\n" + report)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")
    print(f"\n报告已存: {REPORT}")


if __name__ == "__main__":
    main()
