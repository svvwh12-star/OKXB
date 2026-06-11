#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI 事件模块独立测试 (不触达 OKX, 不下单)。
================================================
验证: SEC EDGAR 真实拉取 + 分类器(规则降级或 Claude) + veto 逻辑。
EDGAR 免费无 key; Finnhub/Claude 无 key 时自动降级。

用法:  python scripts/test_events.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from okxb.config import Secrets                       # noqa: E402
from okxb.core.enums import Side                      # noqa: E402
from okxb.events.service import AIEventService, ActiveEvent  # noqa: E402


async def main() -> None:
    secrets = Secrets()
    from okxb.config import Config
    cfg = Config.load()
    tickers = {"NVDA", "AAPL", "MSFT"}

    svc = AIEventService(cfg, secrets, tickers)
    print("== 1) 加载 EDGAR ticker 映射 ==")
    await svc.setup()
    for t in tickers:
        print(f"   {t} -> CIK {svc.edgar.cik_for(t)}")

    print("\n== 2) 拉取 NVDA 最近 filings (真实 SEC 数据) ==")
    cik = svc.edgar.cik_for("NVDA")
    filings = await svc.edgar.recent_filings(cik, forms={"8-K", "4", "10-Q", "10-K"})
    for f in filings[:6]:
        print(f"   {f['form']:<6} {f['date']} items={f['items'] or '-':<10} {f['accession']}")

    print("\n== 3) 分类器测试 (LLM 或规则降级) ==")
    for desc, kind, items in [
        ("NVIDIA announces departure of Chief Financial Officer", "8-K", "5.02"),
        ("Quarterly results of operations released", "8-K", "2.02"),
        ("Company to be acquired in all-stock merger", "news", ""),
        ("Routine investor conference appearance", "news", ""),
    ]:
        label = await svc.llm.classify("NVDA", desc, kind, items)
        print(f"   [{kind} {items or '-'}] {desc[:40]:<42} -> {label['event_type']}/"
              f"{label['severity']}/{label['action']}")

    print("\n== 4) 轮询一次 (财报日历 + 新 filing) ==")
    await svc.poll_once()
    print(f"   当前活跃事件缓存: {dict((k, [e.mode for e in v]) for k, v in svc._events.items()) or '空'}")

    print("\n== 5) veto 逻辑测试 (手动注入一个财报窗口) ==")
    import time
    svc._add("NVDA", ActiveEvent("block_new", "earnings", "medium",
                                 int(time.time() * 1000) + 600_000, confidence=0.8))
    for inst, side in [("NVDA-USDT-SWAP", Side.BUY), ("NVDA-USDT-SWAP", Side.SELL),
                       ("BTC-USDT-SWAP", Side.BUY)]:
        v = svc.get_veto(inst, side)
        print(f"   {inst} {side.value} -> {v.action.value if v else 'None (放行)'}")

    await svc.aclose()
    print("\n[test_events] 完成。")


if __name__ == "__main__":
    asyncio.run(main())
