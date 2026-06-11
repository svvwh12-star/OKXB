#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 1 前瞻收益分析。
================================================
读取 run_phase1.py 产出的 ticks + signals, 评估假设信号在【扣费后】是否有 edge:
  - 入场后 30s/60s/180s 的有方向 gross / net 收益
  - triple-barrier 标签胜率 (用信号自带的 SL/TP)
  - profit factor / 平均盈亏比

用法:  python scripts/analyze_phase1.py            (自动取最新一组)
       python scripts/analyze_phase1.py <stamp>    (指定时间戳)

诚实提醒: demo 数据的成交/滑点不真实; 小样本下结论不可靠 (RESEARCH_BRIEF §7)。
本分析衡量的是"信号后价格是否朝预期方向走", 是 edge 的必要非充分条件。
"""
from __future__ import annotations

import json
import sys
from bisect import bisect_left
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from okxb.research import labeling as lb   # noqa: E402

REC = ROOT / "recordings"
HORIZONS = [30_000, 60_000, 180_000]
MAX_HOLD_MS = 180_000


def _latest(prefix: str) -> Path | None:
    files = sorted(REC.glob(f"{prefix}_*.jsonl"))
    return files[-1] if files else None


def _load_ticks(path: Path) -> dict[str, tuple[list[int], list[float]]]:
    by_inst: dict[str, tuple[list[int], list[float]]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            ts_list, mids = by_inst.setdefault(d["inst"], ([], []))
            ts_list.append(int(d["ts"]))
            mids.append(float(d["mid"]))
    return by_inst


def main() -> None:
    stamp = sys.argv[1] if len(sys.argv) > 1 else None
    if stamp:
        tick_path = REC / f"phase1_ticks_{stamp}.jsonl"
        sig_path = REC / f"phase1_signals_{stamp}.jsonl"
    else:
        tick_path = _latest("phase1_ticks")
        sig_path = _latest("phase1_signals")

    if not tick_path or not tick_path.exists():
        print("未找到 tick 记录。请先跑 python scripts/run_phase1.py 收集数据。")
        return
    if not sig_path or not sig_path.exists():
        print(f"找到 ticks 但无 signals ({sig_path}); 说明扫描期内没有达到阈值的信号。")
        return

    ticks = _load_ticks(tick_path)
    signals = []
    with open(sig_path, encoding="utf-8") as f:
        for line in f:
            try:
                signals.append(json.loads(line))
            except ValueError:
                pass

    print(f"ticks: {tick_path.name}  signals: {sig_path.name} ({len(signals)} 条)\n")
    if not signals:
        print("无信号可分析。")
        return

    # 收集每个 horizon 的 net 收益, 以及 triple-barrier 结果
    net_by_h: dict[int, list[float]] = {h: [] for h in HORIZONS}
    tb_returns: list[float] = []
    tb_labels: list[int] = []

    for s in signals:
        inst = s["inst"]
        if inst not in ticks:
            continue
        ts_list, mids = ticks[inst]
        if len(ts_list) < 5:
            continue
        side = 1 if s["side"] == "buy" else -1
        entry_ts = int(s["ts"])
        entry_mid = float(s.get("mid") or 0) or None
        if entry_mid is None:
            idx0 = bisect_left(ts_list, entry_ts)
            if idx0 >= len(mids):
                continue
            entry_mid = mids[idx0]
        cost = float(s.get("cost_pct", 0))

        for h in HORIZONS:
            fr = lb.forward_return(ts_list, mids, entry_ts, entry_mid, h, side)
            if fr is not None:
                net_by_h[h].append(fr - cost)

        idx = bisect_left(ts_list, entry_ts)
        if idx < len(ts_list) - 1:
            br = lb.triple_barrier(ts_list, mids, idx, side,
                                   float(s.get("tp_pct", 0.004)),
                                   float(s.get("sl_pct", 0.002)), MAX_HOLD_MS)
            tb_returns.append(br.ret_pct - cost)
            tb_labels.append(br.label)

    print("=== 前瞻 net 收益 (扣往返成本) ===")
    print(f"{'horizon':>9}{'n':>6}{'均值bps':>10}{'胜率':>8}{'PF':>7}")
    for h in HORIZONS:
        rs = net_by_h[h]
        if not rs:
            continue
        mean_bps = sum(rs) / len(rs) * 1e4
        print(f"{h//1000:>7}s{len(rs):>6}{mean_bps:>10.2f}{lb.win_rate(rs):>8.2%}"
              f"{lb.profit_factor(rs):>7.2f}")

    if tb_returns:
        print("\n=== Triple-barrier (用信号 SL/TP, 扣成本) ===")
        n = len(tb_returns)
        tp = sum(1 for x in tb_labels if x == 1)
        sl = sum(1 for x in tb_labels if x == -1)
        to = sum(1 for x in tb_labels if x == 0)
        print(f"样本 {n}  先触TP {tp}  先触SL {sl}  超时 {to}")
        print(f"净胜率 {lb.win_rate(tb_returns):.2%}  PF {lb.profit_factor(tb_returns):.2f}  "
              f"平均盈亏比 {lb.avg_rr(tb_returns):.2f}  Sharpe(每笔) {lb.sharpe(tb_returns):.3f}  "
              f"最大回撤 {lb.max_drawdown(tb_returns)*1e4:.1f}bps")
        print("\n上线门槛 (§15.4): 样本外>=300笔, PF>=1.25, 扣费胜率>=55%, 平均盈亏比>=0.8。")
        print("当前为 demo/小样本, 仅作链路验证, 不能据此上实盘。")


if __name__ == "__main__":
    main()
