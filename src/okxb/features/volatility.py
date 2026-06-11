"""波动因子 (纯函数)。用于动态止盈止损与仓位缩放 (RESEARCH_BRIEF §7)。"""
from __future__ import annotations

import math


def realized_vol(mids: list[float]) -> float | None:
    """窗口内对数收益标准差 (per-step, 非年化)。用作短周期波动度量。"""
    if len(mids) < 3:
        return None
    rets = [math.log(mids[i] / mids[i - 1]) for i in range(1, len(mids)) if mids[i - 1] > 0]
    if len(rets) < 2:
        return None
    mu = sum(rets) / len(rets)
    var = sum((r - mu) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var)


def atr_proxy(mids: list[float], window: int = 60) -> float | None:
    """无 K 线时的 ATR 近似: 滚动窗口内极差 / 均价 (返回百分比小数)。
    真正的 ATR 需要 OHLC; 后续接入 K 线后替换。"""
    if len(mids) < 3:
        return None
    w = mids[-window:]
    avg = sum(w) / len(w)
    if avg <= 0:
        return None
    return (max(w) - min(w)) / avg
