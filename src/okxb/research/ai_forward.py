"""AI 前向验证候选: 测【AI 自己的方向判断】在 N 分钟后是否有净 edge (前向 IC)。

定位: 预登记 + 只记录, 不实盘。流程: 现在记下 AI 对某标的的 long/short 判断 + 当时中间价 ->
HORIZON 分钟后用实价结算 net (扣成本) -> 累计判 AI 方向是否前向有效。
诚实预期: AI 读盘口/价/资金费给方向, 多半与未来无关 -> 长期 PENDING/KILL。这正是要测的:
用未来数据如实证伪"AI 能不能预测", 而不是凭感觉相信它。

纯标准库 + 项目内 forward_integrity/labeling 原语; 无网络 (取价 + 调 AI 在 runner)。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .forward_integrity import bonferroni_t
from .labeling import deflated_sharpe, profit_factor

CODE = "AIFWD"
HORIZON_MIN = 60                      # AI 判断后 60 分钟用实价结算
COST_BPS_STRESS = 15.0
COST_BPS_MILD = 10.0
MIN_FWD_TS = 100                      # PASS 需 >=100 个已结算前向样本
WATCHLIST = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"]
FAMILY_TRIALS = 1                     # 单一假设: "AI 方向有 edge" (跨标的合并为一个候选)


def net_bps(direction: int, entry: float, exit_: float, cost_bps: float) -> float:
    if entry <= 0:
        return 0.0
    return direction * (exit_ / entry - 1.0) * 1e4 - cost_bps


@dataclass(slots=True)
class Verdict:
    verdict: str
    reason: str
    metrics: dict


def evaluate(net15: list, net10: list, dir_ret_bps: list, *,
             family_trials: int = FAMILY_TRIALS, already_dead: bool = False) -> Verdict:
    """dir_ret_bps = direction*forward_ret*1e4 (无成本; 其均值>0 = AI 方向与未来正相关)。"""
    n = len(net15)
    m15 = (sum(net15) / n) if n else 0.0
    m10 = (sum(net10) / n) if n else 0.0
    ic = (sum(dir_ret_bps) / n) if n else 0.0
    metrics = {"n_ts": n, "net15_mean_bps": round(m15, 3), "net10_mean_bps": round(m10, 3),
               "ai_ic_bps": round(ic, 3)}
    if already_dead:
        return Verdict("KILL", "already_dead_sticky", metrics)

    t = None
    if n >= 2:
        var = sum((x - m15) ** 2 for x in net15) / (n - 1)
        sd = math.sqrt(var)
        t = (m15 / (sd / math.sqrt(n))) if sd > 1e-12 else None
    thr = bonferroni_t(family_trials)
    dsr = deflated_sharpe(net15, family_trials) if n >= 10 else None
    pf = profit_factor(net15) if n else 0.0
    metrics.update({"t": (round(t, 3) if t is not None else None), "t_threshold": round(thr, 3),
                    "dsr": (round(dsr, 4) if dsr is not None else None),
                    "pf": (round(pf, 3) if pf != float("inf") else None)})

    if n >= 30 and m10 <= 0.0:
        return Verdict("KILL", "net10<=0 after >=30 (AI 方向无 edge, even at 10bps)", metrics)
    if n < MIN_FWD_TS:
        return Verdict("PENDING", f"insufficient forward ts ({n}/{MIN_FWD_TS})", metrics)

    gates = []
    if m15 <= 0:
        gates.append("net15<=0")
    if ic <= 0:
        gates.append("ai_ic<=0")
    if t is None or t < thr:
        gates.append("t<bonferroni")
    if dsr is None or dsr < 0.95:
        gates.append("dsr<0.95")
    if pf < 1.2:
        gates.append("pf<1.2")
    if gates:
        return Verdict("PENDING", "failed gates: " + ",".join(gates), metrics)
    return Verdict("PASS", "AI direction is forward-validated (net edge survives cost)", metrics)
