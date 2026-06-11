"""回测评估与上线门槛 (用户方案 §15.4 / RESEARCH_BRIEF §7)。

本模块提供:
  - GoLiveGate: 用一组交易收益判定是否满足上线门槛。
  - evaluate_trades: 汇总 PF/胜率/盈亏比/Sharpe/回撤。

完整的事件驱动 L2 回放回测 (用录制的 books 增量重建订单簿、按保守队尾成交 + 滑点
重跑策略) 需要先录制完整盘口快照, 是下一步 (当前 run_phase1 只录 mid/spread)。
诚实声明: 任何回测都会高估成交 (队列位置/touch即成/无逆选), 实盘结果通常显著更差。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Config
from . import labeling as lb


@dataclass
class TradeStats:
    n: int
    profit_factor: float
    win_rate: float
    avg_rr: float
    sharpe: float
    max_drawdown: float
    total_return: float


def evaluate_trades(returns: list[float]) -> TradeStats:
    return TradeStats(
        n=len(returns),
        profit_factor=lb.profit_factor(returns),
        win_rate=lb.win_rate(returns),
        avg_rr=lb.avg_rr(returns),
        sharpe=lb.sharpe(returns),
        max_drawdown=lb.max_drawdown(returns),
        total_return=sum(returns),
    )


@dataclass
class GateResult:
    passed: bool
    checks: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)


class GoLiveGate:
    """样本外交易需全部满足才允许进入下一阶段。"""

    def __init__(self, config: Config):
        g = config.section("go_live_gate")
        self.min_trades = int(g.get("min_oos_trades", 300))
        self.min_pf = float(g.get("min_profit_factor", 1.25))
        self.min_win = float(g.get("min_win_rate_pct", 55)) / 100.0
        self.min_rr = float(g.get("min_avg_rr", 0.8))
        self.max_dd_vs_profit = float(g.get("max_drawdown_vs_expected_profit_pct", 50)) / 100.0

    def check(self, returns: list[float]) -> GateResult:
        s = evaluate_trades(returns)
        checks, reasons = {}, []

        def chk(name, ok, detail):
            checks[name] = ok
            if not ok:
                reasons.append(detail)

        chk("trades", s.n >= self.min_trades, f"样本 {s.n} < {self.min_trades}")
        chk("profit_factor", s.profit_factor >= self.min_pf,
            f"PF {s.profit_factor:.2f} < {self.min_pf}")
        chk("win_rate", s.win_rate >= self.min_win,
            f"胜率 {s.win_rate:.2%} < {self.min_win:.0%}")
        chk("avg_rr", s.avg_rr >= self.min_rr, f"盈亏比 {s.avg_rr:.2f} < {self.min_rr}")
        if s.total_return > 0:
            chk("drawdown", abs(s.max_drawdown) <= self.max_dd_vs_profit * s.total_return,
                f"回撤 {abs(s.max_drawdown):.4f} > 预期利润 {s.total_return:.4f} 的 "
                f"{self.max_dd_vs_profit:.0%}")
        else:
            chk("drawdown", False, "总收益<=0")
        return GateResult(passed=all(checks.values()), checks=checks, reasons=reasons)
