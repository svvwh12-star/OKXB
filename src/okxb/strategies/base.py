"""策略基类。策略消费 FeatureSet + CompositeResult, 决定是否产出交易 Signal。
策略只负责"该不该做、做多还是做空、止盈止损建议"; 仓位大小由 RiskEngine 决定。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..config import Config
from ..core.enums import StrategyId
from ..core.models import FeatureSet, Signal
from ..signal.composite import CompositeResult, CompositeScorer


class Strategy(ABC):
    id: StrategyId

    def __init__(self, config: Config):
        self.cfg = config
        sig = config.section("signal")
        # 对称尺度(多+空=100, 50中性): 入场门槛 66≈方向0.32; 强信号 74≈方向0.48 (由『校准』按数据重定)
        self.min_composite = float(sig.get("min_composite_score", 66))
        self.strong_composite = float(sig.get("strong_composite_score", 74))
        self.min_prob = float(sig.get("min_model_probability", 0.58))
        self.min_edge_cost = float(sig.get("min_edge_to_cost_ratio", 1.5))
        self.confirm_min = float(sig.get("confirm_min", 0.20))
        self.persist_ticks = int(sig.get("persist_ticks", 3))
        self.cooldown_ms = float(sig.get("cooldown_seconds", 20)) * 1000.0
        # 可交易性独立门槛(与方向分解耦): 正常行情≈1, 仅价差过宽/深度过薄/波动死或过烈才<此值
        # (对称尺度下"反向分上限 opp_max"恒满足 -> 已删除冗余项, 改由可交易性 + 方向闩锁过滤冲突)
        self.min_tradability = float(sig.get("min_tradability", 0.5))
        self.max_entry_spread_bps = float(sig.get("max_entry_spread_bps", 50.0))  # 绝对硬门: 价差超此bps直接不开新仓(软分之外的硬风控)
        # 波动自适应 + 容错 (与去噪信号配合)
        self.regime_rv_lo = float(sig.get("regime_rv_lo", 5e-5))   # 低于此=真冻结不做(原2e-4过高); 行情淡由期望edge门管
        self.regime_rv_hi = float(sig.get("regime_rv_hi", 1.2e-3))  # 高于此=剧烈, 多等几拍
        self.regime_persist_bonus = int(sig.get("regime_persist_bonus", 1))
        self.persist_miss_grace = int(sig.get("persist_miss_grace", 1))  # 容忍单拍坏点不清零

    @abstractmethod
    def evaluate(self, fs: FeatureSet, comp: CompositeResult,
                 scorer: CompositeScorer) -> Optional[Signal]:
        ...

    def note_exit(self, inst_id: str) -> None:
        """执行器平仓后回调, 触发该标的冷却 (默认无状态策略忽略)。"""

    @staticmethod
    def conf(comp: CompositeResult, direction: int) -> float:
        """单一确认值: 流向 + 趋势 与方向一致的合并强度, 范围[0,1]。"""
        f = max(0.0, direction * comp.order_flow_dir)
        t = max(0.0, direction * comp.trend_dir)
        return 0.5 * f + 0.5 * t

    def is_strong(self, composite: float, sig: Signal) -> bool:
        # 占位 model_prob 不再参与; 强信号 = 综合分达强阈值 (用于放大仓位/允许taker)
        return composite >= self.strong_composite
