"""强趋势突破策略 (用户方案 §10)。

唯一允许少量 taker (optimal_limit_ioc) 的策略, 但触发门槛最高:
综合分≥90 + 订单流强烈单边 + 趋势一致。风控会再用 taker 风险上限与"仅强信号"限制把关。
"""
from __future__ import annotations

from typing import Optional

from ..core.enums import Side, StrategyId
from ..core.models import FeatureSet, Signal
from ..signal.composite import CompositeResult, CompositeScorer
from .base import Strategy


class Breakout(Strategy):
    id = StrategyId.BREAKOUT

    def evaluate(self, fs: FeatureSet, comp: CompositeResult,
                 scorer: CompositeScorer) -> Optional[Signal]:
        if (fs.obi_5_z is None or fs.ofi_z is None or fs.trade_imbalance_3s is None
                or fs.mid_return_5s is None or fs.mid_return_15s is None):
            return None
        if (comp.warmup or min(comp.tradability, comp.tradability_raw) < self.min_tradability
                or (fs.spread_bps is not None and fs.spread_bps > self.max_entry_spread_bps)):
            return None                                              # 预热/差行情/价差硬门: 不做

        # 做多突破
        if (comp.long_score >= self.strong_composite
                and fs.obi_5_z >= 1.5 and fs.ofi_z >= 1.5 and fs.trade_imbalance_3s >= 0.35
                and fs.mid_return_5s > 0 and fs.mid_return_15s > 0):
            sig = self._mk(fs, scorer, Side.BUY, comp.long_score)
            if sig:
                return sig
        # 做空突破
        if (comp.short_score >= self.strong_composite
                and fs.obi_5_z <= -1.5 and fs.ofi_z <= -1.5 and fs.trade_imbalance_3s <= -0.35
                and fs.mid_return_5s < 0 and fs.mid_return_15s < 0):
            sig = self._mk(fs, scorer, Side.SELL, comp.short_score)
            if sig:
                return sig
        return None

    def _mk(self, fs, scorer, side, score) -> Optional[Signal]:
        sig = scorer.build_signal(fs, self.id, side, score, taker=True)
        if self.is_strong(score, sig):     # 综合分≥90 + 模型概率≥0.62 + edge/cost≥3.0
            return sig
        return None
