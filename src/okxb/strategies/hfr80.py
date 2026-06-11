"""HFR-80 高频反转 / 流动性回补策略 (副策略, 用户方案 §8)。

价格短时间快速偏离 (按已实现波动衡量), 且订单流开始衰竭/反向吸收 -> 做均值回归。
全程 post-only maker。频率应低于顺势策略。
"""
from __future__ import annotations

import math
from typing import Optional

from ..core.enums import Side, StrategyId
from ..core.models import FeatureSet, Signal
from ..signal.composite import CompositeResult, CompositeScorer
from .base import Strategy

EXT_TRIGGER = 2.0   # 60s 移动达到 ~2σ 视为过度偏离


class HFR80(Strategy):
    id = StrategyId.HFR80

    def evaluate(self, fs: FeatureSet, comp: CompositeResult,
                 scorer: CompositeScorer) -> Optional[Signal]:
        if (fs.realized_vol_60s is None or fs.mid_return_60s is None
                or fs.ofi_z is None or fs.trade_imbalance_3s is None):
            return None
        if (comp.warmup or min(comp.tradability, comp.tradability_raw) < self.min_tradability
                or (fs.spread_bps is not None and fs.spread_bps > self.max_entry_spread_bps)):
            return None                                              # 预热/差行情/价差硬门: 不做
        rv = fs.realized_vol_60s
        if rv <= 1e-9:
            return None
        # 60s 移动相对 60s 波动的标准化幅度 (≈ z)
        ext = fs.mid_return_60s / (rv * math.sqrt(120.0) + 1e-12)

        # 做空反转: 急涨 + 订单流衰竭/转弱
        if ext >= EXT_TRIGGER and fs.ofi_z <= -0.5 and fs.trade_imbalance_3s <= 0.0:
            score = min(96.0, 76.0 + 6.0 * (ext - EXT_TRIGGER) + 6.0 * min(1.0, abs(fs.ofi_z)))
            sig = self._mk(fs, scorer, Side.SELL, score)
            if sig:
                return sig
        # 做多反转: 急跌 + 买盘衰竭/转强
        if ext <= -EXT_TRIGGER and fs.ofi_z >= 0.5 and fs.trade_imbalance_3s >= 0.0:
            score = min(96.0, 76.0 + 6.0 * (abs(ext) - EXT_TRIGGER) + 6.0 * min(1.0, abs(fs.ofi_z)))
            sig = self._mk(fs, scorer, Side.BUY, score)
            if sig:
                return sig
        return None

    def _mk(self, fs, scorer, side, score) -> Optional[Signal]:
        if score < self.min_composite:
            return None
        sig = scorer.build_signal(fs, self.id, side, score)
        # 期望净edge门槛 (真过滤器, 不依赖占位 model_prob)
        if sig.total_cost_pct > 0 and sig.edge_to_cost >= self.min_edge_cost:
            return sig
        return None
