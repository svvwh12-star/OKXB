"""股票永续 Basis 均值回归策略 (用户方案 §9)。

basis = (perp_mid - 标记/指数价)/指数价。当 basis 偏离过大 (z 高/低)、且订单流不反向时,
做反向 (perp 高于指数 -> 做空, 预期回落)。全程 post-only。
注: 需 app 周期注入标记价 (fe.set_marks); 无标记价则不触发。
持仓本应 5–60min, 当前执行器全局时间止损 180s 偏短 —— TODO 按策略设独立持仓时长。
"""
from __future__ import annotations

from typing import Optional

from ..core.enums import Side, StrategyId
from ..core.models import FeatureSet, Signal
from ..risk.engine import is_stock_perp
from ..signal.composite import CompositeResult, CompositeScorer
from .base import Strategy

Z_TRIGGER = 2.2


class BasisMeanRev(Strategy):
    id = StrategyId.BASIS

    def evaluate(self, fs: FeatureSet, comp: CompositeResult,
                 scorer: CompositeScorer) -> Optional[Signal]:
        if not is_stock_perp(fs.inst_id) or fs.basis_z is None:
            return None
        if (comp.warmup or min(comp.tradability, comp.tradability_raw) < self.min_tradability
                or (fs.spread_bps is not None and fs.spread_bps > self.max_entry_spread_bps)):
            return None                                              # 预热/差行情/价差硬门: 不做
        z = fs.basis_z
        ofi = fs.ofi_z if fs.ofi_z is not None else 0.0

        # 做空: perp 显著高于指数 (basis_z 高) 且订单流不继续上行
        if z >= Z_TRIGGER and ofi <= 0.0:
            score = min(95.0, 78.0 + 5.0 * (z - Z_TRIGGER))
            sig = self._mk(fs, scorer, Side.SELL, score)
            if sig:
                return sig
        # 做多: perp 显著低于指数 且订单流不继续下行
        if z <= -Z_TRIGGER and ofi >= 0.0:
            score = min(95.0, 78.0 + 5.0 * (abs(z) - Z_TRIGGER))
            sig = self._mk(fs, scorer, Side.BUY, score)
            if sig:
                return sig
        return None

    def _mk(self, fs, scorer, side, score) -> Optional[Signal]:
        if score < self.min_composite:
            return None
        sig = scorer.build_signal(fs, self.id, side, score)
        # basis 策略要求更高的期望净edge (§9.2: 净edge >= 2.5×往返成本; 真过滤器, 不依赖占位 model_prob)
        if sig.total_cost_pct > 0 and sig.edge_to_cost >= 2.5:
            return sig
        return None
