"""HFM-80 高频 maker 顺势主策略 (深度优化版, 2026-06)。

旧版几乎不触发: 综合分>=80 近乎不可达, 还叠加 OBI/OFI/TradeImb/收益/model_prob 多个"且", 单拍判定。
新版入场 = 综合分>=门槛(主) + 1个确认(流向/趋势一致 conf>=confirm_min)
        + 连续 persist_ticks 拍持续(去抖) + 扣费净 edge>=门槛。
去掉对各原始因子的独立硬门槛(冗余, 综合分已含)与占位 model_prob 硬门槛; 多空对称。
冷却由编排器统一管理(对所有策略生效)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..core.enums import Side, StrategyId
from ..core.models import FeatureSet, Signal
from ..signal.composite import CompositeResult, CompositeScorer
from .base import Strategy


@dataclass
class _State:
    side: Optional[Side] = None
    run_len: int = 0
    misses: int = 0          # 连续坏拍计数 (容错, 不立即清零)


class HFM80(Strategy):
    id = StrategyId.HFM80

    def __init__(self, config):
        super().__init__(config)
        self._st: dict[str, _State] = {}

    def _qualifies(self, comp: CompositeResult, side: Side) -> bool:
        """对称尺度(多+空=100, 50中性): 方向分>=门槛 ∧ 可交易性>=门槛 ∧ 确认 ∧ 闩锁方向一致。
        可交易性已与方向分解耦, 必须独立判定 (否则会在流动性差/死盘里照样开仓)。"""
        d = 1 if side == Side.BUY else -1
        score = comp.long_score_s if d > 0 else comp.short_score_s
        # 可交易性入场用"快下慢上": 取 raw 与 EMA 的较小者 -> 流动性骤坏立即反应(EMA只用于面板显示, 不拖慢刹车)
        trad_gate = min(comp.tradability, comp.tradability_raw)
        return (score >= self.min_composite
                and not comp.warmup                         # 预热中(分位历史不足)不开仓
                and trad_gate >= self.min_tradability
                and self.conf(comp, d) >= self.confirm_min
                and comp.dir_latch == d)          # 闩锁方向须一致(死区内 latch=0 -> 不做)

    def evaluate(self, fs: FeatureSet, comp: CompositeResult,
                 scorer: CompositeScorer) -> Optional[Signal]:
        # 死盘抑制: 波动过低没有 edge, 不做
        rv = fs.realized_vol_60s or 0.0
        if rv < self.regime_rv_lo:
            self._st.pop(fs.inst_id, None)
            return None
        if fs.spread_bps is not None and fs.spread_bps > self.max_entry_spread_bps:  # 绝对价差硬门
            self._st.pop(fs.inst_id, None)
            return None
        eff_persist = self.persist_ticks + (self.regime_persist_bonus if rv > self.regime_rv_hi else 0)

        st = self._st.setdefault(fs.inst_id, _State())
        cand = Side.BUY if comp.long_score_s >= comp.short_score_s else Side.SELL
        if self._qualifies(comp, cand):
            if st.side == cand:
                st.run_len += 1
                st.misses = 0
            else:                                  # 起始/换边 -> 重置计数
                st.side, st.run_len, st.misses = cand, 1, 0
        else:
            # 容错: 单拍不达标先不清零(噪声坏点), 超过容忍才重置
            if st.side is not None and st.misses < self.persist_miss_grace:
                st.misses += 1
                return None
            st.side, st.run_len, st.misses = None, 0, 0
            return None
        if st.run_len < eff_persist:                # 持续未达, 等待 (去抖)
            return None
        score = comp.long_score if cand == Side.BUY else comp.short_score   # 录制用原始分
        sig = scorer.build_signal(fs, self.id, cand, score)
        # 期望净edge门槛(真过滤器): 期望幅度(方向强度×持有期波动)−成本, 再/成本 >= 门槛;
        # 取代旧的 tp/cost 恒成立门(因 tp=max(rr·sl,3·cost) 必有 tp/cost≥3, 永远通过)。
        if sig.total_cost_pct <= 0 or sig.edge_to_cost < self.min_edge_cost:
            return None
        st.run_len, st.misses = 0, 0                 # 消费触发
        return sig
