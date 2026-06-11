"""综合信号评分 (v1, 透明启发式)。

产出方向性 long_score / short_score (0-100) + 各组分。关键原则:
  - OFI/OBI/TradeImbalance 是【同一个订单流因子】, 取均值合成一个方向值, 不叠加放大 (RESEARCH_BRIEF §7)。
  - 缺失的组 (basis/event/funding 在 Phase1 未接) 自动从分母剔除并重归一化, 不拖低分。
  - 质量组 (波动/流动性) 不分方向, 对多空同等加分。

注意: 本评分的权重与阈值是【先验】, 真实有效性必须由 Phase1 前瞻收益 + 样本外回测校准。
model_prob 在 Phase1 是占位 (由 composite 单调映射); v1.5 起由 meta-labeling 模型替代。
所有 *_pct 内部用小数分数 (0.002 = 0.2%)。
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from ..config import Config
from ..core.enums import Side, StrategyId
from ..core.models import FeatureSet, Signal


COMPUTE_S = 0.5      # 扫描周期(秒), 与 app.COMPUTE_S 一致; EMA alpha 据此换算半衰期


def _squash(z: float | None) -> float:
    """z-score -> [-1,1]。"""
    return math.tanh(z / 2.0) if z is not None else 0.0


def _alpha(dt: float, half_life_s: float) -> float:
    """半衰期 -> EMA 系数。half_life<=0 表示不平滑(alpha=1, 直通)。"""
    if half_life_s <= 0:
        return 1.0
    return 1.0 - 0.5 ** (dt / half_life_s)


@dataclass
class CompositeResult:
    long_score: float              # 原始(未平滑), 多+空=100、50中性 - 供录制/校准诚实记录
    short_score: float
    order_flow_dir: float          # [-1,1] 已平滑 (conf/出场共用此去噪视图)
    trend_dir: float               # [-1,1] 已平滑
    quality: float                 # [0,1]
    components: dict = field(default_factory=dict)
    order_flow_dir_raw: float = 0.0   # 未平滑原值
    trend_dir_raw: float = 0.0
    long_score_s: float = 0.0         # 已平滑, 多+空=100 - 入场门槛/面板用此 (稳定可操作)
    short_score_s: float = 0.0
    dir_latch: int = 0                # 双阈值闩锁方向: +1多/-1空/0无
    direction: float = 0.0            # 合成方向(平滑) [-1,1]: 负偏空/正偏多 (与质量解耦)
    direction_raw: float = 0.0        # 合成方向(未平滑)
    tradability: float = 0.0          # 可交易性 [0,1] 已平滑: 数据×流动性×波动 (与方向解耦, 门槛/面板用)
    tradability_raw: float = 0.0      # 可交易性(未平滑) - 录制/重调EMA用
    warmup: bool = False              # 预热中(价差/波动分位历史<30样本): 只记录不开仓


class CompositeScorer:
    def __init__(self, config: Config, meta_model=None):
        self._cfg = config
        self._meta = meta_model        # MetaModel 或 None; 有则替换占位 model_prob
        w = config.section("signal").get("weights", {})
        # 方向组
        self.w_flow = float(w.get("microstructure", 20)) + float(w.get("order_flow", 15))
        self.w_trend = float(w.get("trend", 15))
        # 质量组
        self.w_vol = float(w.get("volatility_regime", 10))
        self.w_liq = float(w.get("liquidity", 10))
        fees = config.section("fees")
        self.maker_fee = float(fees.get("crypto_maker_pct", 0.02)) / 100.0
        self.taker_fee = float(fees.get("crypto_taker_pct", 0.05)) / 100.0
        # 止盈盈亏比 (可由『策略校准』写入); 默认 1.6:1
        self.tp_rr = float(config.section("signal").get("tp_rr", 1.6))
        # 止损至少为往返成本的多少倍 (防止亚成本止损被噪声秒扫)
        self.sl_cost_mult = float(config.section("signal").get("sl_min_cost_mult", 2.5))
        # 期望净edge门槛参数 (取代旧的 tp/cost 恒成立门): 期望幅度 = k × 方向强度 × 持有期波动
        self.edge_move_k = float(config.section("signal").get("edge_move_k", 1.5))
        self.edge_horizon_s = float(config.section("signal").get("edge_horizon_s", 30.0))
        self._regime_rv_lo = float(config.section("signal").get("regime_rv_lo", 5e-5))  # 绝对冻结门(与策略死盘一致), 用于质量一致性
        # ---- 信号去噪 (修复每0.5秒在20-80狂跳): EMA平滑 + 双阈值闩锁 + 波动自适应 ----
        sg = config.section("signal")
        dt = float(COMPUTE_S)
        self.a_flow = _alpha(dt, float(sg.get("ema_flow_half_life_s", 1.0)))
        self.a_trend = _alpha(dt, float(sg.get("ema_trend_half_life_s", 2.0)))
        self.a_score = _alpha(dt, float(sg.get("ema_score_half_life_s", 0.5)))
        self.a_trad = _alpha(dt, float(sg.get("ema_trad_half_life_s", 1.5)))   # 可交易性EMA(去抖)
        self.dir_enter = float(sg.get("dir_hyst_enter", 0.12))
        self.dir_exit = float(sg.get("dir_hyst_exit", 0.04))
        self.rv_hi = float(sg.get("regime_rv_hi", 1.2e-3))
        self.hv_scale = float(sg.get("regime_hv_alpha_scale", 0.6))
        self._ema: dict = {}          # inst -> {'f','t'}  方向分量EMA
        self._dir_ema: dict = {}      # inst -> {'d'}      合成方向再EMA
        self._dir_latch: dict = {}    # inst -> -1/0/+1
        self._trad_ema: dict = {}     # inst -> 可交易性EMA (平滑可做分, 去除每拍价差/深度抖动)
        # 逐标的滚动历史 (用于波动/流动性的"分位"度量, 取代全局魔法常数)
        self._rv_hist: dict = {}
        self._spread_hist: dict = {}
        self._depth_hist: dict = {}

    @staticmethod
    def _ratio(store: dict, inst: str, x: float) -> float | None:
        """当前值 x 相对该标的滚动历史【中位数】的倍数 (x/median); 预热(<30样本)或中位数≈0 返回 None。
        关键: 用"相对中位数倍数"而非"分位排名"——分位排名对幅度不敏感(当前最高值永远=1分位),
        会让可做在价差只动一点点时就乱跳; 倍数对小抖动稳定, 只在真的变宽/变薄数倍时才反应。"""
        d = store.get(inst)
        if d is None:
            d = deque(maxlen=400)
            store[inst] = d
        r = None
        if len(d) >= 30:
            srt = sorted(d)
            n = len(srt)
            med = srt[n // 2] if n % 2 else (srt[n // 2 - 1] + srt[n // 2]) / 2.0
            r = (x / med) if med > 1e-12 else None
        d.append(x)
        return r

    @staticmethod
    def _vol_band(ratio: float) -> float:
        """波动/中位数: 仅【近乎死盘】(远低于常态)才降; 高波动不罚
        (对顺势策略高波动是机会, 其风险由 期望edge门 + 价差门 管理, 不该在此重复扣)。"""
        if ratio < 0.2:
            return 0.0
        if ratio < 0.5:
            return (ratio - 0.2) / 0.3
        return 1.0

    @staticmethod
    def _gate_low_good(ratio: float) -> float:
        """价差/中位数(低=好): <=1.5倍中位数都算正常=1; 仅明显变宽(>1.5倍)才降, >=3倍到0.3。"""
        if ratio <= 1.5:
            return 1.0
        if ratio <= 3.0:
            return 1.0 - 0.7 * (ratio - 1.5) / 1.5
        return 0.3

    @staticmethod
    def _gate_high_good(ratio: float) -> float:
        """深度/中位数(高=好): >=0.6倍中位数都算正常=1; 仅明显变薄(<0.6倍)才降, <=0.2倍到0.3。"""
        if ratio >= 0.6:
            return 1.0
        if ratio >= 0.2:
            return 0.3 + 0.7 * (ratio - 0.2) / 0.4
        return 0.3

    def _is_warmup(self, inst: str) -> bool:
        """预热判定: 价差/波动滚动历史 < 30 样本 -> 中位数不稳, 策略应只记录不开仓 (防冷启动盲开)。"""
        sp = self._spread_hist.get(inst)
        rvh = self._rv_hist.get(inst)
        return (sp is None or len(sp) < 30) or (rvh is None or len(rvh) < 30)

    def _tradability(self, fs) -> float:
        """可交易性 [0,1]: 正常行情≈1, 只在【价差明显变宽/深度明显变薄/近乎死盘】时降。
        用"相对该标的中位数的倍数"度量(非分位排名), 对小抖动不敏感; 外层再 EMA 去抖。"""
        q = 1.0
        if fs.spread_bps is not None:
            r = self._ratio(self._spread_hist, fs.inst_id, fs.spread_bps)
            if r is not None:
                q *= self._gate_low_good(r)
            else:                                   # 预热: 仅在价差绝对偏大时才轻罚
                q *= 1.0 - min(0.6, max(0.0, (fs.spread_bps - 10.0) / 40.0))
        if fs.depth_5bps is not None and fs.depth_5bps > 0:
            r = self._ratio(self._depth_hist, fs.inst_id, fs.depth_5bps)
            if r is not None:
                q *= self._gate_high_good(r)
        rv = fs.realized_vol_60s or 0.0
        if rv > 0:
            r = self._ratio(self._rv_hist, fs.inst_id, rv)
            if r is not None:
                q *= self._vol_band(r)
            if rv < self._regime_rv_lo:        # 绝对近乎冻结 -> 质量也归0, 与"死盘"一致(消除"质量100却死盘")
                q = 0.0
        return max(0.0, min(1.0, q))

    def _ema_step(self, inst: str, raw_flow: float, raw_trend: float, scale: float = 1.0):
        st = self._ema.setdefault(inst, {"f": None, "t": None})
        af, at = self.a_flow * scale, self.a_trend * scale
        st["f"] = raw_flow if (af <= 0.0 or st["f"] is None) else st["f"] + af * (raw_flow - st["f"])
        st["t"] = raw_trend if (at <= 0.0 or st["t"] is None) else st["t"] + at * (raw_trend - st["t"])
        return st["f"], st["t"]

    def _latch_step(self, inst: str, d: float) -> int:
        """双阈值(Schmitt)闩锁: 超过 dir_enter 才认方向, 跌破 dir_exit 才松手, 中间死区不翻。"""
        cur = self._dir_latch.get(inst, 0)
        if cur == 0:
            cur = 1 if d >= self.dir_enter else (-1 if d <= -self.dir_enter else 0)
        elif cur > 0:
            cur = -1 if d <= -self.dir_enter else (0 if d < self.dir_exit else 1)
        else:
            cur = 1 if d >= self.dir_enter else (0 if d > -self.dir_exit else -1)
        self._dir_latch[inst] = cur
        return cur

    # ----------------- 评分 -----------------

    def score(self, fs: FeatureSet) -> CompositeResult:
        """专业版 (Stage A'): 方向 与 可交易性 解耦, 评分对称(多+空=100)。
        方向 = 加权(订单流, 趋势) ∈[-1,1]; 多分 = 50×(1+方向) ∈[0,100], 空分 = 100−多分。
        50=中性、>50偏多、<50偏空。可交易性[0,1]独立输出, 由策略单独门控(不再乘进分数,
        否则正常行情也会把分数整体压低到个位数)。"""
        # --- 订单流方向 (raw): OBI/OFI(经典,已深度归一) z 压缩 + 成交失衡, 取均值(同一压力因子) ---
        flow_parts = [p for p in (_squash(fs.obi_5_z), _squash(fs.ofi_z),
                                  (0.7 * fs.trade_imbalance_3s) if fs.trade_imbalance_3s is not None else None)
                      if p is not None]
        flow_raw = sum(flow_parts) / len(flow_parts) if flow_parts else 0.0

        # --- 趋势方向 (raw): 加权中间价收益 / 该周期波动σ (波动归一化, 取代魔法×4000) ---
        rv = fs.realized_vol_60s or 0.0
        # 每步σ估计 = max(已实现波动, 半价差噪声地板, 绝对地板): 防低波时单个tick跳动被放大成"强趋势"
        spread_frac = (fs.spread_bps / 1e4) if fs.spread_bps is not None else 0.0
        sig_step = max(rv, 0.5 * spread_frac, 1.0e-4)
        trend_parts = []
        for r, steps in ((fs.mid_return_5s, 10), (fs.mid_return_15s, 30)):
            if r is not None:
                sigma_h = sig_step * math.sqrt(steps)          # h步收益的σ ≈ 每步σ×√步数
                trend_parts.append(math.tanh(r / sigma_h) if sigma_h > 1e-12 else 0.0)
        trend_raw = sum(trend_parts) / len(trend_parts) if trend_parts else 0.0

        # --- 合成方向 (raw) ∈[-1,1] ---
        wdir = (self.w_flow + self.w_trend) or 1.0
        dir_raw = max(-1.0, min(1.0, (self.w_flow * flow_raw + self.w_trend * trend_raw) / wdir))

        # --- 去噪: 方向分量EMA -> 合成 -> 再EMA -> 双阈值闩锁 ---
        scale = self.hv_scale if rv > self.rv_hi else 1.0
        flow_s, trend_s = self._ema_step(fs.inst_id, flow_raw, trend_raw, scale)
        dir_s = max(-1.0, min(1.0, (self.w_flow * flow_s + self.w_trend * trend_s) / wdir))
        de = self._dir_ema.setdefault(fs.inst_id, {"d": None})
        de["d"] = dir_s if (self.a_score <= 0.0 or de["d"] is None) else de["d"] + self.a_score * (dir_s - de["d"])
        dir_smooth = de["d"]
        latch = self._latch_step(fs.inst_id, dir_smooth)

        # --- 可交易性 (与方向解耦, 独立门槛) + EMA去抖 + 预热判定 (预热中只记录不开仓) ---
        warm = self._is_warmup(fs.inst_id)
        trad_raw = self._tradability(fs)
        prev_t = self._trad_ema.get(fs.inst_id)
        trad = trad_raw if (self.a_trad <= 0.0 or prev_t is None) else prev_t + self.a_trad * (trad_raw - prev_t)
        self._trad_ema[fs.inst_id] = trad

        # 对称概率式: 多+空恒=100, 50中性 (修复"平盘=0 / 个位数"). 方向越偏多, 多分越高于50。
        # 可交易性不再相乘(那会把分数压低), 改为独立输出 trad -> 策略做"分数门槛 ∧ 可交易性门槛"双判定。
        long_raw = 50.0 * (1.0 + dir_raw)
        short_raw = 100.0 - long_raw
        long_sm = 50.0 * (1.0 + dir_smooth)
        short_sm = 100.0 - long_sm

        return CompositeResult(
            long_score=round(long_raw, 1),           # 原始(录制/校准诚实记录)
            short_score=round(short_raw, 1),
            order_flow_dir=round(flow_s, 3),         # 平滑流向 (conf/出场共用)
            trend_dir=round(trend_s, 3),             # 平滑趋势
            quality=round(trad, 3),
            components={
                "obi_5_z": fs.obi_5_z, "ofi_z": fs.ofi_z,
                "trade_imb_3s": fs.trade_imbalance_3s,
                "ret_5s": fs.mid_return_5s, "ret_15s": fs.mid_return_15s,
                "spread_bps": fs.spread_bps, "rvol_60s": fs.realized_vol_60s,
            },
            order_flow_dir_raw=round(flow_raw, 3),
            trend_dir_raw=round(trend_raw, 3),
            long_score_s=round(long_sm, 1),          # 平滑(入场门槛/面板用)
            short_score_s=round(short_sm, 1),
            dir_latch=latch,
            direction=round(dir_smooth, 3),
            direction_raw=round(dir_raw, 3),
            tradability=round(trad, 3),
            tradability_raw=round(trad_raw, 3),
            warmup=warm,
        )

    # ----------------- 止盈止损 / 成本 / edge -----------------

    def sl_tp(self, fs: FeatureSet) -> tuple[float, float]:
        """返回 (sl_pct, tp_pct) 小数分数。v1 由 ATR 近似 + 价差推导, 需回测校准。"""
        atr = fs.atr_1m or 0.002
        spread = (fs.spread_bps or 5.0) / 1e4
        total_cost = self.total_cost_pct(fs)
        sl = max(1.2 * atr, 3.0 * spread, 0.0010,
                 self.sl_cost_mult * total_cost)          # 下限含"成本感知"(不低于成本的若干倍)
        sl = min(sl, 0.012)                               # 上限 1.2%
        tp = max(self.tp_rr * sl, 3.0 * total_cost)       # tp_rr:1 盈亏比(可校准), 使净edge稳过门槛
        return sl, tp

    def total_cost_pct(self, fs: FeatureSet, taker: bool = False) -> float:
        entry = self.taker_fee if taker else self.maker_fee
        exit_ = self.taker_fee                            # 退出可能吃单, 保守按 taker
        spread_cost = (fs.spread_bps or 5.0) / 1e4 * 0.5  # 半价差
        slippage = 0.0003                                 # 估计, 后续由实测替换
        return entry + exit_ + spread_cost + slippage

    def expected_edge(self, composite: float, fs: FeatureSet, cost: float) -> tuple[float, float]:
        """期望净 edge (暂用启发式, 非拟合): 期望幅度 = k × 方向强度 × 持有期波动; 净edge = 期望幅度 − 成本。
        取代旧的"占位 model_prob edge"与"tp/cost≥1.2 恒成立门"(因 tp=max(rr·sl, 3·cost) 必有 tp/cost≥3, 无过滤力)。
        Stage B 将用 拟合的 E[扣费后净收益|因子] 替换 k。返回 (期望幅度, 净edge) 小数分数。"""
        dir_strength = abs(composite - 50.0) / 50.0          # 由对称分还原方向强度 ∈[0,1]
        rv = fs.realized_vol_60s or 0.0
        h_steps = max(1.0, self.edge_horizon_s / COMPUTE_S)
        sigma_h = rv * math.sqrt(h_steps)                    # 持有期波动 ≈ 每步σ×√步数
        exp_move = self.edge_move_k * dir_strength * sigma_h
        return exp_move, exp_move - cost

    def build_signal(self, fs: FeatureSet, strategy: StrategyId,
                     side: Side, composite: float, taker: bool = False) -> Signal:
        sl, tp = self.sl_tp(fs)
        cost = self.total_cost_pct(fs, taker=taker)
        # 真实(暂用)期望净edge: 方向强度×持有期波动 估期望幅度, 封顶到自身止盈(防极端波动把edge吹爆), 再扣成本
        exp_move, _ = self.expected_edge(composite, fs, cost)
        exp_move = min(exp_move, tp)
        net_edge = exp_move - cost
        # model_prob 仅保留为展示占位, 不参与任何决策 (入场/出场/仓位均不依赖)
        model_prob = self._meta.predict_prob(fs, composite) if self._meta is not None else None
        if model_prob is None:
            model_prob = 0.5 + 0.2 * max(0.0, (composite - 50.0) / 50.0)
        return Signal(
            inst_id=fs.inst_id, ts=fs.ts, strategy=strategy, side=side,
            composite_score=composite, model_prob=round(model_prob, 3),
            expected_edge_pct=round(net_edge, 6), total_cost_pct=round(cost, 6),
            sl_pct=round(sl, 6), tp_pct=round(tp, 6), features=fs,
            signal_id=f"{fs.inst_id}-{fs.ts}-{side.value}", taker=taker,
        )
