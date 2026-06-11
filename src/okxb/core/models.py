"""领域模型。

热路径对象 (行情/因子) 用 dataclass(slots=True) 以降低开销;
跨模块传递的"意图/裁决/事件"也用 dataclass, 保持可序列化、可审计。
价格/数量统一用 Decimal 处理交易所精度, 因子用 float。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

from .enums import (
    EventAction,
    OrderState,
    OrderType,
    PosSide,
    RiskAction,
    Side,
    StrategyId,
)

# ----------------------------- 行情 -----------------------------

@dataclass(slots=True)
class BBO:
    """最优买卖 (bbo-tbt)。"""
    inst_id: str
    ts: int                 # 交易所毫秒时间戳
    bid_px: float
    bid_sz: float
    ask_px: float
    ask_sz: float

    @property
    def mid(self) -> float:
        return (self.bid_px + self.ask_px) / 2.0

    @property
    def microprice(self) -> float:
        """size-weighted mid (Stoikov micro-price近似): 买量大→价偏向卖价(上行压力)。
        = (bid_px*ask_sz + ask_px*bid_sz)/(bid_sz+ask_sz)。比裸中间价更少受单边报价抖动污染。"""
        tot = self.bid_sz + self.ask_sz
        return (self.bid_px * self.ask_sz + self.ask_px * self.bid_sz) / tot if tot > 0 else self.mid

    @property
    def spread_bps(self) -> float:
        m = self.mid
        return (self.ask_px - self.bid_px) / m * 1e4 if m else float("inf")


@dataclass(slots=True)
class BookLevel:
    px: float
    sz: float


@dataclass(slots=True)
class BookSnapshot:
    """N 档盘口快照 (books5 / 本地重建的 books)。"""
    inst_id: str
    ts: int
    bids: list[BookLevel]   # 价格降序
    asks: list[BookLevel]   # 价格升序

    def depth(self, side: str, levels: int) -> float:
        book = self.bids if side == "bid" else self.asks
        return sum(l.sz for l in book[:levels])


@dataclass(slots=True)
class Trade:
    """逐笔成交 (trades 频道)。side = 主动方。"""
    inst_id: str
    ts: int
    px: float
    sz: float
    side: Side              # buy=主动买, sell=主动卖


@dataclass(slots=True)
class FundingInfo:
    inst_id: str
    funding_rate: float
    next_funding_time: int  # ms


# ----------------------------- 因子 / 信号 -----------------------------

@dataclass(slots=True)
class FeatureSet:
    """某标的某时刻的全部因子 (§6)。None = 暂不可得。"""
    inst_id: str
    ts: int
    # 微观结构
    obi_5: Optional[float] = None
    obi_5_z: Optional[float] = None
    ofi: Optional[float] = None
    ofi_z: Optional[float] = None
    trade_imbalance_3s: Optional[float] = None
    spread_bps: Optional[float] = None
    depth_5bps: Optional[float] = None
    # 趋势
    mid_return_5s: Optional[float] = None
    mid_return_15s: Optional[float] = None
    mid_return_60s: Optional[float] = None
    # 波动
    realized_vol_60s: Optional[float] = None     # 基于 micro-price (实盘消费用)
    rv_mid: Optional[float] = None               # 基于纯中间价 (录制对比用; Stage B 决定是否切换为主)
    atr_1m: Optional[float] = None
    # 股票永续 basis
    basis_index: Optional[float] = None
    basis_z: Optional[float] = None
    # 资金费
    funding_rate: Optional[float] = None
    seconds_to_funding: Optional[float] = None


@dataclass(slots=True)
class Signal:
    """SignalService 产出的候选信号。"""
    inst_id: str
    ts: int
    strategy: StrategyId
    side: Side
    composite_score: float          # 0-100
    model_prob: float               # 模型胜率概率
    expected_edge_pct: float
    total_cost_pct: float
    sl_pct: float                   # 建议止损 %
    tp_pct: float                   # 建议止盈 %
    features: FeatureSet
    signal_id: str = ""
    taker: bool = False             # True=用 optimal_limit_ioc 主动成交 (突破策略); 否则 post-only maker

    @property
    def edge_to_cost(self) -> float:
        return self.expected_edge_pct / self.total_cost_pct if self.total_cost_pct else 0.0


# ----------------------------- 交易 -----------------------------

@dataclass(slots=True)
class OrderIntent:
    """策略 -> 风控 -> 执行 的交易意图 (尚未下单)。"""
    inst_id: str
    side: Side
    pos_side: PosSide
    order_type: OrderType
    notional_usdt: Decimal          # 名义价值
    px: Optional[Decimal]           # 限价; market 为 None
    reduce_only: bool
    strategy: StrategyId
    signal_id: str
    sl_pct: float
    tp_pct: float
    max_loss_usdt: Decimal
    ttl_ms: int


@dataclass(slots=True)
class Order:
    """已提交订单的本地追踪状态。"""
    client_oid: str
    inst_id: str
    side: Side
    pos_side: PosSide
    order_type: OrderType
    px: Optional[Decimal]
    sz: Decimal                     # 合约张数
    reduce_only: bool
    state: OrderState
    strategy: StrategyId
    signal_id: str
    created_ms: int
    okx_ord_id: Optional[str] = None
    filled_sz: Decimal = Decimal("0")
    avg_px: Optional[Decimal] = None


@dataclass(slots=True)
class Position:
    inst_id: str
    pos_side: PosSide
    size: Decimal                   # 张数 (净/方向)
    avg_px: Decimal
    notional_usdt: Decimal
    upl: Decimal                    # 未实现盈亏
    strategy: Optional[StrategyId] = None
    entry_ms: int = 0
    initial_sl_px: Optional[Decimal] = None
    initial_tp_px: Optional[Decimal] = None
    max_loss_usdt: Optional[Decimal] = None


@dataclass(slots=True)
class RiskDecision:
    """RiskEngine 的裁决结果。"""
    action: RiskAction
    approved_notional_usdt: Decimal
    reason: str
    intent: Optional[OrderIntent] = None


# ----------------------------- 事件 -----------------------------

@dataclass(slots=True)
class MarketEvent:
    """AI 事件模块输出 (§11.2), 结构化、带有效期。不进下单路径。"""
    ticker: str
    event_type: str                 # earnings / SEC_8K / lawsuit / macro / ...
    sentiment: float                # -1.0 ~ +1.0
    confidence: float               # 0 ~ 1
    severity: str                   # low / medium / high
    time_horizon: str
    source_quality: str             # official / tier1_news / social
    action: EventAction
    valid_until_ms: int
    raw_ref: str = ""               # 来源链接/原文摘要
