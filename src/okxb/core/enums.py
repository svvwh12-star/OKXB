"""领域枚举。集中定义, 避免散落的魔法字符串。"""
from __future__ import annotations

from enum import Enum


class Mode(str, Enum):
    DEMO = "demo"      # OKX 模拟盘 (x-simulated-trading: 1)
    LIVE = "live"      # 实盘


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class PosSide(str, Enum):
    LONG = "long"
    SHORT = "short"
    NET = "net"


class InstType(str, Enum):
    SWAP = "SWAP"
    FUTURES = "FUTURES"
    SPOT = "SPOT"
    # 股票永续的 instType 待调研核验; 占位以便后续替换。
    STOCK_PERP = "SWAP"


class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"                 # 系统默认禁用 (无保护; 受 maxMktSz 限)
    POST_ONLY = "post_only"           # 只做 maker, 会立即成交则被拒 (仅限价)
    FOK = "fok"
    IOC = "ioc"
    OPTIMAL_LIMIT_IOC = "optimal_limit_ioc"  # 永续/期货专用市价式 IOC; 紧急退出/强信号 taker 用此而非 market


class OrderState(str, Enum):
    PENDING = "pending"        # 本地待发
    LIVE = "live"              # 已挂单
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


class StrategyId(str, Enum):
    HFM80 = "hfm80_momentum_maker"
    HFR80 = "hfr80_reversion"
    BASIS = "basis_meanrev"
    BREAKOUT = "breakout_taker"
    RECONCILED = "reconciled"          # 对账重建的孤儿仓 (交易所有/本地无 -> 接管管理)


class RiskAction(str, Enum):
    """RiskEngine 对一个交易意图的裁决。"""
    APPROVE = "approve"
    REJECT = "reject"
    REDUCE_SIZE = "reduce_size"
    REDUCE_ONLY = "reduce_only"   # 只允许减仓
    HALT = "halt"                 # 触发熔断


class SystemState(str, Enum):
    """全局运行状态机 (受 kill switch / 回撤阶梯驱动)。"""
    NORMAL = "normal"
    HIGH_VOL = "high_vol"
    REDUCED = "reduced"           # 连亏/回撤后降仓
    STRONG_ONLY = "strong_only"   # 仅最强信号
    CLOSE_ONLY = "close_only"     # 只允许平仓
    HALTED = "halted"             # 熔断停机


class EventAction(str, Enum):
    """AI 事件模块输出的风控动作 (§11)。"""
    NO_ACTION = "no_action"
    BLOCK_LONG = "block_long"
    BLOCK_SHORT = "block_short"
    REDUCE_ONLY = "reduce_only"
    CLOSE_ALL = "close_all"
