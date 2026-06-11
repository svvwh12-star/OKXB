"""订单与受管持仓的内存追踪。

成交后为每个持仓建立内部风控状态 (entry/SL/TP/time-stop/max_loss), 所有平仓单
强制 reduce-only, 防止误反手 (用户方案 §14.3)。
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from ..core.enums import Side, StrategyId


@dataclass
class ManagedPosition:
    inst_id: str
    side: Side                  # 持仓方向 (BUY=多)
    contracts: Decimal
    entry_px: Decimal
    sl_px: Decimal
    tp_px: Decimal
    strategy: StrategyId
    signal_id: str
    entry_ms: int
    max_loss_usdt: Decimal
    tp_order_oid: Optional[str] = None
    closing: bool = False
    hwm: float = 0.0           # 最高/最低水位 (移动止盈用)
    rev_run: int = 0           # 反转/衰减 连续拍计数


class OrderManager:
    def __init__(self) -> None:
        self.positions: dict[str, ManagedPosition] = {}
        self.pending_entries: dict[str, dict] = {}   # client_oid -> meta

    def add_position(self, p: ManagedPosition) -> None:
        self.positions[p.inst_id] = p

    def remove_position(self, inst_id: str) -> Optional[ManagedPosition]:
        return self.positions.pop(inst_id, None)

    def get(self, inst_id: str) -> Optional[ManagedPosition]:
        return self.positions.get(inst_id)

    def has(self, inst_id: str) -> bool:
        return inst_id in self.positions

    def all(self) -> list[ManagedPosition]:
        return list(self.positions.values())
