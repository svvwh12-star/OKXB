"""本地 L2 订单簿 (books 频道, 400 档增量)。

完整性走 seqId/prevSeqId 连续性 (checksum 2026-06-23 起恒为 0, 已废弃):
  - 每条 update 的 prevSeqId 必须等于上一条的 seqId; 否则视为断号 -> 丢弃并重订阅重拉快照。
  - prevSeqId == seqId 表示无变化 (心跳)。
  - seqId 变小 (维护后重置) 视为断号。
参考 docs/RESEARCH_BRIEF.md §0.5 / §4。
"""
from __future__ import annotations

from sortedcontainers import SortedDict  # type: ignore

from ..core.models import BBO, BookLevel, BookSnapshot


class OrderBook:
    """单标的本地订单簿。bids/asks 用 SortedDict(px->sz) 维护有序。"""

    def __init__(self, inst_id: str):
        self.inst_id = inst_id
        self._bids: "SortedDict[float, float]" = SortedDict()   # 价升序; 取最高需 peekitem(-1)
        self._asks: "SortedDict[float, float]" = SortedDict()   # 价升序; 取最低需 peekitem(0)
        self._last_seq: int | None = None
        self.ts: int = 0
        self.ready: bool = False

    def reset(self) -> None:
        self._bids.clear()
        self._asks.clear()
        self._last_seq = None
        self.ready = False

    def _apply_side(self, book: "SortedDict[float, float]", levels: list[list]) -> None:
        for lvl in levels:
            px = float(lvl[0])
            sz = float(lvl[1])
            if sz == 0.0:
                book.pop(px, None)
            else:
                book[px] = sz

    def apply_snapshot(self, data: dict) -> bool:
        self.reset()
        self._apply_side(self._bids, data.get("bids", []))
        self._apply_side(self._asks, data.get("asks", []))
        self._last_seq = int(data.get("seqId", -1))
        self.ts = int(data.get("ts", 0))
        self.ready = True
        return True

    def apply_update(self, data: dict) -> bool:
        """返回 True=已应用; False=断号 (调用方须重订阅+重拉快照)。"""
        seq = int(data.get("seqId", -1))
        prev = int(data.get("prevSeqId", -1))
        if self._last_seq is None:
            return False                          # 未初始化, 需快照
        if prev != self._last_seq:
            if seq == self._last_seq:
                return True                       # 无变化心跳
            return False                          # 断号
        self._apply_side(self._bids, data.get("bids", []))
        self._apply_side(self._asks, data.get("asks", []))
        self._last_seq = seq
        self.ts = int(data.get("ts", self.ts))
        return True

    # ----------------- 读取 -----------------

    def best_bid(self) -> tuple[float, float] | None:
        if not self._bids:
            return None
        px, sz = self._bids.peekitem(-1)
        return px, sz

    def best_ask(self) -> tuple[float, float] | None:
        if not self._asks:
            return None
        px, sz = self._asks.peekitem(0)
        return px, sz

    def bbo(self) -> BBO | None:
        b, a = self.best_bid(), self.best_ask()
        if not b or not a:
            return None
        return BBO(self.inst_id, self.ts, b[0], b[1], a[0], a[1])

    def snapshot(self, levels: int = 5) -> BookSnapshot | None:
        if not self.ready:
            return None
        bids = [BookLevel(px, self._bids[px]) for px in list(self._bids.keys())[::-1][:levels]]
        asks = [BookLevel(px, self._asks[px]) for px in list(self._asks.keys())[:levels]]
        return BookSnapshot(self.inst_id, self.ts, bids, asks)

    def depth_notional(self, side: str, bps: float) -> float:
        """从最优价向内 bps 范围内的名义深度 (USDT 近似 = sum px*sz)。"""
        bbo = self.bbo()
        if not bbo:
            return 0.0
        mid = bbo.mid
        if side == "bid":
            lo = mid * (1 - bps / 1e4)
            return sum(px * sz for px, sz in self._bids.items() if px >= lo)
        hi = mid * (1 + bps / 1e4)
        return sum(px * sz for px, sz in self._asks.items() if px <= hi)
