"""行情 WS 网关 (公共频道, 只读, 无需登录)。

订阅 bbo-tbt(顶档) + books(400档增量) + trades(逐笔流向), 维护:
  - 每标的本地订单簿 (seqId 完整性, 断号自动重订阅+重拉快照)
  - 最新 BBO
  - 近 N 笔成交 (供 trade imbalance)
并跟踪数据老化 (data_age_ms) 与断线自动重连。

用原生 websockets 而非 SDK: 需要对 seqId 重组、app 级 ping/pong、重连重订阅做精确控制。
demo 用 wspap.okx.com:8443, live 用 ws.okx.com:8443 (见 config.exchange)。
"""
from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from typing import Awaitable, Callable, Optional

import websockets

from ..config import Config, Secrets
from ..core.enums import Side
from ..core.models import BBO, Trade
from .orderbook import OrderBook

UpdateCallback = Callable[[str], Awaitable[None]]  # 收到某标的更新时回调 (传 inst_id)


class MarketDataGateway:
    def __init__(self, config: Config, secrets: Secrets, inst_ids: list[str],
                 on_update: Optional[UpdateCallback] = None,
                 trades_maxlen: int = 500):
        self._cfg = config
        self._inst_ids = inst_ids
        self._on_update = on_update
        ch = config.section("data").get("channels", {})
        self._ch_trades = ch.get("trades", "trades")
        # 扫描频道: 默认 books5(5档快照,轻量,可扩展到几十标的) + trades; BBO 由盘口派生。
        self._channels = list(config.get("data.scan_channels", ["books5", "trades"])
                              or ["books5", "trades"])
        self._book_channels = {"books", "books5"}
        self._ch_book = next((c for c in self._channels if c in self._book_channels), "books5")
        self._keepalive = float(config.get("exchange.ws_limits.keepalive_seconds", 20))
        # 行情源: 默认始终用实盘公共WS (全标的/低延迟); 公共数据免鉴权, 与下单模式解耦。
        feed = str(config.get("data.market_data_feed", "live")).lower()
        if feed == "auto":
            key = "demo" if secrets.is_demo else "live"
        else:
            key = "demo" if feed == "demo" else "live"
        self._url = config.get(f"exchange.ws_public.{key}",
                               "wss://ws.okx.com:8443/ws/v5/public")

        self.books: dict[str, OrderBook] = {i: OrderBook(i) for i in inst_ids}
        self.bbos: dict[str, BBO] = {}
        self.trades: dict[str, deque[Trade]] = {i: deque(maxlen=trades_maxlen) for i in inst_ids}
        # 逐事件 OFI 累计器 (修复"仅取0.5s首尾两点快照差"): 每次盘口更新都算一次增量并累加,
        # compute 拍时一次取走 -> OFI 反映该拍内的【全部】最优价变动 (经典 Cont 要求逐事件累计)。
        self._ofi: dict[str, dict] = {}
        self._ofi_from_bbo = "bbo-tbt" in self._channels   # 有bbo-tbt(~10ms)则用它驱动, 否则用books5(~100ms)
        self._last_msg_ms: int = 0
        self._last_data_ms: dict[str, int] = {}            # 逐标的最近一次"真实行情数据"到达时戳 (H-3)
        self._running = False
        self._ws = None

    # ----------------- 公共读取 -----------------

    def get_bbo(self, inst_id: str) -> Optional[BBO]:
        return self.bbos.get(inst_id)

    def get_book(self, inst_id: str) -> OrderBook:
        return self.books[inst_id]

    def recent_trades(self, inst_id: str, within_ms: int) -> list[Trade]:
        now = self._last_msg_ms or int(time.time() * 1000)
        return [t for t in self.trades[inst_id] if now - t.ts <= within_ms]

    def data_age_ms(self) -> int:
        if not self._last_msg_ms:
            return 10 ** 9
        return int(time.time() * 1000) - self._last_msg_ms

    def symbol_age_ms(self, inst_id: str) -> int:
        """单标的行情老化 (H-3): 某频道掉线/单标的冻结时, 全局 data_age 可能仍新, 须逐标的判。"""
        t = self._last_data_ms.get(inst_id)
        if not t:
            return 10 ** 9
        return int(time.time() * 1000) - t

    def take_ofi(self, inst_id: str) -> Optional[float]:
        """取走自上次以来【逐事件累计】的 OFI(各次增量已按 L1 深度归一化后求和)并清零; 无新事件返回 None。
        使 OFI 反映 0.5s 计算拍内的全部最优价变动, 而非仅首尾两点之差。"""
        st = self._ofi.get(inst_id)
        if not st or st["n"] == 0:
            return None
        val = st["sum"]
        st["sum"] = 0.0
        st["n"] = 0
        return val

    def _accum_ofi(self, inst_id: str, bbo: Optional[BBO]) -> None:
        """每次盘口更新累加一次经典 L1 OFI 增量 (Cont-Kukanov-Stoikov, 已深度归一化)。"""
        from ..features.microstructure import ofi_tick     # 局部导入, 避免包初始化期循环依赖
        if bbo is None:
            return
        st = self._ofi.get(inst_id)
        if st is None:
            st = {"sum": 0.0, "n": 0, "prev": None}
            self._ofi[inst_id] = st
        prev = st["prev"]
        if prev is not None:
            if bbo.ts and prev.ts and bbo.ts < prev.ts:   # 乱序到达: 丢弃本次增量, 保留较新的 prev
                return
            e = ofi_tick(prev, bbo)
            if e is not None:
                st["sum"] += e
                st["n"] += 1
        st["prev"] = bbo

    # ----------------- 运行 -----------------

    async def run(self) -> None:
        self._running = True
        backoff = 1.0
        while self._running:
            try:
                async with websockets.connect(self._url, ping_interval=None,
                                              max_size=2 ** 22) as ws:
                    self._ws = ws
                    await self._subscribe(ws)
                    print(f"[gateway] 已连接 {self._url}; 订阅 {len(self._inst_ids)} 标的 x "
                          f"{len(self._channels)} 频道 ({'+'.join(self._channels)})")
                    backoff = 1.0
                    ping_task = asyncio.create_task(self._ping_loop(ws))
                    try:
                        async for raw in ws:
                            await self._handle(raw)
                    finally:
                        ping_task.cancel()
            except Exception as e:                  # 断线/异常 -> 重连
                if not self._running:
                    break
                print(f"[gateway] 连接异常: {e!r}; {backoff:.0f}s 后重连")
                for ob in self.books.values():
                    ob.reset()
                self._ofi.clear()              # 重连后清OFI累计器, 避免跨断点的虚假大增量
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def stop(self) -> None:
        self._running = False

    async def _subscribe(self, ws) -> None:
        args = [{"channel": chn, "instId": i}
                for i in self._inst_ids for chn in self._channels]
        # 分批发送, 避免单条订阅消息超过 64KB 上限 (支持几十~上百标的)
        chunk = 120
        for k in range(0, len(args), chunk):
            await ws.send(json.dumps({"op": "subscribe", "args": args[k:k + chunk]}))

    async def _resubscribe_book(self, ws, inst_id: str) -> None:
        """断号后: 退订再订阅 books, 触发新的 snapshot。"""
        self.books[inst_id].reset()
        self._ofi.pop(inst_id, None)           # 断号: 清该标的OFI累计, 避免跨gap虚假增量
        arg = [{"channel": self._ch_book, "instId": inst_id}]
        await ws.send(json.dumps({"op": "unsubscribe", "args": arg}))
        await ws.send(json.dumps({"op": "subscribe", "args": arg}))

    async def _ping_loop(self, ws) -> None:
        while True:
            await asyncio.sleep(self._keepalive)
            try:
                await ws.send("ping")
            except Exception:
                return

    async def _handle(self, raw) -> None:
        # H-3: 心跳 pong 与订阅/错误回执【不】刷新行情老化时戳, 否则行情冻结时 staleness 闸门形同虚设
        if raw == "pong":
            return
        try:
            msg = json.loads(raw)
        except (ValueError, TypeError):
            return
        if "event" in msg:                          # 订阅/错误回执 (非行情数据)
            if msg.get("event") == "error":
                print(f"[gateway] 订阅错误: {msg}")
            return
        arg = msg.get("arg", {})
        channel = arg.get("channel")
        inst_id = arg.get("instId")
        data = msg.get("data")
        if not channel or not inst_id or inst_id not in self.books or not data:
            return
        now = int(time.time() * 1000)               # 仅真实行情数据到达才刷新老化时戳
        self._last_msg_ms = now
        self._last_data_ms[inst_id] = now

        if channel in self._book_channels:
            await self._on_book(inst_id, msg.get("action"), data[0],
                                snapshot_only=(channel == "books5"))
        elif channel == self._ch_trades:
            for d in data:
                self._on_trade(inst_id, d)
        elif channel == "bbo-tbt":
            self._on_bbo(inst_id, data[0])

        if self._on_update:
            await self._on_update(inst_id)

    def _on_bbo(self, inst_id: str, d: dict) -> None:
        bids = d.get("bids") or []
        asks = d.get("asks") or []
        if not bids or not asks:
            return
        bbo = BBO(inst_id, int(d.get("ts", 0)),
                  float(bids[0][0]), float(bids[0][1]),
                  float(asks[0][0]), float(asks[0][1]))
        self.bbos[inst_id] = bbo
        if self._ofi_from_bbo:
            self._accum_ofi(inst_id, bbo)          # 逐事件OFI: bbo-tbt 驱动(~10ms)

    async def _on_book(self, inst_id: str, action: Optional[str], d: dict,
                       snapshot_only: bool = False) -> None:
        ob = self.books[inst_id]
        if snapshot_only or action == "snapshot":
            ob.apply_snapshot(d)               # books5: 每条都是全量5档快照
        else:
            if not ob.apply_update(d) and self._ws is not None:
                # books 增量断号: 退订+重订阅触发新 snapshot
                await self._resubscribe_book(self._ws, inst_id)
                return
        if not self._ofi_from_bbo:
            self._accum_ofi(inst_id, ob.bbo())     # 逐事件OFI: 每次盘口快照(~100ms)累计一次

    def _on_trade(self, inst_id: str, d: dict) -> None:
        side = Side.BUY if d.get("side") == "buy" else Side.SELL
        self.trades[inst_id].append(
            Trade(inst_id, int(d.get("ts", 0)), float(d["px"]), float(d["sz"]), side)
        )
