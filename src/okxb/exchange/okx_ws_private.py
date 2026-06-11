"""OKX V5 私有 WebSocket: 登录 + 订单/持仓实时流 + WS 下单/撤单/改单。

优势: 一条常驻已登录连接, 省去每次 REST 的 TLS 握手, 下单延迟更低; 且实时拿到成交/持仓。
登录签名: HMAC-SHA256( ts + 'GET' + '/users/self/verify' ), ts 为 UNIX 秒(非毫秒), base64。
demo 用 wspap.okx.com (跟随交易模式, 与公共行情走实盘 WS 解耦)。
place/cancel/amend 与 REST 客户端同签名, 返回同结构, 便于执行器透明切换; 失败时执行器回落 REST。
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import time
from typing import Optional

import websockets

from ..config import Config, Secrets
from .okx_rest import OkxError


class OkxPrivateWS:
    def __init__(self, secrets: Secrets, config: Config, alert_fn=None):
        self._key = secrets.okx_api_key
        self._secret = secrets.okx_secret_key
        self._pass = secrets.okx_passphrase
        self._alert = alert_fn or (lambda m: None)
        key = "demo" if secrets.is_demo else "live"
        self._url = config.get(f"exchange.ws_private.{key}",
                               "wss://ws.okx.com:8443/ws/v5/private")
        self._keepalive = float(config.get("exchange.ws_limits.keepalive_seconds", 20))
        self.logged_in = False
        self.orders: dict[str, dict] = {}      # clOrdId/ordId -> 最新订单
        self.positions: dict[str, dict] = {}   # instId -> 持仓
        self._ws = None
        self._pending: dict[str, asyncio.Future] = {}
        self._req = 0
        self._running = False

    # ----------------- 连接 / 登录 -----------------

    async def run(self) -> None:
        self._running = True
        backoff = 1.0
        while self._running:
            try:
                async with websockets.connect(self._url, ping_interval=None,
                                              max_size=2 ** 22) as ws:
                    self._ws = ws
                    self.logged_in = False
                    await self._send_login(ws)
                    backoff = 1.0
                    ping = asyncio.create_task(self._ping_loop(ws))
                    try:
                        async for raw in ws:
                            await self._handle(raw, ws)
                    finally:
                        ping.cancel()
            except Exception as e:
                if not self._running:
                    break
                self.logged_in = False
                print(f"[priv_ws] 连接异常: {e!r}; {backoff:.0f}s 后重连")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    def stop(self) -> None:
        self._running = False

    def _sign_login(self, ts: str) -> str:
        mac = hmac.new(self._secret.encode(),
                       f"{ts}GET/users/self/verify".encode(), hashlib.sha256)
        return base64.b64encode(mac.digest()).decode()

    async def _send_login(self, ws) -> None:
        ts = str(int(time.time()))               # UNIX 秒
        await ws.send(json.dumps({"op": "login", "args": [{
            "apiKey": self._key, "passphrase": self._pass,
            "timestamp": ts, "sign": self._sign_login(ts)}]}))

    async def _subscribe(self, ws) -> None:
        await ws.send(json.dumps({"op": "subscribe", "args": [
            {"channel": "orders", "instType": "SWAP"},
            {"channel": "positions", "instType": "SWAP"}]}))

    async def _ping_loop(self, ws) -> None:
        while True:
            await asyncio.sleep(self._keepalive)
            try:
                await ws.send("ping")
            except Exception:
                return

    # ----------------- 消息处理 -----------------

    async def _handle(self, raw, ws) -> None:
        if raw == "pong":
            return
        try:
            msg = json.loads(raw)
        except (ValueError, TypeError):
            return
        ev = msg.get("event")
        if ev == "login":
            if msg.get("code") == "0":
                self.logged_in = True
                await self._subscribe(ws)
                self._alert("🔐 私有WS已登录 (实时订单/持仓; WS下单已启用)")
            else:
                print(f"[priv_ws] 登录失败: {msg}")
            return
        if ev == "error":
            print(f"[priv_ws] 错误: {msg}")
            return
        if ev in ("subscribe", "unsubscribe"):
            return
        # 下单/撤单/改单 ack (按 id 关联)
        rid = msg.get("id")
        if rid and rid in self._pending:
            fut = self._pending.pop(rid)
            if not fut.done():
                fut.set_result(msg)
            return
        # 频道数据
        arg = msg.get("arg", {})
        channel = arg.get("channel")
        data = msg.get("data") or []
        if channel == "orders":
            for o in data:
                self._on_order(o)
        elif channel == "positions":
            for p in data:
                self._on_position(p)

    def _on_order(self, o: dict) -> None:
        key = o.get("clOrdId") or o.get("ordId")
        if key:
            self.orders[key] = o
        state = o.get("state")
        if state == "filled":
            self._alert(f"💰 成交 {o.get('instId')} {o.get('side')} "
                        f"{o.get('accFillSz')}@{o.get('avgPx')} ({o.get('ordType')})")

    def _on_position(self, p: dict) -> None:
        inst = p.get("instId")
        if not inst:
            return
        if float(p.get("pos", 0) or 0) == 0:
            self.positions.pop(inst, None)
        else:
            self.positions[inst] = p

    # ----------------- WS 下单 (与 REST 同签名) -----------------

    async def _op(self, op: str, arg: dict, endpoint: str) -> dict:
        if not (self._ws and self.logged_in):
            raise OkxError("WS", "私有WS未登录", endpoint)
        self._req += 1
        rid = f"{op[:3]}{self._req}"
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[rid] = fut
        await self._ws.send(json.dumps({"id": rid, "op": op, "args": [arg]}))
        try:
            resp = await asyncio.wait_for(fut, timeout=5.0)
        except asyncio.TimeoutError:
            self._pending.pop(rid, None)
            raise OkxError("WS", "下单响应超时", endpoint)
        d = (resp.get("data") or [{}])[0]
        if d.get("sCode") not in (None, "0"):
            raise OkxError(d.get("sCode", "?"), d.get("sMsg", ""), endpoint)
        return d

    async def place_order(self, *, inst_id: str, td_mode: str, side: str,
                          ord_type: str, sz: str, px: Optional[str] = None,
                          pos_side: Optional[str] = None,
                          reduce_only: bool = False,
                          cl_ord_id: Optional[str] = None) -> dict:
        arg = {"instId": inst_id, "tdMode": td_mode, "side": side,
               "ordType": ord_type, "sz": sz}
        if px is not None:
            arg["px"] = px
        if pos_side:
            arg["posSide"] = pos_side
        if reduce_only:
            arg["reduceOnly"] = True
        if cl_ord_id:
            arg["clOrdId"] = cl_ord_id
        return await self._op("order", arg, "ws/order")

    async def cancel_order(self, inst_id: str, *, ord_id: Optional[str] = None,
                           cl_ord_id: Optional[str] = None) -> dict:
        arg = {"instId": inst_id}
        if ord_id:
            arg["ordId"] = ord_id
        if cl_ord_id:
            arg["clOrdId"] = cl_ord_id
        return await self._op("cancel-order", arg, "ws/cancel-order")
