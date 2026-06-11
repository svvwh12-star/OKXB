"""OKX V5 异步 REST 客户端。

为何不直接用 python-okx 的 REST: 官方 SDK 的 REST 是同步(requests)的, 放进 asyncio
下单热路径会阻塞事件循环。这里用 httpx.AsyncClient + 官方同款 HMAC 签名 (与
verify_account.py 一致), 获得异步、低延迟、字段完全可控。
行情/私有 WS 流式仍计划使用 python-okx 的异步 WS 客户端 (那部分 SDK 是异步的)。

所有下单/撤单调用前先 await 限速器, 防止触发 OKX 频率上限。
错误统一抛 OkxError, 调用方/风控可据此降级。
"""
from __future__ import annotations

import base64
import datetime
import hashlib
import hmac
import json
from typing import Any, Optional

import httpx

from ..config import Config, Secrets
from .rate_limiter import OkxRateLimiter

# 区域路由 (已核验 2026-06): 账户所属地区决定 base host, 用错会路由/鉴权失败。
REST_HOSTS = {
    "global": "https://www.okx.com",
    "us": "https://us.okx.com",
    "eea": "https://eea.okx.com",
}


class OkxError(RuntimeError):
    def __init__(self, code: str, msg: str, endpoint: str = ""):
        self.code = code
        self.msg = msg
        self.endpoint = endpoint
        super().__init__(f"OKX {code} @ {endpoint}: {msg}")


def _utc_ms_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class OkxRestClient:
    def __init__(self, secrets: Secrets, config: Config,
                 rate_limiter: Optional[OkxRateLimiter] = None):
        # 不在此强制要求密钥: 公共接口(时间/合约/行情)无需鉴权;
        # 私有调用时若缺密钥会在 _headers 抛错。
        self._key = secrets.okx_api_key
        self._secret = secrets.okx_secret_key
        self._passphrase = secrets.okx_passphrase
        self._demo = secrets.is_demo
        self._cfg = config
        self._rl = rate_limiter or OkxRateLimiter.from_config(
            config.get("execution.rate_limit", {}) or {}
        )
        region = getattr(secrets, "region", None) or config.get("exchange.region", "global")
        base_url = (config.get("exchange.rest_hosts", {}) or REST_HOSTS).get(
            region, REST_HOSTS["global"]
        )
        self.region = region
        self._client = httpx.AsyncClient(base_url=base_url, timeout=10.0)

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------- 签名与请求 -------------------------

    def _sign(self, ts: str, method: str, path: str, body: str) -> str:
        mac = hmac.new(self._secret.encode(), f"{ts}{method}{path}{body}".encode(),
                       hashlib.sha256)
        return base64.b64encode(mac.digest()).decode()

    def _headers(self, method: str, path: str, body: str, auth: bool) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._demo:
            h["x-simulated-trading"] = "1"
        if auth:
            if not self._key or not self._secret or not self._passphrase:
                raise OkxError("AUTH", "私有接口需要 OKX 密钥, 请在 .env 配置", path)
            ts = _utc_ms_iso()
            h.update({
                "OK-ACCESS-KEY": self._key,
                "OK-ACCESS-SIGN": self._sign(ts, method, path, body),
                "OK-ACCESS-TIMESTAMP": ts,
                "OK-ACCESS-PASSPHRASE": self._passphrase,
            })
        return h

    async def _request(self, method: str, path: str, *,
                       params: Optional[dict] = None,
                       body: Optional[dict] = None,
                       auth: bool = True) -> list[dict]:
        if params:
            from urllib.parse import urlencode
            query = "?" + urlencode(params)
            path = path + query
        body_str = json.dumps(body) if body is not None else ""
        headers = self._headers(method, path, body_str, auth)
        resp = await self._client.request(method, path, headers=headers,
                                          content=body_str or None)
        data = resp.json()
        if data.get("code") != "0":
            # code=1(操作失败)时真实原因在 data[0].sCode/sMsg, 必须带出, 否则只看到"All operations failed"
            msg = data.get("msg", "")
            arr = data.get("data") or []
            if arr and isinstance(arr, list) and isinstance(arr[0], dict):
                sc, sm = arr[0].get("sCode"), arr[0].get("sMsg")
                if sc and sc != "0":
                    msg = f"{msg} [{sc}: {sm}]".strip() if msg else f"{sc}: {sm}"
            raise OkxError(data.get("code", "?"), msg, path)
        return data.get("data", [])

    # ------------------------- 公共数据 -------------------------

    async def get_server_time_ms(self) -> int:
        d = await self._request("GET", "/api/v5/public/time", auth=False)
        return int(d[0]["ts"])

    async def get_instruments(self, inst_type: str = "SWAP",
                              inst_id: Optional[str] = None) -> list[dict]:
        params = {"instType": inst_type}
        if inst_id:                       # 单合约查询: 比拉全市场(~500条)快得多
            params["instId"] = inst_id
        return await self._request("GET", "/api/v5/public/instruments",
                                   params=params, auth=False)

    async def get_tickers(self, inst_type: str = "SWAP") -> list[dict]:
        """全市场行情快照 (含 24h 成交量), 用于按流动性自动选标的。"""
        return await self._request("GET", "/api/v5/market/tickers",
                                   params={"instType": inst_type}, auth=False)

    async def get_ticker(self, inst_id: str) -> dict:
        """单标的行情 (last/bid/ask/24h开盘/成交量)。免鉴权。"""
        d = await self._request("GET", "/api/v5/market/ticker",
                                params={"instId": inst_id}, auth=False)
        return d[0] if d else {}

    async def get_funding_rate(self, inst_id: str) -> dict:
        d = await self._request("GET", "/api/v5/public/funding-rate",
                                params={"instId": inst_id}, auth=False)
        return d[0] if d else {}

    async def get_mark_price(self, inst_id: str) -> dict:
        d = await self._request("GET", "/api/v5/public/mark-price",
                                params={"instType": "SWAP", "instId": inst_id}, auth=False)
        return d[0] if d else {}

    async def get_mark_prices(self, inst_type: str = "SWAP") -> list[dict]:
        """全市场标记价 (一次取全部, 供 basis 计算)。"""
        return await self._request("GET", "/api/v5/public/mark-price",
                                   params={"instType": inst_type}, auth=False)

    async def get_candles(self, inst_id: str, bar: str = "1m",
                          after: Optional[str] = None, before: Optional[str] = None,
                          limit: int = 300) -> list[list[str]]:
        """最近K线 (公共, 免鉴权)。bar: 1m/3m/5m/15m/30m/1H/2H/4H...
        每条 [ts, o, h, l, c, vol(张), volCcy(基币), volCcyQuote(计价币/USDT), confirm]; 新→旧。"""
        params: dict[str, str] = {"instId": inst_id, "bar": bar, "limit": str(limit)}
        if after:
            params["after"] = str(after)
        if before:
            params["before"] = str(before)
        return await self._request("GET", "/api/v5/market/candles",
                                   params=params, auth=False)  # type: ignore[return-value]

    async def get_history_candles(self, inst_id: str, bar: str = "1m",
                                  after: Optional[str] = None, before: Optional[str] = None,
                                  limit: int = 100) -> list[list[str]]:
        """历史K线 (公共, 免鉴权, 用于深度回填; limit≤100)。
        after = 返回【早于】该 ts 的更旧数据 (向过去翻页); 返回新→旧。"""
        params: dict[str, str] = {"instId": inst_id, "bar": bar, "limit": str(limit)}
        if after:
            params["after"] = str(after)
        if before:
            params["before"] = str(before)
        return await self._request("GET", "/api/v5/market/history-candles",
                                   params=params, auth=False)  # type: ignore[return-value]

    # ------------------------- 账户 -------------------------

    async def get_account_config(self) -> dict:
        d = await self._request("GET", "/api/v5/account/config")
        return d[0] if d else {}

    async def get_balance(self) -> dict:
        d = await self._request("GET", "/api/v5/account/balance")
        return d[0] if d else {}

    async def get_positions(self, inst_type: str = "SWAP") -> list[dict]:
        return await self._request("GET", "/api/v5/account/positions",
                                   params={"instType": inst_type})

    async def get_positions_history(self, inst_type: str = "SWAP", limit: int = 100) -> list[dict]:
        """已平仓历史 (每条含 realizedPnl/pnl + uTime), 用于 日/周/月 已实现盈亏统计 (默认近3月)。"""
        return await self._request("GET", "/api/v5/account/positions-history",
                                   params={"instType": inst_type, "limit": str(limit)})

    async def set_leverage(self, inst_id: str, lever: str, mgn_mode: str = "isolated",
                           pos_side: Optional[str] = None) -> list[dict]:
        body = {"instId": inst_id, "lever": str(lever), "mgnMode": mgn_mode}
        if pos_side:
            body["posSide"] = pos_side
        return await self._request("POST", "/api/v5/account/set-leverage", body=body)

    # ------------------------- 交易 -------------------------

    async def place_order(self, *, inst_id: str, td_mode: str, side: str,
                          ord_type: str, sz: str, px: Optional[str] = None,
                          pos_side: Optional[str] = None,
                          reduce_only: bool = False,
                          cl_ord_id: Optional[str] = None,
                          attach_algo: Optional[list] = None) -> dict:
        """下单。ord_type: limit/post_only/fok/ioc/market。
        系统默认禁止无保护 market (见 config.execution.use_market_order)。
        attach_algo: 附带止盈止损(OCO), 入场成交后自动激活, 形如
          [{"tpTriggerPx","tpOrdPx","slTriggerPx","slOrdPx", ...}] (-1 表示市价触发)。"""
        await self._rl.acquire_place(inst_id)
        body: dict[str, Any] = {
            "instId": inst_id, "tdMode": td_mode, "side": side,
            "ordType": ord_type, "sz": sz,
        }
        if px is not None:
            body["px"] = px
        if pos_side:
            body["posSide"] = pos_side
        if reduce_only:
            body["reduceOnly"] = True
        if cl_ord_id:
            body["clOrdId"] = cl_ord_id
        if attach_algo:
            body["attachAlgoOrds"] = attach_algo
        d = await self._request("POST", "/api/v5/trade/order", body=body)
        res = d[0] if d else {}
        if res.get("sCode") not in (None, "0"):
            raise OkxError(res.get("sCode", "?"), res.get("sMsg", ""), "trade/order")
        return res

    async def place_algo_order(self, *, inst_id: str, td_mode: str, side: str, sz: str,
                               ord_type: str = "conditional", reduce_only: bool = True,
                               pos_side: str = "net",
                               tp_trigger_px: Optional[str] = None, tp_ord_px: Optional[str] = None,
                               sl_trigger_px: Optional[str] = None, sl_ord_px: Optional[str] = None,
                               trigger_px_type: str = "last") -> dict:
        """独立策略委托(止盈或止损), reduce-only。用于手动『单独挂止盈/止损』。"""
        await self._rl.acquire_place(inst_id)
        body: dict[str, Any] = {"instId": inst_id, "tdMode": td_mode, "side": side,
                                "ordType": ord_type, "sz": sz, "posSide": pos_side}
        if reduce_only:
            body["reduceOnly"] = True
        if tp_trigger_px:
            body["tpTriggerPx"] = tp_trigger_px
            body["tpOrdPx"] = tp_ord_px or "-1"
            body["tpTriggerPxType"] = trigger_px_type
        if sl_trigger_px:
            body["slTriggerPx"] = sl_trigger_px
            body["slOrdPx"] = sl_ord_px or "-1"
            body["slTriggerPxType"] = trigger_px_type
        d = await self._request("POST", "/api/v5/trade/order-algo", body=body)
        res = d[0] if d else {}
        if res.get("sCode") not in (None, "0"):
            raise OkxError(res.get("sCode", "?"), res.get("sMsg", ""), "trade/order-algo")
        return res

    async def get_order(self, inst_id: str, *, ord_id: Optional[str] = None,
                        cl_ord_id: Optional[str] = None) -> dict:
        params = {"instId": inst_id}
        if ord_id:
            params["ordId"] = ord_id
        if cl_ord_id:
            params["clOrdId"] = cl_ord_id
        d = await self._request("GET", "/api/v5/trade/order", params=params)
        return d[0] if d else {}

    async def get_pending_orders(self, inst_type: str = "SWAP") -> list[dict]:
        return await self._request("GET", "/api/v5/trade/orders-pending",
                                   params={"instType": inst_type})

    async def get_algo_pending(self, inst_type: str = "SWAP",
                               ord_type: str = "oco") -> list[dict]:
        """未触发的策略委托 (止盈止损/OCO/条件单)。ord_type: oco/conditional/trigger 等。"""
        return await self._request("GET", "/api/v5/trade/orders-algo-pending",
                                   params={"instType": inst_type, "ordType": ord_type})

    async def cancel_algos(self, orders: list[dict]) -> list[dict]:
        """撤策略委托。orders: [{algoId, instId}, ...]。"""
        return await self._request("POST", "/api/v5/trade/cancel-algos", body=orders)

    async def get_fills(self, inst_type: str = "SWAP",
                        inst_id: Optional[str] = None) -> list[dict]:
        """近期成交明细 (近3天)。用于手动页『历史成交』。"""
        params = {"instType": inst_type, "limit": "30"}
        if inst_id:
            params["instId"] = inst_id
        return await self._request("GET", "/api/v5/trade/fills", params=params)

    async def cancel_order(self, inst_id: str, *, ord_id: Optional[str] = None,
                           cl_ord_id: Optional[str] = None) -> dict:
        await self._rl.acquire_cancel(inst_id)
        body = {"instId": inst_id}
        if ord_id:
            body["ordId"] = ord_id
        if cl_ord_id:
            body["clOrdId"] = cl_ord_id
        d = await self._request("POST", "/api/v5/trade/cancel-order", body=body)
        return d[0] if d else {}

    async def cancel_batch(self, orders: list[dict]) -> list[dict]:
        """orders: [{instId, ordId|clOrdId}, ...] 一次最多 20 条。撤单不计入子账户上限。"""
        await self._rl.acquire_batch(len(orders), counts_subaccount=False)
        return await self._request("POST", "/api/v5/trade/cancel-batch-orders", body=orders)

    async def amend_order(self, inst_id: str, *, ord_id: Optional[str] = None,
                          cl_ord_id: Optional[str] = None,
                          new_px: Optional[str] = None,
                          new_sz: Optional[str] = None) -> dict:
        await self._rl.acquire_amend(inst_id)
        body: dict[str, Any] = {"instId": inst_id}
        if ord_id:
            body["ordId"] = ord_id
        if cl_ord_id:
            body["clOrdId"] = cl_ord_id
        if new_px is not None:
            body["newPx"] = new_px
        if new_sz is not None:
            body["newSz"] = new_sz
        d = await self._request("POST", "/api/v5/trade/amend-order", body=body)
        return d[0] if d else {}

    async def close_position(self, inst_id: str, mgn_mode: str = "isolated",
                             pos_side: Optional[str] = None) -> list[dict]:
        """市价全平 (紧急用)。常规退出应走 reduce-only 限价。"""
        body = {"instId": inst_id, "mgnMode": mgn_mode}
        if pos_side:
            body["posSide"] = pos_side
        return await self._request("POST", "/api/v5/trade/close-position", body=body)

    async def cancel_all_after(self, timeout_seconds: int, tag: Optional[str] = None) -> dict:
        """死手开关 (cancel-on-disconnect)。启动时武装、每轮重置;
        断线/崩溃 timeout_seconds 后交易所自动撤掉本连接的所有挂单。
        timeout=0 解除。须 > 主循环周期。无人值守 bot 最重要的安全机制。"""
        body: dict[str, Any] = {"timeOut": str(timeout_seconds)}
        if tag:
            body["tag"] = tag
        d = await self._request("POST", "/api/v5/trade/cancel-all-after", body=body)
        return d[0] if d else {}

    async def mass_cancel(self, inst_type: str, inst_family: str) -> dict:
        """一次性批量撤单 (与 cancel-all-after 不同, 这是立即执行的单次操作)。"""
        body = {"instType": inst_type, "instFamily": inst_family}
        d = await self._request("POST", "/api/v5/trade/mass-cancel", body=body)
        return d[0] if d else {}
