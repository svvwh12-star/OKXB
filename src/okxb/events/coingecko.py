"""CoinGecko 行情/趋势数据 (免费 demo key), fail-closed。

这是【客观行情/大盘数据】源, 不是新闻: 全球加密总市值 / 24h 变化 / BTC 占比 / 热搜趋势币。
作为 AI 选品的"加密大盘 regime + 散户注意力"背景, 补 OKX 单标的数据之外的全市场视角。
demo key 走 api.coingecko.com + x_cg_demo_api_key 参数; 无 key 即休眠 (opt-in, 与其它源一致)。
"""
from __future__ import annotations

from typing import Optional

import httpx

from .provenance import Evidence, now_ms

DEMO_BASE = "https://api.coingecko.com/api/v3"


class CoinGeckoClient:
    def __init__(self, api_key: str = ""):
        self._key = (api_key or "").strip()
        self._client = httpx.AsyncClient(base_url=DEMO_BASE, timeout=15.0) if self._key else None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @property
    def label(self) -> str:
        return "CoinGecko(demo)" if self._key else "off"

    async def aclose(self) -> None:
        if self._client:
            await self._client.aclose()

    async def _get(self, path: str, params: Optional[dict] = None):
        if not self._client:
            return None
        try:
            p = dict(params or {})
            p["x_cg_demo_api_key"] = self._key
            r = await self._client.get(path, params=p)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            print(f"[coingecko] {path} 失败: {e!r}")
            return None

    async def market_context(self) -> list[Evidence]:
        """全球加密市值/BTC占比/24h + 热搜趋势币 -> Evidence(kind=market)。失败返回 []。"""
        if not self._client:
            return []
        out: list[Evidence] = []
        ing = now_ms()
        g = await self._get("/global")
        data = (g or {}).get("data") if isinstance(g, dict) else None
        if isinstance(data, dict):
            mc = (data.get("total_market_cap") or {}).get("usd")
            chg = data.get("market_cap_change_percentage_24h_usd")
            dom = (data.get("market_cap_percentage") or {}).get("btc")
            try:
                title = (f"全球加密总市值 ${float(mc) / 1e9:,.0f}B, "
                         f"24h {float(chg):+.2f}%, BTC占比 {float(dom):.1f}%")
                out.append(Evidence("market", "CoinGecko", title,
                                    url="https://www.coingecko.com/en/global-charts",
                                    publish_ms=ing, ingest_ms=ing, extra="加密大盘 regime"))
            except (TypeError, ValueError):
                pass
        tr = await self._get("/search/trending")
        coins = (tr or {}).get("coins") if isinstance(tr, dict) else None
        if isinstance(coins, list) and coins:
            names = []
            for c in coins[:7]:
                item = c.get("item") if isinstance(c, dict) else None
                if isinstance(item, dict) and item.get("symbol"):
                    names.append(str(item["symbol"]).upper())
            if names:
                out.append(Evidence("market", "CoinGecko", "热搜趋势币: " + ", ".join(names),
                                    url="https://www.coingecko.com/en/highlights/trending-crypto",
                                    publish_ms=ing, ingest_ms=ing, extra="散户注意力, 非买卖建议"))
        return out
