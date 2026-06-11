"""Finnhub 客户端 (免费档: 财报日历 + 公司新闻)。

免费档限制 (RESEARCH_BRIEF §8): 60 calls/min + 账户级 30/s; 美股 only; 情绪打分需 Premium。
用途: 补 SEC EDGAR 没有的"财报日期"一等字段, 以及公司新闻头条。
无 key 时所有方法返回空 (优雅降级)。
"""
from __future__ import annotations

from typing import Optional

import httpx

BASE = "https://finnhub.io/api/v1"


class FinnhubClient:
    def __init__(self, api_key: str):
        self._key = api_key
        self._client = httpx.AsyncClient(base_url=BASE, timeout=15.0) if api_key else None

    async def aclose(self) -> None:
        if self._client:
            await self._client.aclose()

    @property
    def enabled(self) -> bool:
        return self._client is not None

    async def _get(self, path: str, params: dict) -> Optional[dict]:
        if not self._client:
            return None
        try:
            params = {**params, "token": self._key}
            r = await self._client.get(path, params=params)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[finnhub] {path} 失败: {e!r}")
            return None

    async def earnings_calendar(self, from_date: str, to_date: str,
                                symbol: Optional[str] = None) -> list[dict]:
        """[{symbol, date, hour, epsEstimate, ...}]。hour: bmo/amc/dmh。"""
        params = {"from": from_date, "to": to_date}
        if symbol:
            params["symbol"] = symbol
        data = await self._get("/calendar/earnings", params)
        return (data or {}).get("earningsCalendar", []) if data else []

    async def company_news(self, symbol: str, from_date: str, to_date: str) -> list[dict]:
        """[{headline, summary, datetime, source, url, id, category}]。"""
        data = await self._get("/company-news",
                               {"symbol": symbol, "from": from_date, "to": to_date})
        return data if isinstance(data, list) else []

    async def general_news(self, category: str = "general") -> list[dict]:
        """市场新闻头条。category: general / crypto / forex / merger。[{headline, source, datetime, ...}]。"""
        data = await self._get("/news", {"category": category})
        return data if isinstance(data, list) else []
