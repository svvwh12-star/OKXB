"""SEC EDGAR 客户端 (data.sec.gov, 免费无 key)。

合规要求 (RESEARCH_BRIEF §8): 必须带描述性 User-Agent (缺则 403); 限速 <=10 req/s/IP。
只轮询关注的 CIK (股票永续对应的公司), 不做全量爬取。
关注: 8-K(重大事件, 含 item 码)、Form 4(内部人交易)、10-Q/10-K。
所有方法对失败做降级 (返回空 + 记录), 不抛出, 以免拖垮交易循环。
"""
from __future__ import annotations

import asyncio
from typing import Optional

import httpx

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"


class EdgarClient:
    def __init__(self, user_agent: str, min_interval_s: float = 0.15):
        # 默认 ~6.7 req/s, 安全低于 10/s 上限
        self._ua = user_agent or "OKXB research contact@example.com"
        self._client = httpx.AsyncClient(
            headers={"User-Agent": self._ua, "Accept-Encoding": "gzip, deflate"},
            timeout=15.0,
        )
        self._interval = min_interval_s
        self._last_req = 0.0
        self._ticker_to_cik: dict[str, str] = {}

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _throttle(self) -> None:
        loop = asyncio.get_event_loop()
        wait = self._interval - (loop.time() - self._last_req)
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_req = loop.time()

    async def load_ticker_map(self) -> dict[str, str]:
        """ticker(大写) -> 10 位零填充 CIK。"""
        try:
            await self._throttle()
            r = await self._client.get(TICKER_MAP_URL)
            r.raise_for_status()
            data = r.json()
            self._ticker_to_cik = {
                v["ticker"].upper(): str(v["cik_str"]).zfill(10)
                for v in data.values()
            }
        except Exception as e:
            print(f"[edgar] 加载 ticker 映射失败: {e!r}")
        return self._ticker_to_cik

    def cik_for(self, ticker: str) -> Optional[str]:
        return self._ticker_to_cik.get(ticker.upper())

    async def recent_filings(self, cik: str, forms: Optional[set[str]] = None,
                             limit: int = 20) -> list[dict]:
        """返回最近 filings: [{form, date, accession, primaryDoc, items, acceptance}]。"""
        try:
            await self._throttle()
            r = await self._client.get(SUBMISSIONS_URL.format(cik=cik))
            r.raise_for_status()
            recent = r.json().get("filings", {}).get("recent", {})
        except Exception as e:
            print(f"[edgar] CIK {cik} submissions 失败: {e!r}")
            return []

        forms_arr = recent.get("form", [])
        out = []
        for i in range(min(len(forms_arr), limit)):
            form = forms_arr[i]
            if forms and form not in forms:
                continue
            out.append({
                "form": form,
                "date": recent.get("filingDate", [None])[i] if i < len(recent.get("filingDate", [])) else None,
                "accession": recent.get("accessionNumber", [None])[i] if i < len(recent.get("accessionNumber", [])) else None,
                "primaryDoc": recent.get("primaryDocument", [None])[i] if i < len(recent.get("primaryDocument", [])) else None,
                "items": recent.get("items", [""])[i] if i < len(recent.get("items", [])) else "",
                "acceptance": recent.get("acceptanceDateTime", [None])[i] if i < len(recent.get("acceptanceDateTime", [])) else None,
            })
        return out
