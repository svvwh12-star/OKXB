"""美股新闻 RSS (Google News + Yahoo Finance), 无需 key, fail-closed。

按股票代码拉一手新闻头条(带来源+链接+时间), 供【单标的分析】的背景事件。
注: 真正的实时美股逐笔价不免费; 但交易用的是 OKX 股票永续自身盘口(已 live), 这里只补新闻/背景。
铁律: 只收有标题的条目, 每条尽量带链接; 不含任何裸社媒。
"""
from __future__ import annotations

from email.utils import parsedate_to_datetime
from typing import Optional
from xml.etree import ElementTree as ET

import httpx

from .provenance import Evidence, now_ms

GOOGLE_NEWS = "https://news.google.com/rss/search"
YAHOO_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline"


def _rfc822_ms(s: str) -> Optional[int]:
    try:
        return int(parsedate_to_datetime(s).timestamp() * 1000)
    except Exception:
        return None


def _parse_rss(text: str, default_source: str, ref_ms: int, limit: int) -> list[Evidence]:
    """RSS xml -> Evidence 列表 (纯函数, 可离线测试)。<source> 子元素优先做来源名。"""
    out: list[Evidence] = []
    try:
        root = ET.fromstring(text)
    except Exception:
        return out
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        if not title:
            continue
        src_el = it.find("source")
        source = (src_el.text.strip() if src_el is not None and src_el.text else default_source)
        out.append(Evidence(
            kind="news", source=source[:24], title=title[:140],
            url=(it.findtext("link") or "").strip(),
            publish_ms=_rfc822_ms(it.findtext("pubDate") or ""), ingest_ms=ref_ms))
        if len(out) >= limit:
            break
    return out


class StockNewsClient:
    """无需 key; 默认 enabled。按 ticker 拉 Google News + Yahoo Finance RSS, 去重后返回。"""

    def __init__(self, enabled: bool = True):
        self._client = (httpx.AsyncClient(timeout=15.0,
                                          headers={"User-Agent": "Mozilla/5.0 OKXB"})
                        if enabled else None)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    async def aclose(self) -> None:
        if self._client:
            await self._client.aclose()

    async def headlines(self, ticker: str, limit: int = 6) -> list[Evidence]:
        if not self._client:
            return []
        ref = now_ms()
        merged = await self._google(ticker, ref, limit) + await self._yahoo(ticker, ref, limit)
        seen, dedup = set(), []
        for e in sorted(merged, key=lambda x: x.publish_ms or 0, reverse=True):
            k = e.title[:40]
            if k in seen:
                continue
            seen.add(k)
            dedup.append(e)
        return dedup[:limit]

    async def _google(self, ticker, ref, limit) -> list[Evidence]:
        try:
            r = await self._client.get(GOOGLE_NEWS, params={
                "q": f"{ticker} stock", "hl": "en-US", "gl": "US", "ceid": "US:en"})
            r.raise_for_status()
            return _parse_rss(r.text, "GoogleNews", ref, limit)
        except Exception as e:  # noqa: BLE001
            print(f"[stocknews] google {ticker} 失败: {e!r}")
            return []

    async def _yahoo(self, ticker, ref, limit) -> list[Evidence]:
        try:
            r = await self._client.get(YAHOO_RSS,
                                       params={"s": ticker, "region": "US", "lang": "en-US"})
            r.raise_for_status()
            return _parse_rss(r.text, "Yahoo", ref, limit)
        except Exception as e:  # noqa: BLE001
            print(f"[stocknews] yahoo {ticker} 失败: {e!r}")
            return []
