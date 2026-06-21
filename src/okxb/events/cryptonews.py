"""加密新闻源 (CryptoPanic 编辑聚合 + 可配置 RSS, 默认 CoinDesk), fail-closed。

契约与 FinnhubClient 一致: 无 key/url 即 disabled, 任何网络/解析失败返回 [] (绝不拖垮上层)。
- CryptoPanic: 需 auth_token (免费档可申请); 编辑聚合, 比裸社媒干净。
- RSS: 无需 key (默认 CoinDesk 一手编辑标题)。
铁律 (风控评审): 只收【有标题 + 有时间】的条目, 每条带一手链接供核对; 不含任何裸社媒(X/TG/Reddit)。
"""
from __future__ import annotations

import datetime as dt
from email.utils import parsedate_to_datetime
from typing import Optional
from xml.etree import ElementTree as ET

import httpx

from .provenance import Evidence, now_ms

CRYPTOPANIC = "https://cryptopanic.com/api/v1/posts/"
DEFAULT_RSS = "https://www.coindesk.com/arc/outboundfeeds/rss/"


def _iso_ms(s: str) -> Optional[int]:
    try:
        return int(dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000)
    except Exception:
        return None


def _rfc822_ms(s: str) -> Optional[int]:
    try:
        return int(parsedate_to_datetime(s).timestamp() * 1000)
    except Exception:
        return None


class CryptoNewsClient:
    """api_key=CryptoPanic token; rss_url 留空则不取 RSS。两者皆空 => disabled。"""

    def __init__(self, api_key: str = "", rss_url: str = ""):
        self._key = (api_key or "").strip()
        self._rss = (rss_url or "").strip()
        self._client = httpx.AsyncClient(timeout=15.0) if (self._key or self._rss) else None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @property
    def label(self) -> str:
        parts = []
        if self._key:
            parts.append("CryptoPanic")
        if self._rss:
            parts.append("RSS")
        return "+".join(parts) if parts else "off"

    async def aclose(self) -> None:
        if self._client:
            await self._client.aclose()

    async def headlines(self, currencies: Optional[list[str]] = None,
                        limit: int = 12) -> list[Evidence]:
        if not self._client:
            return []
        out: list[Evidence] = []
        if self._key:
            out += await self._cryptopanic(currencies, limit)
        if self._rss and len(out) < limit:
            out += await self._rss_feed(limit - len(out))
        return out[:limit]

    async def _cryptopanic(self, currencies, limit) -> list[Evidence]:
        try:
            params = {"auth_token": self._key, "public": "true", "kind": "news"}
            if currencies:
                params["currencies"] = ",".join(c.upper() for c in currencies[:10])
            r = await self._client.get(CRYPTOPANIC, params=params)
            r.raise_for_status()
            ing = now_ms()
            items = []
            for p in (r.json().get("results") or [])[:limit]:
                title = str(p.get("title", "")).strip()
                if not title:
                    continue
                src = ((p.get("source") or {}).get("title")) or "CryptoPanic"
                items.append(Evidence(
                    kind="news", source=f"CryptoPanic/{src}"[:32], title=title[:140],
                    url=str(p.get("url", "")),
                    publish_ms=_iso_ms(str(p.get("published_at", ""))), ingest_ms=ing,
                    symbol=",".join(c.get("code", "") for c in (p.get("currencies") or [])[:4]),
                ))
            return items
        except Exception as e:  # noqa: BLE001
            print(f"[cryptonews] cryptopanic 失败: {e!r}")
            return []

    async def _rss_feed(self, limit) -> list[Evidence]:
        try:
            r = await self._client.get(self._rss)
            r.raise_for_status()
            root = ET.fromstring(r.text)
            ing = now_ms()
            items = []
            for it in root.iter("item"):
                title = (it.findtext("title") or "").strip()
                if not title:
                    continue
                items.append(Evidence(
                    kind="news", source="RSS", title=title[:140],
                    url=(it.findtext("link") or "").strip(),
                    publish_ms=_rfc822_ms(it.findtext("pubDate") or ""), ingest_ms=ing,
                ))
                if len(items) >= limit:
                    break
            return items
        except Exception as e:  # noqa: BLE001
            print(f"[cryptonews] rss 失败: {e!r}")
            return []
