"""经济日历 (TradingEconomics), fail-closed。

只提供【已排定的发布时间】(CPI/NFP/FOMC 等) —— 这是事前唯一可靠为真的东西, 用于
"别在重大数据公布前盲目开仓"的【风险提示】, 不是 edge / 不是方向预测。
无 key 即 disabled (用户预算未含此源时整模块休眠, 不影响其它功能)。
TradingEconomics 也接受 guest 档 (c=guest:guest), 但条目极少; 留空默认休眠。
"""
from __future__ import annotations

import datetime as dt
from typing import Optional

import httpx

from .provenance import Evidence, now_ms

BASE = "https://api.tradingeconomics.com"
HIGH = {"3", "high", "High", "HIGH"}


def _te_ms(s: str) -> Optional[int]:
    s = (s or "").strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return int(dt.datetime.strptime(s, fmt)
                       .replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
        except Exception:
            continue
    return None


class EconCalendarClient:
    def __init__(self, api_key: str = "", countries: Optional[list[str]] = None):
        self._key = (api_key or "").strip()
        self._countries = countries or ["united states"]
        self._client = httpx.AsyncClient(base_url=BASE, timeout=15.0) if self._key else None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    async def aclose(self) -> None:
        if self._client:
            await self._client.aclose()

    async def upcoming(self, hours_ahead: float = 48.0, high_only: bool = True) -> list[Evidence]:
        """未来 hours_ahead 小时内的(高重要度)排定事件; 失败返回 []。"""
        if not self._client:
            return []
        try:
            today = dt.datetime.now(dt.timezone.utc).date()
            end = today + dt.timedelta(days=int(hours_ahead // 24) + 2)
            params = {"c": self._key, "d1": str(today), "d2": str(end),
                      "country": ",".join(self._countries), "f": "json"}
            r = await self._client.get("/calendar", params=params)
            r.raise_for_status()
            ing = ref = now_ms()
            horizon = ref + int(hours_ahead * 3600 * 1000)
            out = []
            for e in (r.json() or []):
                imp = str(e.get("Importance", ""))
                if high_only and imp not in HIGH:
                    continue
                when = _te_ms(str(e.get("Date", "")))
                if when is None or when < ref or when > horizon:
                    continue
                ev = str(e.get("Event", "")).strip()
                ctry = str(e.get("Country", "")).strip()
                out.append(Evidence(
                    kind="econ", source="TradingEconomics", title=f"{ctry}: {ev}",
                    url="https://tradingeconomics.com/calendar",
                    publish_ms=when, ingest_ms=ing,
                    extra=f"重要度{imp} 预期{e.get('Forecast', '')} 前值{e.get('Previous', '')}",
                ))
            out.sort(key=lambda x: x.publish_ms or 0)
            return out
        except Exception as e:  # noqa: BLE001
            print(f"[econ] tradingeconomics 失败: {e!r}")
            return []
