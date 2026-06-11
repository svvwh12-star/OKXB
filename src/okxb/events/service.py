"""AI 事件服务: 编排 EDGAR + Finnhub + Claude, 维护每标的 veto 缓存。

数据流(异步, 不在下单路径): poll_loop 周期刷新 -> 更新 self._events 缓存。
RiskEngine 同步调用 get_veto(inst_id, side) 读缓存。AI 永不下单。

veto 优先级: close_all > reduce_only > block(对应方向) > block_new(财报窗口) > none。
失败全程降级, 不抛出。
"""
from __future__ import annotations

import asyncio
import datetime as dt
from dataclasses import dataclass
from typing import Optional

from ..config import Config, Secrets
from ..core.enums import EventAction, Side
from ..core.models import MarketEvent
from ..risk.engine import is_stock_perp
from .edgar import EdgarClient
from .finnhub import FinnhubClient
from .llm_classifier import LLMClassifier

WATCH_FORMS = {"8-K", "4", "10-Q", "10-K"}
FRESH_MS = 2 * 3600 * 1000      # 首轮只处理 2h 内的新 filing


def ticker_of(inst_id: str) -> str:
    return inst_id.split("-")[0].upper()


def _now_ms() -> int:
    return int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)


def _parse_dt(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    try:
        return int(dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000)
    except Exception:
        return None


def earnings_window(date_str: str, hour: Optional[str],
                    before_min: float, after_min: float) -> Optional[tuple[int, int]]:
    """财报阻断窗口 [start_ms, end_ms]。hour: bmo盘前/amc盘后/dmh盘中。
    近似: bmo~13:00 UTC, amc~20:30 UTC, 其他=阻断整个交易日。DST/精度待细化。"""
    try:
        d = dt.datetime.fromisoformat(date_str).replace(tzinfo=dt.timezone.utc)
    except Exception:
        return None
    if hour == "bmo":
        t = d.replace(hour=13, minute=0)
    elif hour == "amc":
        t = d.replace(hour=20, minute=30)
    else:
        start = d.replace(hour=13, minute=0)
        end = d.replace(hour=21, minute=0)
        return (int(start.timestamp() * 1000), int(end.timestamp() * 1000))
    tm = int(t.timestamp() * 1000)
    return (tm - int(before_min * 60_000), tm + int(after_min * 60_000))


@dataclass
class ActiveEvent:
    mode: str            # close_all / reduce_only / block_long / block_short / block_new
    event_type: str
    severity: str
    valid_until_ms: int
    sentiment: float = 0.0
    confidence: float = 0.5
    raw_ref: str = ""


_PRIORITY = {"close_all": 0, "reduce_only": 1, "block_long": 2,
             "block_short": 2, "block_new": 3}


class AIEventService:
    def __init__(self, config: Config, secrets: Secrets, tickers: set[str]):
        self.tickers = {t.upper() for t in tickers}
        self._cfg = config
        er = config.section("event_risk")
        self._poll_s = float(er.get("poll_interval_seconds", 60))
        self._earn_before = float(er.get("earnings_block_before_minutes", 30))
        self._earn_after = float(er.get("earnings_block_after_minutes", 90))
        self._8k_min = float(er.get("sec_8k_high_severity_reduce_only_minutes", 60))
        self.edgar = EdgarClient(secrets.edgar_user_agent)
        self.finnhub = FinnhubClient(secrets.finnhub_api_key)
        self.llm = LLMClassifier.from_secrets(secrets)
        self._events: dict[str, list[ActiveEvent]] = {}
        self._seen: set[str] = set()
        self._running = False

    async def setup(self) -> None:
        await self.edgar.load_ticker_map()
        missing = [t for t in self.tickers if not self.edgar.cik_for(t)]
        print(f"[events] 监控 {len(self.tickers)} 标的; AI={self.llm.label}; "
              f"Finnhub={'on' if self.finnhub.enabled else 'off'}; 未找到CIK: {missing}")

    # ----------------- 轮询 -----------------

    async def run(self) -> None:
        self._running = True
        while self._running:
            try:
                await self.poll_once()
            except Exception as e:
                print(f"[events] 轮询异常: {e!r}")
            await asyncio.sleep(self._poll_s)

    def stop(self) -> None:
        self._running = False

    async def poll_once(self) -> None:
        now = _now_ms()
        await self._poll_earnings(now)
        for ticker in self.tickers:
            await self._poll_filings(ticker, now)
        self._prune(now)

    async def _poll_earnings(self, now: int) -> None:
        if not self.finnhub.enabled:
            return
        today = dt.datetime.now(dt.timezone.utc).date()
        cal = await self.finnhub.earnings_calendar(
            str(today), str(today + dt.timedelta(days=7)))
        for e in cal:
            sym = str(e.get("symbol", "")).upper()
            if sym not in self.tickers:
                continue
            win = earnings_window(e.get("date", ""), e.get("hour"),
                                  self._earn_before, self._earn_after)
            if not win:
                continue
            start, end = win
            if now > end:
                continue
            self._add(sym, ActiveEvent("block_new", "earnings", "medium", end,
                                       confidence=0.8, raw_ref=f"earnings {e.get('date')}"),
                      replace_kind="earnings")

    async def _poll_filings(self, ticker: str, now: int) -> None:
        cik = self.edgar.cik_for(ticker)
        if not cik:
            return
        filings = await self.edgar.recent_filings(cik, forms=WATCH_FORMS)
        for f in filings:
            acc = f.get("accession")
            if not acc or acc in self._seen:
                continue
            self._seen.add(acc)
            accepted = _parse_dt(f.get("acceptance"))
            if accepted is None or (now - accepted) > FRESH_MS:
                continue   # 旧 filing: 只入 seen 不动作
            form = f.get("form", "")
            kind = "8-K" if form == "8-K" else "form4" if form == "4" else form
            label = await self.llm.classify(
                ticker, f.get("primaryDoc") or form, kind, f.get("items", ""))
            mode = label.get("action", "no_action")
            if mode == "no_action":
                continue
            dur = self._duration_ms(mode, label.get("severity", "medium"))
            self._add(ticker, ActiveEvent(
                mode=mode, event_type=label.get("event_type", "SEC_8K"),
                severity=label.get("severity", "medium"), valid_until_ms=now + dur,
                sentiment=float(label.get("sentiment", 0.0)),
                confidence=float(label.get("confidence", 0.5)),
                raw_ref=f"{form} {f.get('date')} {acc}"))

    def _duration_ms(self, mode: str, severity: str) -> int:
        if mode == "close_all":
            return 6 * 3600 * 1000           # 公司行为: 6h 后需人工复核
        base = self._8k_min if severity == "high" else self._8k_min * 0.5
        return int(base * 60_000)

    def _add(self, ticker: str, ev: ActiveEvent, replace_kind: Optional[str] = None) -> None:
        lst = self._events.setdefault(ticker, [])
        if replace_kind:
            lst[:] = [e for e in lst if e.event_type != replace_kind]
        lst.append(ev)

    def _prune(self, now: int) -> None:
        for t, lst in list(self._events.items()):
            lst[:] = [e for e in lst if e.valid_until_ms > now]
            if not lst:
                self._events.pop(t, None)

    # ----------------- 供 RiskEngine 调用 -----------------

    def get_veto(self, inst_id: str, side: Side) -> Optional[MarketEvent]:
        if not is_stock_perp(inst_id):
            return None
        ticker = ticker_of(inst_id)
        now = _now_ms()
        active = [e for e in self._events.get(ticker, []) if e.valid_until_ms > now]
        if not active:
            return None
        ev = min(active, key=lambda e: _PRIORITY.get(e.mode, 9))
        if ev.mode == "close_all":
            action = EventAction.CLOSE_ALL
        elif ev.mode == "reduce_only":
            action = EventAction.REDUCE_ONLY
        elif ev.mode == "block_long":
            action = EventAction.BLOCK_LONG
        elif ev.mode == "block_short":
            action = EventAction.BLOCK_SHORT
        elif ev.mode == "block_new":
            action = EventAction.BLOCK_LONG if side == Side.BUY else EventAction.BLOCK_SHORT
        else:
            return None
        return MarketEvent(
            ticker=ticker, event_type=ev.event_type, sentiment=ev.sentiment,
            confidence=ev.confidence, severity=ev.severity, time_horizon="intraday",
            source_quality="official", action=action, valid_until_ms=ev.valid_until_ms,
            raw_ref=ev.raw_ref)

    async def aclose(self) -> None:
        await self.edgar.aclose()
        await self.finnhub.aclose()
