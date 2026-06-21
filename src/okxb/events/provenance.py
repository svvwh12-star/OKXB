"""信息出处与时效标注 (provenance + freshness)。

风控评审硬性要求: 喂给 AI / 展示给用户的每一条外部信息, 都必须可追溯【来源 + 发布时间 + 时延】,
否则一律不可信 (防幻觉 / 防陈旧 / 防 look-ahead)。本模块是纯标准库的小工具:
  - Evidence: 一条带出处的证据 (新闻/财报/filing/经济日历/资金费/宏观)。
  - stamp / render_brief: 把证据渲染成"带出处与时效"的文本, 供 LLM 阅读与用户核对。
  - fmt_data_age / fmt_eta: 盘口数据时效 / 未来事件倒计时。
时间一律以"调用时传入的 ref_ms"为基准, 便于确定性测试 (不在内部偷偷读 wall clock 来比较)。
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional


def now_ms() -> int:
    return int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)


@dataclass(slots=True)
class Evidence:
    kind: str                          # news / earnings / filing / econ / funding / macro / onchain
    source: str                        # CryptoPanic / CoinDesk / Finnhub / SEC EDGAR / TradingEconomics / OKX
    title: str                         # 人类可读标题/摘要
    url: str = ""                      # 一手链接 (可点开核对); 空 = 无法核对
    publish_ms: Optional[int] = None   # source 发布时间 (UTC ms); 未来事件=排定时间
    ingest_ms: Optional[int] = None    # 本地抓取时间 (UTC ms)
    symbol: str = ""                   # 关联标的 (可空)
    extra: str = ""                    # 附加 (importance / actual vs forecast 等)

    def age_min(self, ref_ms: int) -> Optional[float]:
        base = self.publish_ms if self.publish_ms is not None else self.ingest_ms
        if base is None:
            return None
        return max(0.0, (ref_ms - base) / 60000.0)


def _local(ms: int) -> str:
    """本地时区显示 (用户能直接对照盘面时间)。"""
    return dt.datetime.fromtimestamp(ms / 1000.0).strftime("%m-%d %H:%M")


def fmt_age(minutes: Optional[float]) -> str:
    if minutes is None:
        return "时间未知"
    if minutes < 1:
        return "刚刚"
    if minutes < 60:
        return f"{int(minutes)}分前"
    if minutes < 60 * 48:
        return f"{minutes / 60:.1f}小时前"
    return f"{minutes / 1440:.1f}天前"


def fmt_eta(target_ms: int, ref_ms: int) -> str:
    """未来事件倒计时 (经济日历用): 距发布还有多久。"""
    mins = (target_ms - ref_ms) / 60000.0
    if mins < 0:
        return "已过"
    if mins < 60:
        return f"还有{int(mins)}分钟"
    if mins < 60 * 48:
        return f"还有{mins / 60:.1f}小时"
    return f"还有{mins / 1440:.1f}天"


def fmt_data_age(age_ms: Optional[float]) -> str:
    """实时盘口数据时效标注。"""
    if age_ms is None:
        return "时效未知"
    s = age_ms / 1000.0
    if s < 3:
        return "实时"
    if s < 60:
        return f"{int(s)}秒前"
    if s < 3600:
        return f"{int(s / 60)}分前"
    return "陈旧(>1h)"


def stamp(ev: Evidence, ref_ms: int) -> str:
    """单条证据 -> 一行 (带出处与时效)。无链接的明确标注, 供调用方/AI 降低信任。"""
    when = ev.publish_ms if ev.publish_ms is not None else ev.ingest_ms
    t = _local(when) if when else "时间未知"
    age = fmt_age(ev.age_min(ref_ms))
    sym = f" [{ev.symbol}]" if ev.symbol else ""
    extra = f" ({ev.extra})" if ev.extra else ""
    link = "" if ev.url else " ⚠无链接"
    return f"· {t}({age}) {ev.source}{sym}: {ev.title}{extra}{link}"


def render_brief(items: list[Evidence], ref_ms: int, *, max_items: int = 12,
                 require_url: bool = False) -> str:
    """多条【已发生】证据 -> 给 LLM 的 brief (按发布时间倒序, 每条带出处与时效)。
    require_url=True 时丢弃无链接的条目 (无法核对 = 不喂)。"""
    usable = [e for e in items if e.title and (e.url or not require_url)]
    usable.sort(key=lambda e: (e.publish_ms or e.ingest_ms or 0), reverse=True)
    return "\n".join(stamp(e, ref_ms) for e in usable[:max_items])


def render_schedule(items: list[Evidence], ref_ms: int, *, max_items: int = 8) -> str:
    """多条【未来排定】事件 (经济日历) -> 倒计时文本。仅保留尚未发生的, 按时间升序。"""
    upcoming = [e for e in items if e.title and e.publish_ms and e.publish_ms >= ref_ms]
    upcoming.sort(key=lambda e: e.publish_ms or 0)
    out = []
    for e in upcoming[:max_items]:
        eta = fmt_eta(e.publish_ms or ref_ms, ref_ms)
        extra = f" ({e.extra})" if e.extra else ""
        out.append(f"· {_local(e.publish_ms)} {eta} · {e.source}: {e.title}{extra}")
    return "\n".join(out)
