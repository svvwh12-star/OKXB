"""P0–P3: provenance stamping + fail-closed news/econ clients.

These guard the risk-review's hard requirement that every item fed to the AI carries a
source + timestamp, and that optional feeds stay dormant (return []) when unconfigured.
"""
import asyncio

from okxb.events import coingecko, cryptonews, econ_calendar, stocknews
from okxb.events.llm_classifier import LLMClassifier, _parse_json, _render_picks
from okxb.events.provenance import (Evidence, fmt_age, fmt_data_age, fmt_eta,
                                     render_brief, render_schedule, stamp)

NOW = 1_700_000_000_000          # fixed ref ms
MIN = 60_000
HOUR = 60 * MIN
DAY = 24 * HOUR


def test_evidence_age_prefers_publish_then_ingest():
    assert Evidence("news", "X", "t", publish_ms=NOW - 5 * MIN).age_min(NOW) == 5.0
    assert Evidence("news", "X", "t", ingest_ms=NOW - 2 * MIN).age_min(NOW) == 2.0
    assert Evidence("news", "X", "t").age_min(NOW) is None
    # future publish clamps to 0 (never negative age)
    assert Evidence("econ", "X", "t", publish_ms=NOW + HOUR).age_min(NOW) == 0.0


def test_fmt_helpers():
    assert fmt_age(None) == "时间未知"
    assert fmt_age(0.5) == "刚刚"
    assert fmt_age(30) == "30分前"
    assert "小时前" in fmt_age(120)
    assert "天前" in fmt_age(3 * 1440)
    assert fmt_eta(NOW + 30 * MIN, NOW) == "还有30分钟"
    assert "小时" in fmt_eta(NOW + 3 * HOUR, NOW)
    assert fmt_eta(NOW - MIN, NOW) == "已过"
    assert fmt_data_age(None) == "时效未知"
    assert fmt_data_age(1000) == "实时"
    assert fmt_data_age(30_000) == "30秒前"
    assert fmt_data_age(2 * HOUR) == "陈旧(>1h)"


def test_stamp_marks_missing_link_and_source():
    s = stamp(Evidence("news", "CoinDesk", "BTC surges", url="http://x", publish_ms=NOW - MIN), NOW)
    assert "CoinDesk" in s and "BTC surges" in s and "⚠无链接" not in s
    s2 = stamp(Evidence("news", "RSS", "no link item", publish_ms=NOW - MIN), NOW)
    assert "⚠无链接" in s2


def test_render_brief_sorts_desc_caps_and_can_require_url():
    items = [
        Evidence("news", "A", "old", url="u", publish_ms=NOW - 10 * MIN),
        Evidence("news", "B", "new", url="u", publish_ms=NOW - 1 * MIN),
        Evidence("news", "C", "nolink", publish_ms=NOW - 2 * MIN),
    ]
    out = render_brief(items, NOW, max_items=2)
    lines = out.splitlines()
    assert len(lines) == 2
    assert "new" in lines[0] and "nolink" in lines[1]      # newest first, then 2nd newest
    only_linked = render_brief(items, NOW, require_url=True)
    assert "nolink" not in only_linked


def test_render_schedule_only_future_ascending():
    items = [
        Evidence("econ", "TE", "FOMC", publish_ms=NOW + 3 * HOUR),
        Evidence("econ", "TE", "CPI", publish_ms=NOW + 1 * HOUR),
        Evidence("econ", "TE", "past", publish_ms=NOW - HOUR),
    ]
    out = render_schedule(items, NOW)
    lines = out.splitlines()
    assert len(lines) == 2 and "CPI" in lines[0] and "FOMC" in lines[1]
    assert "还有" in lines[0]


# ----------------- fail-closed contracts -----------------

def test_cryptonews_disabled_without_key_or_rss():
    c = cryptonews.CryptoNewsClient(api_key="", rss_url="")
    assert not c.enabled and c.label == "off"
    assert asyncio.run(c.headlines()) == []


def test_cryptonews_enabled_label():
    c = cryptonews.CryptoNewsClient(api_key="k", rss_url="http://feed")
    assert c.enabled and "CryptoPanic" in c.label and "RSS" in c.label
    asyncio.run(c.aclose())


def test_cryptonews_time_parsers():
    assert cryptonews._iso_ms("2026-06-20T12:00:00Z") is not None
    assert cryptonews._iso_ms("garbage") is None
    assert cryptonews._rfc822_ms("Sat, 20 Jun 2026 12:00:00 GMT") is not None
    assert cryptonews._rfc822_ms("nope") is None


def test_econ_disabled_without_key():
    c = econ_calendar.EconCalendarClient(api_key="")
    assert not c.enabled
    assert asyncio.run(c.upcoming()) == []


def test_econ_time_parser():
    assert econ_calendar._te_ms("2026-06-20T12:30:00") is not None
    assert econ_calendar._te_ms("2026-06-20 12:30:00") is not None
    assert econ_calendar._te_ms("2026-06-20") is not None
    assert econ_calendar._te_ms("") is None


def test_coingecko_disabled_without_key():
    c = coingecko.CoinGeckoClient(api_key="")
    assert not c.enabled and c.label == "off"
    assert asyncio.run(c.market_context()) == []


def test_coingecko_enabled_label():
    c = coingecko.CoinGeckoClient(api_key="CG-demo")
    assert c.enabled and c.label == "CoinGecko(demo)"
    asyncio.run(c.aclose())


# ----------------- stock news RSS (keyless) -----------------

_SAMPLE_RSS = """<?xml version="1.0"?><rss version="2.0"><channel>
<item><title>Apple hits record high - Reuters</title><link>http://g/abc</link>
<pubDate>Sat, 20 Jun 2026 12:00:00 GMT</pubDate>
<source url="http://reuters.com">Reuters</source></item>
<item><title>AAPL earnings preview</title><link>http://y/def</link>
<pubDate>Fri, 19 Jun 2026 09:00:00 GMT</pubDate></item>
<item><link>http://no/title</link></item>
</channel></rss>"""


def test_stocknews_parse_rss_extracts_source_link_time():
    evs = stocknews._parse_rss(_SAMPLE_RSS, "GoogleNews", NOW, limit=10)
    assert len(evs) == 2                                  # 第3条无标题被丢弃
    assert evs[0].source == "Reuters" and evs[0].url == "http://g/abc"
    assert evs[0].publish_ms is not None
    assert evs[1].source == "GoogleNews"                 # 无 <source> 用默认
    bad = stocknews._parse_rss("not xml", "Yahoo", NOW, limit=5)
    assert bad == []


def test_stocknews_disabled_returns_empty():
    c = stocknews.StockNewsClient(enabled=False)
    assert not c.enabled
    assert asyncio.run(c.headlines("AAPL")) == []


def test_stocknews_rfc822_parser():
    assert stocknews._rfc822_ms("Sat, 20 Jun 2026 12:00:00 GMT") is not None
    assert stocknews._rfc822_ms("nope") is None


# ----------------- llm json parsing + selection robustness + tier policy -----------------

def test_parse_json_prefers_last_object():
    # reasoning-style: prose + early placeholder object, then the real answer last
    s = ('thinking... {"picks": []} 继续推理 '
         '{"picks": [{"inst":"BTC-USDT-SWAP"}], "note":"ok"}')
    d = _parse_json(s)
    assert d.get("note") == "ok" and len(d.get("picks")) == 1


def test_parse_json_fenced_and_garbage():
    assert _parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _parse_json("no json at all") == {}


def test_render_picks_empty_surfaces_reason():
    out = _render_picks("加密永续", [], "市场不确定, 不推荐")
    assert "未给出推荐" in out and "市场不确定" in out
    out2 = _render_picks("加密永续", [], "")
    assert "未给出推荐" in out2 and "选型策略" in out2     # 无 note 时给排查提示


def test_json_model_respects_tier_policy():
    pro = LLMClassifier("deepseek", "k", "", "flash-m", "pro-m", tier_policy="pro")
    assert pro._json_model() == "pro-m"
    flash = LLMClassifier("deepseek", "k", "", "flash-m", "pro-m", tier_policy="flash")
    assert flash._json_model() == "flash-m"
    auto = LLMClassifier("deepseek", "k", "", "flash-m", "pro-reasoner", tier_policy="auto")
    assert auto._json_model() == "flash-m"                # auto: 推理模型回退 flash
