"""P0–P3: provenance stamping + fail-closed news/econ clients.

These guard the risk-review's hard requirement that every item fed to the AI carries a
source + timestamp, and that optional feeds stay dormant (return []) when unconfigured.
"""
import asyncio

from okxb.events import coingecko, cryptonews, econ_calendar
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
