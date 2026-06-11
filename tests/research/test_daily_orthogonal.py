import numpy as np
import pandas as pd

from okxb.research.daily_orthogonal import merge_pit, run_daily

DAY = 86_400_000


def test_merge_pit_no_lookahead_and_lag():
    panel_ts = np.arange(1, 6, dtype=np.int64) * DAY      # decision days 1..5
    src = pd.DataFrame({"ts": np.arange(1, 6, dtype=np.int64) * DAY, "value": [10.0, 20, 30, 40, 50]})
    out = merge_pit(panel_ts, src, publish_lag_ms=DAY)    # published one day after close
    assert np.isnan(out[0])      # day1 source not yet available (1d lag)
    assert out[1] == 10.0        # day2: source(day1) now available
    assert out[4] == 40.0        # day5: latest available is source(day4)


def test_merge_pit_empty_source_is_nan():
    out = merge_pit(np.array([DAY, 2 * DAY]), pd.DataFrame({"ts": [], "value": []}), 0)
    assert np.isnan(out).all()


def test_run_daily_returns_verdict_structure():
    rng = np.random.default_rng(0)
    dfs = {}
    for inst in ("BTC-USDT-SWAP", "ETH-USDT-SWAP"):
        n = 800
        ts = (np.arange(n) * DAY + 1_600_000_000_000).astype(np.int64)
        price = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
        dfs[inst] = pd.DataFrame({"ts": ts, "o": price, "h": price * 1.01, "l": price * 0.99,
                                  "c": price, "vol": 1.0, "volccy": 1.0, "volquote": 1e6})
    res = run_daily(dfs, horizons_min=(1440,))
    assert "by_h" in res and 1440 in res["by_h"]
    assert "position" in res["by_h"][1440]
    # pure noise must not be tradable
    assert res["by_h"][1440]["position"]["tradable"] is False
