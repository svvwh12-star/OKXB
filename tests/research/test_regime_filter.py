import numpy as np
import pandas as pd

from okxb.research.regime_filter import build_reversal_panel, verdict


def _mk(prices):
    ts = (np.arange(len(prices)) * 900_000 + 1_600_000_000_000).astype(np.int64)
    return pd.DataFrame({"ts": ts, "c": np.asarray(prices, dtype=float)})


def test_reversal_is_cross_sectionally_demeaned():
    dfs = {"A": _mk([100, 101, 102, 103, 104, 105, 106, 107]),
           "B": _mk([100, 99, 98, 97, 96, 95, 94, 93])}
    p = build_reversal_panel(dfs, "15m", k_lookback=1, h_fwd=1)
    s = p.groupby("ts")["signal"].sum()
    assert np.allclose(s.values, 0.0, atol=1e-9)   # market-neutral per timestamp


def test_verdict_detects_winner_and_null():
    win = [{"regime": "ALL", "n": 500, "net_taker": -2.0, "t_taker": -0.5},
           {"regime": "r=1", "n": 300, "net_taker": 5.0, "t_taker": 2.5}]
    ok, _ = verdict(win)
    assert ok

    null = [{"regime": "ALL", "n": 500, "net_taker": -2.0, "t_taker": -0.5},
            {"regime": "r=1", "n": 300, "net_taker": 1.0, "t_taker": 1.2}]  # t<2
    ok2, _ = verdict(null)
    assert not ok2
