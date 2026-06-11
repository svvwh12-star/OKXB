import numpy as np

from okxb.research.daily_panel import pit_asof_join, DAY_MS


def test_value_invisible_before_publish_lag():
    # one daily series point closing at day D, published 1 day later
    close = np.array([10 * DAY_MS], dtype=np.int64)
    val = np.array([1.23])
    lag = DAY_MS  # published one full day after close
    # decision exactly at close -> NOT yet available -> NaN
    out_at_close = pit_asof_join(np.array([10 * DAY_MS]), close, val, lag)
    assert np.isnan(out_at_close[0])
    # decision one day later -> available
    out_after = pit_asof_join(np.array([11 * DAY_MS]), close, val, lag)
    assert out_after[0] == 1.23


def test_asof_takes_latest_available_not_future():
    close = np.array([1, 2, 3], dtype=np.int64) * DAY_MS
    val = np.array([10.0, 20.0, 30.0])
    lag = 0
    # decisions between points pick the latest already-available value
    dec = np.array([2 * DAY_MS + DAY_MS // 2, 2 * DAY_MS], dtype=np.int64)
    out = pit_asof_join(dec, close, val, lag)
    assert out[0] == 20.0  # at 2.5D, latest available is the 2D value
    assert out[1] == 20.0  # at exactly 2D (lag 0), the 2D value is available


def test_decision_before_any_data_is_nan():
    close = np.array([5], dtype=np.int64) * DAY_MS
    val = np.array([9.0])
    out = pit_asof_join(np.array([1 * DAY_MS]), close, val, 0)
    assert np.isnan(out[0])
