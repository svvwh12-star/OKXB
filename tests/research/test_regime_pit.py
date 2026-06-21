"""RV-6/RV-8: daily regime attached to an intraday panel must be point-in-time.

attach_regime now REQUIRES an explicit publish_lag_ms (no silent 0), and with a daily lag
an intraday bar must see the *prior* day's value, never the same (not-yet-closed) day's.
"""
import numpy as np
import pandas as pd
import pytest

from okxb.research import regime_filter as rf

DAY = 86_400_000


def _panel():
    return pd.DataFrame({"ts": [DAY + 3_600_000], "inst": ["X"], "signal": [0.0], "fwd": [0.0]})


def test_attach_regime_requires_explicit_lag():
    daily = pd.DataFrame({"ts": [0, DAY], "value": [-1.0, 5.0]})
    with pytest.raises(TypeError):                 # publish_lag_ms no longer defaults to 0
        rf.attach_regime(_panel(), daily, "r", mode="positive")


def test_attach_regime_daily_lag_blocks_same_day_leak():
    daily = pd.DataFrame({"ts": [0, DAY], "value": [-1.0, 5.0]})   # day-1 negative, day-0 positive
    lagged = rf.attach_regime(_panel(), daily, "r", publish_lag_ms=rf.DAY_MS, mode="positive")
    assert lagged["r"].iloc[0] == 0.0             # 1h into day-1 sees PRIOR day (-1) -> regime 0
    leaky = rf.attach_regime(_panel(), daily, "r", publish_lag_ms=0, mode="positive")
    assert leaky["r"].iloc[0] == 1.0             # lag 0 would leak same-day (+5) -> regime 1 (the bug)


def test_daily_orthogonal_dvol_lag_is_conservative_by_default():
    from okxb.research import daily_orthogonal as do
    assert do.DVOL_PUBLISH_LAG_MS == do.DAY_MS    # daily IV defaults to "available next period", not 0
    assert do.FUNDING_PUBLISH_LAG_MS == 0         # funding realized at settlement ts -> 0 correct
