import numpy as np
import pandas as pd

from okxb.research.daily_panel import (
    FeatureSpec, build_daily_panel, assert_no_leakage, DAY_MS,
)


def _toy_sources():
    # 5 daily closes for a price series and one daily on-chain series
    closes = np.arange(1, 6, dtype=np.int64) * DAY_MS
    price = pd.DataFrame({"ts": closes, "value": [100.0, 101, 102, 103, 104]})
    onchain = pd.DataFrame({"ts": closes, "value": [1.0, 2, 3, 4, 5]})
    return price, onchain


def test_panel_columns_are_point_in_time():
    price, onchain = _toy_sources()
    specs = [
        FeatureSpec("px", source="price", publish_lag_ms=0, min_horizon_min=1440),
        FeatureSpec("oc", source="onchain", publish_lag_ms=DAY_MS, min_horizon_min=1440),
    ]
    frames = {"px": price, "oc": onchain}
    panel = build_daily_panel(frames, specs, horizon_min=1440)
    # on-chain has 1-day publish lag -> its first usable row is shifted by a day
    assert panel.loc[panel["ts"] == 1 * DAY_MS, "oc"].isna().all()
    assert panel.loc[panel["ts"] == 2 * DAY_MS, "oc"].iloc[0] == 1.0  # value from day1, visible day2
    assert_no_leakage(panel, specs, frames)  # must not raise


def test_daily_feature_forbidden_from_sub_1d_horizon():
    price, onchain = _toy_sources()
    specs = [FeatureSpec("oc", source="onchain", publish_lag_ms=DAY_MS, min_horizon_min=1440)]
    panel = build_daily_panel({"oc": onchain}, specs, horizon_min=240)  # 4h horizon
    assert "oc" not in panel.columns  # excluded: daily cadence into 4h label


def test_leakage_guard_raises_on_future_value():
    price, _ = _toy_sources()
    specs = [FeatureSpec("px", source="price", publish_lag_ms=0, min_horizon_min=1440)]
    panel = build_daily_panel({"px": price}, specs, horizon_min=1440)
    # corrupt: paste a future value into an earlier row
    panel.loc[0, "px"] = 999.0
    try:
        assert_no_leakage(panel, specs, {"px": price})
    except AssertionError:
        return
    raise AssertionError("leakage guard failed to detect injected look-ahead value")
