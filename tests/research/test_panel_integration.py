import numpy as np
import pandas as pd

from okxb.research.daily_panel import FeatureSpec, build_daily_panel, assert_no_leakage, DAY_MS


def test_multi_source_panel_is_leakage_free_at_daily_horizon():
    closes = np.arange(1, 11, dtype=np.int64) * DAY_MS
    frames = {
        "px": pd.DataFrame({"ts": closes, "value": np.linspace(100, 110, 10)}),
        "oi": pd.DataFrame({"ts": closes, "value": np.linspace(5000, 5100, 10)}),
        "dvol": pd.DataFrame({"ts": closes, "value": np.linspace(50, 55, 10)}),
        "oc": pd.DataFrame({"ts": closes, "value": np.linspace(1, 2, 10)}),
        "vix": pd.DataFrame({"ts": closes, "value": np.linspace(13, 15, 10)}),
    }
    specs = [
        FeatureSpec("px", "price", 0, 240),
        FeatureSpec("oi", "okx", 0, 240),
        FeatureSpec("dvol", "deribit", 12 * 3_600_000, 720),
        FeatureSpec("oc", "onchain", DAY_MS, 1440),
        FeatureSpec("vix", "macro", DAY_MS, 1440),
    ]
    panel = build_daily_panel(frames, specs, horizon_min=1440)
    assert set(panel.columns) == {"ts", "px", "oi", "dvol", "oc", "vix"}
    assert_no_leakage(panel, specs, frames)  # must not raise

    # at a 4h horizon, daily on-chain/macro are excluded (cadence coarser than label)
    panel_4h = build_daily_panel(frames, specs, horizon_min=240)
    assert "oc" not in panel_4h.columns and "vix" not in panel_4h.columns
