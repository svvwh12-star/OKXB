"""Point-in-time daily panel assembly for the v2 longer-horizon model.

Every external value becomes visible only at (period_close + publish_lag).
Daily-cadence features are forbidden from sub-1-day label horizons.
This module is the single place that enforces no-look-ahead.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DAY_MS = 86_400_000


def daily_grid(start_ms: int, end_ms: int) -> np.ndarray:
    """Contiguous UTC-midnight grid [start_ms, end_ms] inclusive, both snapped down to midnight."""
    s = (int(start_ms) // DAY_MS) * DAY_MS
    e = (int(end_ms) // DAY_MS) * DAY_MS
    n = (e - s) // DAY_MS + 1
    return s + np.arange(max(n, 0), dtype=np.int64) * DAY_MS


def pit_asof_join(
    decision_ts: np.ndarray,
    series_close_ms: np.ndarray,
    series_val: np.ndarray,
    publish_lag_ms: int,
) -> np.ndarray:
    """As-of join with publish lag. For each decision_ts, return the most recent
    series value whose availability time (close + publish_lag) is <= decision_ts.
    Decisions before any available value return NaN. No look-ahead by construction.
    """
    decision_ts = np.asarray(decision_ts, dtype=np.int64)
    avail = np.asarray(series_close_ms, dtype=np.int64) + int(publish_lag_ms)
    val = np.asarray(series_val, dtype=float)
    order = np.argsort(avail, kind="mergesort")
    avail_s, val_s = avail[order], val[order]
    idx = np.searchsorted(avail_s, decision_ts, side="right") - 1
    out = np.full(decision_ts.shape, np.nan, dtype=float)
    ok = idx >= 0
    out[ok] = val_s[idx[ok]]
    return out


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    source: str            # "price" | "okx" | "deribit" | "onchain" | "macro"
    publish_lag_ms: int    # delay between period close and real availability
    min_horizon_min: int   # smallest label horizon (minutes) this feature may feed


def build_daily_panel(
    frames: dict[str, pd.DataFrame],
    specs: list[FeatureSpec],
    horizon_min: int,
) -> pd.DataFrame:
    """Assemble a UTC-daily panel of point-in-time features for a given label horizon.

    frames[name] is a tidy DataFrame with int-ms 'ts' (period close) and 'value'.
    A feature is EXCLUDED when its cadence is coarser than the horizon
    (min_horizon_min > horizon_min) -- daily features never feed sub-1d labels.
    """
    usable = [s for s in specs if s.min_horizon_min <= horizon_min]
    if not usable:
        return pd.DataFrame({"ts": np.array([], dtype=np.int64)})
    closes = np.sort(np.unique(np.concatenate(
        [frames[s.name]["ts"].to_numpy(dtype=np.int64) for s in usable]
    )))
    grid = daily_grid(int(closes.min()), int(closes.max()))
    out = pd.DataFrame({"ts": grid})
    for s in usable:
        f = frames[s.name]
        out[s.name] = pit_asof_join(
            grid, f["ts"].to_numpy(dtype=np.int64), f["value"].to_numpy(dtype=float), s.publish_lag_ms
        )
    return out


def assert_no_leakage(
    panel: pd.DataFrame,
    specs: list[FeatureSpec],
    frames: dict[str, pd.DataFrame],
) -> None:
    """Fail if any panel cell holds a value that was not yet available at its row ts.
    Recomputes the point-in-time value independently and compares."""
    by_name = {s.name: s for s in specs}
    for col in panel.columns:
        if col == "ts" or col not in by_name:
            continue
        s = by_name[col]
        f = frames[s.name]
        expected = pit_asof_join(
            panel["ts"].to_numpy(dtype=np.int64),
            f["ts"].to_numpy(dtype=np.int64), f["value"].to_numpy(dtype=float), s.publish_lag_ms,
        )
        got = panel[col].to_numpy(dtype=float)
        both_nan = np.isnan(expected) & np.isnan(got)
        mismatch = ~both_nan & ~np.isclose(expected, got, equal_nan=False)
        if mismatch.any():
            i = int(np.argmax(mismatch))
            raise AssertionError(
                f"leakage in '{col}' at ts={int(panel['ts'].iloc[i])}: "
                f"panel={got[i]} but point-in-time value={expected[i]}"
            )
