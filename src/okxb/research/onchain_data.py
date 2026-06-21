"""Coin Metrics community (no-key) daily on-chain metrics.

Daily cadence + retroactive revision -> consume only with a publish lag
(FeatureSpec.publish_lag_ms >= 1 day) and only for >=1d horizons.

Community-tier metrics confirmed available (2026-06): AdrActCnt, TxCnt,
CapMrktCurUSD, CapMVRVCur, SplyCur, FlowInExNtv/USD, FlowOutExNtv/USD.
Per-metric fetch is resilient: an unavailable metric for a given asset is skipped.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import pandas as pd

DEFAULT_METRICS = [
    "AdrActCnt", "TxCnt", "CapMrktCurUSD", "CapMVRVCur", "SplyCur",
    "FlowInExNtv", "FlowOutExNtv", "FlowInExUSD", "FlowOutExUSD",
]


def _log(m: str) -> None:
    print(m, flush=True)


def to_tidy_ms(raw: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Coin Metrics frame -> tidy df(ts int-ms, value float) ascending, for one metric."""
    df = raw[["time", metric]].copy()
    df["ts"] = (pd.to_datetime(df["time"], utc=True).astype("int64") // 1_000_000).astype("int64")
    df["value"] = pd.to_numeric(df[metric], errors="coerce")
    return df[["ts", "value"]].dropna().sort_values("ts").reset_index(drop=True)


def fetch_onchain(asset: str, metrics: Optional[list[str]], days: float,
                  root: Path, *, force: bool = False, log: Callable[[str], None] = _log) -> dict[str, pd.DataFrame]:
    """Return {metric: tidy df}. Cached per (asset, metric) CSV. Skips metrics unavailable for the asset."""
    metrics = metrics or DEFAULT_METRICS
    root.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.utcnow()
    need_start_ms = int((now - pd.Timedelta(days=days)).timestamp() * 1000)
    start = (now - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    out: dict[str, pd.DataFrame] = {}
    client = None
    for m in metrics:
        f = root / f"{asset}_{m}.csv"
        if f.exists() and not force:
            cached = pd.read_csv(f)
            # reuse only if the cache actually covers the requested window (avoid silent truncation)
            if len(cached) and int(cached["ts"].iloc[0]) <= need_start_ms + 2 * 86_400_000:
                out[m] = cached
                continue
        try:
            if client is None:
                from coinmetrics.api_client import CoinMetricsClient
                client = CoinMetricsClient()
            raw = client.get_asset_metrics(
                assets=asset, metrics=[m], frequency="1d", start_time=start
            ).to_dataframe()
            tidy = to_tidy_ms(raw, m)
            tidy.to_csv(f, index=False)
            out[m] = tidy
        except Exception as e:  # noqa: BLE001
            log(f"  onchain {asset}/{m}: unavailable ({str(e)[:60]})")
    return out
