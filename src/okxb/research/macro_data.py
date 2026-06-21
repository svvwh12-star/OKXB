"""Macro regime series (VIX / equity indices / yields) via stooq.

Used only as band (c)/(d) regime conditioners (NOT raw alpha). Daily series
publish after the US close -> treat as available next UTC day (publish_lag_ms
>= 1 day in the FeatureSpec).

stooq is used because FRED (fred.stlouisfed.org) is unreachable from some
networks; this fetcher returns whatever symbols resolve and skips the rest, so
a missing macro feed never blocks the panel. ALFRED point-in-time vintages are
a later hardening step if a reachable FRED route exists.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Callable

import pandas as pd

# stooq symbols
SERIES = {"vix": "^VIX", "spx": "^SPX", "ndx": "^NDX", "ust10y": "^TNX"}


def _log(m: str) -> None:
    print(m, flush=True)


def stooq_to_tidy_ms(raw: pd.DataFrame, col: str = "Close") -> pd.DataFrame:
    """stooq OHLC frame (DatetimeIndex) -> tidy df(ts int-ms, value) ascending, NaN dropped."""
    s = pd.to_numeric(raw[col], errors="coerce").dropna()
    ts = (pd.DatetimeIndex(s.index).tz_localize("UTC").astype("int64") // 1_000_000).astype("int64")
    return pd.DataFrame({"ts": ts, "value": s.to_numpy(dtype=float)}).sort_values("ts").reset_index(drop=True)


def fetch_macro(days: float, root: Path, *, force: bool = False,
                log: Callable[[str], None] = _log) -> dict[str, pd.DataFrame]:
    """Return {name: tidy df} for resolvable SERIES via stooq. Skips unreachable symbols."""
    root.mkdir(parents=True, exist_ok=True)
    start = (dt.datetime.utcnow() - dt.timedelta(days=days)).date()
    need_start_ms = int((dt.datetime.utcnow() - dt.timedelta(days=days)).timestamp() * 1000)
    out: dict[str, pd.DataFrame] = {}
    reader = None
    for name, sym in SERIES.items():
        f = root / f"macro_{name}.csv"
        if f.exists() and not force:
            cached = pd.read_csv(f)
            # reuse only if the cache covers the requested window (avoid silent truncation)
            if len(cached) and int(cached["ts"].iloc[0]) <= need_start_ms + 5 * 86_400_000:
                out[name] = cached
                continue
        try:
            if reader is None:
                from pandas_datareader import data as pdr
                reader = pdr
            raw = reader.DataReader(sym, "stooq", start, dt.date.today())
            tidy = stooq_to_tidy_ms(raw)
            if len(tidy):
                tidy.to_csv(f, index=False)
                out[name] = tidy
        except Exception as e:  # noqa: BLE001
            log(f"  macro {name}({sym}): unreachable ({str(e)[:60]})")
    return out
