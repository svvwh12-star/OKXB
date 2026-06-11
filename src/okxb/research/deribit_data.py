"""Deribit public (no-auth) options data for daily IV/skew/term/OI features.

Offline research fetcher: httpx + CSV cache, UTC-ms timestamps, no keys.

Two horizons of availability:
  - DVOL (implied-vol index) is BACKFILLABLE for years via get_volatility_index_data
    -> the historical options signal (DVOL level/change, and VRP = DVOL^2 - RV^2).
  - Per-strike smile/skew/term/OI/gamma come from get_book_summary_by_currency,
    which is a CURRENT SNAPSHOT only -> build that archive FORWARD with a daily
    snapshot job (snapshot_daily); features computed point-in-time downstream.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

import httpx
import pandas as pd

BASE = "https://www.deribit.com/api/v2/public/"
_MONTHS = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
           "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}


def _log(m: str) -> None:
    print(m, flush=True)


def parse_dvol(result: dict) -> pd.DataFrame:
    """get_volatility_index_data result -> ascending, de-duplicated df(ts, dvol=close)."""
    rows = result.get("data", []) if result else []
    out = [(int(r[0]), float(r[4])) for r in rows if len(r) >= 5]
    df = pd.DataFrame(sorted(set(out)), columns=["ts", "dvol"])
    if len(df):
        df["ts"] = df["ts"].astype("int64")
    return df


def expiry_days_from_name(instrument_name: str, asof_ms: int) -> int:
    """Whole days from asof to the option expiry encoded in a Deribit name like BTC-27JUN25-100000-C.
    Deribit options expire at 08:00 UTC."""
    tok = instrument_name.split("-")[1]  # e.g. 27JUN25
    day = int(tok[:2]); mon = _MONTHS[tok[2:5]]; yr = 2000 + int(tok[5:])
    exp = pd.Timestamp(year=yr, month=mon, day=day, hour=8, tz="UTC")
    return int((exp.value // 1_000_000 - int(asof_ms)) // 86_400_000)


def fetch_dvol(currency: str, days: float, *, resolution: str = "86400",
               log: Callable[[str], None] = _log) -> pd.DataFrame:
    """DVOL history (resolution seconds as string; 86400 = 1D default for the daily study,
    ~1000-row cap => ~2.7yr; 43200 = 12h). Ascending df(ts, dvol).
    Follows the API `continuation` cursor to cover long windows."""
    now = int(time.time() * 1000)
    start = now - int(days * 86_400_000)
    rows: list = []
    cont = None
    with httpx.Client(timeout=20.0, headers={"User-Agent": "okxb-research/1"}) as cli:
        for _ in range(60):
            params = {"currency": currency, "start_timestamp": start,
                      "end_timestamp": now, "resolution": resolution}
            if cont:
                params["continuation"] = cont
            try:
                res = cli.get(BASE + "get_volatility_index_data", params=params).json().get("result", {})
            except Exception as e:  # noqa: BLE001
                log(f"  dvol {currency}: {type(e).__name__} {e}")
                break
            data = res.get("data", [])
            rows.extend(data)
            cont = res.get("continuation")
            if not cont or not data:
                break
            time.sleep(0.1)
    return parse_dvol({"data": rows})


def fetch_option_summary(currency: str, *, log: Callable[[str], None] = _log) -> pd.DataFrame:
    """Current per-strike option summary snapshot. Columns include
    instrument_name, mark_iv, open_interest, underlying_price, mid_price."""
    with httpx.Client(timeout=20.0, headers={"User-Agent": "okxb-research/1"}) as cli:
        rows = cli.get(BASE + "get_book_summary_by_currency",
                       params={"currency": currency, "kind": "option"}).json().get("result", [])
    keep = ["instrument_name", "mark_iv", "open_interest", "underlying_price", "mid_price"]
    df = pd.DataFrame(rows)
    for k in keep:
        if k not in df.columns:
            df[k] = float("nan")
    return df[keep]


def snapshot_daily(currency: str, root: Path, snapshot_ms: int) -> Path:
    """Persist a raw option-summary snapshot tagged with snapshot_ms (forward archive for PIT features)."""
    root.mkdir(parents=True, exist_ok=True)
    df = fetch_option_summary(currency)
    df["snapshot_ts"] = int(snapshot_ms)
    f = root / f"deribit_{currency}_{int(snapshot_ms)}.csv"
    df.to_csv(f, index=False)
    return f
