# P0 — 四源数据层 + Point-in-Time + 泄漏测试 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and test the v2 daily-cadence data foundation — OKX (daily bars + OI history), Deribit options, Coin Metrics on-chain, and macro — assembled into a single point-in-time-correct daily panel with an automated leakage test, so the Stage-1 net-edge research (P1) runs on data that cannot look ahead.

**Architecture:** Each source gets a small, offline, public-data fetcher mirroring the proven `candle_data.py` pattern (httpx + disk cache + UTC-ms timestamps, no keys). A central `daily_panel.py` does an as-of (point-in-time) join: every external value becomes visible only at `period_close + publish_lag`, and daily-cadence features are config-forbidden from sub-1-day horizons. A leakage unit test fails the build if any feature's availability postdates its decision bar.

**Tech Stack:** Python 3.11, httpx (installed), pandas/numpy (installed), pytest (installed); new free deps `coinmetrics-api-client`, `pandas-datareader`. CSV caches under `dist/daily/` (consistent with existing `dist/candles/`).

**Branch:** `feat/v2-daily-orthogonal` (already created; `master` holds the clean baseline).

**Conventions to follow (from existing code):**
- Timestamps are `int` epoch **milliseconds**, UTC, columns named `ts`.
- Fetchers use `httpx.Client(timeout=15, headers={"User-Agent": "okxb-research/1"})` with host failover, throttle ~0.1s, and a CSV disk cache that is reused if it covers the requested range (see `src/okxb/research/candle_data.py`).
- Research modules never import `.env`/Config/the trading client; they call public endpoints directly.

---

## File Structure

| File | Create/Modify | Responsibility |
|---|---|---|
| `conftest.py` (repo root) | Create | Put `src/` on `sys.path` so `from okxb...` imports work under pytest |
| `src/okxb/research/candle_data.py` | Modify | Add `12H`/`1D`/`1W` bars to `BAR_MS`; add `fetch_oi_history()` |
| `src/okxb/research/deribit_data.py` | Create | Deribit public REST: DVOL history + per-strike option summary → daily IV/skew/term/OI/gamma frame |
| `src/okxb/research/onchain_data.py` | Create | Coin Metrics community daily on-chain metrics, with `(asof_ts, value)` and publish lag |
| `src/okxb/research/macro_data.py` | Create | FRED daily macro (VIX/broad-USD/yields) with publish lag |
| `src/okxb/research/daily_panel.py` | Create | `FeatureSpec`, `pit_asof_join()`, `daily_grid()`, `build_daily_panel()` — the point-in-time spine |
| `tests/research/test_daily_bars.py` | Create | Daily-grid + bar-ms correctness |
| `tests/research/test_pit_join.py` | Create | As-of join obeys publish lag (no look-ahead) |
| `tests/research/test_leakage.py` | Create | Panel-level leakage guard + daily-feature-into-sub-1d-horizon guard |
| `scripts/fetch_daily_data.py` | Create | Runner: pull 4 sources for the universe, build panel, write data-quality report |

---

## Task 1: Daily/weekly bars + UTC daily grid helper

**Files:**
- Modify: `src/okxb/research/candle_data.py` (the `BAR_MS` dict near line 25)
- Create: `src/okxb/research/daily_panel.py` (start it here with `daily_grid()`)
- Create: `conftest.py` (repo root)
- Test: `tests/research/test_daily_bars.py`

- [ ] **Step 1: Create the repo-root pytest path shim**

Create `conftest.py` at the repository root:

```python
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
```

- [ ] **Step 2: Write the failing test**

Create `tests/research/test_daily_bars.py`:

```python
from okxb.research import candle_data as cd
from okxb.research.daily_panel import daily_grid


def test_daily_and_weekly_bars_registered():
    assert cd.BAR_MS["1D"] == 86_400_000
    assert cd.BAR_MS["12H"] == 43_200_000
    assert cd.BAR_MS["1W"] == 604_800_000


def test_daily_grid_is_contiguous_utc_midnights():
    # 2024-01-01..2024-01-04 inclusive (UTC midnights, ms)
    start = 1_704_067_200_000  # 2024-01-01T00:00:00Z
    end = start + 3 * 86_400_000
    grid = daily_grid(start, end)
    assert list(grid) == [start + i * 86_400_000 for i in range(4)]
    assert (grid % 86_400_000 == 0).all()  # every point is a UTC midnight
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/research/test_daily_bars.py -v`
Expected: FAIL — `KeyError: '1D'` (or ImportError for `daily_grid`).

- [ ] **Step 4: Add daily/weekly bars to `BAR_MS`**

In `src/okxb/research/candle_data.py`, extend the `BAR_MS` dict to include:

```python
BAR_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1H": 3_600_000, "2H": 7_200_000, "4H": 14_400_000,
    "6H": 21_600_000, "12H": 43_200_000,
    "1D": 86_400_000, "1W": 604_800_000,
}
```

- [ ] **Step 5: Implement `daily_grid()` in the new `daily_panel.py`**

Create `src/okxb/research/daily_panel.py`:

```python
"""Point-in-time daily panel assembly for the v2 longer-horizon model.

Every external value becomes visible only at (period_close + publish_lag).
Daily-cadence features are forbidden from sub-1-day label horizons.
This module is the single place that enforces no-look-ahead.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DAY_MS = 86_400_000


def daily_grid(start_ms: int, end_ms: int) -> np.ndarray:
    """Contiguous UTC-midnight grid [start_ms, end_ms] inclusive, both snapped down to midnight."""
    s = (int(start_ms) // DAY_MS) * DAY_MS
    e = (int(end_ms) // DAY_MS) * DAY_MS
    n = (e - s) // DAY_MS + 1
    return s + np.arange(max(n, 0), dtype=np.int64) * DAY_MS
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/research/test_daily_bars.py -v`
Expected: PASS (2 passed).

- [ ] **Step 7: Commit**

```bash
git add conftest.py tests/research/test_daily_bars.py src/okxb/research/candle_data.py src/okxb/research/daily_panel.py
git commit -m "feat(data): add daily/weekly bars and UTC daily grid helper"
```

---

## Task 2: Point-in-time as-of join (the spine)

**Files:**
- Modify: `src/okxb/research/daily_panel.py`
- Test: `tests/research/test_pit_join.py`

- [ ] **Step 1: Write the failing test**

Create `tests/research/test_pit_join.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/research/test_pit_join.py -v`
Expected: FAIL — `ImportError: cannot import name 'pit_asof_join'`.

- [ ] **Step 3: Implement `pit_asof_join()`**

Append to `src/okxb/research/daily_panel.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/research/test_pit_join.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add tests/research/test_pit_join.py src/okxb/research/daily_panel.py
git commit -m "feat(data): point-in-time as-of join with publish lag"
```

---

## Task 3: FeatureSpec + panel builder + leakage guard (the crown jewel)

**Files:**
- Modify: `src/okxb/research/daily_panel.py`
- Test: `tests/research/test_leakage.py`

- [ ] **Step 1: Write the failing test**

Create `tests/research/test_leakage.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/research/test_leakage.py -v`
Expected: FAIL — `ImportError` for `FeatureSpec`/`build_daily_panel`/`assert_no_leakage`.

- [ ] **Step 3: Implement `FeatureSpec`, `build_daily_panel`, `assert_no_leakage`**

Append to `src/okxb/research/daily_panel.py`:

```python
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
    (min_horizon_min > horizon_min) — daily features never feed sub-1d labels.
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/research/test_leakage.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest tests/research/ -v`
Expected: PASS (8 passed total).

- [ ] **Step 6: Commit**

```bash
git add tests/research/test_leakage.py src/okxb/research/daily_panel.py
git commit -m "feat(data): FeatureSpec + point-in-time panel builder + leakage guard"
```

---

## Task 4: OKX open-interest history fetcher

**Files:**
- Modify: `src/okxb/research/candle_data.py` (add `fetch_oi_history()`)
- Test: `tests/research/test_oi_history.py`

- [ ] **Step 1: Confirm the live endpoint (probe before coding)**

Run this one-off probe to confirm the exact OKX OI-history path/params and response shape:

```bash
python - <<'PY'
import httpx, json
# OKX historical open interest (contracts). Confirm field names from the live payload.
r = httpx.get("https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-history",
              params={"instId": "BTC-USDT-SWAP", "period": "1D", "limit": "5"}, timeout=15)
print(r.status_code); print(json.dumps(r.json(), indent=2)[:1200])
PY
```

Expected: `code == "0"` and `data` rows of `[ts, oi, oiCcy, oiUsd]` (newest first). **If the path or field order differs from this, adjust the code in Step 3 to match the live payload — do not code against an assumed shape.**

- [ ] **Step 2: Write the failing test (pure parser, no network)**

Create `tests/research/test_oi_history.py`:

```python
from okxb.research.candle_data import parse_oi_rows


def test_parse_oi_rows_sorts_ascending_and_floats():
    # OKX returns newest-first [ts, oi, oiCcy, oiUsd]
    raw = [["1700000172000", "5000", "50", "1.0e9"],
           ["1700000086400000", "5100", "51", "1.1e9"]]
    df = parse_oi_rows(raw)
    assert list(df.columns) == ["ts", "oi", "oi_usd"]
    assert df["ts"].is_monotonic_increasing
    assert df["oi"].iloc[0] == 5000.0
```

- [ ] **Step 3: Implement `parse_oi_rows()` + `fetch_oi_history()`**

Add to `src/okxb/research/candle_data.py` (mirror `fetch_funding_series` paging/host-failover):

```python
def parse_oi_rows(rows: list) -> "pd.DataFrame":
    """Parse OKX open-interest-history rows [ts, oi, oiCcy, oiUsd] -> ascending df(ts, oi, oi_usd)."""
    out = []
    for r in rows:
        try:
            out.append((int(r[0]), float(r[1]), float(r[3]) if len(r) > 3 else float("nan")))
        except (TypeError, ValueError, IndexError):
            continue
    df = pd.DataFrame(sorted(set(out)), columns=["ts", "oi", "oi_usd"])
    df["ts"] = df["ts"].astype("int64")
    return df


def fetch_oi_history(inst_id: str, period: str, days: float, *,
                     host: Optional[str] = None, log: Callable[[str], None] = _log) -> "pd.DataFrame":
    """Open-interest history for a perp. period in OKX codes (e.g. '1H','1D'). Ascending df."""
    now_ms = int(time.time() * 1000)
    start = now_ms - int(days * 86_400_000)
    hosts = [host] if host else list(HOSTS)
    rows: list = []
    chosen: Optional[str] = None
    after: Optional[int] = None
    with httpx.Client(timeout=15.0, headers={"User-Agent": "okxb-research/1"}) as cli:
        for _ in range(200):
            data = None
            for h in (hosts if chosen is None else [chosen]):
                try:
                    params = {"instId": inst_id, "period": period, "limit": "100"}
                    if after is not None:
                        params["after"] = str(after)
                    r = cli.get(h + "/api/v5/rubik/stat/contracts/open-interest-history", params=params)
                    j = r.json()
                    if j.get("code") == "0":
                        data = j.get("data", []); chosen = h; break
                except Exception:  # noqa: BLE001
                    continue
            if not data:
                break
            rows.extend(data)
            oldest = min(int(x[0]) for x in data)
            if oldest <= start:
                break
            after = oldest
            time.sleep(0.1)
    df = parse_oi_rows(rows)
    return df[df["ts"] >= start].reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/research/test_oi_history.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Live smoke check**

Run: `python -c "from okxb.research.candle_data import fetch_oi_history; df=fetch_oi_history('BTC-USDT-SWAP','1D',30); print(len(df), df.tail(2).to_dict('records'))"`
Expected: non-zero row count, ascending `ts`, plausible `oi`/`oi_usd`.

- [ ] **Step 6: Commit**

```bash
git add tests/research/test_oi_history.py src/okxb/research/candle_data.py
git commit -m "feat(data): OKX open-interest history fetcher"
```

---

## Task 5: Deribit options fetcher (DVOL + smile/skew/term/OI)

**Files:**
- Create: `src/okxb/research/deribit_data.py`
- Test: `tests/research/test_deribit_data.py`

- [ ] **Step 1: Probe the live endpoints**

```bash
python - <<'PY'
import httpx, json
b="https://www.deribit.com/api/v2/public/"
v=httpx.get(b+"get_volatility_index_data",
            params={"currency":"BTC","start_timestamp":1700000000000,
                    "end_timestamp":1700600000000,"resolution":"43200"},timeout=15).json()
print("DVOL keys:", list(v.get("result",{}).keys()))
s=httpx.get(b+"get_book_summary_by_currency",params={"currency":"BTC","kind":"option"},timeout=15).json()
print("summary n:", len(s.get("result",[])), "sample fields:", list((s.get("result") or [{}])[0].keys()))
PY
```

Expected: DVOL result has `data` = rows `[ts, open, high, low, close]`; summary items have `instrument_name`, `mark_iv`, `open_interest`, `underlying_price` (and a `mid_price`/greeks set). **Match the parser below to the live field names.**

- [ ] **Step 2: Write the failing test (pure transforms)**

Create `tests/research/test_deribit_data.py`:

```python
from okxb.research.deribit_data import parse_dvol, expiry_days_from_name


def test_parse_dvol_to_ascending_close():
    res = {"data": [[1700000000000, 50, 55, 49, 52], [1700043200000, 52, 58, 51, 57]]}
    df = parse_dvol(res)
    assert list(df.columns) == ["ts", "dvol"]
    assert df["ts"].is_monotonic_increasing
    assert df["dvol"].iloc[-1] == 57.0


def test_expiry_days_from_instrument_name():
    # Deribit option name: BTC-27JUN25-100000-C
    assert expiry_days_from_name("BTC-27JUN25-100000-C", asof_ms=_ms("2025-06-20")) == 7


def _ms(d):
    import pandas as pd
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)
```

- [ ] **Step 3: Implement `deribit_data.py`**

Create `src/okxb/research/deribit_data.py`:

```python
"""Deribit public (no-auth) options data for daily IV/skew/term/OI features.

Offline research fetcher: httpx + CSV cache, UTC-ms timestamps, no keys.
Daily snapshot job persists RAW payloads with the snapshot ts; features are
computed point-in-time downstream (see daily_panel.py).
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
    """get_volatility_index_data result -> ascending df(ts, dvol=close)."""
    rows = result.get("data", []) if result else []
    out = [(int(r[0]), float(r[4])) for r in rows if len(r) >= 5]
    df = pd.DataFrame(sorted(set(out)), columns=["ts", "dvol"])
    if len(df):
        df["ts"] = df["ts"].astype("int64")
    return df


def expiry_days_from_name(instrument_name: str, asof_ms: int) -> int:
    """Whole days from asof to the option expiry encoded in a Deribit name like BTC-27JUN25-100000-C."""
    tok = instrument_name.split("-")[1]  # e.g. 27JUN25
    day = int(tok[:2]); mon = _MONTHS[tok[2:5]]; yr = 2000 + int(tok[5:])
    exp = pd.Timestamp(year=yr, month=mon, day=day, hour=8, tz="UTC")  # Deribit expiry 08:00 UTC
    return int((exp.value // 1_000_000 - asof_ms) // 86_400_000)


def fetch_dvol(currency: str, days: float, *, resolution: str = "43200",
               log: Callable[[str], None] = _log) -> pd.DataFrame:
    """DVOL history (resolution seconds as string; 43200 = 12h). Ascending df(ts, dvol)."""
    now = int(time.time() * 1000)
    start = now - int(days * 86_400_000)
    with httpx.Client(timeout=15.0, headers={"User-Agent": "okxb-research/1"}) as cli:
        r = cli.get(BASE + "get_volatility_index_data",
                    params={"currency": currency, "start_timestamp": start,
                            "end_timestamp": now, "resolution": resolution})
        return parse_dvol(r.json().get("result", {}))


def fetch_option_summary(currency: str, *, log: Callable[[str], None] = _log) -> pd.DataFrame:
    """Current per-strike option summary snapshot. Columns include
    instrument_name, mark_iv, open_interest, underlying_price, mid_price."""
    with httpx.Client(timeout=15.0, headers={"User-Agent": "okxb-research/1"}) as cli:
        r = cli.get(BASE + "get_book_summary_by_currency",
                    params={"currency": currency, "kind": "option"})
        rows = r.json().get("result", [])
    keep = ["instrument_name", "mark_iv", "open_interest", "underlying_price", "mid_price"]
    df = pd.DataFrame(rows)
    for k in keep:
        if k not in df.columns:
            df[k] = float("nan")
    return df[keep]


def snapshot_daily(currency: str, root: Path, snapshot_ms: int) -> Path:
    """Persist a raw option-summary snapshot tagged with snapshot_ms (for point-in-time features)."""
    root.mkdir(parents=True, exist_ok=True)
    df = fetch_option_summary(currency)
    df["snapshot_ts"] = int(snapshot_ms)
    f = root / f"deribit_{currency}_{int(snapshot_ms)}.csv"
    df.to_csv(f, index=False)
    return f
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/research/test_deribit_data.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Live smoke check**

Run: `python -c "from okxb.research.deribit_data import fetch_dvol, fetch_option_summary; print(len(fetch_dvol('BTC',10))); s=fetch_option_summary('BTC'); print(len(s), s['mark_iv'].notna().mean())"`
Expected: non-zero DVOL rows; hundreds of option rows; most `mark_iv` populated.

- [ ] **Step 6: Commit**

```bash
git add tests/research/test_deribit_data.py src/okxb/research/deribit_data.py
git commit -m "feat(data): Deribit public DVOL + option summary fetcher"
```

---

## Task 6: Coin Metrics community on-chain fetcher (point-in-time)

**Files:**
- Create: `src/okxb/research/onchain_data.py`
- Test: `tests/research/test_onchain_data.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Install the free client**

Run: `pip install coinmetrics-api-client` then add to `requirements.txt` under the data section:

```
coinmetrics-api-client>=2024.1.0   # community (no-key) daily on-chain metrics
```

- [ ] **Step 2: Probe available community metrics**

```bash
python - <<'PY'
from coinmetrics.api_client import CoinMetricsClient
c = CoinMetricsClient()
df = c.get_asset_metrics(assets="btc", metrics=["AdrActCnt","FlowInExNtv","FlowOutExNtv"],
                         frequency="1d", start_time="2024-01-01", end_time="2024-01-05").to_dataframe()
print(df.columns.tolist()); print(df.head().to_dict("records"))
PY
```

Expected: a daily frame with `asset`, `time`, and the requested metric columns. **If a metric name is rejected, drop it to the supported community set and update `DEFAULT_METRICS`.**

- [ ] **Step 3: Write the failing test (pure transform)**

Create `tests/research/test_onchain_data.py`:

```python
import pandas as pd
from okxb.research.onchain_data import to_tidy_ms


def test_to_tidy_ms_converts_time_to_ms_and_renames():
    raw = pd.DataFrame({"asset": ["btc", "btc"],
                        "time": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
                        "AdrActCnt": ["100", "110"]})
    out = to_tidy_ms(raw, "AdrActCnt")
    assert list(out.columns) == ["ts", "value"]
    assert out["ts"].iloc[0] == int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)
    assert out["value"].iloc[1] == 110.0
    assert out["ts"].is_monotonic_increasing
```

- [ ] **Step 4: Implement `onchain_data.py`**

Create `src/okxb/research/onchain_data.py`:

```python
"""Coin Metrics community (no-key) daily on-chain metrics.

Daily cadence + retroactive revision -> consume only with a publish lag
(see FeatureSpec.publish_lag_ms; default >= 1 day) and only for >=1d horizons.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

DEFAULT_METRICS = ["AdrActCnt", "FlowInExNtv", "FlowOutExNtv", "TxCnt", "CapMVRVCur"]


def to_tidy_ms(raw: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Coin Metrics frame -> tidy df(ts int-ms, value float) ascending, for one metric."""
    df = raw[["time", metric]].copy()
    df["ts"] = (pd.to_datetime(df["time"], utc=True).astype("int64") // 1_000_000).astype("int64")
    df["value"] = pd.to_numeric(df[metric], errors="coerce")
    df = df[["ts", "value"]].dropna().sort_values("ts").reset_index(drop=True)
    return df


def fetch_onchain(asset: str, metrics: Optional[list[str]], days: float,
                  root: Path, *, force: bool = False) -> dict[str, pd.DataFrame]:
    """Return {metric: tidy df}. Cached per (asset, metric) CSV under root."""
    from coinmetrics.api_client import CoinMetricsClient
    metrics = metrics or DEFAULT_METRICS
    root.mkdir(parents=True, exist_ok=True)
    start = (pd.Timestamp.utcnow() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    out: dict[str, pd.DataFrame] = {}
    need = []
    for m in metrics:
        f = root / f"{asset}_{m}.csv"
        if f.exists() and not force:
            out[m] = pd.read_csv(f)
        else:
            need.append(m)
    if need:
        raw = CoinMetricsClient().get_asset_metrics(
            assets=asset, metrics=need, frequency="1d", start_time=start
        ).to_dataframe()
        for m in need:
            tidy = to_tidy_ms(raw, m)
            tidy.to_csv(root / f"{asset}_{m}.csv", index=False)
            out[m] = tidy
    return out
```

- [ ] **Step 5: Run tests + live smoke**

Run: `python -m pytest tests/research/test_onchain_data.py -v`
Expected: PASS (1 passed).
Run: `python -c "from pathlib import Path; from okxb.research.onchain_data import fetch_onchain; d=fetch_onchain('btc',None,30,Path('dist/daily/onchain')); print({k:len(v) for k,v in d.items()})"`
Expected: each metric has ~30 daily rows.

- [ ] **Step 6: Commit**

```bash
git add tests/research/test_onchain_data.py src/okxb/research/onchain_data.py requirements.txt
git commit -m "feat(data): Coin Metrics community on-chain fetcher (point-in-time)"
```

---

## Task 7: Macro fetcher (FRED daily, with publish lag)

**Files:**
- Create: `src/okxb/research/macro_data.py`
- Test: `tests/research/test_macro_data.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Install + probe**

Run: `pip install pandas-datareader` and add to `requirements.txt`:

```
pandas-datareader>=0.10.0   # FRED macro (VIX / broad USD / yields)
```

Probe: `python -c "from pandas_datareader import data as p; import datetime as d; print(p.DataReader('VIXCLS','fred', d.date(2024,1,1), d.date(2024,1,10)).tail())"`
Expected: a daily VIX series (NaN on holidays).

- [ ] **Step 2: Write the failing test (pure transform)**

Create `tests/research/test_macro_data.py`:

```python
import pandas as pd
from okxb.research.macro_data import fred_to_tidy_ms


def test_fred_to_tidy_ms_drops_nan_and_sorts():
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    raw = pd.DataFrame({"VIXCLS": [13.2, None, 13.5]}, index=idx)
    out = fred_to_tidy_ms(raw, "VIXCLS")
    assert list(out.columns) == ["ts", "value"]
    assert len(out) == 2  # NaN holiday dropped
    assert out["ts"].iloc[0] == int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)
```

- [ ] **Step 3: Implement `macro_data.py`**

Create `src/okxb/research/macro_data.py`:

```python
"""FRED daily macro series (VIX / broad-USD / yields) for regime conditioning.

Daily series publish AFTER the US close; treat as available next UTC day
(publish_lag_ms >= 1 day in the FeatureSpec). v1 uses latest values; ALFRED
vintages are a later point-in-time hardening step.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

SERIES = {"vix": "VIXCLS", "usd_broad": "DTWEXBGS", "ust10y": "DGS10", "ust2y": "DGS2"}


def fred_to_tidy_ms(raw: pd.DataFrame, code: str) -> pd.DataFrame:
    """FRED DataReader frame (DatetimeIndex) -> tidy df(ts int-ms, value) ascending, NaN dropped."""
    s = pd.to_numeric(raw[code], errors="coerce").dropna()
    ts = (pd.DatetimeIndex(s.index).tz_localize("UTC").astype("int64") // 1_000_000).astype("int64")
    return pd.DataFrame({"ts": ts, "value": s.to_numpy(dtype=float)}).sort_values("ts").reset_index(drop=True)


def fetch_macro(days: float, root: Path, *, force: bool = False) -> dict[str, pd.DataFrame]:
    """Return {name: tidy df} for SERIES. Cached per-name CSV under root."""
    from pandas_datareader import data as pdr
    root.mkdir(parents=True, exist_ok=True)
    start = (dt.datetime.utcnow() - dt.timedelta(days=days)).date()
    out: dict[str, pd.DataFrame] = {}
    for name, code in SERIES.items():
        f = root / f"macro_{name}.csv"
        if f.exists() and not force:
            out[name] = pd.read_csv(f)
            continue
        raw = pdr.DataReader(code, "fred", start, dt.date.today())
        tidy = fred_to_tidy_ms(raw, code)
        tidy.to_csv(f, index=False)
        out[name] = tidy
    return out
```

- [ ] **Step 4: Run tests + live smoke**

Run: `python -m pytest tests/research/test_macro_data.py -v`
Expected: PASS (1 passed).
Run: `python -c "from pathlib import Path; from okxb.research.macro_data import fetch_macro; d=fetch_macro(60,Path('dist/daily/macro')); print({k:len(v) for k,v in d.items()})"`
Expected: each series has ~40 business-day rows over 60 days.

- [ ] **Step 5: Commit**

```bash
git add tests/research/test_macro_data.py src/okxb/research/macro_data.py requirements.txt
git commit -m "feat(data): FRED macro fetcher (daily, publish-lagged)"
```

---

## Task 8: End-to-end runner + data-quality report + full leakage check

**Files:**
- Create: `scripts/fetch_daily_data.py`
- Test: `tests/research/test_panel_integration.py`

- [ ] **Step 1: Write the failing integration test (synthetic, offline)**

Create `tests/research/test_panel_integration.py`:

```python
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

    # at a 4h horizon, daily on-chain/macro are excluded
    panel_4h = build_daily_panel(frames, specs, horizon_min=240)
    assert "oc" not in panel_4h.columns and "vix" not in panel_4h.columns
```

- [ ] **Step 2: Run test to verify it fails, then passes**

Run: `python -m pytest tests/research/test_panel_integration.py -v`
Expected: FAIL only if `daily_panel` API drifted; otherwise it should PASS against Task 3 code. If it fails for a real reason, fix `daily_panel.py` and re-run until PASS.

- [ ] **Step 3: Write the runner**

Create `scripts/fetch_daily_data.py`:

```python
#!/usr/bin/env python
"""Pull all four v2 data sources for the universe, build a point-in-time daily
panel per horizon, and write a data-quality report. Research-only, no keys."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from okxb.research import candle_data as cd          # noqa: E402
from okxb.research import deribit_data as dd          # noqa: E402
from okxb.research import onchain_data as oc          # noqa: E402
from okxb.research import macro_data as md            # noqa: E402

DAILY = ROOT / "dist" / "daily"


def main() -> None:
    insts = ["BTC-USDT-SWAP", "ETH-USDT-SWAP"] + [f"{s}-USDT-SWAP" for s in ("SOL", "BNB", "XRP")]
    days = 365 * 3
    lines = ["data-quality report (P0 daily sources)", "=" * 48]

    perp = cd.fetch_universe(insts, "1D", days, DAILY / "candles")
    lines.append(f"OKX 1D candles: {len(perp)}/{len(insts)} instruments")
    for inst in perp:
        oi = cd.fetch_oi_history(inst, "1D", days)
        lines.append(f"  {inst}: candles={len(perp[inst])} oi_rows={len(oi)}")

    for ccy in ("BTC", "ETH"):
        dv = dd.fetch_dvol(ccy, days)
        lines.append(f"Deribit {ccy} DVOL rows: {len(dv)}")

    ocd = oc.fetch_onchain("btc", None, days, DAILY / "onchain")
    lines.append("on-chain (btc): " + ", ".join(f"{k}={len(v)}" for k, v in ocd.items()))

    mac = md.fetch_macro(days, DAILY / "macro")
    lines.append("macro: " + ", ".join(f"{k}={len(v)}" for k, v in mac.items()))

    report = "\n".join(lines)
    (DAILY).mkdir(parents=True, exist_ok=True)
    (DAILY / "data_quality_report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the full suite + the live runner**

Run: `python -m pytest tests/research/ -v`
Expected: PASS (all tests green).
Run: `python scripts/fetch_daily_data.py`
Expected: a printed + saved report showing non-zero row counts for OKX candles/OI, Deribit DVOL, on-chain metrics, and macro series. Eyeball that spans cover ~3 years.

- [ ] **Step 5: Commit**

```bash
git add tests/research/test_panel_integration.py scripts/fetch_daily_data.py
git commit -m "feat(data): daily-data runner + data-quality report + integration test"
```

---

## Self-Review (completed by plan author)

- **Spec coverage:** P0 items from the v2.1 spec §2/§13 are all covered — daily/weekly bars (Task 1), OKX OI history (Task 4), Deribit options (Task 5), Coin Metrics on-chain (Task 6), macro (Task 7), point-in-time as-of join + daily-into-sub-1d guard + leakage test (Tasks 2–3, 8), data-quality report (Task 8). Live L2 recording via `cryptofeed` is intentionally deferred to P4 (execution-realism), not P0 — noted in the spec.
- **Placeholder scan:** No TBD/TODO. Every code step shows complete code. Fetcher tasks include a live-probe step because the external response shape MUST be confirmed before trusting field names — the probe is a real step, not a placeholder, and the code is shown to adjust.
- **Type consistency:** `FeatureSpec(name, source, publish_lag_ms, min_horizon_min)`, `pit_asof_join(decision_ts, series_close_ms, series_val, publish_lag_ms)`, `build_daily_panel(frames, specs, horizon_min)`, `assert_no_leakage(panel, specs, frames)`, and tidy `df(ts, value)` for every source are used consistently across Tasks 2–8.
- **Note for executor:** Live-probe steps depend on public endpoints; if a field name differs from the shown parser, adjust the parser to the live payload (the tests assert on the parser, not the network, so they stay deterministic).
