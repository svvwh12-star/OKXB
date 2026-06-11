"""Regime-filter experiment (time-boxed final test of the unexhausted W/L lever).

Does conditioning a thin base signal (short-horizon cross-sectional reversal) on an
orthogonal daily REGIME (DVOL / VRP / funding extreme) concentrate it enough to clear
cost in some regime? Direction prediction is closed; this tests *selectivity*, not direction.

All regimes are point-in-time (trailing-median / sign thresholds, no look-ahead); evaluation
is on a held-out tail. Pass criterion is strict (held-out net-after-taker>0, NW-t>2, beats
unconditional) precisely because regime-subsetting is a multiple-testing minefield.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import candle_research as cr
from .daily_panel import pit_asof_join


def _log(m: str) -> None:
    print(m, flush=True)


def build_reversal_panel(dfs: dict[str, pd.DataFrame], bar: str,
                         k_lookback: int, h_fwd: int) -> pd.DataFrame:
    """Cross-sectional short-horizon reversal. signal = market-neutral (-recent return);
    fwd = h_fwd-bar forward return. Returns long panel(ts, inst, signal, fwd)."""
    parts = []
    for inst, df in dfs.items():
        d = df.sort_values("ts")
        c = pd.Series(d["c"].to_numpy(float), index=d["ts"].to_numpy(np.int64))
        rev = -(c / c.shift(k_lookback) - 1.0)
        fwd = c.shift(-h_fwd) / c - 1.0
        parts.append(pd.DataFrame({"ts": c.index.values, "inst": inst,
                                   "rev_raw": rev.values, "fwd": fwd.values}))
    panel = pd.concat(parts, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    g = panel.groupby("ts")["rev_raw"]
    panel["signal"] = panel["rev_raw"] - g.transform("mean")   # cross-sectionally demeaned
    return panel.dropna(subset=["signal", "fwd"]).reset_index(drop=True)


def attach_regime(panel: pd.DataFrame, daily: pd.DataFrame, name: str, *,
                  publish_lag_ms: int = 0, mode: str = "above_median",
                  min_periods: int = 20) -> pd.DataFrame:
    """Attach a binary market-wide regime column from a daily series (ts,value), point-in-time.
    mode='above_median' (vs expanding trailing median) | 'positive' (value>0) | 'abs_extreme' (|z|>1)."""
    ts = np.sort(panel["ts"].unique())
    val = pit_asof_join(ts, daily["ts"].to_numpy(np.int64), daily["value"].to_numpy(float), publish_lag_ms)
    s = pd.Series(val)
    if mode == "positive":
        reg = (s > 0).astype(float)
    elif mode == "abs_extreme":
        z = (s - s.expanding(min_periods=min_periods).mean()) / s.expanding(min_periods=min_periods).std()
        reg = (z.abs() > 1.0).astype(float)
    else:  # above_median (trailing, look-ahead-safe)
        reg = (s > s.expanding(min_periods=min_periods).median()).astype(float)
    reg = reg.to_numpy()
    reg[~np.isfinite(val)] = np.nan
    panel[name] = panel["ts"].map(dict(zip(ts, reg)))
    return panel


def _net(sub: pd.DataFrame, cost_bps: float) -> Optional[dict]:
    """Portfolio (per-ts mean of sign(signal)*fwd) net after cost, with Newey-West t."""
    signed = np.sign(sub["signal"].to_numpy()) * sub["fwd"].to_numpy()
    port = pd.Series(signed, index=sub["ts"].to_numpy()).groupby(level=0).mean().to_numpy()
    if len(port) < 8:
        return None
    return cr._stats_from_gross(port, cost_bps, 1)


def eval_regimes(panel: pd.DataFrame, regime_cols: list[str], *,
                 maker_bps: float = 4.0, taker_bps: float = 10.0,
                 holdout_frac: float = 0.3, min_n: int = 200) -> list[dict]:
    """Held-out net edge of the reversal signal: unconditional + each regime side."""
    ts_sorted = np.sort(panel["ts"].unique())
    cut = ts_sorted[int(len(ts_sorted) * (1 - holdout_frac))]
    test = panel[panel["ts"] > cut]
    out = []
    segments = [("ALL", test)]
    for rc in regime_cols:
        segments.append((f"{rc}=1", test[test[rc] == 1]))
        segments.append((f"{rc}=0", test[test[rc] == 0]))
    for label, sub in segments:
        if len(sub) < min_n:
            out.append({"regime": label, "n": len(sub), "net_maker": None})
            continue
        m = _net(sub, maker_bps)
        t = _net(sub, taker_bps)
        out.append({"regime": label, "n": len(sub),
                    "net_maker": m["net_bps"] if m else None, "t_maker": m["nw_t"] if m else None,
                    "net_taker": t["net_bps"] if t else None, "t_taker": t["nw_t"] if t else None})
    return out


def verdict(rows: list[dict]) -> tuple[bool, str]:
    """Pass only if some regime beats unconditional AND is held-out net-taker>0 with NW-t>2."""
    base = next((r for r in rows if r["regime"] == "ALL" and r.get("net_taker") is not None), None)
    base_net = base["net_taker"] if base else -1e9
    winners = [r for r in rows if r["regime"] != "ALL" and r.get("net_taker") is not None
               and r["net_taker"] > 0 and (r.get("t_taker") or 0) > 2 and r["net_taker"] > base_net]
    if winners:
        w = max(winners, key=lambda r: r["t_taker"])
        return True, (f"CANDIDATE: regime {w['regime']} held-out net_taker={w['net_taker']:+.1f}bps "
                      f"(t={w['t_taker']:+.1f}, n={w['n']}) beats unconditional → run full DSR/PBO + shadow before believing.")
    return False, "NO EDGE: no regime makes the reversal net-positive after taker on held-out (W/L lever exhausted)."
