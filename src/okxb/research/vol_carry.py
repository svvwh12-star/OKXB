"""Variance-risk-premium (VRP) harvest experiment — defined-risk short-vol on BTC.

The one genuinely-new avenue with a REAL underlying edge: implied vol (DVOL) trades
rich to realized vol on average. Harvest it by rolling weekly DEFINED-RISK iron condors
(sell a strangle, buy further wings so the per-trade loss is bounded), priced off DVOL,
settled against the realized BTC move, net of OKX option fees. Honest question: does it
survive realistic fees + the crash weeks for a small account?

Simplifications (stated honestly): DVOL (~30d ATM IV index) is used as the option IV
(real weekly IV differs by term structure); no rate (crypto r=0); European settle at hold.
"""
from __future__ import annotations

from math import erf, log, sqrt
from typing import Callable

import numpy as np
import pandas as pd


def _log(m: str) -> None:
    print(m, flush=True)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_price(S: float, K: float, T: float, sigma: float, is_call: bool) -> float:
    """Black-Scholes price, no rate (crypto). sigma decimal annualized, T years."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    d1 = (log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if is_call:
        return S * _norm_cdf(d1) - K * _norm_cdf(d2)
    return K * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def _leg_fee(S0: float, premium: float, fee_rate: float, fee_cap_frac: float) -> float:
    """OKX option fee per leg: rate * underlying, capped at fee_cap_frac of the option premium."""
    return min(fee_rate * S0, fee_cap_frac * abs(premium))


def iron_condor_pnl(S0: float, S_T: float, iv: float, T: float,
                    short_d: float, wing_d: float,
                    fee_rate: float = 0.0003, fee_cap_frac: float = 0.125,
                    spread_frac: float = 0.0) -> float:
    """P&L (in underlying units) of a weekly defined-risk iron condor sold at entry IV.

    Sell put/call at +-short_d sigma; buy wings at +-wing_d sigma (wing_d>short_d => bounded loss).
    Collect entry credit, settle intrinsics at S_T, pay 4 leg fees + spread_frac of each leg's
    premium (half bid-ask paid crossing each of the 4 legs). wing_d huge ~= naked strangle.
    """
    m = iv * sqrt(T)
    Kp = S0 * np.exp(-short_d * m); Kc = S0 * np.exp(+short_d * m)        # short strikes
    Kpw = S0 * np.exp(-wing_d * m); Kcw = S0 * np.exp(+wing_d * m)        # long wings (further OTM)
    p_sp = bs_price(S0, Kp, T, iv, False); p_sc = bs_price(S0, Kc, T, iv, True)
    p_lp = bs_price(S0, Kpw, T, iv, False); p_lc = bs_price(S0, Kcw, T, iv, True)
    credit = (p_sp + p_sc) - (p_lp + p_lc)
    fees = (_leg_fee(S0, p_sp, fee_rate, fee_cap_frac) + _leg_fee(S0, p_sc, fee_rate, fee_cap_frac)
            + _leg_fee(S0, p_lp, fee_rate, fee_cap_frac) + _leg_fee(S0, p_lc, fee_rate, fee_cap_frac))
    spread = spread_frac * (abs(p_sp) + abs(p_sc) + abs(p_lp) + abs(p_lc))
    settle = (-max(Kp - S_T, 0.0) - max(S_T - Kc, 0.0)
              + max(Kpw - S_T, 0.0) + max(S_T - Kcw, 0.0))
    return credit + settle - fees - spread


def run_vol_carry(btc_df: pd.DataFrame, dvol_df: pd.DataFrame, *,
                  short_d: float = 1.0, wing_d: float = 2.0, hold_days: int = 7,
                  fee_rate: float = 0.0003, fee_cap_frac: float = 0.125,
                  spread_frac: float = 0.0,
                  log: Callable[[str], None] = _log) -> dict:
    """Roll weekly defined-risk iron condors. Returns net-of-fee stats (return on notional)."""
    b = btc_df.drop_duplicates("ts").sort_values("ts")
    c = pd.Series(b["c"].to_numpy(float), index=b["ts"].to_numpy(np.int64))
    dv = dvol_df.drop_duplicates("ts").sort_values("ts")
    dvser = pd.Series(dv["dvol"].to_numpy(float), index=dv["ts"].to_numpy(np.int64)).reindex(c.index, method="ffill")
    ts = c.index.to_numpy()
    step_ms = hold_days * 86_400_000
    rets, worst = [], None
    i = 0
    while i < len(ts):
        t0 = ts[i]
        j = np.searchsorted(ts, t0 + step_ms)
        if j >= len(ts):
            break
        S0, S_T, iv = float(c.iloc[i]), float(c.iloc[j]), float(dvser.iloc[i]) / 100.0
        if not (np.isfinite(iv) and iv > 0 and S0 > 0 and S_T > 0):
            i = j
            continue
        pnl = iron_condor_pnl(S0, S_T, iv, hold_days / 365.0, short_d, wing_d,
                              fee_rate, fee_cap_frac, spread_frac)
        r = pnl / S0
        rets.append(r)
        if worst is None or r < worst:
            worst = r
        i = j
    if not rets:
        return {"error": "no trades"}
    a = np.array(rets)
    eq = np.cumprod(1.0 + a)
    dd = float((eq / np.maximum.accumulate(eq) - 1.0).min())
    wins = a[a > 0]; losses = a[a < 0]
    pf = float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() != 0 else float("inf")
    return {
        "n_trades": len(a), "hold_days": hold_days, "short_d": short_d, "wing_d": wing_d,
        "mean_bps": float(a.mean() * 1e4), "win_rate": float((a > 0).mean()),
        "total_return": float(eq[-1] - 1.0), "pf": pf, "max_dd": dd,
        "worst_trade_bps": float(worst * 1e4), "ann_sharpe": float(a.mean() / (a.std() + 1e-12) * np.sqrt(365.0 / hold_days)),
    }
