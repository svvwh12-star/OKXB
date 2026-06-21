"""Stage-1 daily orthogonal-feature net-edge study (the MVP that answers 'is there edge').

Fuses the new orthogonal daily signals onto the price/vol base from feature_lab.inst_bank,
strictly point-in-time, then drives the existing audited net-edge gate at daily horizons.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import feature_lab as fl
from . import pro_model_workflow as pmw
from .daily_panel import pit_asof_join

DAY_MS = 86_400_000
# RV-6/RV-8: 外部源 PIT 发布延迟集中声明 (不再散落魔法数 0)。
FUNDING_PUBLISH_LAG_MS = 0          # funding 结算时刻即已实现, ts=结算时间 -> 0 正确
# 日频隐含波动: 保守默认按"周期收盘后才可得"对齐 (跨日前视的安全侧;
# 若确认 deribit DVOL 的 ts 即为当日收盘时刻, 可在调用处把 dvol_lag_ms 设回 0)。
DVOL_PUBLISH_LAG_MS = DAY_MS


def _log(m: str) -> None:
    print(m, flush=True)


def merge_pit(panel_ts: np.ndarray, src: pd.DataFrame, publish_lag_ms: int) -> np.ndarray:
    """Point-in-time as-of merge of a tidy df(ts, value) onto panel decision timestamps."""
    if src is None or len(src) == 0:
        return np.full(len(panel_ts), np.nan)
    return pit_asof_join(np.asarray(panel_ts, dtype=np.int64),
                         src["ts"].to_numpy(dtype=np.int64),
                         src["value"].to_numpy(dtype=float), publish_lag_ms)


def _zscore(x: np.ndarray, win: int) -> np.ndarray:
    s = pd.Series(x)
    return ((s - s.rolling(win, min_periods=max(3, win // 3)).mean())
            / s.rolling(win, min_periods=max(3, win // 3)).std().replace(0, np.nan)).to_numpy()


def build_daily_orthogonal_bank(
    perp_dfs: dict[str, pd.DataFrame],
    H_min: int,
    *,
    spot_dfs: Optional[dict[str, pd.DataFrame]] = None,
    funding_dfs: Optional[dict[str, pd.DataFrame]] = None,
    dvol_by_ccy: Optional[dict[str, pd.DataFrame]] = None,
    onchain_by_asset: Optional[dict[str, dict[str, pd.DataFrame]]] = None,
    onchain_lag_ms: int = DAY_MS,
    dvol_lag_ms: int = DVOL_PUBLISH_LAG_MS,
) -> tuple[pd.DataFrame, int]:
    """Daily bank = inst_bank price/vol base + basis/funding + DVOL/VRP + on-chain + cross-asset.

    perp_dfs: {inst: 1D candle df}. dvol_by_ccy: {'BTC': df(ts,dvol)}. onchain_by_asset:
    {'btc': {metric: df(ts,value)}}. All non-price sources merged point-in-time (no look-ahead).
    """
    bar_min = 1440
    bar_ms = DAY_MS
    spot_dfs = spot_dfs or {}
    funding_dfs = funding_dfs or {}
    dvol_by_ccy = dvol_by_ccy or {}
    onchain_by_asset = onchain_by_asset or {}
    hb = max(1, round(H_min / bar_min))

    def _ccy(inst: str) -> str:
        return inst.split("-")[0]

    def _close_on(df: pd.DataFrame, ts: np.ndarray) -> pd.Series:
        s = df.sort_values("ts")
        return pd.Series(s["c"].to_numpy(float), index=s["ts"].to_numpy(np.int64)).reindex(ts, method="ffill")

    parts: list[pd.DataFrame] = []
    btc = perp_dfs.get("BTC-USDT-SWAP")
    eth = perp_dfs.get("ETH-USDT-SWAP")
    for inst, df in perp_dfs.items():
        p = fl.inst_bank(df, H_min, bar_min, bar_ms)   # ts + ~50 price/vol feats + fwd + y
        p["inst"] = inst
        ts = p["ts"].to_numpy(dtype=np.int64)
        pc = _close_on(df, ts)

        # --- basis (perp vs spot) ---
        spot = spot_dfs.get(inst)
        if spot is not None and len(spot):
            sc = _close_on(spot, ts)
            basis = (pc.to_numpy() / sc.to_numpy() - 1.0)
            p["basis"] = basis
            p["basis_z"] = _zscore(basis, max(6, hb * 4))
        else:
            p["basis"] = np.nan
            p["basis_z"] = np.nan

        # --- funding (already-realized; available at settlement) ---
        f = funding_dfs.get(inst)
        if f is not None and len(f):
            fd = f.sort_values("ts").rename(columns={"funding": "value"})[["ts", "value"]]
            last = merge_pit(ts, fd, FUNDING_PUBLISH_LAG_MS)
            p["funding_last"] = last
            p["funding_z"] = _zscore(last, max(6, hb * 3))
            p["funding_cum_h"] = pd.Series(last).rolling(max(1, hb)).sum().to_numpy()
        else:
            p["funding_last"] = np.nan
            p["funding_z"] = np.nan
            p["funding_cum_h"] = np.nan

        # --- DVOL / VRP (implied vol; daily index) ---
        dv = dvol_by_ccy.get(_ccy(inst))
        if dv is not None and len(dv):
            dvol = merge_pit(ts, dv.rename(columns={"dvol": "value"})[["ts", "value"]], dvol_lag_ms)
            p["dvol"] = dvol
            p["dvol_chg"] = pd.Series(dvol).diff(max(1, hb)).to_numpy()
            logret = np.log(pc).diff()
            rv = (logret.rolling(max(5, hb)).std() * np.sqrt(365.0) * 100.0).to_numpy()
            p["vrp"] = (dvol ** 2 - rv ** 2)
        else:
            p["dvol"] = np.nan
            p["dvol_chg"] = np.nan
            p["vrp"] = np.nan

        # --- on-chain (daily, publish-lagged; only valid for >=1d horizons) ---
        oc = onchain_by_asset.get(_ccy(inst).lower())
        if oc and H_min >= 1440:
            for metric, mdf in oc.items():
                v = merge_pit(ts, mdf[["ts", "value"]], onchain_lag_ms)
                p[f"oc_{metric}_z"] = _zscore(v, 30)
                p[f"oc_{metric}_chg"] = pd.Series(v).pct_change(max(1, hb)).replace([np.inf, -np.inf], np.nan).to_numpy()

        # --- cross-asset leads ---
        for name, s in (("btc", btc), ("eth", eth)):
            if s is None:
                continue
            sc = _close_on(s, ts)
            for mn in (1, hb, hb * 2):
                p[f"{name}_ret_{mn}"] = (sc / sc.shift(mn) - 1.0).to_numpy()
        parts.append(p)

    panel = pd.concat(parts, ignore_index=True)
    # cross-sectional point-in-time context (same timestamp only)
    for col in ("roc", "basis", "funding_last", "vol_z", "dvol"):
        if col in panel.columns:
            g = panel.groupby("ts")[col]
            panel[f"xs_{col}"] = ((panel[col] - g.transform("mean")) / g.transform("std").replace(0, np.nan))
    return panel, bar_ms


def run_daily(
    perp_dfs: dict[str, pd.DataFrame],
    *,
    spot_dfs: Optional[dict[str, pd.DataFrame]] = None,
    funding_dfs: Optional[dict[str, pd.DataFrame]] = None,
    dvol_by_ccy: Optional[dict[str, pd.DataFrame]] = None,
    onchain_by_asset: Optional[dict[str, dict[str, pd.DataFrame]]] = None,
    horizons_min: tuple[int, ...] = (1440, 2880, 4320),
    n_folds: int = 4,
    k_sel: int = 22,
    costs: pmw.WorkflowCosts = pmw.WorkflowCosts(),
    log: Callable[[str], None] = _log,
) -> dict:
    """Daily-horizon net-edge study on the orthogonal bank. Reuses the audited
    purged WF + net-edge gate + sizing from pro_model_workflow."""
    mode = "single_asset" if len(perp_dfs) == 1 else "cross_sectional"
    min_cell_ts = 30 if mode == "single_asset" else 20
    by_h: dict[int, dict] = {}
    for H in horizons_min:
        log(f"H={H}min: build daily orthogonal bank + fold-local selection + model zoo")
        panel, bar_ms = build_daily_orthogonal_bank(
            perp_dfs, H, spot_dfs=spot_dfs, funding_dfs=funding_dfs,
            dvol_by_ccy=dvol_by_ccy, onchain_by_asset=onchain_by_asset,
        )
        feats = pmw.feature_cols(panel)
        n_rows = len(panel.dropna(subset=["fwd", "y"]))
        min_train = min(800 if mode == "single_asset" else 1500, max(300, int(n_rows * 0.35)))
        comp = pmw.purged_model_compare(panel, feats, H, bar_ms, n_folds=n_folds,
                                        k_sel=k_sel, min_train=min_train, preset="fast", log=log)
        if comp.get("error"):
            return comp
        metrics: dict = {}
        best = None
        # RV-5: DSR/PBO 的多重检验试验数 = 模型数 × 置信档数(7) × 周期数 (整个搜索空间)
        n_trials = max(1, len(comp["models"]) * 7 * len(horizons_min))
        for name, (odf, train_abs) in comp["models"].items():
            mm = (pmw.evaluate_scores(odf, H, bar_ms, train_abs, costs=costs, mode=mode, n_trials=n_trials)
                  if len(odf) else {"skip": True})
            metrics[name] = mm
            if mm.get("skip"):
                continue
            cells = [(fr, c) for fr, c in mm["curve"] if c is not None]
            pass_cells = [(fr, c) for fr, c in cells if pmw._cell_pass(c, costs, min_ts=min_cell_ts)]
            stress_cells = [(fr, c) for fr, c in pass_cells if pmw._stress_pass(c, costs)]
            # RV-5: 除净edge闸门外, 还须通过 Deflated Sharpe(>=0.95) 与 PBO(<0.5); 否则不判 edge_ok。
            # 方向: 只会让门更严 (防小样本/多档过拟合的假 PASS), 不会凭空制造 PASS。
            dsr_ok = (mm.get("dsr") is not None and mm["dsr"] >= 0.95)
            pbo_ok = (mm.get("pbo") is None or mm["pbo"] < 0.5)
            gate = (mm.get("primary_ic", 0.0) > 0.01 and len(pass_cells) >= 2 and len(stress_cells) >= 1
                    and dsr_ok and pbo_ok)
            score_cell = max(cells, key=lambda fc: (fc[1].get("net_4", -1e9), fc[1].get("t_4", -1e9)),
                             default=(None, None))
            cand = {"model": name, "gate": gate, "best_frac": score_cell[0],
                    "best_cell": score_cell[1], "metric": mm}
            if best is None or (cand["gate"], (cand["best_cell"] or {}).get("net_4", -1e9)) > (
                    best["gate"], (best["best_cell"] or {}).get("net_4", -1e9)):
                best = cand
        edge_ok = bool(best and best.get("gate"))
        by_h[H] = {
            "n_features": len(feats),
            "selected_freq": comp["selected_freq"][:20],
            "metrics": metrics,
            "best": best,
            "position": pmw._position_from_cell(best.get("best_cell") if best else None, costs, edge_ok=edge_ok),
        }
    return {"mode": mode, "horizons": list(horizons_min), "by_h": by_h, "costs": costs.__dict__}
