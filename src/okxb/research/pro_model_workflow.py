"""15/30min professional modeling workflow.

This module answers the user's requested sequence:
  1) build a broad feature bank from all locally available public data;
  2) select features inside each training fold only, with collinearity pruning;
  3) compare multiple regularized, tree, boosting, neural and regression models;
  4) evaluate OOS win rate and net edge after OKX retail costs;
  5) emit position/leverage only when the statistical gate is passed.

It deliberately does not place orders and does not read API secrets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import candle_research as cr
from . import feature_lab as fl
from . import horizon_model as hm

try:
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
    from sklearn.metrics import roc_auc_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    _HAS_SK = True
except Exception:  # noqa: BLE001
    _HAS_SK = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except Exception:  # noqa: BLE001
    _HAS_LGBM = False


def _log(msg: str) -> None:
    print(msg, flush=True)


@dataclass(frozen=True)
class WorkflowCosts:
    maker_bps: float = 4.0
    taker_bps: float = 10.0
    stress_bps: float = 15.0


def _safe_z(s: pd.Series, win: int) -> pd.Series:
    r = s.rolling(win)
    return (s - r.mean()) / r.std().replace(0, np.nan)


def _grid_series(df: pd.DataFrame, bar_ms: int, col: str = "c") -> pd.Series:
    d = df.sort_values("ts")
    full = np.arange(int(d["ts"].iloc[0]), int(d["ts"].iloc[-1]) + bar_ms, bar_ms)
    s = d.set_index("ts")[col].reindex(full)
    return s.where(s > 0)


def _funding_features(fdf: Optional[pd.DataFrame], idx: pd.Index) -> pd.DataFrame:
    out = pd.DataFrame(index=idx)
    if fdf is None or len(fdf) < 5:
        out["funding_last"] = np.nan
        out["funding_mean_3"] = np.nan
        out["funding_z_21"] = np.nan
        return out
    f = fdf.sort_values("ts").drop_duplicates("ts").set_index("ts")["funding"]
    last = f.reindex(idx, method="ffill")
    out["funding_last"] = last.values
    out["funding_mean_3"] = last.rolling(3 * max(1, round(8 * 60 / ((idx[1] - idx[0]) / 60_000)))).mean().values
    out["funding_z_21"] = _safe_z(last, 21 * max(1, round(8 * 60 / ((idx[1] - idx[0]) / 60_000)))).values
    return out


def build_augmented_bank(
    perp_dfs: dict[str, pd.DataFrame],
    spot_dfs: Optional[dict[str, pd.DataFrame]],
    funding_dfs: Optional[dict[str, pd.DataFrame]],
    bar: str,
    H_min: int,
    *,
    leads_min: tuple[int, ...] = (5, 15, 30, 60),
) -> tuple[pd.DataFrame, int]:
    """Build an H-specific candidate feature panel.

    Sources:
      - perp OHLCV technical features from feature_lab.inst_bank;
      - spot/perp basis when spot data is available;
      - trailing realized funding features when funding data is available;
      - BTC/ETH lead-lag and cross-sectional market context.
    """
    bar_min = cr.BAR_MIN[bar]
    bar_ms = bar_min * 60_000
    spot_dfs = spot_dfs or {}
    funding_dfs = funding_dfs or {}

    btc = _grid_series(perp_dfs["BTC-USDT-SWAP"], bar_ms) if "BTC-USDT-SWAP" in perp_dfs else None
    eth = _grid_series(perp_dfs["ETH-USDT-SWAP"], bar_ms) if "ETH-USDT-SWAP" in perp_dfs else None
    parts: list[pd.DataFrame] = []
    for inst, df in perp_dfs.items():
        p = fl.inst_bank(df, H_min, bar_min, bar_ms)
        p["inst"] = inst
        pc = _grid_series(df, bar_ms).reindex(p["ts"].values)
        idx = pc.index

        # Spot/perp basis and spot movement. Missing spot data remains NaN and is filtered fold-wise.
        spot = spot_dfs.get(inst)
        if spot is not None and len(spot):
            sc = _grid_series(spot, bar_ms).reindex(idx)
            basis = pc / sc - 1.0
            hb = max(1, round(H_min / bar_min))
            p["basis"] = basis.values
            p["basis_z_h"] = _safe_z(basis, max(6, hb * 4)).values
            p["basis_chg_h"] = basis.diff(hb).values
            p["spot_ret_h"] = (sc / sc.shift(hb) - 1.0).values
            p["perp_spot_ret_gap_h"] = ((pc / pc.shift(hb) - 1.0) - (sc / sc.shift(hb) - 1.0)).values
        else:
            p["basis"] = np.nan
            p["basis_z_h"] = np.nan
            p["basis_chg_h"] = np.nan
            p["spot_ret_h"] = np.nan
            p["perp_spot_ret_gap_h"] = np.nan

        ff = _funding_features(funding_dfs.get(inst), idx)
        for col in ff.columns:
            p[col] = ff[col].values

        for name, s in (("btc", btc), ("eth", eth)):
            if s is None:
                continue
            sa = s.reindex(idx)
            for mn in leads_min:
                k = max(1, round(mn / bar_min))
                p[f"{name}_ret_{mn}"] = (sa / sa.shift(k) - 1.0).values
        parts.append(p)

    panel = pd.concat(parts, ignore_index=True)

    # Cross-sectional context is point-in-time: same timestamp only.
    for col in ("ret_1", "roc", "basis", "funding_last", "vol_z"):
        if col in panel.columns:
            g = panel.groupby("ts")[col]
            panel[f"xs_{col}"] = (panel[col] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    if "ret_1" in panel.columns:
        panel["mkt_ret_1_mean"] = panel.groupby("ts")["ret_1"].transform("mean")
    if "roc" in panel.columns:
        panel["mkt_roc_mean"] = panel.groupby("ts")["roc"].transform("mean")
    return panel, bar_ms


def feature_cols(panel: pd.DataFrame) -> list[str]:
    return [c for c in panel.columns if c not in ("ts", "inst", "fwd", "y")]


def _model_factories(random_state: int = 0, preset: str = "fast"):
    fast = preset != "deep"
    models = {
        "logit_l2": ("clf_scale", lambda: LogisticRegression(C=0.5, max_iter=1200)),
        "logit_l1": ("clf_scale", lambda: LogisticRegression(penalty="l1", solver="liblinear", C=0.15, max_iter=1200)),
        "elastic_logit": ("clf_scale", lambda: LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5, C=0.4, max_iter=1000
        )),
        "rf": ("clf_raw", lambda: RandomForestClassifier(
            n_estimators=(120 if fast else 300), max_depth=7, min_samples_leaf=180, n_jobs=-1, random_state=random_state
        )),
        "hist_gbm": ("clf_raw", lambda: HistGradientBoostingClassifier(
            max_depth=3, max_iter=(120 if fast else 250), learning_rate=0.04, l2_regularization=1.0,
            min_samples_leaf=200, early_stopping=True, validation_fraction=0.15,
            random_state=random_state,
        )),
        "mlp": ("clf_scale", lambda: MLPClassifier(
            hidden_layer_sizes=((32,) if fast else (48, 16)), alpha=1e-3, max_iter=(160 if fast else 350),
            early_stopping=True, random_state=random_state,
        )),
        "ridge_ret": ("reg_scale", lambda: Ridge(alpha=10.0)),
        "enet_ret": ("reg_scale", lambda: ElasticNet(alpha=1e-4, l1_ratio=0.2, max_iter=3000, random_state=random_state)),
    }
    if not fast:
        models["extra_trees"] = ("clf_raw", lambda: ExtraTreesClassifier(
            n_estimators=350, max_depth=7, min_samples_leaf=160, n_jobs=-1, random_state=random_state
        ))
        models["rf_ret"] = ("reg_raw", lambda: RandomForestRegressor(
            n_estimators=240, max_depth=7, min_samples_leaf=160, n_jobs=-1, random_state=random_state
        ))
    if _HAS_LGBM:
        models["lightgbm"] = ("clf_raw", lambda: LGBMClassifier(
            n_estimators=(180 if fast else 400), max_depth=3, learning_rate=0.03, subsample=0.85,
            colsample_bytree=0.85, reg_alpha=0.2, reg_lambda=2.0,
            min_child_samples=180, objective="binary", random_state=random_state,
            verbosity=-1,
        ))
    return models


def _score_from_model(kind: str, model, X) -> np.ndarray:
    if kind.startswith("clf"):
        return model.predict_proba(X)[:, 1] - 0.5
    return np.asarray(model.predict(X), dtype=float)


def purged_model_compare(
    panel: pd.DataFrame,
    feats: list[str],
    H_min: int,
    bar_ms: int,
    *,
    n_folds: int = 4,
    k_sel: int = 22,
    min_train: int = 4500,
    preset: str = "fast",
    log: Callable[[str], None] = _log,
) -> dict:
    """Walk-forward comparison with fold-local feature selection."""
    if not _HAS_SK:
        return {"error": "sklearn 未安装"}
    bar_min = bar_ms / 60_000
    embargo_ms = int(round(H_min / bar_min)) * bar_ms
    data = panel[["ts", "inst", "fwd", "y"] + feats].replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["fwd", "y"])
    ts_sorted = np.sort(data["ts"].unique())
    bounds = [int(len(ts_sorted) * (i + 1) / (n_folds + 1)) for i in range(n_folds + 1)]
    zoo = _model_factories(preset=preset)
    preds = {m: [] for m in zoo}
    train_scores = {m: [] for m in zoo}
    selected_freq: dict[str, int] = {}
    fold_reports = []

    for fold in range(n_folds):
        cut = int(ts_sorted[bounds[fold]])
        te_hi = int(ts_sorted[min(bounds[fold + 1], len(ts_sorted) - 1)])
        tr = data[data["ts"] <= cut - embargo_ms]
        te = data[(data["ts"] > cut) & (data["ts"] <= te_hi)]
        if len(tr) < min_train or len(te) < 300:
            continue
        sel, rep = fl.select_features(tr, feats, k=k_sel)
        for f in sel:
            selected_freq[f] = selected_freq.get(f, 0) + 1
        fold_reports.append({"fold": fold + 1, "n_train": len(tr), "n_test": len(te), "selected": sel, "rank": rep})

        med = tr[sel].median()
        Xtr = tr[sel].fillna(med).values
        Xte = te[sel].fillna(med).values
        ytr = tr["y"].values.astype(int)
        rtr = tr["fwd"].values.astype(float)
        for name, (kind, factory) in zoo.items():
            try:
                if kind.endswith("scale"):
                    sc = StandardScaler().fit(Xtr)
                    Xtr_, Xte_ = sc.transform(Xtr), sc.transform(Xte)
                else:
                    Xtr_, Xte_ = Xtr, Xte
                target = ytr if kind.startswith("clf") else rtr
                model = factory().fit(Xtr_, target)
                ste = _score_from_model(kind, model, Xte_)
                strn = _score_from_model(kind, model, Xtr_)
                preds[name].append(pd.DataFrame({
                    "ts": te["ts"].values,
                    "inst": te["inst"].values,
                    "score": ste,
                    "fwd": te["fwd"].values,
                    "y": te["y"].values,
                }))
                train_scores[name].append(np.abs(strn))
            except Exception as exc:  # noqa: BLE001
                log(f"    {name} fold{fold + 1} failed: {type(exc).__name__}: {exc}")

    out = {}
    for name in zoo:
        odf = pd.concat(preds[name], ignore_index=True) if preds[name] else pd.DataFrame()
        ts = np.concatenate(train_scores[name]) if train_scores[name] else np.array([])
        out[name] = (odf, ts)
    return {
        "models": out,
        "selected_freq": sorted(selected_freq.items(), key=lambda kv: kv[1], reverse=True),
        "fold_reports": fold_reports,
        "n_models": len(zoo),
    }


def evaluate_scores(
    oos: pd.DataFrame,
    H_min: int,
    bar_ms: int,
    train_abs_score: np.ndarray,
    *,
    costs: WorkflowCosts = WorkflowCosts(),
    mode: str = "cross_sectional",
    fracs: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 1.0),
) -> dict:
    """Evaluate directional scores. Positive score means long, negative means short."""
    if len(oos) < 300:
        return {"skip": True, "n": len(oos)}
    bar_min = bar_ms / 60_000
    h = max(1, round(H_min / bar_min))
    clean = oos.replace([np.inf, -np.inf], np.nan).dropna(subset=["score", "fwd", "y"])
    if len(clean) < 300:
        return {"skip": True, "n": len(clean)}
    try:
        auc = roc_auc_score(clean["y"].values, clean["score"].values) if clean["y"].nunique() == 2 else np.nan
    except Exception:  # noqa: BLE001
        auc = np.nan
    ts_ic = cr._spearman(clean["score"].values, clean["fwd"].values)
    xs = cr._xs_ic_series(clean.rename(columns={"fwd": "_y"}), "score", "_y")
    xs_ic = float(xs.mean()) if len(xs) else np.nan
    primary_ic = ts_ic if mode == "single_asset" else xs_ic
    winrate_all = float((np.sign(clean["score"].values) == np.sign(clean["fwd"].values)).mean())
    all_ts = np.sort(clean["ts"].unique())
    reb = set(cr._rebalance_ts(all_ts, bar_ms, h).tolist())
    sub = clean[clean["ts"].isin(reb)].copy()
    sub["conf"] = sub["score"].abs()
    have_train = len(train_abs_score) > 100 and np.nanmax(train_abs_score) > 0
    curve = []
    for fr in fracs:
        thr = 0.0 if fr >= 1.0 else (
            float(np.nanquantile(train_abs_score, 1 - fr)) if have_train else float(sub["conf"].quantile(1 - fr))
        )
        s = sub[sub["conf"] >= thr].copy()
        if len(s) < 20:
            curve.append((fr, None))
            continue
        signed = np.sign(s["score"].values) * s["fwd"].values
        port = pd.Series(signed, index=s["ts"].values).groupby(level=0).mean().values
        if len(port) < 8:
            curve.append((fr, None))
            continue
        wins = signed[signed > 0]
        losses = signed[signed < 0]
        cell = {
            "n_ts": len(port),
            "n_tr": len(s),
            "winrate": float((signed > 0).mean()),
            "gross_bps": float(np.mean(signed) * 1e4),
            "avg_win_bps": float(np.mean(wins) * 1e4) if len(wins) else 0.0,
            "avg_loss_bps": float(np.mean(losses) * 1e4) if len(losses) else 0.0,
            "wl": float(np.mean(wins) / -np.mean(losses)) if len(wins) and len(losses) else np.nan,
        }
        for c in (costs.maker_bps, costs.taker_bps, costs.stress_bps):
            st = cr._stats_from_gross(port, c, 1)
            key = str(int(round(c)))
            cell[f"net_{key}"] = st["net_bps"] if st else np.nan
            cell[f"t_{key}"] = st["nw_t"] if st else np.nan
        curve.append((fr, cell))
    return {
        "n": len(clean),
        "auc": float(auc) if np.isfinite(auc) else np.nan,
        "primary_ic": float(primary_ic) if np.isfinite(primary_ic) else np.nan,
        "ts_ic": float(ts_ic) if np.isfinite(ts_ic) else np.nan,
        "xs_ic": xs_ic,
        "xs_t": cr._nw_t(xs.values) if len(xs) > 5 else np.nan,
        "winrate": winrate_all,
        "curve": curve,
    }


def _cell_pass(cell: Optional[dict], costs: WorkflowCosts, *, min_ts: int = 20) -> bool:
    if not cell or cell.get("n_ts", 0) < min_ts:
        return False
    for c in (costs.maker_bps, costs.taker_bps):
        key = str(int(round(c)))
        if cell.get(f"net_{key}", -1e9) <= 0 or (cell.get(f"t_{key}") or 0.0) <= 2:
            return False
    return True


def _stress_pass(cell: Optional[dict], costs: WorkflowCosts) -> bool:
    if not cell:
        return False
    key = str(int(round(costs.stress_bps)))
    return cell.get(f"net_{key}", -1e9) > 0 and (cell.get(f"t_{key}") or 0.0) > 1.5


def _position_from_cell(cell: Optional[dict], costs: WorkflowCosts, *, edge_ok: bool) -> dict:
    """Return sizing policy. If gate fails, all trading size is zero."""
    if not edge_ok or not cell:
        return {
            "tradable": False,
            "reason": "model gate failed",
            "risk_pct_equity": 0.0,
            "max_leverage": 0.0,
            "kelly_raw": 0.0,
            "kelly_used": 0.0,
        }
    p = float(cell.get("winrate", 0.0))
    wl = float(cell.get("wl", np.nan))
    if not np.isfinite(wl) or wl <= 0:
        kelly = 0.0
    else:
        kelly = p - (1 - p) / wl
    kelly_used = max(0.0, min(0.03, 0.25 * kelly))
    # Hard cap remains small because this is still only historical OOS, not live shadow evidence.
    risk_pct = min(0.0025, kelly_used)
    return {
        "tradable": risk_pct > 0,
        "reason": "statistical gate passed; still requires shadow live before real capital",
        "risk_pct_equity": risk_pct,
        "max_leverage": 2.0 if risk_pct > 0 else 0.0,
        "kelly_raw": float(kelly),
        "kelly_used": float(kelly_used),
        "winrate": p,
        "wl": wl,
        "avg_win_bps": cell.get("avg_win_bps"),
        "avg_loss_bps": cell.get("avg_loss_bps"),
    }


def run(
    perp_dfs: dict[str, pd.DataFrame],
    *,
    spot_dfs: Optional[dict[str, pd.DataFrame]] = None,
    funding_dfs: Optional[dict[str, pd.DataFrame]] = None,
    bar: str = "5m",
    horizons_min: tuple[int, ...] = (15, 30),
    n_folds: int = 4,
    k_sel: int = 22,
    min_train: Optional[int] = None,
    preset: str = "fast",
    costs: WorkflowCosts = WorkflowCosts(),
    log: Callable[[str], None] = _log,
) -> dict:
    if not _HAS_SK:
        return {"error": "sklearn 未安装"}
    res = {}
    mode = "single_asset" if len(perp_dfs) == 1 else "cross_sectional"
    min_cell_ts = 50 if mode == "single_asset" else 20
    for H in horizons_min:
        log(f"H={H}min: build augmented bank + fold-local selection + model zoo")
        panel, bar_ms = build_augmented_bank(perp_dfs, spot_dfs, funding_dfs, bar, H)
        feats = feature_cols(panel)
        n_rows = len(panel.dropna(subset=["fwd", "y"]))
        fold_min_train = min_train
        if fold_min_train is None:
            fold_min_train = 1200 if mode == "single_asset" else 4500
            fold_min_train = min(fold_min_train, max(600, int(n_rows * 0.35)))
        comp = purged_model_compare(panel, feats, H, bar_ms, n_folds=n_folds, k_sel=k_sel,
                                    min_train=fold_min_train, preset=preset, log=log)
        if comp.get("error"):
            return comp
        metrics = {}
        best = None
        for name, (odf, train_abs) in comp["models"].items():
            mm = evaluate_scores(odf, H, bar_ms, train_abs, costs=costs, mode=mode) if len(odf) else {"skip": True, "n": 0}
            metrics[name] = mm
            if mm.get("skip"):
                continue
            cells = [(fr, c) for fr, c in mm["curve"] if c is not None]
            pass_cells = [(fr, c) for fr, c in cells if _cell_pass(c, costs, min_ts=min_cell_ts)]
            stress_cells = [(fr, c) for fr, c in pass_cells if _stress_pass(c, costs)]
            gate = (
                mm.get("primary_ic", 0.0) > 0.01
                and len(pass_cells) >= 2
                and len(stress_cells) >= 1
            )
            score_cell = max(cells, key=lambda fc: (fc[1].get("net_4", -1e9), fc[1].get("t_4", -1e9)), default=(None, None))
            candidate = {
                "model": name,
                "gate": gate,
                "pass_count": len(pass_cells),
                "stress_count": len(stress_cells),
                "best_frac": score_cell[0],
                "best_cell": score_cell[1],
                "metric": mm,
            }
            if best is None:
                best = candidate
            else:
                bcell = best.get("best_cell") or {}
                ccell = candidate.get("best_cell") or {}
                if (candidate["gate"], ccell.get("net_4", -1e9), ccell.get("t_4", -1e9)) > (
                    best["gate"], bcell.get("net_4", -1e9), bcell.get("t_4", -1e9)
                ):
                    best = candidate
        edge_ok = bool(best and best.get("gate"))
        res[H] = {
            "n_features": len(feats),
            "selected_freq": comp["selected_freq"][:20],
            "folds": [
                {"fold": r["fold"], "n_train": r["n_train"], "n_test": r["n_test"], "selected": r["selected"][:12]}
                for r in comp["fold_reports"]
            ],
            "metrics": metrics,
            "best": best,
            "position": _position_from_cell(best.get("best_cell") if best else None, costs, edge_ok=edge_ok),
            "bar_ms": bar_ms,
            "min_train": fold_min_train,
        }
    return {
        "bar": bar,
        "n_inst": len(perp_dfs),
        "horizons": list(horizons_min),
        "has_spot": bool(spot_dfs),
        "has_funding": bool(funding_dfs),
        "has_lightgbm": _HAS_LGBM,
        "mode": mode,
        "min_cell_ts": min_cell_ts,
        "preset": preset,
        "costs": costs.__dict__,
        "by_h": res,
    }


def _fmt_cell(cell: Optional[dict], costs: WorkflowCosts) -> str:
    if cell is None:
        return "-"
    vals = []
    for c in (costs.maker_bps, costs.taker_bps, costs.stress_bps):
        key = str(int(round(c)))
        vals.append(f"net{key}={cell.get(f'net_{key}', np.nan):+.1f}(t{cell.get(f't_{key}', np.nan):+.1f})")
    return (
        f"win={cell['winrate']:.1%} gross={cell['gross_bps']:+.1f} "
        + " ".join(vals)
        + f" n_ts={cell['n_ts']}"
    )


def format_report(meta: dict, res: dict) -> str:
    if res.get("error"):
        return f"15/30min modeling workflow failed: {res['error']}"
    costs = WorkflowCosts(**res["costs"])
    min_cell_ts = int(res.get("min_cell_ts", 20))
    lines = [
        "=" * 88,
        "15/30min professional modeling workflow (feature selection + model zoo + OOS net edge + sizing gate)",
        "=" * 88,
        f"data: {res['n_inst']} instruments / bar={res['bar']} / span={meta.get('span_days', '?')} days / "
        f"spot_features={res['has_spot']} funding_features={res['has_funding']} "
        f"lightgbm={res['has_lightgbm']} preset={res.get('preset', 'fast')} mode={res.get('mode', 'cross_sectional')}",
        f"costs: maker={costs.maker_bps:g}bps taker={costs.taker_bps:g}bps stress={costs.stress_bps:g}bps "
        f"min_cell_ts={min_cell_ts}",
        "gate: primary_ic>0.01 AND >=2 confidence buckets pass maker+taker net>0,t>2 AND >=1 bucket survives stress",
        "",
    ]
    any_trade = False
    for H in res["horizons"]:
        d = res["by_h"][H]
        best = d.get("best")
        pos = d.get("position", {})
        any_trade = any_trade or bool(pos.get("tradable"))
        lines.append(f"--- H={H}min candidates={d['n_features']} selected_top22 each fold min_train={d.get('min_train')} ---")
        lines.append("selected often: " + ", ".join(f"{f}x{n}" for f, n in d["selected_freq"][:12]))
        lines.append("fold selected samples: " + " | ".join(
            f"F{fr['fold']}[{','.join(fr['selected'][:5])}]" for fr in d["folds"][:4]
        ))
        lines.append("model comparison:")
        for name, mm in d["metrics"].items():
            if mm.get("skip"):
                lines.append(f"  {name:>12}: skip n={mm.get('n', 0)}")
                continue
            cells = [(fr, c) for fr, c in mm["curve"] if c is not None]
            bc = max(cells, key=lambda fc: fc[1].get("net_4", -1e9), default=(None, None))
            passes = sum(1 for _, c in cells if _cell_pass(c, costs, min_ts=min_cell_ts))
            stress = sum(1 for _, c in cells if _cell_pass(c, costs, min_ts=min_cell_ts) and _stress_pass(c, costs))
            lines.append(
                f"  {name:>12}: auc={mm['auc']:.3f} primary_ic={mm.get('primary_ic', np.nan):+.4f} "
                f"ts_ic={mm.get('ts_ic', np.nan):+.4f} xs_ic={mm.get('xs_ic', np.nan):+.4f} "
                f"win_all={mm['winrate']:.1%} pass={passes}/stress={stress} "
                f"best_top{(bc[0] or 0)*100:g}% {_fmt_cell(bc[1], costs)}"
            )
        if best:
            lines.append(
                f"best: H={H} {best['model']} top{(best.get('best_frac') or 0)*100:g}% "
                f"gate={'PASS' if best['gate'] else 'FAIL'} {_fmt_cell(best.get('best_cell'), costs)}"
            )
        lines.append(
            "sizing: "
            f"tradable={pos.get('tradable')} risk_pct_equity={pos.get('risk_pct_equity', 0.0):.4%} "
            f"max_leverage={pos.get('max_leverage', 0.0):.1f}x "
            f"kelly_raw={pos.get('kelly_raw', 0.0):+.4f} reason={pos.get('reason', '')}"
        )
        lines.append("")
    lines.append("verdict:")
    if any_trade:
        lines.append("  CANDIDATE ONLY: a model passed the historical OOS gate. Run live shadow trading before any real position.")
    else:
        lines.append("  NO TRADE: no 15/30min model passed the OOS net-edge gate; position and leverage remain zero.")
    lines.append("")
    lines.append("notes: all feature selection is fold-local; collinearity pruning is inside the training fold;")
    lines.append("  win rate is not the objective; sizing is disabled unless net edge survives maker, taker and stress costs.")
    lines.append("=" * 88)
    return "\n".join(lines)
