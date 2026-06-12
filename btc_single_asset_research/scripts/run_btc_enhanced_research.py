from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Callable

import httpx
import numpy as np
import pandas as pd

# joblib/loky 在本机无 wmic 时探测物理核失败 -> 噪音 [WinError 2]。给定核数 + 兜底过滤该 warning。
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from okxb.research import candle_data as cd  # noqa: E402
from okxb.research import deribit_data, onchain_data  # noqa: E402
from okxb.research import pro_model_workflow as pmw  # noqa: E402


OUT_ROOT = ROOT / "btc_single_asset_research"
DATA_ROOT = OUT_ROOT / "data"
REPORT_ROOT = OUT_ROOT / "reports"
HOSTS = ["https://www.okx.com", "https://us.okx.com", "https://eea.okx.com"]
DAY_MS = 86_400_000

# Multi-asset: the pipeline was BTC-only. These maps let --asset=eth reuse the SAME feature
# engineering on ETH data. Feature COLUMN names are kept shared across assets (so the forward
# scorer needs no per-asset renames); only the rubik cache files + output files are separated.
ASSETS = {
    "btc": {"perp": "BTC-USDT-SWAP", "spot": "BTC-USDT", "ccy": "BTC", "deribit": "BTC", "cm": "btc"},
    "eth": {"perp": "ETH-USDT-SWAP", "spot": "ETH-USDT", "ccy": "ETH", "deribit": "ETH", "cm": "eth"},
}


def log(msg: str) -> None:
    print(msg, flush=True)


def ts_utc(ms: int | float | None) -> str:
    if ms is None or not np.isfinite(ms):
        return ""
    return pd.to_datetime(int(ms), unit="ms", utc=True).strftime("%Y-%m-%d %H:%M")


def rolling_z(s: pd.Series, w: int = 20) -> pd.Series:
    mu = s.rolling(w, min_periods=max(5, w // 4)).mean()
    sd = s.rolling(w, min_periods=max(5, w // 4)).std().replace(0, np.nan)
    return (s - mu) / sd


def safe_pct(s: pd.Series) -> pd.Series:
    return s.replace(0, np.nan).pct_change()


def cache_read(path: Path, force: bool) -> pd.DataFrame | None:
    if path.exists() and not force:
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def okx_get(path: str, params: dict, *, timeout: float = 20.0) -> dict:
    last_exc: Exception | None = None
    with httpx.Client(timeout=timeout, headers={"User-Agent": "okxb-btc-research/1"}) as cli:
        for host in HOSTS:
            try:
                j = cli.get(host + path, params=params).json()
                if j.get("code") == "0":
                    return j
                last_exc = RuntimeError(f"{path} code={j.get('code')} msg={j.get('msg')}")
            except Exception as exc:
                last_exc = exc
    raise RuntimeError(str(last_exc))


def parse_rows(rows: list, cols: list[str]) -> pd.DataFrame:
    out: list[list[float]] = []
    for r in rows:
        if not isinstance(r, list) or len(r) < len(cols):
            continue
        try:
            out.append([int(r[0])] + [float(x) for x in r[1:len(cols)]])
        except (TypeError, ValueError):
            continue
    df = pd.DataFrame(out, columns=cols)
    if len(df):
        df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
        df["ts"] = df["ts"].astype("int64")
    return df


def fetch_rubik_once(name: str, path: str, params: dict, cols: list[str], *, force: bool,
                     cache_name: str | None = None) -> pd.DataFrame:
    d = DATA_ROOT / "okx_rubik"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{cache_name or name}.csv"
    cached = cache_read(f, force)
    if cached is not None and len(cached):
        return cached
    j = okx_get(path, params)
    df = parse_rows(j.get("data", []), cols)
    df.to_csv(f, index=False)
    return df


def add_basic_transforms(df: pd.DataFrame, cols: list[str], *, prefix: str) -> pd.DataFrame:
    out = df[["ts"]].copy()
    for c in cols:
        if c not in df:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        out[f"{prefix}_{c}"] = s
        out[f"{prefix}_{c}_chg1"] = safe_pct(s)
        out[f"{prefix}_{c}_z20"] = rolling_z(s, 20)
        if (s.dropna() > 0).mean() > 0.95:
            out[f"{prefix}_{c}_log"] = np.log1p(s.clip(lower=0))
    return out


def merge_on_ts(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [x for x in frames if x is not None and len(x)]
    if not frames:
        return pd.DataFrame(columns=["ts"])
    out = frames[0].copy()
    for f in frames[1:]:
        out = out.merge(f, on="ts", how="outer")
    out = out.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    value_cols = [c for c in out.columns if c != "ts"]
    # Each source has its own daily close timestamp. Once a value is published, it
    # remains the latest point-in-time value until that source publishes again.
    if value_cols:
        out[value_cols] = out[value_cols].ffill()
    return out


DAILY_EXTERNAL_CACHE_HOURS = 20.0       # 日频外部源(Rubik/DVOL/链上)最多每 ~20h 重取一次


def fetch_daily_external(days: int, *, force: bool, asset: str = "btc") -> tuple[pd.DataFrame, pd.DataFrame]:
    a = ASSETS[asset]
    pref = "" if asset == "btc" else f"{asset}_"        # btc reuses existing unprefixed rubik cache
    out_csv = DATA_ROOT / "external" / f"{asset}_daily_external_features.csv"
    # 日缓存短路: 组装好的日频外部特征若 <20h 旧, 直接读盘, 完全不碰 Rubik/Deribit/CoinMetrics
    # (这些是境外/日频源, 每轮 evaluate 都重取既慢又易被拒; 日内复用同一份即可)。
    if not force and out_csv.exists():
        try:
            age_h = (time.time() - out_csv.stat().st_mtime) / 3600.0
            if age_h < DAILY_EXTERNAL_CACHE_HOURS:
                cached = pd.read_csv(out_csv)
                if len(cached):
                    return cached, pd.DataFrame([{"source": f"{asset}_daily_external_cache",
                                                  "rows": len(cached), "age_h": round(age_h, 1)}])
        except Exception:  # noqa: BLE001 - 缓存坏了就照常重取
            pass
    frames: list[pd.DataFrame] = []
    inventory: list[dict] = []

    specs = [
        (
            "okx_oi_1d",
            "/api/v5/rubik/stat/contracts/open-interest-history",
            {"instId": a["perp"], "period": "1D", "limit": "100"},
            ["ts", "oi_contracts", "oi_btc", "oi_usd"],
        ),
        (
            "okx_oi_volume_1d",
            "/api/v5/rubik/stat/contracts/open-interest-volume",
            {"ccy": a["ccy"], "period": "1D", "limit": "200"},
            ["ts", "oi_volume_open", "oi_volume_turnover"],
        ),
        (
            "okx_taker_1d",
            "/api/v5/rubik/stat/taker-volume",
            {"ccy": a["ccy"], "instType": "CONTRACTS", "period": "1D", "limit": "200"},
            ["ts", "taker_buy", "taker_sell"],
        ),
        (
            "okx_lsr_global_1d",
            "/api/v5/rubik/stat/contracts/long-short-account-ratio",
            {"ccy": a["ccy"], "period": "1D", "limit": "200"},
            ["ts", "lsr_global"],
        ),
        (
            "okx_lsr_contract_1d",
            "/api/v5/rubik/stat/contracts/long-short-account-ratio-contract",
            {"instId": a["perp"], "period": "1D", "limit": "100"},
            ["ts", "lsr_contract"],
        ),
    ]

    cutoff = int(time.time() * 1000) - int((days + 10) * DAY_MS)
    for name, path, params, cols in specs:
        try:
            raw = fetch_rubik_once(name, path, params, cols, force=force, cache_name=f"{pref}{name}")
            raw = raw[raw["ts"] >= cutoff].reset_index(drop=True)
            if name == "okx_taker_1d" and {"taker_buy", "taker_sell"}.issubset(raw.columns):
                tot = raw["taker_buy"] + raw["taker_sell"]
                raw["taker_total"] = tot
                raw["taker_imb"] = (raw["taker_buy"] - raw["taker_sell"]) / tot.replace(0, np.nan)
                cols2 = ["taker_total", "taker_imb"]
            else:
                cols2 = [c for c in raw.columns if c != "ts"]
            feat = add_basic_transforms(raw, cols2, prefix=name)
            frames.append(feat)
            inventory.append(source_inventory(name, raw, "historical_daily_lag1", path))
        except Exception as exc:
            inventory.append({"source": name, "rows": 0, "error": f"{type(exc).__name__}: {exc}"})

    try:
        dvol = deribit_data.fetch_dvol(a["deribit"], days + 5, resolution="86400", log=log)
        dvol = dvol.rename(columns={"dvol": "dvol_daily"})
        dvol["dvol_daily_chg1"] = dvol["dvol_daily"].diff()
        dvol["dvol_daily_z20"] = rolling_z(dvol["dvol_daily"], 20)
        frames.append(dvol)
        inventory.append(source_inventory("deribit_dvol_1d", dvol, "historical_daily_lag1", "Deribit get_volatility_index_data"))
    except Exception as exc:
        inventory.append({"source": "deribit_dvol_1d", "rows": 0, "error": f"{type(exc).__name__}: {exc}"})

    try:
        onchain = onchain_data.fetch_onchain(a["cm"], None, days + 10, DATA_ROOT / "coinmetrics", force=force, log=log)
        cm_frames = []
        for metric, df in onchain.items():
            m = df.rename(columns={"value": f"cm_{metric}"})
            s = pd.to_numeric(m[f"cm_{metric}"], errors="coerce")
            m[f"cm_{metric}_chg1"] = safe_pct(s)
            m[f"cm_{metric}_z20"] = rolling_z(s, 20)
            if (s.dropna() > 0).mean() > 0.95:
                m[f"cm_{metric}_log"] = np.log1p(s.clip(lower=0))
            cm_frames.append(m)
            inventory.append(source_inventory(f"coinmetrics_{metric}", m, "historical_daily_lag1_regime", "Coin Metrics community asset metrics"))
        if cm_frames:
            frames.append(merge_on_ts(cm_frames))
    except Exception as exc:
        inventory.append({"source": "coinmetrics", "rows": 0, "error": f"{type(exc).__name__}: {exc}"})

    daily = merge_on_ts(frames)
    d = DATA_ROOT / "external"
    d.mkdir(parents=True, exist_ok=True)
    daily.to_csv(d / f"{asset}_daily_external_features.csv", index=False)
    inv = pd.DataFrame(inventory)
    inv.to_csv(d / f"{asset}_source_inventory.csv", index=False)
    return daily, inv


def fetch_short_intraday_external(*, force: bool, asset: str = "btc") -> tuple[pd.DataFrame, pd.DataFrame]:
    a = ASSETS[asset]
    pref = "" if asset == "btc" else f"{asset}_"
    frames: list[pd.DataFrame] = []
    inventory: list[dict] = []
    specs = [
        (
            "okx_oi_5m",
            "/api/v5/rubik/stat/contracts/open-interest-history",
            {"instId": a["perp"], "period": "5m", "limit": "100"},
            ["ts", "oi_contracts_5m", "oi_btc_5m", "oi_usd_5m"],
        ),
        (
            "okx_oi_volume_5m",
            "/api/v5/rubik/stat/contracts/open-interest-volume",
            {"ccy": a["ccy"], "period": "5m", "limit": "600"},
            ["ts", "oi_volume_open_5m", "oi_volume_turnover_5m"],
        ),
        (
            "okx_taker_5m",
            "/api/v5/rubik/stat/taker-volume",
            {"ccy": a["ccy"], "instType": "CONTRACTS", "period": "5m", "limit": "600"},
            ["ts", "taker_buy_5m", "taker_sell_5m"],
        ),
        (
            "okx_lsr_global_5m",
            "/api/v5/rubik/stat/contracts/long-short-account-ratio",
            {"ccy": a["ccy"], "period": "5m", "limit": "600"},
            ["ts", "lsr_global_5m"],
        ),
    ]
    for name, path, params, cols in specs:
        try:
            raw = fetch_rubik_once(name, path, params, cols, force=force, cache_name=f"{pref}{name}")
            if name == "okx_taker_5m" and {"taker_buy_5m", "taker_sell_5m"}.issubset(raw.columns):
                tot = raw["taker_buy_5m"] + raw["taker_sell_5m"]
                raw["taker_total_5m"] = tot
                raw["taker_imb_5m"] = (raw["taker_buy_5m"] - raw["taker_sell_5m"]) / tot.replace(0, np.nan)
                use_cols = ["taker_total_5m", "taker_imb_5m"]
            else:
                use_cols = [c for c in raw.columns if c != "ts"]
            frames.append(add_basic_transforms(raw, use_cols, prefix=name))
            inventory.append(source_inventory(name, raw, "short_history_intraday_lag1bar", path))
        except Exception as exc:
            inventory.append({"source": name, "rows": 0, "error": f"{type(exc).__name__}: {exc}"})
    short = merge_on_ts(frames)
    d = DATA_ROOT / "external"
    d.mkdir(parents=True, exist_ok=True)
    short.to_csv(d / f"{asset}_short_intraday_external_features.csv", index=False)
    return short, pd.DataFrame(inventory)


def source_inventory(name: str, df: pd.DataFrame, use: str, path: str) -> dict:
    if df is None or not len(df):
        return {"source": name, "rows": 0, "use": use, "path": path}
    return {
        "source": name,
        "rows": len(df),
        "start_utc": ts_utc(df["ts"].min()),
        "end_utc": ts_utc(df["ts"].max()),
        "span_days": round((int(df["ts"].max()) - int(df["ts"].min())) / DAY_MS, 2),
        "use": use,
        "path": path,
    }


def load_candles(inst_id: str, bar: str, days: int, *, source: str, force: bool,
                 update: bool = False) -> pd.DataFrame:
    if source == "dist":
        f = ROOT / "dist" / "candles" / f"{inst_id}_{bar}.csv"
        if f.exists() and not force:
            df = pd.read_csv(f)
            if len(df) > 100:
                span = (int(df["ts"].iloc[-1]) - int(df["ts"].iloc[0])) / DAY_MS
                if span >= days - 2:
                    return df
                log(f"{inst_id} {bar}: dist cache span={span:.1f}d, refreshing.")
    return cd.get_candles(inst_id, bar, days, DATA_ROOT / "candles", force=force, update=update, log=log)


def asof_join_features(panel: pd.DataFrame, feat: pd.DataFrame, lag_ms: int, suffix: str) -> pd.DataFrame:
    if feat is None or not len(feat):
        return panel
    f = feat.copy()
    f = f.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    f["ts_avail"] = f["ts"].astype("int64") + int(lag_ms)
    drop = ["ts"]
    keep = [c for c in f.columns if c not in drop]
    tmp = f[keep].sort_values("ts_avail")
    left = panel.sort_values("ts").reset_index(drop=True)
    out = pd.merge_asof(left, tmp, left_on="ts", right_on="ts_avail", direction="backward")
    if "ts_avail" in out:
        out = out.drop(columns=["ts_avail"])
    out = out.rename(columns={c: f"{c}{suffix}" for c in []})
    return out


def add_calendar_and_rv(panel: pd.DataFrame, btc_candles: pd.DataFrame, bar: str) -> pd.DataFrame:
    out = panel.copy()
    dt = pd.to_datetime(out["ts"], unit="ms", utc=True)
    minute = dt.dt.hour * 60 + dt.dt.minute
    out["tod_sin"] = np.sin(2 * np.pi * minute / 1440.0)
    out["tod_cos"] = np.cos(2 * np.pi * minute / 1440.0)
    out["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7.0)

    c = btc_candles[["ts", "c"]].drop_duplicates("ts").sort_values("ts")
    close = pd.Series(c["c"].values, index=c["ts"].values)
    close = close.reindex(out["ts"].values).ffill()
    ret = np.log(close).diff()
    bar_min = int(re.match(r"(\d+)", bar).group(1)) if bar.endswith("m") else 60
    bars_per_day = max(1, round(1440 / bar_min))
    bars_per_year = bars_per_day * 365
    rv = ret.rolling(bars_per_day, min_periods=max(20, bars_per_day // 4)).std() * math.sqrt(bars_per_year) * 100
    out["btc_rv_1d_ann_pct"] = rv.values
    out["btc_rv_1d_ann_pct_z20"] = rolling_z(pd.Series(rv.values), 20 * bars_per_day).values
    return out


def enrich_panel(
    panel: pd.DataFrame,
    btc_candles: pd.DataFrame,
    bar: str,
    daily_external: pd.DataFrame,
    short_external: pd.DataFrame,
    *,
    include_short: bool,
) -> pd.DataFrame:
    out = add_calendar_and_rv(panel, btc_candles, bar)
    out = asof_join_features(out, daily_external, DAY_MS, "")
    if include_short:
        out = asof_join_features(out, short_external, 5 * 60_000, "")
    if "dvol_daily" in out.columns and "btc_rv_1d_ann_pct" in out.columns:
        out["dvol_minus_rv_1d"] = out["dvol_daily"] - out["btc_rv_1d_ann_pct"]
        out["vrp_dvol2_minus_rv2_1d"] = (out["dvol_daily"] ** 2) - (out["btc_rv_1d_ann_pct"] ** 2)
    return out


def funding_sum_over_horizon(ts_values: np.ndarray, funding_df: pd.DataFrame, H_min: int) -> np.ndarray:
    """Realized funding sum in (entry_ts, entry_ts + H]. Positive means longs pay shorts.

    This is label/evaluation information only. It must not be exposed as a model feature.
    """
    if funding_df is None or not len(funding_df):
        return np.zeros(len(ts_values), dtype=float)
    f = funding_df[["ts", "funding"]].dropna().drop_duplicates("ts").sort_values("ts")
    if not len(f):
        return np.zeros(len(ts_values), dtype=float)
    fts = f["ts"].astype("int64").to_numpy()
    vals = f["funding"].astype(float).to_numpy()
    csum = np.concatenate([[0.0], np.cumsum(vals)])
    starts = np.searchsorted(fts, ts_values.astype("int64"), side="right")
    ends = np.searchsorted(fts, ts_values.astype("int64") + H_min * 60_000, side="right")
    return csum[ends] - csum[starts]


def apply_funding_hold_cost(panel: pd.DataFrame, funding_df: pd.DataFrame, H_min: int) -> pd.DataFrame:
    """Adjust the forward target to include directional funding over the hold window.

    For a long, net return is price_return - funding_sum. For a short, net return is
    -price_return + funding_sum, which is equivalent to using adjusted fwd =
    price_return - funding_sum before multiplying by the position sign.
    """
    out = panel.copy()
    fsum = funding_sum_over_horizon(out["ts"].to_numpy(), funding_df, H_min)
    out["fwd"] = out["fwd"].astype(float) - fsum
    out["y"] = (out["fwd"] > 0).astype(float)
    out.loc[~np.isfinite(out["fwd"]) | (out["fwd"] == 0.0), "y"] = np.nan
    return out


def run_variant(
    name: str,
    perp_dfs: dict[str, pd.DataFrame],
    spot_dfs: dict[str, pd.DataFrame],
    funding_dfs: dict[str, pd.DataFrame],
    *,
    bar: str,
    horizons: tuple[int, ...],
    n_folds: int,
    k_sel: int,
    preset: str,
    costs: pmw.WorkflowCosts,
    daily_external: pd.DataFrame | None,
    short_external: pd.DataFrame | None,
    include_short: bool,
    funding_hold_cost_mode: str,
    save_panels: bool,
    perp_key: str = "BTC-USDT-SWAP",
) -> tuple[dict, pd.DataFrame]:
    res = {}
    coverage_rows = []
    min_cell_ts = 50
    btc = perp_dfs[perp_key]
    for H in horizons:
        log(f"{name}: H={H} build panel")
        panel, bar_ms = pmw.build_augmented_bank(perp_dfs, spot_dfs, funding_dfs, bar, H)
        use_funding_hold_cost = (
            funding_hold_cost_mode == "always"
            or (funding_hold_cost_mode == "auto" and H >= 240)
        )
        if use_funding_hold_cost:
            panel = apply_funding_hold_cost(panel, funding_dfs.get(perp_key), H)
        base_cols = set(panel.columns)
        if daily_external is not None:
            short_df = short_external if short_external is not None else pd.DataFrame()
            panel = enrich_panel(panel, btc, bar, daily_external, short_df, include_short=include_short)
        new_cols = sorted([c for c in panel.columns if c not in base_cols and c not in ("ts", "inst", "fwd", "y")])
        for c in new_cols:
            coverage_rows.append({"variant": name, "H_min": H, "feature": c, "coverage": float(panel[c].notna().mean())})
        if save_panels:
            d = DATA_ROOT / "panels"
            d.mkdir(parents=True, exist_ok=True)
            panel.to_csv(d / f"{name}_H{H}_panel.csv.gz", index=False, compression="gzip")

        feats = pmw.feature_cols(panel)
        n_rows = len(panel.dropna(subset=["fwd", "y"]))
        fold_min_train = min(1200, max(600, int(n_rows * 0.35)))
        comp = pmw.purged_model_compare(
            panel, feats, H, bar_ms, n_folds=n_folds, k_sel=k_sel,
            min_train=fold_min_train, preset=preset, log=log,
        )
        if comp.get("error"):
            return comp, pd.DataFrame(coverage_rows)
        metrics = {}
        best = None
        for model_name, (odf, train_abs) in comp["models"].items():
            mm = pmw.evaluate_scores(odf, H, bar_ms, train_abs, costs=costs, mode="single_asset") if len(odf) else {"skip": True, "n": 0}
            metrics[model_name] = mm
            if mm.get("skip"):
                continue
            cells = [(fr, c) for fr, c in mm["curve"] if c is not None]
            pass_cells = [(fr, c) for fr, c in cells if pmw._cell_pass(c, costs, min_ts=min_cell_ts)]
            stress_cells = [(fr, c) for fr, c in pass_cells if pmw._stress_pass(c, costs)]
            gate = mm.get("primary_ic", 0.0) > 0.01 and len(pass_cells) >= 2 and len(stress_cells) >= 1
            score_cell = max(
                cells,
                key=lambda fc: (fc[1].get("net_10", -1e9), fc[1].get("t_10", -1e9), fc[1].get("net_4", -1e9)),
                default=(None, None),
            )
            cand = {
                "model": model_name,
                "gate": gate,
                "pass_count": len(pass_cells),
                "stress_count": len(stress_cells),
                "best_frac": score_cell[0],
                "best_cell": score_cell[1],
                "metric": mm,
            }
            if best is None:
                best = cand
            else:
                b = best.get("best_cell") or {}
                c = cand.get("best_cell") or {}
                if (cand["gate"], c.get("net_10", -1e9), c.get("t_10", -1e9), c.get("net_4", -1e9)) > (
                    best["gate"], b.get("net_10", -1e9), b.get("t_10", -1e9), b.get("net_4", -1e9)
                ):
                    best = cand
        edge_ok = bool(best and best.get("gate"))
        res[H] = {
            "n_features": len(feats),
            "selected_freq": comp["selected_freq"][:30],
            "folds": [
                {"fold": r["fold"], "n_train": r["n_train"], "n_test": r["n_test"], "selected": r["selected"][:15]}
                for r in comp["fold_reports"]
            ],
            "metrics": metrics,
            "best": best,
            "position": pmw._position_from_cell(best.get("best_cell") if best else None, costs, edge_ok=edge_ok),
            "bar_ms": bar_ms,
            "min_train": fold_min_train,
        }
    out = {
        "bar": bar,
        "n_inst": len(perp_dfs),
        "horizons": list(horizons),
        "has_spot": bool(spot_dfs),
        "has_funding": bool(funding_dfs),
        "has_lightgbm": pmw._HAS_LGBM,
        "mode": "single_asset",
        "min_cell_ts": min_cell_ts,
        "preset": preset,
        "costs": costs.__dict__,
        "funding_hold_cost_mode": funding_hold_cost_mode,
        "by_h": res,
    }
    return out, pd.DataFrame(coverage_rows)


def summary_rows(res: dict, variant: str) -> list[dict]:
    rows = []
    if res.get("error"):
        return rows
    for H in res["horizons"]:
        best = res["by_h"][H].get("best") or {}
        cell = best.get("best_cell") or {}
        rows.append({
            "variant": variant,
            "H_min": H,
            "model": best.get("model"),
            "gate": bool(best.get("gate")),
            "best_top_frac": best.get("best_frac"),
            "primary_ic": (best.get("metric") or {}).get("primary_ic"),
            "auc": (best.get("metric") or {}).get("auc"),
            "winrate": cell.get("winrate"),
            "gross_bps": cell.get("gross_bps"),
            "net4_bps": cell.get("net_4"),
            "t4": cell.get("t_4"),
            "net10_bps": cell.get("net_10"),
            "t10": cell.get("t_10"),
            "net15_bps": cell.get("net_15"),
            "t15": cell.get("t_15"),
            "n_ts": cell.get("n_ts"),
            "n_features": res["by_h"][H].get("n_features"),
        })
    return rows


def parse_existing_btc_report() -> pd.DataFrame:
    p = ROOT / "dist" / "pro_model_workflow_btc_report.txt"
    if not p.exists():
        return pd.DataFrame()
    txt = p.read_text(encoding="utf-8", errors="ignore")
    pat = re.compile(
        r"best: H=(?P<H>\d+) (?P<model>\S+) top(?P<top>[\d.]+)% gate=(?P<gate>\w+) "
        r"win=(?P<win>[\d.]+)% gross=(?P<gross>[+-][\d.]+) "
        r"net4=(?P<net4>[+-][\d.]+)\(t(?P<t4>[+-][\d.]+)\) "
        r"net10=(?P<net10>[+-][\d.]+)\(t(?P<t10>[+-][\d.]+)\) "
        r"net15=(?P<net15>[+-][\d.]+)\(t(?P<t15>[+-][\d.]+)\) n_ts=(?P<n>\d+)"
    )
    rows = []
    for m in pat.finditer(txt):
        rows.append({
            "variant": "existing_report",
            "H_min": int(m.group("H")),
            "model": m.group("model"),
            "gate": m.group("gate") == "PASS",
            "best_top_frac": float(m.group("top")) / 100.0,
            "winrate": float(m.group("win")) / 100.0,
            "gross_bps": float(m.group("gross")),
            "net4_bps": float(m.group("net4")),
            "t4": float(m.group("t4")),
            "net10_bps": float(m.group("net10")),
            "t10": float(m.group("t10")),
            "net15_bps": float(m.group("net15")),
            "t15": float(m.group("t15")),
            "n_ts": int(m.group("n")),
        })
    return pd.DataFrame(rows)


def write_comparison(existing: pd.DataFrame, base: pd.DataFrame, enhanced: pd.DataFrame, coverage: pd.DataFrame, inventory: pd.DataFrame, asset: str = "btc") -> str:
    A = asset.upper()
    lines = [
        "=" * 92,
        f"{A} single-asset enhanced variable audit",
        "=" * 92,
        "Decision rule: trade only if OOS primary_ic>0.01, at least two confidence buckets pass maker+taker net>0 with NW-t>2, and at least one bucket survives 15 bps stress.",
        "The selected best row is ranked by taker net edge first, then taker t-stat, then maker edge.",
        "",
        "Existing failure baseline from dist/pro_model_workflow_btc_report.txt:",
    ]
    if len(existing):
        for _, r in existing.iterrows():
            lines.append(
                f"  H={int(r.H_min):2d} {r.model:>12} gate={r.gate} "
                f"net4={r.net4_bps:+.1f}(t{r.t4:+.1f}) net10={r.net10_bps:+.1f}(t{r.t10:+.1f}) "
                f"net15={r.net15_bps:+.1f}(t{r.t15:+.1f}) n={int(r.n_ts)}"
            )
    else:
        lines.append("  not found or not parseable")

    lines.extend(["", "Same-code rerun summary:"])
    all_rows = pd.concat([base, enhanced], ignore_index=True)
    for _, r in all_rows.iterrows():
        lines.append(
            f"  {r.variant:>9} H={int(r.H_min):2d} {str(r.model):>12} gate={bool(r.gate)} "
            f"features={int(r.n_features)} primary_ic={r.primary_ic:+.4f} auc={r.auc:.3f} "
            f"net4={r.net4_bps:+.1f}(t{r.t4:+.1f}) net10={r.net10_bps:+.1f}(t{r.t10:+.1f}) "
            f"net15={r.net15_bps:+.1f}(t{r.t15:+.1f}) n={int(r.n_ts) if pd.notna(r.n_ts) else 0}"
        )

    lines.extend(["", "Enhanced minus same-code baseline:"])
    for H in sorted(set(base["H_min"]).intersection(set(enhanced["H_min"]))):
        b = base[base["H_min"] == H].iloc[0]
        e = enhanced[enhanced["H_min"] == H].iloc[0]
        lines.append(
            f"  H={int(H):2d}: d_net10={e.net10_bps - b.net10_bps:+.1f}bps "
            f"d_t10={e.t10 - b.t10:+.2f} d_net4={e.net4_bps - b.net4_bps:+.1f}bps "
            f"gate {bool(b.gate)} -> {bool(e.gate)}"
        )

    lines.extend(["", "External source inventory:"])
    if len(inventory):
        for _, r in inventory.fillna("").iterrows():
            err = f" error={r.get('error')}" if r.get("error") else ""
            lines.append(
                f"  {r.get('source')}: rows={r.get('rows')} span={r.get('span_days','')} "
                f"{r.get('start_utc','')} -> {r.get('end_utc','')} use={r.get('use','')}{err}"
            )

    lines.extend(["", "New-feature coverage in enhanced panels (top 25 by coverage):"])
    if len(coverage):
        cov = coverage[coverage["variant"] == "enhanced"].sort_values(["coverage", "feature"], ascending=[False, True]).head(25)
        for _, r in cov.iterrows():
            lines.append(f"  H={int(r.H_min):2d} {r.feature}: {r.coverage:.1%}")

    tradable = bool(enhanced["gate"].any()) if len(enhanced) else False
    lines.extend(["", "Verdict:"])
    if tradable:
        lines.append("  CANDIDATE ONLY: an enhanced model passed the historical OOS fee gate. It still needs live shadow validation on the raw L2/trade recorder.")
    else:
        lines.append(f"  NO TRADE: the enhanced {A}-specific variables did not prove fee-adjusted out-of-sample net edge. Best model is research-only; size and leverage remain zero.")
    lines.append("=" * 92)
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Single-asset enhanced variable audit (BTC/ETH).")
    p.add_argument("--asset", default="btc", choices=list(ASSETS.keys()),
                   help="标的: btc(默认) 或 eth。eth 用同一套特征工程在 ETH 数据上跑。")
    p.add_argument("--days", type=int, default=150)
    p.add_argument("--bar", default="5m")
    p.add_argument("--horizons", default="15,30")
    p.add_argument("--preset", default="fast", choices=["fast", "deep"])
    p.add_argument("--n-folds", type=int, default=4)
    p.add_argument("--k-sel", type=int, default=30)
    p.add_argument("--candle-source", default="dist", choices=["dist", "refresh"])
    p.add_argument("--force", action="store_true")
    p.add_argument("--include-short-intraday", action="store_true", help="Include short-window 5m Rubik data. Usually too short for 150d OOS.")
    p.add_argument(
        "--funding-hold-cost",
        default="none",
        choices=["none", "auto", "always"],
        help="Adjust labels/OOS returns by realized funding in the hold window; auto applies to H>=240min.",
    )
    p.add_argument("--save-panels", action="store_true")
    args = p.parse_args()

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    asset = args.asset
    a = ASSETS[asset]
    perp_inst, spot_inst = a["perp"], a["spot"]
    horizons = tuple(int(x) for x in args.horizons.split(",") if x.strip())
    source = "refresh" if args.candle_source == "refresh" else "dist"
    log(f"load candles ({asset})")
    btc = load_candles(perp_inst, args.bar, args.days, source=source, force=args.force)
    spot = load_candles(spot_inst, args.bar, args.days, source=source, force=args.force)
    funding = cd.fetch_funding_series(perp_inst, args.days + 5, log=log)

    span_days = (int(btc["ts"].iloc[-1]) - int(btc["ts"].iloc[0])) / DAY_MS
    meta = {"span_days": round(span_days, 1)}
    pd.DataFrame([{
        "inst_id": perp_inst,
        "bar": args.bar,
        "rows": len(btc),
        "start_utc": ts_utc(btc["ts"].iloc[0]),
        "end_utc": ts_utc(btc["ts"].iloc[-1]),
        "span_days": span_days,
    }]).to_csv(DATA_ROOT / f"{asset}_candle_inventory.csv", index=False)

    log("fetch external variables")
    daily_external, inv_daily = fetch_daily_external(args.days, force=args.force, asset=asset)
    short_external, inv_short = fetch_short_intraday_external(force=args.force, asset=asset)
    inventory = pd.concat([inv_daily, inv_short], ignore_index=True)
    inventory.to_csv(DATA_ROOT / f"{asset}_source_inventory_all.csv", index=False)

    costs = pmw.WorkflowCosts()
    perp_dfs = {perp_inst: btc}
    spot_dfs = {perp_inst: spot}
    funding_dfs = {perp_inst: funding}

    log("run same-code baseline")
    base_res, base_cov = run_variant(
        "baseline", perp_dfs, spot_dfs, funding_dfs,
        bar=args.bar, horizons=horizons, n_folds=args.n_folds, k_sel=args.k_sel,
        preset=args.preset, costs=costs, daily_external=None, short_external=None,
        include_short=False, funding_hold_cost_mode=args.funding_hold_cost, save_panels=False,
        perp_key=perp_inst,
    )

    log("run enhanced variable model")
    enh_res, enh_cov = run_variant(
        "enhanced", perp_dfs, spot_dfs, funding_dfs,
        bar=args.bar, horizons=horizons, n_folds=args.n_folds, k_sel=args.k_sel,
        preset=args.preset, costs=costs, daily_external=daily_external,
        short_external=short_external, include_short=args.include_short_intraday,
        funding_hold_cost_mode=args.funding_hold_cost,
        save_panels=args.save_panels, perp_key=perp_inst,
    )

    label = f"{asset.upper()} horizon modeling workflow"
    base_report = pmw.format_report(meta, base_res).replace(
        "15/30min professional modeling workflow", label)
    enhanced_report = pmw.format_report(meta, enh_res).replace(
        "15/30min professional modeling workflow", label)
    (REPORT_ROOT / f"{asset}_same_code_baseline_report.txt").write_text(base_report, encoding="utf-8")
    (REPORT_ROOT / f"{asset}_enhanced_report.txt").write_text(enhanced_report, encoding="utf-8")

    base_summary = pd.DataFrame(summary_rows(base_res, "baseline"))
    enhanced_summary = pd.DataFrame(summary_rows(enh_res, "enhanced"))
    existing = parse_existing_btc_report() if asset == "btc" else pd.DataFrame()
    coverage = pd.concat([base_cov, enh_cov], ignore_index=True)
    coverage.to_csv(DATA_ROOT / f"{asset}_feature_coverage.csv", index=False)
    base_summary.to_csv(DATA_ROOT / f"{asset}_baseline_summary.csv", index=False)
    enhanced_summary.to_csv(DATA_ROOT / f"{asset}_enhanced_summary.csv", index=False)

    comp = write_comparison(existing, base_summary, enhanced_summary, coverage, inventory, asset=asset)
    (REPORT_ROOT / f"{asset}_enhanced_comparison.txt").write_text(comp, encoding="utf-8")
    run_meta = {
        "asset": asset,
        "days": args.days,
        "bar": args.bar,
        "horizons": horizons,
        "preset": args.preset,
        "n_folds": args.n_folds,
        "k_sel": args.k_sel,
        "candle_source": args.candle_source,
        "include_short_intraday": args.include_short_intraday,
        "funding_hold_cost": args.funding_hold_cost,
        "has_lightgbm": pmw._HAS_LGBM,
        "outputs": {
            "comparison": str(REPORT_ROOT / f"{asset}_enhanced_comparison.txt"),
            "baseline_report": str(REPORT_ROOT / f"{asset}_same_code_baseline_report.txt"),
            "enhanced_report": str(REPORT_ROOT / f"{asset}_enhanced_report.txt"),
        },
    }
    (REPORT_ROOT / f"{asset}_run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    print(comp)


if __name__ == "__main__":
    main()
