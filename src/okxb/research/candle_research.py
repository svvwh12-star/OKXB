"""分钟~小时级方向预测研究 (H 对齐, 扣费后净 edge) —— 用 K 线, 不是秒级 tick。

目的: 严格回答『用合适的分钟级特征, 能否预测未来 H(5~240min)的方向, 且扣费后为正?』
这是秒级 validate.py【无法】检验的问题。换数据(K线)、换特征(分钟动量/波动/量/相对强弱)、
换标签(未来H收益)、换回测(逐H、非重叠、扣费、样本外)。

—— 经 4-Agent 对抗审计后加固 (v2)。修掉了会【凭空造出 edge】的偏差: ——
  [防前视] 1) 每个标的先 reindex 到完整 bar 网格, 再造特征/标签 → shift(k) 恰好 = k×bar 真分钟,
            缺K线的窗口变 NaN 丢弃 (否则 fwd_H 会横跨缺口、虚增幅度)。
           2) 横截面标准化只在单 ts 内; 训练/测试按时间切; 组合信号符号只由训练段定, 且
            embargo 训练段尾部 hb 根 (防标签泄漏到测试边界)。
  [真显著] 3) 回测 t 值一律【组合层面】: 先把每个 rebalance 时刻聚成 1 个组合收益, 再算 t
            (而非把跨币高度相关的 (币,时) 拉平成伪 i.i.d., 那会把 n 灌水 ~√N倍、造假显著)。
           4) 用 Newey-West(HAC) 抗自相关 t; IC 的 IR 用【非重叠 H 间隔】采样 (不是逐bar)。
  [真成本] 5) 方向择时书计入【资金费】(H≥2h 单边、与动量同号、不抵消); 横截面多空书资金费
            多空相抵, 不计。费率为"仅手续费地板"(maker4/taker10bps), 半价差/滑点未计, 已注明。
  [真稳健] 6) 主判据 = 横截面多空的【走步(walk-forward)合并样本外】NW-t, 不是单次切分;
            逐 H 报 rebalance 次数, 样本太薄不下结论。方向择时书降级为【探索性】, 不进 go/no-go。

三类 IC: pooled(池化,受市场beta混淆,仅参考) / ts_ic(每币自身择时) /
  xs_ic(每个ts横截面排名相关=选币alpha, 对全市场共涨跌天然中性)。
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

BAR_MIN = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1H": 60, "2H": 120, "4H": 240}
_MOM_LOOKBACKS_MIN = (15, 30, 60, 120, 240)
DEFAULT_HORIZONS_MIN = (5, 15, 30, 60, 120, 240)
FUNDING_INTERVAL_MIN = 480.0   # OKX 资金费每 8h 一次


def _log(m: str) -> None:
    print(m, flush=True)


# ----------------------------- 相关性 / 统计工具 -----------------------------

def _pearson(x, y) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    x = x[m] - x[m].mean(); y = y[m] - y[m].mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 1e-15 else np.nan


def _spearman(x, y) -> float:
    x = pd.Series(np.asarray(x, float)); y = pd.Series(np.asarray(y, float))
    m = x.notna() & y.notna()
    if m.sum() < 3:
        return np.nan
    return _pearson(x[m].rank().values, y[m].rank().values)


def _nw_se(x: np.ndarray, lag: int) -> float:
    """均值的 Newey-West(HAC) 标准误 (抗序列自相关)。"""
    x = np.asarray(x, float)
    n = len(x)
    if n < 3:
        return np.nan
    u = x - x.mean()
    s = float((u * u).mean())
    for l in range(1, min(lag, n - 1) + 1):
        w = 1.0 - l / (lag + 1.0)
        s += 2.0 * w * float((u[l:] * u[:-l]).mean())
    s = max(s, 1e-30)
    return float(np.sqrt(s / n))


def _nw_t(x: np.ndarray, lag: Optional[int] = None) -> float:
    x = np.asarray(x, float)
    n = len(x)
    if n < 5:
        return np.nan
    if lag is None:
        lag = max(1, int(n ** 0.25))
    se = _nw_se(x, lag)
    return float(x.mean() / se) if se and se > 0 else np.nan


def _autocorr1(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    if len(x) < 5:
        return np.nan
    return _pearson(x[1:], x[:-1])


def _xs_ic_series(df: pd.DataFrame, fcol: str, ycol: str, min_n: int = 5) -> pd.Series:
    """每个 ts 的横截面秩相关 (向量化)。返回 index=ts 的 IC 序列。"""
    s = df[["ts", fcol, ycol]].dropna()
    if s.empty:
        return pd.Series(dtype=float)
    s = s.assign(fr=s.groupby("ts")[fcol].rank(), yr=s.groupby("ts")[ycol].rank())
    cnt = s.groupby("ts")["fr"].transform("count")
    s = s[cnt >= min_n]
    if s.empty:
        return pd.Series(dtype=float)
    frm = s["fr"] - s.groupby("ts")["fr"].transform("mean")
    yrm = s["yr"] - s.groupby("ts")["yr"].transform("mean")
    num = (frm * yrm).groupby(s["ts"]).sum()
    den = np.sqrt((frm * frm).groupby(s["ts"]).sum() * (yrm * yrm).groupby(s["ts"]).sum())
    return (num / den.replace(0, np.nan)).dropna()


# ----------------------------- 特征 / 标签 (网格对齐) -----------------------------

def build_inst_panel(df: pd.DataFrame, bar_min: int, bar_ms: int,
                     horizons_min=DEFAULT_HORIZONS_MIN) -> tuple[pd.DataFrame, int]:
    """单标的: 先 reindex 到完整 bar 网格(缺K线→NaN), 再造分钟级特征 + 各 H 未来收益标签。
    返回 (panel, gap_count)。全 point-in-time; shift(k)=k×bar 真分钟。"""
    d = df.sort_values("ts").reset_index(drop=True)
    full = np.arange(int(d["ts"].iloc[0]), int(d["ts"].iloc[-1]) + bar_ms, bar_ms)
    d = d.set_index("ts").reindex(full)
    # 坏 bar 清洗: 非正价格 → NaN (否则 log(c<=0)=-inf/nan 污染 vol/收益与排序); 体量列负值清零
    for col in ("o", "h", "l", "c"):
        d[col] = d[col].where(d[col] > 0)
    d["volquote"] = d["volquote"].where(d["volquote"] >= 0)
    gap = int(d["c"].isna().sum())
    c, h_, l_, vq = d["c"], d["h"], d["l"], d["volquote"]
    logret = np.log(c).diff()

    def bars(minutes: int) -> int:
        return max(1, round(minutes / bar_min))

    out = pd.DataFrame({"ts": full})
    for mn in _MOM_LOOKBACKS_MIN:
        k = bars(mn)
        out[f"mom_{mn}"] = (c / c.shift(k) - 1.0).values

    k60 = bars(60)
    vol60 = logret.rolling(k60).std()
    out["vol_60"] = vol60.values
    out["madist_60"] = (c / c.rolling(k60).mean() - 1.0).values
    lo = l_.rolling(k60).min(); hi = h_.rolling(k60).max()
    out["rngpos_60"] = ((c - lo) / (hi - lo).replace(0, np.nan) - 0.5).values
    vqp = vq.replace(0, np.nan)   # 0成交量 bar → NaN, 避免 log(0)=-inf 污染横截面 z
    out["volsurge_60"] = np.log(vqp / vqp.rolling(k60).mean().replace(0, np.nan)).values
    out["vnmom_60"] = ((c / c.shift(k60) - 1.0) / (vol60 * np.sqrt(k60)).replace(0, np.nan)).values
    out["accel"] = (out["mom_30"] - out["mom_120"]).values
    delta = c.diff(); up = delta.clip(lower=0); dn = (-delta).clip(lower=0)
    rs = up.rolling(14).mean() / dn.rolling(14).mean().replace(0, np.nan)
    out["rsi_14"] = (100 - 100 / (1 + rs) - 50).values

    for H in horizons_min:
        hb = max(1, round(H / bar_min))
        out[f"fwd_{H}"] = (c.shift(-hb) / c - 1.0).values
    return out, gap


def feature_cols(panel: pd.DataFrame) -> list[str]:
    return [c for c in panel.columns
            if c not in ("ts", "inst") and not c.startswith("fwd_") and not c.startswith("xs_")]


def directional_feats(feat_cols: list[str]) -> list[str]:
    return [f for f in feat_cols if f != "vol_60"]   # 纯波动幅度不是方向信号


def build_panel(dfs: dict[str, pd.DataFrame], bar: str,
                horizons_min=DEFAULT_HORIZONS_MIN) -> tuple[pd.DataFrame, dict]:
    bar_min = BAR_MIN[bar]; bar_ms = bar_min * 60_000
    parts = []
    gaps = {}
    for inst, df in dfs.items():
        p, gap = build_inst_panel(df, bar_min, bar_ms, horizons_min)
        p["inst"] = inst
        parts.append(p)
        gaps[inst] = (gap, len(p))
    panel = pd.concat(parts, ignore_index=True)
    fcols = feature_cols(panel)
    g = panel.groupby("ts")
    for f in fcols:
        mu = g[f].transform("mean"); sd = g[f].transform("std")
        panel["xs_" + f] = (panel[f] - mu) / sd.replace(0, np.nan)
    tot_gap = sum(v[0] for v in gaps.values())
    tot_rows = sum(v[1] for v in gaps.values())
    diag = {"gap_pct": 100.0 * tot_gap / max(1, tot_rows),
            "worst": sorted(((v[0] / max(1, v[1]), k) for k, v in gaps.items()), reverse=True)[:3]}
    return panel, diag


# ----------------------------- IC 表 -----------------------------

def _rebalance_ts(all_ts: np.ndarray, bar_ms: int, h: int,
                  lo: Optional[int] = None, hi: Optional[int] = None) -> np.ndarray:
    """在全局 bar 时钟上每 h 根取一个 rebalance 时刻 (保证相邻持仓非重叠)。"""
    if len(all_ts) == 0:
        return all_ts
    ts0 = int(all_ts[0])
    sel = all_ts[((all_ts - ts0) // bar_ms) % h == 0]
    if lo is not None:
        sel = sel[sel > lo]
    if hi is not None:
        sel = sel[sel <= hi]
    return sel


def ic_table(panel: pd.DataFrame, feat_cols: list[str], horizons_min, bar_ms: int) -> pd.DataFrame:
    all_ts = np.sort(panel["ts"].unique())
    insts = panel["inst"].unique()
    rows = []
    for f in feat_cols:
        for H in horizons_min:
            y = f"fwd_{H}"
            sub = panel[["ts", "inst", f, y]].dropna()
            if len(sub) < 100:
                continue
            pooled = _spearman(sub[f], sub[y])
            ts_ics = []
            for inst in insts:
                gg = sub[sub["inst"] == inst]
                if len(gg) >= 30:
                    v = _spearman(gg[f], gg[y])
                    if np.isfinite(v):
                        ts_ics.append(v)
            ts_ic = float(np.mean(ts_ics)) if ts_ics else np.nan
            xs = _xs_ic_series(sub, f, y)
            xs_ic = float(xs.mean()) if len(xs) else np.nan
            # IR 用【非重叠 H 间隔】采样 (匹配可交易频率, 不灌水)
            h = max(1, round(H / (bar_ms / 60_000)))
            reb = _rebalance_ts(all_ts, bar_ms, h)
            xs_h = xs.reindex(reb).dropna()
            xs_ir = (float(xs_h.mean() / xs_h.std() * np.sqrt(len(xs_h)))
                     if len(xs_h) > 2 and xs_h.std() > 0 else np.nan)
            rows.append(dict(feature=f, H=H, n=len(sub), pooled_ic=pooled, ts_ic=ts_ic,
                             xs_ic=xs_ic, xs_ir_h=xs_ir, n_ic=len(xs_h)))
    return pd.DataFrame(rows)


# ----------------------------- 回测 (网格非重叠 · 组合层面 · 扣费) -----------------------------

def xs_ls_gross(panel_seg: pd.DataFrame, signal_col: str, h: int, bar_ms: int,
                q: float, lo=None, hi=None, all_ts: Optional[np.ndarray] = None):
    """横截面多空: 每个 rebalance 时刻聚成 1 个组合毛收益(多腿均 − 空腿均)。
    返回 (gross_per_reb 数组, 每次的标的数列表)。"""
    sub = panel_seg[["ts", signal_col, f"_y"]].dropna()
    if sub.empty:
        return np.array([]), []
    if all_ts is None:
        all_ts = np.sort(panel_seg["ts"].unique())
    reb = _rebalance_ts(all_ts, bar_ms, h, lo, hi)
    by_ts = {ts: g for ts, g in sub.groupby("ts")}
    gross, ncoins = [], []
    for ts in reb:
        g = by_ts.get(ts)
        if g is None or len(g) < 5:
            continue
        g = g.sort_values(signal_col)
        k = max(1, int(len(g) * q))
        val = g["_y"].tail(k).mean() - g["_y"].head(k).mean()
        if np.isfinite(val):
            gross.append(val); ncoins.append(len(g))
    return np.array(gross), ncoins


def ts_dir_gross(panel_seg: pd.DataFrame, signal_col: str, h: int, bar_ms: int,
                 funding: dict, ebd: float, thr: float,
                 lo=None, hi=None, all_ts: Optional[np.ndarray] = None) -> np.ndarray:
    """每标的方向择时, 但【组合层面】聚合: 每个 rebalance 时刻把所有出手币的
    (sign×fwd − 资金费) 取均值成 1 个组合收益。资金费已扣(单边、与方向同号)。返回组合收益序列(未扣手续费)。"""
    sub = panel_seg[["ts", "inst", signal_col, "_y"]].dropna()
    if sub.empty:
        return np.array([])
    if all_ts is None:
        all_ts = np.sort(panel_seg["ts"].unique())
    reb = set(_rebalance_ts(all_ts, bar_ms, h, lo, hi).tolist())
    sub = sub[sub["ts"].isin(reb)]
    if sub.empty:
        return np.array([])
    sig = sub[signal_col].values
    fr = sub["_y"].values
    insts = sub["inst"].values
    fund = np.array([(funding.get(i) or 0.0) for i in insts])
    sgn = np.sign(sig)
    fire = np.abs(sig) > thr
    # 每币净(扣资金费, 未扣手续费): sign×fwd − sign×funding×E[boundaries]
    pertrade = sgn * fr - sgn * fund * ebd
    d = pd.DataFrame({"ts": sub["ts"].values, "r": pertrade, "fire": fire})
    d = d[d["fire"]]
    if d.empty:
        return np.array([])
    port = d.groupby("ts")["r"].mean()
    return port.values


def _stats_from_gross(gross: np.ndarray, cost_rt_bps: float, n_legs: int) -> Optional[dict]:
    """gross=每个rebalance的组合毛收益(分数); 扣 n_legs×往返成本后给净bps/NW-t/自相关。"""
    if len(gross) < 8:
        return None
    net = gross - n_legs * cost_rt_bps / 1e4
    return dict(n=len(net), gross_bps=float(gross.mean() * 1e4), net_bps=float(net.mean() * 1e4),
                nw_t=_nw_t(net), ac1=_autocorr1(net), pos=float((net > 0).mean()))


def _train_signs(panel: pd.DataFrame, dir_feats: list[str], H: int, embargo_cut: int) -> dict:
    tr = panel[panel["ts"] <= embargo_cut]
    signs = {}
    for f in dir_feats:
        xs = _xs_ic_series(tr.rename(columns={f"fwd_{H}": "_yf"}), f, "_yf") \
            if f"fwd_{H}" in tr.columns else pd.Series(dtype=float)
        m = xs.mean() if len(xs) else 0.0
        signs[f] = float(np.sign(m)) if np.isfinite(m) and abs(m) > 0 else 0.0
    return signs


def _composite(panel: pd.DataFrame, dir_feats: list[str], signs: dict) -> pd.Series:
    return sum(signs[f] * panel["xs_" + f].fillna(0.0) for f in dir_feats) / max(1, len(dir_feats))


def walk_forward(panel: pd.DataFrame, dir_feats: list[str], H: int, bar_min: int, bar_ms: int,
                 *, n_folds: int = 5, q: float = 0.2) -> dict:
    """走步样本外: 锚定扩窗。每折用训练段定特征符号(embargo), 在该折测试段做横截面多空。
    合并所有折的测试 rebalance 收益 → 主判据的毛收益序列。"""
    h = max(1, round(H / bar_min))
    all_ts = np.sort(panel["ts"].unique())
    n = len(all_ts)
    if n < (n_folds + 1) * h * 4:                # 太薄: 减折
        n_folds = max(2, n // (h * 8))
    bounds = [int(n * (i + 1) / (n_folds + 1)) for i in range(n_folds + 1)]  # 训练终点索引
    pooled_gross = []
    fold_stats = []
    for i in range(n_folds):
        tr_end = bounds[i]; te_end = bounds[i + 1]
        if tr_end - h <= 0:
            continue
        embargo_cut = int(all_ts[max(0, tr_end - h)])
        cut = int(all_ts[tr_end]); te_hi = int(all_ts[min(te_end, n - 1)])
        signs = _train_signs(panel, dir_feats, H, embargo_cut)
        comp = _composite(panel, dir_feats, signs)
        p = panel.assign(_comp=comp.values, _y=panel[f"fwd_{H}"].values)
        g, nc = xs_ls_gross(p, "_comp", h, bar_ms, q, lo=cut, hi=te_hi, all_ts=all_ts)
        if len(g):
            pooled_gross.append(g)
            fold_stats.append(dict(fold=i + 1, n=len(g),
                                   net_taker=float((g.mean() - 2 * 10 / 1e4) * 1e4)))
    pooled = np.concatenate(pooled_gross) if pooled_gross else np.array([])
    return dict(pooled_gross=pooled, n_folds=len(fold_stats), folds=fold_stats)


def single_split(panel: pd.DataFrame, dir_feats: list[str], H: int, bar_min: int, bar_ms: int,
                 *, train_frac: float = 0.7, q: float = 0.2):
    """单次 70/30 切分 (作对照 + 方向择时探索通道)。返回 (test_panel_with_comp, cut, all_ts)。"""
    h = max(1, round(H / bar_min))
    all_ts = np.sort(panel["ts"].unique()); n = len(all_ts)
    ti = int(n * train_frac)
    cut = int(all_ts[ti]); embargo_cut = int(all_ts[max(0, ti - h)])
    signs = _train_signs(panel, dir_feats, H, embargo_cut)
    comp = _composite(panel, dir_feats, signs)
    p = panel.assign(_comp=comp.values, _y=panel[f"fwd_{H}"].values)
    return p, cut, all_ts, signs


# ----------- 选择性曲线 + 逐H拟合模型 (回答: 只做最自信的top-X%, 扣费后转正吗?) -----------

def _ridge_fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0):
    Xc = X - X.mean(0); yc = y - y.mean()
    k = X.shape[1]
    coef = np.linalg.solve(Xc.T @ Xc + l2 * np.eye(k), Xc.T @ yc)
    return coef, float(y.mean() - X.mean(0) @ coef)


def fit_predict_oos(panel: pd.DataFrame, dir_feats: list[str], H: int,
                    train_frac: float = 0.7, l2: float = 1.0):
    """逐 H 岭回归: 训练段 xs_z 特征 → fwd_H; 全样本预测 = 期望收益估计 _pred (point-in-time)。
    返回 (panel+_pred, cut, test_ic)。这是用户要的 'H 专用 expected_net 模型' 的最简形态。"""
    cols = ["xs_" + f for f in dir_feats]
    ts_sorted = np.sort(panel["ts"].unique())
    cut = int(ts_sorted[int(len(ts_sorted) * train_frac)])
    tr = panel[panel["ts"] <= cut][cols + [f"fwd_{H}"]].dropna()
    if len(tr) < 500:
        return None, cut, np.nan
    coef, b0 = _ridge_fit(tr[cols].values, tr[f"fwd_{H}"].values, l2)
    p = panel.copy()
    p["_pred"] = panel[cols].fillna(0.0).values @ coef + b0
    te = p[p["ts"] > cut]
    xs = _xs_ic_series(te, "_pred", f"fwd_{H}")
    return p, cut, (float(xs.mean()) if len(xs) else np.nan)


def selectivity_curve(panel_pred: pd.DataFrame, H: int, bar_min: int, bar_ms: int, cut: int,
                      cost_rt_bps: float, fracs=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 1.0)):
    """只交易 |_pred|(期望收益) 在【训练分布】top-X% 的信号(因果阈值), 方向=sign(_pred), 持有H,
    扣单边往返费, 组合层面(每ts聚合)NW-t。回答: 越稀疏越自信地做, OOS 扣费后能否转正?"""
    h = max(1, round(H / bar_min))
    all_ts = np.sort(panel_pred["ts"].unique())
    reb = set(_rebalance_ts(all_ts, bar_ms, h, lo=cut).tolist())
    trabs = panel_pred[panel_pred["ts"] <= cut]["_pred"].abs().dropna()
    te = panel_pred[(panel_pred["ts"] > cut) & (panel_pred["ts"].isin(reb))]
    sub = te[["ts", "_pred", f"fwd_{H}"]].dropna()
    out = []
    for fr in fracs:
        thr = 0.0 if fr >= 1.0 else float(trabs.quantile(1 - fr))
        s = sub[sub["_pred"].abs() >= thr]
        if len(s) < 20:
            out.append((fr, None)); continue
        r = np.sign(s["_pred"].values) * s[f"fwd_{H}"].values - cost_rt_bps / 1e4
        port = pd.Series(r, index=s["ts"].values).groupby(level=0).mean().values
        if len(port) < 8:
            out.append((fr, None)); continue
        out.append((fr, dict(n_ts=len(port), n_tr=len(s), net_bps=float(port.mean() * 1e4),
                             nw_t=_nw_t(port), pos=float((port > 0).mean()))))
    return out


# ----------------------------- 编排 + 报告 -----------------------------

def run(dfs: dict[str, pd.DataFrame], bar: str, *,
        horizons_min=DEFAULT_HORIZONS_MIN, funding: Optional[dict] = None,
        cost_taker_bps: float = 10.0, cost_maker_bps: float = 4.0,
        q: float = 0.2, n_folds: int = 5, log: Callable[[str], None] = _log) -> dict:
    bar_min = BAR_MIN[bar]; bar_ms = bar_min * 60_000
    funding = funding or {}
    log(f"组装面板(网格对齐): {len(dfs)} 标的 / bar={bar} ...")
    panel, pdiag = build_panel(dfs, bar, horizons_min)
    fcols = feature_cols(panel); dfeats = directional_feats(fcols)
    log(f"面板 {len(panel)} 行(含网格NaN), 缺K线 {pdiag['gap_pct']:.2f}% ; 计算 IC ...")
    ict = ic_table(panel, fcols, horizons_min, bar_ms)

    ctx = {}
    for H in horizons_min:
        fr = panel[f"fwd_{H}"].dropna().values
        if len(fr):
            ctx[H] = dict(absmean_bps=float(np.mean(np.abs(fr)) * 1e4),
                          std_bps=float(np.std(fr) * 1e4))

    res_ls = {}     # 主判据: 横截面多空走步OOS
    res_dir = {}    # 探索: 方向择时(单split测试段, 含资金费)
    for H in horizons_min:
        log(f"  H={H}min: 走步多空 + 方向择时 ...")
        wf = walk_forward(panel, dfeats, H, bar_min, bar_ms, n_folds=n_folds, q=q)
        g = wf["pooled_gross"]
        res_ls[H] = {"taker": _stats_from_gross(g, cost_taker_bps, 2),
                     "maker": _stats_from_gross(g, cost_maker_bps, 2),
                     "n_folds": wf["n_folds"], "folds": wf["folds"]}
        # 方向择时探索通道: 单split, 测试段, 组合信号(已OOS符号), 资金费扣减, 组合层面 t
        p, cut, all_ts, _ = single_split(panel, dfeats, H, bar_min, bar_ms, q=q)
        h = max(1, round(H / bar_min)); ebd = H / FUNDING_INTERVAL_MIN
        dg = ts_dir_gross(p, "_comp", h, bar_ms, funding, ebd, thr=0.3,
                          lo=cut, hi=int(all_ts[-1]), all_ts=all_ts)
        res_dir[H] = {"taker": _stats_from_gross(dg, cost_taker_bps, 1),
                      "maker": _stats_from_gross(dg, cost_maker_bps, 1),
                      "ebd": ebd}

    # 选择性曲线 + 逐H岭回归模型 (回答: 只做最自信的 top-X%, 扣费后转正吗?)
    res_sel = {}
    for H in horizons_min:
        pp, cut, te_ic = fit_predict_oos(panel, dfeats, H)
        if pp is None:
            res_sel[H] = None
            continue
        res_sel[H] = {"test_ic": te_ic,
                      "maker": selectivity_curve(pp, H, bar_min, bar_ms, cut, cost_maker_bps),
                      "taker": selectivity_curve(pp, H, bar_min, bar_ms, cut, cost_taker_bps)}

    return dict(panel_rows=len(panel), n_inst=len(dfs), bar=bar, feat_cols=fcols,
                dir_feats=dfeats, ic=ict, ctx=ctx, ls=res_ls, dir=res_dir, sel=res_sel, pdiag=pdiag,
                cost_taker=cost_taker_bps, cost_maker=cost_maker_bps, q=q,
                n_funding=sum(1 for v in funding.values() if v is not None),
                horizons=list(horizons_min))


def _hl(H: int) -> str:
    return f"{H}min" if H < 60 else f"{H // 60}h"


def _flag(s: Optional[dict]) -> str:
    if s is None:
        return "样本不足"
    if s["net_bps"] > 0 and np.isfinite(s["nw_t"]) and s["nw_t"] > 2:
        return "净正✓(显著)"
    if s["net_bps"] > 0:
        return "弱正(不显著)"
    return "净负✗"


def format_report(meta: dict, res: dict) -> str:
    L = ["=" * 72,
         "分钟~小时级方向预测检验 (K线 · H对齐 · 扣费后净edge · 走步OOS · 已过对抗审计v2)",
         "=" * 72,
         f"数据: {res['n_inst']} 加密永续 / bar={res['bar']} / 面板 {res['panel_rows']} 行 / "
         f"跨 {meta.get('span_days','?')} 天 / 缺K线 {res['pdiag']['gap_pct']:.2f}%",
         f"成本: maker往返 {res['cost_maker']:.0f}bps | taker往返 {res['cost_taker']:.0f}bps "
         f"(仅手续费地板, 半价差/滑点未计) ; 资金费已扣的标的 {res['n_funding']}/{res['n_inst']}", ""]

    L.append("① 各 H 市场移动幅度 (扣费参照: 长周期移动远大于成本, 瓶颈在能否预测方向):")
    for H in res["horizons"]:
        c = res["ctx"].get(H)
        if c:
            L.append(f"   {_hl(H):>5}: |均移动|={c['absmean_bps']:7.1f}bps  σ={c['std_bps']:7.1f}bps")
    L.append("")

    ict: pd.DataFrame = res["ic"]
    L.append("② 横截面选币 IC (xs_ic, 对市场beta中性) —— 每 H 最强前3特征:")
    L.append("   [IR 用非重叠H间隔采样; 注意: IC 小但 IR 大 ≠ 可交易, 经济幅度看③]")
    for H in res["horizons"]:
        sub = ict[ict["H"] == H].copy()
        if sub.empty:
            continue
        sub["abs"] = sub["xs_ic"].abs()
        top = sub.sort_values("abs", ascending=False).head(3)
        cells = "  ".join(f"{r['feature']}={r['xs_ic']:+.3f}(IR{r['xs_ir_h']:+.1f})"
                          for _, r in top.iterrows())
        L.append(f"  {_hl(H):>5}: {cells}")
    L.append("")

    L.append("③ ★主判据: 横截面多空『走步合并样本外』(符号仅训练段定·组合层面NW-t·资金费多空相抵):")
    any_edge = False
    for H in res["horizons"]:
        r = res["ls"][H]
        for cn in ("maker", "taker"):
            s = r[cn]
            if s is None:
                L.append(f"   {_hl(H):>5} {cn:5}: 样本不足(folds={r['n_folds']})")
                continue
            fl = _flag(s)
            if "✓" in fl:
                any_edge = True
            L.append(f"   {_hl(H):>5} {cn:5}: 净{s['net_bps']:+6.1f}bps "
                     f"(NW-t={s['nw_t']:+.2f}, 毛{s['gross_bps']:+.1f}, 正占{s['pos']:.0%}, "
                     f"n={s['n']}, ac1={s['ac1']:+.2f}) {fl}")
    L.append("")

    L.append("④ [探索性, 不进 go/no-go] 方向择时书 (组合OOS信号·组合层面t·含资金费):")
    for H in res["horizons"]:
        r = res["dir"][H]
        for cn in ("maker", "taker"):
            s = r[cn]
            if s is None:
                continue
            L.append(f"   {_hl(H):>5} {cn:5}: 净{s['net_bps']:+6.1f}bps "
                     f"(NW-t={s['nw_t']:+.2f}, 正占{s['pos']:.0%}, n={s['n']}) {_flag(s)}")
    L.append("")

    # ⑤ 选择性曲线 (逐H岭回归 expected_net, 只做最自信 top-X%)
    #   多重检验稳健门: 单格 t>2 在 ~84 格里是噪声(5%×84≈4个假阳); 真 edge 须:
    #   模型有预测力(test_ic>0.01) ∧ maker 至少2个相邻档位过 ∧ taker 也至少1档过。
    def _passes(curve):
        return [s for fr, s in curve if s and fr < 1.0 and s["net_bps"] > 0
                and np.isfinite(s["nw_t"]) and s["nw_t"] > 2 and s["n_ts"] >= 20]
    sel_edge = False
    L.append("⑤ ★选择性曲线 (逐H岭回归期望收益, 只做 |pred| 最自信 top-X%, OOS·单边扣费·组合NW-t):")
    L.append("   [回答: 越稀疏越自信地做, 扣费后能否转正? 下行 maker费; ic=测试段拟合模型横截面IC]")
    L.append("   [判决门(抗多重检验): 模型ic>0.01 ∧ maker≥2档过 ∧ taker≥1档过; 单格t>2是噪声不算]")
    for H in res["horizons"]:
        sc = res["sel"].get(H)
        if not sc:
            continue
        H_edge = (sc["test_ic"] > 0.01) and len(_passes(sc["maker"])) >= 2 and len(_passes(sc["taker"])) >= 1
        sel_edge = sel_edge or H_edge
        cells = []
        for fr, s in sc["maker"]:
            if s is None:
                continue
            mark = "✓" if (s["net_bps"] > 0 and np.isfinite(s["nw_t"]) and s["nw_t"] > 2 and s["n_ts"] >= 20) else ""
            cells.append(f"top{fr * 100:g}%={s['net_bps']:+.1f}(t{s['nw_t']:+.1f}){mark}")
        L.append(f"  {_hl(H):>5} (模型ic={sc['test_ic']:+.3f}{' ⟵稳健edge' if H_edge else ''}): " + "  ".join(cells))
    L.append("")

    edge = any_edge or sel_edge
    L.append("◆ 判决 (样本外、扣费后、组合层面NW-t>2 才算数; ③横截面多空 或 ⑤选择性 任一过都算):")
    if edge:
        L.append("  ✅ 存在【扣费后净正且 NW-t>2】的方向 edge (横截面多空 或 高置信选择性) → 可深入。")
        L.append("     下一步: 该 H 做 Stage B-2(权重重拟合+概率校准)→ 加半价差/滑点压力 → DSR/PBO → 影子盘。")
    else:
        L.append("  ❌ 横截面多空(③)与高置信选择性(⑤)在走步OOS【扣费后净正且显著】的都没有 →")
        L.append("     这组标准分钟特征无稳健可交易方向 edge, 连'只做最自信 top-X%'也救不回。")
        L.append("     不等于'分钟级绝对不可做': 可换更强因子(资金费/基差/链上/宏观)或承认不可行(省下亏损也是赢)。")
    L.append("")
    L.append(f"诚实边界: 共 {len(res['feat_cols'])}特征×{len(res['horizons'])}周期×2成本 (多重检验) → 只认③走步OOS,")
    L.append("  不挑单格。IC 必要非充分; 费率为地板(未计半价差/滑点); 长H样本薄, 看 n 与 NW-t。")
    L.append("=" * 72)
    return "\n".join(L)
