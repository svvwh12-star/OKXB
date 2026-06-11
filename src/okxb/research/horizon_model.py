"""逐周期(H)专用 ML 胜率模型搜索 (Phase 1) —— 找"5/10/15/30/60min 各自最优的预测模型"。

与 candle_research 的区别(回应"不同周期要不同指标, 不能一概而论"):
  · 每个 H 用【按 H 缩放的专用指标库】: MA/BOLL/RSI/MACD/ATR/Stoch/量/K线形态/高周期背景;
  · 用【非线性梯度提升模型 HistGBM】预测 P(涨@H), 而非单一岭回归/等权;
  · 概率校准(可选) + 【净化 walk-forward 样本外】(训练/测试间 embargo H 根, 防重叠标签泄漏)。

判决铁律(与全项目一致): 胜率/AUC 只是手段; 真正的门是【扣费后样本外净 edge, 组合层 Newey-West t>2】。
  E = p·W − (1−p)·L − cost: 胜率>50% 但移动≈成本仍是亏。组合层 t 防跨币相关把 n 灌水(已踩过的坑)。

防前视: 特征只用 ≤当前bar(trailing); 标签 fwd_H 严格未来; 网格 reindex; 训练严格早于测试且 embargo H 根;
  概率/阈值只由训练段定。多重检验: 多 H × 多阈值 → 看跨阈一致, 单格过=噪声。
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import candle_research as cr   # _nw_t, _rebalance_ts, _stats_from_gross, _spearman

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    _HAS_SK = True
except Exception:   # noqa: BLE001
    _HAS_SK = False


def _log(m: str) -> None:
    print(m, flush=True)


# ----------------------------- 逐 H 专用特征库 -----------------------------

def _grid_ohlcv(df: pd.DataFrame, bar_ms: int) -> pd.DataFrame:
    d = df.sort_values("ts")
    full = np.arange(int(d["ts"].iloc[0]), int(d["ts"].iloc[-1]) + bar_ms, bar_ms)
    d = d.set_index("ts").reindex(full)
    for col in ("o", "h", "l", "c"):
        d[col] = d[col].where(d[col] > 0)
    d["volquote"] = d["volquote"].where(d["volquote"] >= 0)
    return d


def _rsi(c: pd.Series, n: int) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d).clip(lower=0).rolling(n).mean().replace(0, np.nan)
    return 100 - 100 / (1 + up / dn) - 50.0


FEATURES = ["ret_h", "ret_half", "mom_fast", "ma_fast", "ma_slow", "ma_slope",
            "bb_pctb", "bb_bw", "rsi", "macd_h", "atr", "stoch", "vol_z",
            "body", "upwick", "lowick", "htf", "vnmom", "accel",
            # 扩充常规TA (支撑/承压·VWAP·OBV·ADX/DMI·CCI·BOLL挤压·K线形态)
            "res_dist", "sup_dist", "pivot_dist", "vwap_dist", "obv_z",
            "adx", "dmi", "cci", "bb_squeeze", "engulf", "hammer"]


def inst_features(df: pd.DataFrame, H_min: int, bar_min: int, bar_ms: int) -> pd.DataFrame:
    """单标的、给定 H 的【H 缩放】专用特征 + fwd_H + y_dir。全 point-in-time。"""
    d = _grid_ohlcv(df, bar_ms)
    c, h_, l_, o_, vq = d["c"], d["h"], d["l"], d["o"], d["volquote"]
    hb = max(1, round(H_min / bar_min))
    nf = max(5, hb)            # 快窗 ~H
    ns = max(10, hb * 3)       # 慢窗 ~3H
    logret = np.log(c).diff()
    out = pd.DataFrame({"ts": d.index.values})
    out["ret_h"] = (c / c.shift(hb) - 1.0).values
    out["ret_half"] = (c / c.shift(max(1, hb // 2)) - 1.0).values
    out["mom_fast"] = (c / c.shift(max(1, hb // 3)) - 1.0).values
    maf = c.rolling(nf).mean(); mas = c.rolling(ns).mean()
    out["ma_fast"] = (c / maf - 1.0).values
    out["ma_slow"] = (c / mas - 1.0).values
    out["ma_slope"] = (maf.diff(nf) / c).values
    sd = c.rolling(nf).std()
    out["bb_pctb"] = ((c - (maf - 2 * sd)) / (4 * sd).replace(0, np.nan) - 0.5).values
    out["bb_bw"] = (4 * sd / maf.replace(0, np.nan)).values
    out["rsi"] = _rsi(c, nf).values
    ema_f = c.ewm(span=nf, adjust=False).mean(); ema_s = c.ewm(span=ns, adjust=False).mean()
    macd = ema_f - ema_s; sig = macd.ewm(span=max(3, nf // 2), adjust=False).mean()
    out["macd_h"] = ((macd - sig) / c).values
    tr = pd.concat([h_ - l_, (h_ - c.shift()).abs(), (l_ - c.shift()).abs()], axis=1).max(axis=1)
    out["atr"] = (tr.rolling(nf).mean() / c).values
    lo = l_.rolling(nf).min(); hi = h_.rolling(nf).max()
    out["stoch"] = ((c - lo) / (hi - lo).replace(0, np.nan) - 0.5).values
    out["vol_z"] = ((vq - vq.rolling(ns).mean()) / vq.rolling(ns).std().replace(0, np.nan)).values
    rng = (h_ - l_).replace(0, np.nan)
    out["body"] = ((c - o_) / rng).values
    out["upwick"] = ((h_ - np.maximum(o_, c)) / rng).values
    out["lowick"] = ((np.minimum(o_, c) - l_) / rng).values
    out["htf"] = (c / c.rolling(ns * 2).mean() - 1.0).values
    out["vnmom"] = ((c / c.shift(hb) - 1.0) / (logret.rolling(nf).std() * np.sqrt(hb)).replace(0, np.nan)).values
    out["accel"] = (out["mom_fast"] - out["ret_h"]).values

    # --- 扩充常规 TA (全部 trailing, point-in-time) ---
    res = h_.rolling(ns).max(); sup = l_.rolling(ns).min()
    out["res_dist"] = ((res - c) / c).values                       # 到承压(近高)距离
    out["sup_dist"] = ((c - sup) / c).values                       # 到支撑(近低)距离
    out["pivot_dist"] = ((c - (res + sup + c) / 3) / c).values     # 相对枢轴
    vwap = (c * vq).rolling(nf).sum() / vq.rolling(nf).sum().replace(0, np.nan)
    out["vwap_dist"] = ((c - vwap) / vwap).values                  # 相对 VWAP
    obv = (np.sign(c.diff().fillna(0.0)) * vq).cumsum()
    out["obv_z"] = ((obv - obv.rolling(ns).mean()) / obv.rolling(ns).std().replace(0, np.nan)).values
    up_move = h_.diff(); dn_move = -l_.diff()
    pdm = pd.Series(np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0), index=c.index)
    ndm = pd.Series(np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0), index=c.index)
    trn = tr.rolling(nf).sum().replace(0, np.nan)
    pdi = 100 * pdm.rolling(nf).sum() / trn
    ndi = 100 * ndm.rolling(nf).sum() / trn
    dx = (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan) * 100
    out["adx"] = dx.rolling(nf).mean().values                      # 趋势强度
    out["dmi"] = ((pdi - ndi) / (pdi + ndi).replace(0, np.nan)).values   # 方向(+DI vs -DI)
    tp = (h_ + l_ + c) / 3
    sma_tp = tp.rolling(nf).mean()
    mad = (tp - sma_tp).abs().rolling(nf).mean().replace(0, np.nan)
    out["cci"] = ((tp - sma_tp) / (0.015 * mad)).clip(-300, 300).values
    bw = out["bb_bw"]
    out["bb_squeeze"] = (bw.rolling(ns * 2).rank(pct=True) - 0.5).values  # 低=挤压蓄势
    body = c - o_; prev_body = body.shift(1)
    engulf = np.where((body > 0) & (prev_body < 0) & (c > o_.shift(1)) & (o_ < c.shift(1)), 1.0,
                      np.where((body < 0) & (prev_body > 0) & (c < o_.shift(1)) & (o_ > c.shift(1)), -1.0, 0.0))
    out["engulf"] = engulf
    lw = np.minimum(o_, c) - l_; uw = h_ - np.maximum(o_, c); ab = body.abs()
    hammer = np.where((lw > 2 * ab) & (uw < ab), 1.0,
                      np.where((uw > 2 * ab) & (lw < ab), -1.0, 0.0))
    out["hammer"] = hammer

    fwd = (c.shift(-hb) / c - 1.0).values
    out["fwd"] = fwd
    out["y"] = (fwd > 0).astype(float)
    out.loc[~np.isfinite(fwd) | (fwd == 0.0), "y"] = np.nan    # 平拍不计入胜率标签
    return out


def build_panel(dfs: dict, bar: str, H_min: int) -> tuple[pd.DataFrame, int]:
    bar_min = cr.BAR_MIN[bar]; bar_ms = bar_min * 60_000
    parts = []
    for inst, df in dfs.items():
        p = inst_features(df, H_min, bar_min, bar_ms)
        p["inst"] = inst
        parts.append(p)
    return pd.concat(parts, ignore_index=True), bar_ms


# ----------------------------- 净化 walk-forward OOS -----------------------------

def purged_oos_predict(panel: pd.DataFrame, H_min: int, bar_ms: int, *,
                       n_folds: int = 4, min_train: int = 3000):
    """逐折扩窗: 训练=ts ≤ cut−embargo(embargo=H), 测试=该折; HistGBM 预测 P(涨)。
    返回 (测试段 OOS 预测 DataFrame, 训练段置信分布 array) —— 后者供因果(非测试集)阈值。"""
    bar_min = bar_ms / 60_000
    embargo_ms = int(round(H_min / bar_min)) * bar_ms
    cols = FEATURES
    data = panel[["ts", "inst", "fwd", "y"] + cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < min_train + 500 or not _HAS_SK:
        return pd.DataFrame(columns=["ts", "inst", "p", "fwd", "y"]), np.array([])
    ts_sorted = np.sort(data["ts"].unique())
    bounds = [int(len(ts_sorted) * (i + 1) / (n_folds + 1)) for i in range(n_folds + 1)]
    preds = []
    train_conf = []
    for i in range(n_folds):
        cut = int(ts_sorted[bounds[i]]); te_hi = int(ts_sorted[min(bounds[i + 1], len(ts_sorted) - 1)])
        tr = data[data["ts"] <= cut - embargo_ms]
        te = data[(data["ts"] > cut) & (data["ts"] <= te_hi)]
        if len(tr) < min_train or len(te) < 100:
            continue
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05,
                                             l2_regularization=1.0, min_samples_leaf=200,
                                             early_stopping=True, validation_fraction=0.15,
                                             random_state=0)
        clf.fit(tr[cols].values, tr["y"].values.astype(int))
        p = clf.predict_proba(te[cols].values)[:, 1]
        ptr = clf.predict_proba(tr[cols].values)[:, 1]            # 训练段置信(供因果阈值)
        train_conf.append(np.abs(ptr - 0.5))
        preds.append(pd.DataFrame({"ts": te["ts"].values, "inst": te["inst"].values,
                                   "p": p, "fwd": te["fwd"].values, "y": te["y"].values}))
    oos = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(columns=["ts", "inst", "p", "fwd", "y"])
    tc = np.concatenate(train_conf) if train_conf else np.array([])
    return oos, tc


def evaluate(oos: pd.DataFrame, H_min: int, bar_ms: int, train_conf: np.ndarray, *,
             costs=(4.0, 10.0), fracs=(0.02, 0.05, 0.1, 0.2, 1.0)) -> dict:
    """OOS 评估: AUC(池化,beta混淆) / xs_ic(每ts横截面,beta中性) / 胜率 + 扣费后净edge
    (按置信 top-X% 出手, 阈值取自【训练段】置信分布=因果可成交, 组合层非重叠 NW-t)。"""
    if len(oos) < 200:
        return {"skip": True, "n": len(oos)}
    bar_min = bar_ms / 60_000
    h = max(1, round(H_min / bar_min))
    auc = (roc_auc_score(oos["y"].values, oos["p"].values)
           if oos["y"].nunique() == 2 else np.nan)
    ic = cr._spearman(oos["p"].values, oos["fwd"].values)               # 池化(beta混淆)
    xs = cr._xs_ic_series(oos.rename(columns={"fwd": "_y"}), "p", "_y")  # 横截面(beta中性)
    xs_ic = float(xs.mean()) if len(xs) else np.nan
    winrate = float(((oos["p"] > 0.5) == (oos["fwd"] > 0)).mean())
    all_ts = np.sort(oos["ts"].unique())
    reb = set(cr._rebalance_ts(all_ts, bar_ms, h).tolist())
    sub = oos[oos["ts"].isin(reb)].copy()
    sub["conf"] = (sub["p"] - 0.5).abs()
    have_tc = len(train_conf) > 50
    curve = []
    for fr in fracs:
        if fr >= 1.0:
            thr = 0.0
        else:                       # 因果阈值: 取训练段置信分位(非测试集自身), 使每档为可成交规则
            thr = float(np.quantile(train_conf, 1 - fr)) if have_tc else float(sub["conf"].quantile(1 - fr))
        s = sub[sub["conf"] >= thr]
        if len(s) < 20:
            curve.append((fr, None)); continue
        r = np.sign(s["p"].values - 0.5) * s["fwd"].values
        port = pd.Series(r, index=s["ts"].values).groupby(level=0).mean().values
        if len(port) < 8:
            curve.append((fr, None)); continue
        wr = float((s["fwd"].values * np.sign(s["p"].values - 0.5) > 0).mean())
        cell = {"n_ts": len(port), "n_tr": len(s), "winrate": wr}
        for c in costs:
            st = cr._stats_from_gross(port, c, 1)
            cell[f"net_{int(c)}"] = (st["net_bps"] if st else np.nan)
            cell[f"t_{int(c)}"] = (st["nw_t"] if st else np.nan)
        curve.append((fr, cell))
    return {"n": len(oos), "auc": auc, "ic": ic, "xs_ic": xs_ic, "winrate": winrate, "curve": curve}


def _cell_edge(cell: Optional[dict]) -> bool:
    if not cell or cell.get("n_ts", 0) < 20:
        return False
    return all(cell.get(f"net_{c}", -1) > 0 and (cell.get(f"t_{c}") or 0) > 2 for c in (4, 10))


# ----------------------------- 编排 + 报告 -----------------------------

def run(dfs: dict, bar: str = "5m", horizons_min=(5, 10, 15, 30, 60),
        n_folds: int = 4, log: Callable[[str], None] = _log) -> dict:
    if not _HAS_SK:
        return {"error": "sklearn 未安装"}
    res = {}
    for H in horizons_min:
        log(f"  H={H}min: 造特征 + 净化walk-forward GBM ...")
        panel, bar_ms = build_panel(dfs, bar, H)
        oos, tconf = purged_oos_predict(panel, H, bar_ms, n_folds=n_folds)
        res[H] = evaluate(oos, H, bar_ms, tconf) if len(oos) else {"skip": True, "n": 0}
        res[H]["bar_ms"] = bar_ms
    return {"bar": bar, "n_inst": len(dfs), "horizons": list(horizons_min), "by_h": res}


def format_report(meta: dict, res: dict) -> str:
    if res.get("error"):
        return f"逐周期模型搜索失败: {res['error']}"
    L = ["=" * 78, "逐周期(H)专用 ML 胜率模型搜索 (HistGBM · 净化walk-forward OOS · 扣费后净edge)", "=" * 78,
         f"数据: {res['n_inst']} 标的 / bar={res['bar']} / 跨 {meta.get('span_days','?')} 天 / "
         f"特征 {len(FEATURES)} 个(按H缩放)",
         "判决门: 扣费后(maker4∧taker10)净正 ∧ 组合NW-t>2 ∧ 同H≥2个置信档过; 胜率/AUC仅参考(E=pW−(1−p)L−cost)", ""]
    any_edge = False
    for H in res["horizons"]:
        d = res["by_h"][H]
        if d.get("skip"):
            L.append(f"━ H={H}min ━ 样本不足({d.get('n')})"); continue
        L.append(f"━ 未来 {H}min ━ AUC={d['auc']:.3f}(池化,beta混淆) xs_ic={d.get('xs_ic', float('nan')):+.4f}(beta中性) "
                 f"全样本胜率={d['winrate']:.1%}")
        passes = 0
        for fr, cell in d["curve"]:
            if cell is None:
                continue
            ok = _cell_edge(cell); passes += int(ok)
            L.append(f"   top{fr*100:>4g}%: 胜率{cell['winrate']:.1%} | "
                     f"净@maker{cell['net_4']:+6.1f}(t{cell['t_4']:+.1f}) | "
                     f"净@taker{cell['net_10']:+6.1f}(t{cell['t_10']:+.1f}) | n{cell['n_ts']} {'✓' if ok else ''}")
        if passes >= 2:
            any_edge = True
            L.append(f"   ⟶ H={H}min: {passes} 个置信档过门 (候选)")
        L.append("")
    L.append("◆ 判决:")
    if any_edge:
        L.append("  ⚠ 有 H ≥2档过门 → 进 Phase 2(概率校准+Kelly仓位/杠杆)前, 先加 DSR/PBO + 跨regime + 半价差/滑点压力。")
    else:
        L.append("  ❌ 5/10/15/30/60min 各自的最优 GBM 模型, 扣费后样本外仍无稳健净 edge。")
        L.append("     即使换成 per-H 专用指标 + 非线性ML + 概率阈值选择, 也翻不过成本/预测墙。")
    L.append("")
    L.append("诚实边界: 净edge按【定向单边费】(maker4/taker10), 未计半价差/滑点; 胜率=方向命中≠盈利;")
    L.append("  阈值取自训练段(因果可成交); p 为 GBM 原始分(未概率校准, Phase2 Kelly前须校准);")
    L.append("  多H×多阈值多重检验→只认≥2档一致, 任何正格上线前还需 DSR/PBO + 可成交验证。")
    L.append("=" * 78)
    return "\n".join(L)
