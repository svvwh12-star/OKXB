"""专业特征筛选 + 多模型比较工作流 —— 回应"变量筛选 + 多模型择优"。

流程 (全程 point-in-time, 筛选只在每折训练段内做, 防泄漏):
  A. 大候选特征库 (~50): 多窗 MA/动量/RSI/BOLL/ROC/Williams/CCI/ADX/DMI/ATR/RV/量/OBV/MFI/
     Donchian/Keltner/K线形态/支撑承压/VWAP/HTF 等。
  B. 特征筛选 (训练段): ①覆盖率过滤 ②共线性聚类(|corr|>0.92 只留单变量IC最强者)
     ③重要性排名聚合 = RandomForest重要性 + Lasso(L1)系数 + 单变量秩IC, 取 top-K。
  C. 模型动物园: 逻辑回归(L2) / ElasticNet / 随机森林 / HistGBM / 小型神经网(MLP), 同一筛选特征上比较。
  D. 净化 walk-forward 样本外, 逐模型报 AUC / xs_ic / 胜率 / 扣费后净edge, 按【扣费后净】择优。

判决铁律不变: 只认扣费后样本外净正(组合NW-t>2, ≥2置信档); AUC/胜率仅参考(E=pW−(1−p)L−cost)。
诚实: diffusion 模型不适合此(生成式, 30天数据会过拟); 多模型×多H 本身是多重检验, 故只认 OOS 净edge。
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from . import candle_research as cr
from . import horizon_model as hm

try:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    _HAS_SK = True
except Exception:   # noqa: BLE001
    _HAS_SK = False


def _log(m: str) -> None:
    print(m, flush=True)


# ----------------------------- A. 大候选特征库 -----------------------------

def inst_bank(df: pd.DataFrame, H_min: int, bar_min: int, bar_ms: int) -> pd.DataFrame:
    """单标的、给定 H 的大候选特征库 (~50, 多窗多指标)。全 trailing。"""
    d = hm._grid_ohlcv(df, bar_ms)
    c, h_, l_, o_, vq = d["c"], d["h"], d["l"], d["o"], d["volquote"]
    hb = max(1, round(H_min / bar_min))
    logret = np.log(c).diff()
    tr = pd.concat([h_ - l_, (h_ - c.shift()).abs(), (l_ - c.shift()).abs()], axis=1).max(axis=1)
    out = pd.DataFrame({"ts": d.index.values})
    wins = sorted(set([max(2, hb // 2), hb, hb * 2, hb * 3, max(10, hb), max(20, hb * 4)]))

    for k in sorted(set([1, max(1, hb // 3), max(1, hb // 2), hb, hb * 2])):
        out[f"ret_{k}"] = (c / c.shift(k) - 1.0).values
    for w in wins:
        out[f"ma_{w}"] = (c / c.rolling(w).mean() - 1.0).values
    for w in (max(7, hb), hb * 2):
        out[f"rsi_{w}"] = hm._rsi(c, w).values
    for w in wins[:3]:
        sd = c.rolling(w).std(); ma = c.rolling(w).mean()
        out[f"bbpb_{w}"] = ((c - (ma - 2 * sd)) / (4 * sd).replace(0, np.nan) - 0.5).values
        out[f"bbw_{w}"] = (4 * sd / ma.replace(0, np.nan)).values
    for w in wins:
        out[f"rv_{w}"] = (logret.rolling(w).std()).values
        out[f"atr_{w}"] = (tr.rolling(w).mean() / c).values
    for w in wins:
        lo = l_.rolling(w).min(); hi = h_.rolling(w).max()
        out[f"don_{w}"] = ((c - lo) / (hi - lo).replace(0, np.nan) - 0.5).values   # Donchian/range pos
        out[f"res_{w}"] = ((hi - c) / c).values                                    # 承压距离
        out[f"sup_{w}"] = ((c - lo) / c).values                                    # 支撑距离
    # MACD
    ema_f = c.ewm(span=max(6, hb), adjust=False).mean(); ema_s = c.ewm(span=max(13, hb * 3), adjust=False).mean()
    out["macd_h"] = ((ema_f - ema_s - (ema_f - ema_s).ewm(span=max(3, hb // 2), adjust=False).mean()) / c).values
    # ROC / Williams%R / CCI
    out["roc"] = (c / c.shift(hb) - 1.0).values
    hi_n = h_.rolling(max(10, hb)).max(); lo_n = l_.rolling(max(10, hb)).min()
    out["willr"] = ((hi_n - c) / (hi_n - lo_n).replace(0, np.nan) - 0.5).values
    tp = (h_ + l_ + c) / 3; smatp = tp.rolling(max(10, hb)).mean()
    madv = (tp - smatp).abs().rolling(max(10, hb)).mean().replace(0, np.nan)
    out["cci"] = ((tp - smatp) / (0.015 * madv)).clip(-300, 300).values
    # ADX / DMI
    upm = h_.diff(); dnm = -l_.diff()
    pdm = pd.Series(np.where((upm > dnm) & (upm > 0), upm, 0.0), index=c.index)
    ndm = pd.Series(np.where((dnm > upm) & (dnm > 0), dnm, 0.0), index=c.index)
    trn = tr.rolling(max(7, hb)).sum().replace(0, np.nan)
    pdi = 100 * pdm.rolling(max(7, hb)).sum() / trn; ndi = 100 * ndm.rolling(max(7, hb)).sum() / trn
    out["adx"] = ((pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan) * 100).rolling(max(7, hb)).mean().values
    out["dmi"] = ((pdi - ndi) / (pdi + ndi).replace(0, np.nan)).values
    # 量 / OBV / MFI / VWAP
    out["vol_z"] = ((vq - vq.rolling(hb * 3).mean()) / vq.rolling(hb * 3).std().replace(0, np.nan)).values
    obv = (np.sign(c.diff().fillna(0.0)) * vq).cumsum()
    out["obv_z"] = ((obv - obv.rolling(hb * 3).mean()) / obv.rolling(hb * 3).std().replace(0, np.nan)).values
    mf = tp * vq; pos = mf.where(tp.diff() > 0, 0.0).rolling(max(10, hb)).sum()
    neg = mf.where(tp.diff() < 0, 0.0).rolling(max(10, hb)).sum().replace(0, np.nan)
    out["mfi"] = (100 - 100 / (1 + pos / neg) - 50).values
    vwap = (c * vq).rolling(hb * 2).sum() / vq.rolling(hb * 2).sum().replace(0, np.nan)
    out["vwap_d"] = ((c - vwap) / vwap).values
    # K线形态 + vol归一动量 + HTF
    rng = (h_ - l_).replace(0, np.nan); body = c - o_
    out["body"] = (body / rng).values
    out["upwick"] = ((h_ - np.maximum(o_, c)) / rng).values
    out["lowick"] = ((np.minimum(o_, c) - l_) / rng).values
    eng = np.where((body > 0) & (body.shift(1) < 0) & (c > o_.shift(1)) & (o_ < c.shift(1)), 1.0,
                   np.where((body < 0) & (body.shift(1) > 0) & (c < o_.shift(1)) & (o_ > c.shift(1)), -1.0, 0.0))
    out["engulf"] = eng
    out["vnmom"] = (out["roc"] / (logret.rolling(max(5, hb)).std() * np.sqrt(hb)).replace(0, np.nan).values)
    out["htf"] = (c / c.rolling(hb * 8).mean() - 1.0).values
    fwd = (c.shift(-hb) / c - 1.0).values
    out["fwd"] = fwd
    out["y"] = (fwd > 0).astype(float)
    out.loc[~np.isfinite(fwd) | (fwd == 0.0), "y"] = np.nan
    return out


def bank_cols(panel: pd.DataFrame) -> list[str]:
    return [c for c in panel.columns if c not in ("ts", "inst", "fwd", "y")]


def build_bank(dfs: dict, bar: str, H_min: int) -> tuple[pd.DataFrame, int]:
    bar_min = cr.BAR_MIN[bar]; bar_ms = bar_min * 60_000
    parts = []
    for inst, df in dfs.items():
        p = inst_bank(df, H_min, bar_min, bar_ms); p["inst"] = inst
        parts.append(p)
    return pd.concat(parts, ignore_index=True), bar_ms


# ----------------------------- B. 特征筛选 (训练段内) -----------------------------

def select_features(tr: pd.DataFrame, feats: list[str], k: int = 18,
                    corr_thr: float = 0.92) -> tuple[list[str], pd.DataFrame]:
    X = tr[feats].replace([np.inf, -np.inf], np.nan)
    cov = X.notna().mean()
    feats = [f for f in feats if cov[f] > 0.6]                     # 覆盖率过滤
    Xm = X[feats].fillna(X[feats].median())
    y = tr["y"].values.astype(int); fwd = tr["fwd"].values
    # 单变量秩 IC
    uic = {f: abs(cr._spearman(Xm[f].values, fwd)) for f in feats}
    # 共线性聚类: 按 |IC| 降序贪心保留, 丢与已留者 |corr|>thr 的
    order = sorted(feats, key=lambda f: uic[f], reverse=True)
    corr = Xm[feats].corr().abs()
    kept = []
    for f in order:
        if all(corr.loc[f, g] <= corr_thr for g in kept):
            kept.append(f)
    # RF 重要性 + Lasso(L1) 系数, 在去共线后的集合上
    Xk = Xm[kept].values
    imp = {}
    try:
        rf = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=200,
                                    n_jobs=-1, random_state=0).fit(Xk, y)
        for f, v in zip(kept, rf.feature_importances_):
            imp[f] = v
    except Exception:   # noqa: BLE001
        imp = {f: 0.0 for f in kept}
    lcoef = {}
    try:
        sc = StandardScaler().fit(Xk)
        lr = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, max_iter=500).fit(sc.transform(Xk), y)
        for f, v in zip(kept, np.abs(lr.coef_[0])):
            lcoef[f] = v
    except Exception:   # noqa: BLE001
        lcoef = {f: 0.0 for f in kept}
    # 排名聚合 (RF + Lasso + 单变量IC 各自排名平均)
    def ranks(dct):
        s = pd.Series(dct); return s.rank(ascending=True)
    agg = (ranks(imp) + ranks(lcoef) + ranks({f: uic[f] for f in kept})).sort_values(ascending=False)
    sel = list(agg.head(k).index)
    rep = pd.DataFrame({"feature": agg.index, "rf_imp": [imp.get(f, 0) for f in agg.index],
                        "lasso": [lcoef.get(f, 0) for f in agg.index],
                        "uni_ic": [uic.get(f, 0) for f in agg.index]})
    return sel, rep


# ----------------------------- C. 模型动物园 -----------------------------

def _models():
    return {
        "logit": ("scale", lambda: LogisticRegression(C=0.5, max_iter=1000)),
        "enet": ("scale", lambda: LogisticRegression(penalty="elasticnet", solver="saga",
                                                      l1_ratio=0.5, C=0.5, max_iter=800)),
        "rf": ("raw", lambda: RandomForestClassifier(n_estimators=250, max_depth=6,
                                                     min_samples_leaf=200, n_jobs=-1, random_state=0)),
        "gbm": ("raw", lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=200,
                learning_rate=0.05, l2_regularization=1.0, min_samples_leaf=200,
                early_stopping=True, validation_fraction=0.15, random_state=0)),
        "mlp": ("scale", lambda: MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-3,
                max_iter=300, early_stopping=True, random_state=0)),
    }


# ----------------------------- D. 净化 walk-forward 比较 -----------------------------

def purged_compare(panel: pd.DataFrame, feats: list[str], H_min: int, bar_ms: int, *,
                   n_folds: int = 4, k_sel: int = 18, min_train: int = 4000,
                   log: Callable[[str], None] = _log) -> dict:
    bar_min = bar_ms / 60_000
    embargo_ms = int(round(H_min / bar_min)) * bar_ms
    data = panel[["ts", "inst", "fwd", "y"] + feats].replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["fwd", "y"])
    ts_sorted = np.sort(data["ts"].unique())
    bounds = [int(len(ts_sorted) * (i + 1) / (n_folds + 1)) for i in range(n_folds + 1)]
    zoo = _models()
    oos = {m: [] for m in zoo}; tconf = {m: [] for m in zoo}
    sel_counter: dict[str, int] = {}
    for i in range(n_folds):
        cut = int(ts_sorted[bounds[i]]); te_hi = int(ts_sorted[min(bounds[i + 1], len(ts_sorted) - 1)])
        tr = data[data["ts"] <= cut - embargo_ms]
        te = data[(data["ts"] > cut) & (data["ts"] <= te_hi)]
        if len(tr) < min_train or len(te) < 100:
            continue
        sel, _ = select_features(tr, feats, k=k_sel)
        for f in sel:
            sel_counter[f] = sel_counter.get(f, 0) + 1
        med = tr[sel].median()
        Xtr = tr[sel].fillna(med); Xte = te[sel].fillna(med)
        ytr = tr["y"].values.astype(int)
        for name, (mode, factory) in zoo.items():
            try:
                if mode == "scale":
                    sc = StandardScaler().fit(Xtr.values)
                    Xtr_, Xte_ = sc.transform(Xtr.values), sc.transform(Xte.values)
                else:
                    Xtr_, Xte_ = Xtr.values, Xte.values
                clf = factory().fit(Xtr_, ytr)
                p = clf.predict_proba(Xte_)[:, 1]
                ptr = clf.predict_proba(Xtr_)[:, 1]
                tconf[name].append(np.abs(ptr - 0.5))
                oos[name].append(pd.DataFrame({"ts": te["ts"].values, "p": p,
                                               "fwd": te["fwd"].values, "y": te["y"].values}))
            except Exception as e:   # noqa: BLE001
                log(f"    {name} 失败: {e}")
    out = {}
    for m in zoo:
        odf = pd.concat(oos[m], ignore_index=True) if oos[m] else pd.DataFrame()
        tc = np.concatenate(tconf[m]) if tconf[m] else np.array([])
        out[m] = (odf, tc)
    return {"models": out, "selected_freq": sorted(sel_counter.items(), key=lambda kv: kv[1], reverse=True)}


def run(dfs: dict, bar: str = "5m", horizons_min=(10, 15), n_folds: int = 4,
        log: Callable[[str], None] = _log) -> dict:
    if not _HAS_SK:
        return {"error": "sklearn 未安装"}
    res = {}
    for H in horizons_min:
        log(f"  H={H}min: 建库({'?'}) + 折内筛选 + 5模型比较 ...")
        panel, bar_ms = build_bank(dfs, bar, H)
        feats = bank_cols(panel)
        comp = purged_compare(panel, feats, H, bar_ms, n_folds=n_folds, log=log)
        model_metrics = {}
        for m, (odf, tc) in comp["models"].items():
            model_metrics[m] = hm.evaluate(odf, H, bar_ms, tc) if len(odf) else {"skip": True}
        res[H] = {"n_feats": len(feats), "selected_freq": comp["selected_freq"][:12],
                  "metrics": model_metrics, "bar_ms": bar_ms}
    return {"bar": bar, "n_inst": len(dfs), "horizons": list(horizons_min), "by_h": res}


def format_report(meta: dict, res: dict) -> str:
    if res.get("error"):
        return f"特征筛选+多模型 失败: {res['error']}"
    L = ["=" * 80, "专业特征筛选 + 多模型比较 (折内筛选·5模型·净化walk-forward OOS·扣费后净edge)", "=" * 80,
         f"数据: {res['n_inst']} 标的 / bar={res['bar']} / 跨 {meta.get('span_days','?')} 天", ""]
    any_edge = False
    for H in res["horizons"]:
        d = res["by_h"][H]
        L.append(f"━━━ 未来 {H}min (候选特征 {d['n_feats']} → 折内筛 top18) ━━━")
        topf = ", ".join(f"{f}×{n}" for f, n in d["selected_freq"][:10])
        L.append(f"  常被选中特征: {topf}")
        L.append("  [模型 | AUC | xs_ic | 全胜率 | 最佳置信档: 胜率/净@maker(t)/净@taker(t)/n | 过门]")
        best = None
        for m, mm in d["metrics"].items():
            if mm.get("skip"):
                L.append(f"   {m:>6}: 样本不足"); continue
            # 找该模型最佳(扣费后 maker 净)且 n_ts>=20 的置信档
            cells = [(fr, c) for fr, c in mm["curve"] if c and c.get("n_ts", 0) >= 20]
            bc = max(cells, key=lambda x: x[1]["net_4"]) if cells else (None, None)
            passes = sum(1 for fr, c in cells if hm._cell_edge(c))
            if passes >= 2:
                any_edge = True
            tag = f"✓{passes}档" if passes >= 2 else ""
            if bc[1]:
                fr, c = bc
                cells = f"top{fr*100:g}%: 胜{c['winrate']:.0%}/净{c['net_4']:+.1f}(t{c['t_4']:+.1f})/净{c['net_10']:+.1f}(t{c['t_10']:+.1f})/n{c['n_ts']}"
            else:
                cells = "—"
            L.append(f"   {m:>6}: {mm['auc']:.3f} | {mm.get('xs_ic', float('nan')):+.4f} | "
                     f"{mm['winrate']:.1%} | {cells} {tag}")
        L.append("")
    L.append("◆ 判决:")
    if any_edge:
        L.append("  ⚠ 某模型某H ≥2置信档扣费后净正且显著 → 上 DSR/PBO + 跨regime + 半价差/滑点压力再定。")
    else:
        L.append("  ❌ 系统化特征筛选 + 5 个模型(逻辑/ElasticNet/随机森林/HistGBM/神经网)比较后,")
        L.append("     10/15min 扣费后样本外仍无稳健净 edge。变量筛选与模型选择都不是瓶颈 —— 成本墙是。")
    L.append("")
    L.append("诚实边界: 筛选在折内训练段做(防泄漏); 净edge按单边费(maker4/taker10)未计滑点;")
    L.append("  胜率≠盈利; 多模型×多H多重检验→只认扣费后OOS净edge且≥2档一致。")
    L.append("=" * 80)
    return "\n".join(L)
