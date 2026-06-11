"""正交因子离线验证 (方向型) —— 影子盘计划的第一步: 先用数据证伪, 再决定建不建系统。

测三个【与价量动量正交、数据可得、且未测过】的方向假设:
  ① 资金费做方向 (funding-as-direction): 拥挤多头(高正费)是否预示未来【价格】下跌? (横截面, 8h/1d/3d)
  ② 基差做方向 (basis-as-direction): perp 对现货升水(premium)是否预示回落? (横截面, 15/30/60/120min)
  ③ BTC/ETH lead-lag: 最近一段大盘(BTC/ETH)涨跌是否预测山寨【下一段】涨跌? (时序/beta择时, 5/15/30min)

复用 `candle_research` 已过 4-Agent 审计的统计原语 (网格 reindex、point-in-time、非重叠 rebalance、
组合层 Newey-West t、OOS、抗多重检验判决门)。本模块只新增【因子构造】, 故审计重点放在防前视。

诚实: 资金费carry(结构型)已实测比费墙薄; 这里测的是把这些量当【方向信号】(要翻越"预测方向"墙)。
判决门同 candle_research: 扣费后净正 ∧ 组合NW-t>2 ∧ 样本够 ∧ 跨档一致; 单格 t>2 视为多重检验噪声。
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import candle_research as cr   # 复用 _nw_t/_xs_ic_series/_rebalance_ts/_spearman/BAR_MIN

FUND_INTERVAL_MS = 8 * 3_600_000


def _log(m: str) -> None:
    print(m, flush=True)


def _grid_close(df: pd.DataFrame, bar_ms: int, col: str = "c") -> pd.Series:
    """单标的收盘价 reindex 到完整 bar 网格, 非正价→NaN (防前视/坏bar)。index=ts。"""
    d = df.sort_values("ts")
    full = np.arange(int(d["ts"].iloc[0]), int(d["ts"].iloc[-1]) + bar_ms, bar_ms)
    s = d.set_index("ts")[col].reindex(full)
    return s.where(s > 0)


# ============================ ①+② basis & lead-lag 面板 (5m) ============================

def build_price_factor_panel(perp_dfs: dict, spot_dfs: dict, bar: str,
                             horizons_min, btc="BTC-USDT-SWAP", eth="ETH-USDT-SWAP",
                             leads_min=(5, 15, 30)) -> tuple[pd.DataFrame, int]:
    bar_min = cr.BAR_MIN[bar]; bar_ms = bar_min * 60_000
    btc_c = _grid_close(perp_dfs[btc], bar_ms) if btc in perp_dfs else None
    eth_c = _grid_close(perp_dfs[eth], bar_ms) if eth in perp_dfs else None
    zk = max(4, round(240 / bar_min))   # basis z 的 trailing 窗 (~4h)
    parts = []
    for inst, pdf in perp_dfs.items():
        sdf = spot_dfs.get(inst)
        if sdf is None:
            continue
        pc = _grid_close(pdf, bar_ms)
        sc = _grid_close(sdf, bar_ms).reindex(pc.index)
        df = pd.DataFrame({"ts": pc.index.values, "perp": pc.values, "spot": sc.values})
        df["inst"] = inst
        df["basis"] = df["perp"] / df["spot"] - 1.0
        roll = df["basis"].rolling(zk)
        df["basis_z"] = ((df["basis"] - roll.mean()) / roll.std().replace(0, np.nan)
                         ).replace([np.inf, -np.inf], np.nan)          # point-in-time z, 防 std=0→inf
        for H in horizons_min:                                          # 严格未来 perp 收益
            hb = max(1, round(H / bar_min))
            df[f"fwd_{H}"] = (df["perp"].shift(-hb) / df["perp"] - 1.0).values
        # 大盘 lead 因子 (同一 ts 对所有标的相同 = 共同因子, 只能做时序/beta择时)
        for nm, s in (("btc", btc_c), ("eth", eth_c)):
            if s is None:
                continue
            sa = s.reindex(pc.index)
            for mn in leads_min:
                k = max(1, round(mn / bar_min))
                df[f"{nm}_ret_{mn}"] = (sa / sa.shift(k) - 1.0).values
        parts.append(df)
    return pd.concat(parts, ignore_index=True), bar_ms


def _xs_z(panel: pd.DataFrame, col: str) -> pd.Series:
    g = panel.groupby("ts")[col]
    return (panel[col] - g.transform("mean")) / g.transform("std").replace(0, np.nan)


def test_cross_sectional(panel: pd.DataFrame, factor: str, H: int, bar_ms: int, *,
                         costs=(4.0, 10.0), q: float = 0.2, train_frac: float = 0.7) -> dict:
    """横截面方向因子(basis/funding): xs IC + 训练定符号的 OOS 多空, 组合NW-t, 扣费。"""
    bar_min = bar_ms / 60_000
    h = max(1, round(H / bar_min))
    sub = panel[["ts", "inst", factor, f"fwd_{H}"]].dropna()
    if len(sub) < 200:
        return {"H": H, "n": len(sub), "skip": True}
    xs = cr._xs_ic_series(sub, factor, f"fwd_{H}")
    all_ts = np.sort(panel["ts"].unique())
    ts_sorted = np.sort(sub["ts"].unique())
    ci = int(len(ts_sorted) * train_frac)
    cut = int(ts_sorted[ci])
    embargo = int(ts_sorted[max(0, ci - h)])   # 训练段尾部 embargo h 根, 防 fwd 标签泄漏到测试边界
    sign = np.sign(cr._xs_ic_series(sub[sub["ts"] <= embargo], factor, f"fwd_{H}").mean() or 0.0)
    p = panel.assign(_sig=sign * _xs_z(panel, factor).fillna(0.0), _y=panel[f"fwd_{H}"].values)
    te = p[p["ts"] > cut]
    g, _ = cr.xs_ls_gross(te, "_sig", h, bar_ms, q, lo=cut, all_ts=all_ts)
    out = {"H": H, "n": len(sub), "xs_ic": float(xs.mean()) if len(xs) else np.nan,
           "xs_ic_ir": cr._nw_t(xs.values) if len(xs) > 5 else np.nan, "sign": float(sign),
           "n_reb": len(g)}
    for c in costs:
        s = cr._stats_from_gross(g, c, 2)
        out[f"net_{int(c)}"] = (s["net_bps"] if s else np.nan)
        out[f"t_{int(c)}"] = (s["nw_t"] if s else np.nan)
    return out


def test_leadlag(panel: pd.DataFrame, lead_col: str, H: int, bar_ms: int, *,
                 costs=(4.0, 10.0), thr: float = 0.0) -> dict:
    """大盘 lead-lag (beta择时): 每个非重叠 ts, 若 lead 信号>thr 则【所有山寨同向】持H, 组合层聚合→NW-t。
    sign 由因子本身定 (动量假设: 大盘刚涨→山寨跟涨); 这是方向/beta 暴露, 故组合层算 t 防跨币相关灌水。"""
    bar_min = bar_ms / 60_000
    h = max(1, round(H / bar_min))
    lead_inst = {"btc": "BTC-USDT-SWAP", "eth": "ETH-USDT-SWAP"}.get(lead_col.split("_")[0])
    sub = panel[["ts", "inst", lead_col, f"fwd_{H}"]].dropna()
    if lead_inst:                                   # 排除领先资产自身, 否则 BTC 自动量污染"山寨跟随BTC"
        sub = sub[sub["inst"] != lead_inst]
    if len(sub) < 200:
        return {"H": H, "n": len(sub), "skip": True}
    all_ts = np.sort(panel["ts"].unique())
    reb = set(cr._rebalance_ts(all_ts, bar_ms, h).tolist())
    s = sub[sub["ts"].isin(reb)]
    s = s[np.abs(s[lead_col].values) > thr]
    if s.empty:
        return {"H": H, "n": 0, "skip": True}
    s = s.assign(r=np.sign(s[lead_col].values) * s[f"fwd_{H}"].values)
    port = s.groupby("ts")["r"].mean().values     # 每 ts 一个组合收益 (所有山寨同向均值)
    out = {"H": H, "n": len(sub), "n_reb": len(port),
           "ic": cr._spearman(sub[lead_col].values, sub[f"fwd_{H}"].values)}
    for c in costs:
        st = cr._stats_from_gross(port, c, 1)
        out[f"net_{int(c)}"] = (st["net_bps"] if st else np.nan)
        out[f"t_{int(c)}"] = (st["nw_t"] if st else np.nan)
    return out


# ============================ ③ 资金费做方向 (8h 网格) ============================

def build_funding_dir_panel(perp_dfs: dict, funding_dfs: dict) -> pd.DataFrame:
    """对齐到 8h 资金费网格: 每标的 [ts, perp(资金费时刻价), funding]。供'资金费→未来价格方向'检验。"""
    parts = []
    for inst, fdf in funding_dfs.items():
        pdf = perp_dfs.get(inst)
        if pdf is None or len(fdf) < 20:
            continue
        p = pdf[["ts", "c"]].sort_values("ts").rename(columns={"c": "perp"})
        f = fdf.sort_values("ts")
        m = pd.merge_asof(f, p, on="ts", direction="backward", tolerance=FUND_INTERVAL_MS)
        m = m.dropna(subset=["perp"]); m = m[m["perp"] > 0]
        m["inst"] = inst
        parts.append(m[["ts", "inst", "perp", "funding"]])
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def test_funding_direction(fpanel: pd.DataFrame, hold_steps: int, *,
                           costs=(4.0, 10.0), q: float = 0.2, train_frac: float = 0.7) -> dict:
    """资金费做【方向】: 因子=t0 当期 funding(已结算/已知); 标签=未来 perp 价格收益(非carry)。
    按【真实时间】非重叠 rebalance(兼容混合周期/缺口, 不用位置偏移); 横截面多空(price方向, carry相抵),
    训练段(含embargo)定符号, 组合 NW-t, 扣 2 腿往返费。"""
    P = {i: g.set_index("ts").sort_index() for i, g in fpanel.groupby("inst")}
    all_ts = np.sort(fpanel["ts"].unique())
    if len(all_ts) < hold_steps + 10:
        return {"skip": True, "n": 0}
    hold_ms = hold_steps * FUND_INTERVAL_MS
    tol = FUND_INTERVAL_MS                       # 入/出锚点容差 = 1 个结算
    lo, hi = int(all_ts[0]), int(all_ts[-1])
    reb_t0 = list(range(lo, hi - hold_ms + 1, hold_ms))   # 真实时间、非重叠
    rows = []
    for t0 in reb_t0:
        t1 = t0 + hold_ms
        for inst, g in P.items():
            idx = g.index.values
            le = idx[idx <= t0]
            if len(le) == 0 or t0 - int(le[-1]) > tol:
                continue
            ts0 = int(le[-1])
            lx = idx[(idx <= t1) & (idx > ts0)]
            if len(lx) == 0 or t1 - int(lx[-1]) > tol:
                continue
            ts1 = int(lx[-1])
            rows.append((t0, inst, float(g.at[ts0, "funding"]),
                         g.at[ts1, "perp"] / g.at[ts0, "perp"] - 1.0))
    d = pd.DataFrame(rows, columns=["ts", "inst", "funding", "fwd"])
    if len(d) < 150:
        return {"skip": True, "n": len(d)}
    xsd = d.rename(columns={"fwd": "_y"})
    xs = cr._xs_ic_series(xsd, "funding", "_y")
    ts_sorted = np.sort(d["ts"].unique())
    ci = int(len(ts_sorted) * train_frac)
    cut = int(ts_sorted[ci])
    embargo = int(ts_sorted[max(0, ci - 1)])     # embargo 1 个 rebalance(=hold) 防泄漏
    sign = np.sign(cr._xs_ic_series(xsd[xsd["ts"] <= embargo], "funding", "_y").mean() or 0.0)
    d["_sig"] = sign * _xs_z(d, "funding").fillna(0.0)
    d["_y"] = d["fwd"]
    te = d[d["ts"] > cut]
    gross = []
    for ts, g in te.groupby("ts"):
        if len(g) < 5:
            continue
        g = g.sort_values("_sig")
        k = max(1, int(len(g) * q))
        gross.append(g["_y"].tail(k).mean() - g["_y"].head(k).mean())
    gross = np.array([x for x in gross if np.isfinite(x)])
    out = {"hold_steps": hold_steps, "n": len(d), "xs_ic": float(xs.mean()) if len(xs) else np.nan,
           "xs_ic_ir": cr._nw_t(xs.values) if len(xs) > 5 else np.nan,
           "sign": float(sign), "n_reb": len(gross)}
    for c in costs:
        s = cr._stats_from_gross(gross, c, 2)
        out[f"net_{int(c)}"] = (s["net_bps"] if s else np.nan)
        out[f"t_{int(c)}"] = (s["nw_t"] if s else np.nan)
    return out


# ============================ 编排 + 报告 ============================

def run(perp_dfs: dict, spot_dfs: dict, funding_dfs: dict, *, bar: str = "5m",
        price_horizons=(15, 30, 60, 120), leads_min=(5, 15, 30),
        funding_holds_steps=(1, 3, 9), log: Callable[[str], None] = _log) -> dict:
    log("组装价格因子面板(basis + lead-lag) ...")
    ppanel, bar_ms = build_price_factor_panel(perp_dfs, spot_dfs, bar, price_horizons, leads_min=leads_min)
    res_basis = [test_cross_sectional(ppanel, "basis_z", H, bar_ms) for H in price_horizons]
    res_lead = {}
    for nm in ("btc", "eth"):
        for mn in leads_min:
            col = f"{nm}_ret_{mn}"
            if col in ppanel.columns:
                res_lead[col] = [test_leadlag(ppanel, col, H, bar_ms) for H in price_horizons]
    log("组装资金费方向面板(8h) ...")
    fpanel = build_funding_dir_panel(perp_dfs, funding_dfs)
    res_fund = [test_funding_direction(fpanel, hs) for hs in funding_holds_steps] if len(fpanel) else []
    return dict(bar=bar, n_inst=len(perp_dfs), price_horizons=list(price_horizons),
                leads_min=list(leads_min), funding_holds_steps=list(funding_holds_steps),
                basis=res_basis, lead=res_lead, funding=res_fund,
                n_basis_inst=ppanel["inst"].nunique(), n_fund_inst=(fpanel["inst"].nunique() if len(fpanel) else 0))


def _edge(r: dict, min_n: int = 20) -> bool:
    """单格(展示用): 两种费下都净正且 t>2 且样本够。最终判决=同一因子≥2格过(抗多重检验)。"""
    if r.get("skip") or r.get("n_reb", 0) < min_n:
        return False
    return all(r.get(f"net_{c}", -1) > 0 and (r.get(f"t_{c}") or 0) > 2 for c in (4, 10))


def format_report(meta: dict, res: dict) -> str:
    L = ["=" * 76, "正交因子离线验证 (方向型 · 扣费 · OOS · 组合NW-t · 复用已审计原语)", "=" * 76,
         f"数据: {res['n_inst']} 标的 / bar={res['bar']} / 跨 {meta.get('span_days','?')} 天 "
         f"(basis可用{res['n_basis_inst']}, funding可用{res['n_fund_inst']})",
         "判决门(抗多重检验): 同一因子【≥2 个H/持有格】两种费下均 净正 ∧ 组合NW-t>2 ∧ n_reb够; "
         "单格过=噪声; basis/funding n≥20, funding长持 n≥40; lead-lag 仅探索不进门", ""]
    basis_pass = 0
    fund_pass = 0

    L.append("① 基差做方向 (basis_z 横截面: 升水→回落? 训练定符号+embargo OOS 多空):")
    L.append("   [H | xs_ic(IR) | 符号 | 净@maker4 (t) | 净@taker10 (t) | n_reb]")
    for r in res["basis"]:
        if r.get("skip"):
            L.append(f"   {cr._hl(r['H'])}: 样本不足({r.get('n')})"); continue
        ok = _edge(r); basis_pass += int(ok)
        L.append(f"   {cr._hl(r['H']):>5}: ic={r['xs_ic']:+.4f}(IR{r['xs_ic_ir']:+.1f}) sgn{r['sign']:+.0f} | "
                 f"{r['net_4']:+6.1f}(t{r['t_4']:+.1f}) | {r['net_10']:+6.1f}(t{r['t_10']:+.1f}) | "
                 f"n{r['n_reb']} {'✓' if ok else ''}")
    L.append("")

    L.append("② 大盘 lead-lag (BTC/ETH 近端涨跌→山寨下一段; 已排除领先资产自身; 探索性, 不进判决门):")
    L.append("   [因子 | H | ic | 净@maker4 (t) | 净@taker10 (t) | n_reb]")
    for col, rows in res["lead"].items():
        for r in rows:
            if r.get("skip"):
                continue
            ok = _edge(r)        # 仅展示标记, 不进 gate (符号为假设固定的momentum, 非OOS拟合)
            L.append(f"   {col:>10} {cr._hl(r['H']):>5}: ic={r['ic']:+.4f} | "
                     f"{r['net_4']:+6.1f}(t{r['t_4']:+.1f}) | {r['net_10']:+6.1f}(t{r['t_10']:+.1f}) | "
                     f"n{r['n_reb']} {'(单格过)' if ok else ''}")
    L.append("")

    L.append("③ 资金费做方向 (funding 横截面: 高费/拥挤→未来价格? 真实时间非重叠+embargo OOS 多空):")
    L.append("   [持有(8h步) | xs_ic(IR) | 符号 | 净@maker4 (t) | 净@taker10 (t) | n_reb (长持需≥40)]")
    for r in res["funding"]:
        if r.get("skip"):
            L.append(f"   {r.get('hold_steps','?')}步: 样本不足({r.get('n')})"); continue
        ok = _edge(r, 40); fund_pass += int(ok)
        days = r["hold_steps"] * 8 / 24
        nflag = "" if r["n_reb"] >= 40 else " ⚠n<40不判决"
        L.append(f"   {r['hold_steps']}步(~{days:g}天): ic={r['xs_ic']:+.4f}(IR{r['xs_ic_ir']:+.1f}) "
                 f"sgn{r['sign']:+.0f} | {r['net_4']:+6.1f}(t{r['t_4']:+.1f}) | "
                 f"{r['net_10']:+6.1f}(t{r['t_10']:+.1f}) | n{r['n_reb']}{nflag} {'✓' if ok else ''}")
    L.append("")

    edge = (basis_pass >= 2) or (fund_pass >= 2)
    L.append(f"◆ 判决 (basis过{basis_pass}格 / funding过{fund_pass}格; 需某因子≥2格):")
    if edge:
        L.append("  ⚠ 某因子 ≥2 格过门 → 再上 DSR/PBO + 跨regime 后才值得建影子盘。")
    else:
        L.append("  ❌ 三类正交方向因子扣费后均无稳健 edge (两种费 × OOS × 组合NW-t × ≥2格)。")
        L.append("     与方向研究一致: 基差/大盘lead/资金费 作【方向信号】也翻不过成本/预测墙。")
    L.append("")
    L.append("诚实边界: basis/lead 用5m; funding真实时间对齐裸多空(price方向,carry相抵)按2腿费; 流动性冲击需L2深度(K线无,未测)。")
    L.append("  结论必要非充分; 单格过=多重检验噪声; lead-lag符号为假设固定故仅探索。")
    L.append("=" * 76)
    return "\n".join(L)
