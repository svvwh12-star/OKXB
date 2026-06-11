"""结构性/carry edge 数据验证 (只验证数据, 不建系统) —— funding 套利 + perp-spot basis。

动机: 秒级 + 分钟价量方向均已实证【扣费后无 edge】(撞"预测方向"墙)。carry/basis 不预测方向,
而是收【资金费】并用现货对冲 → delta 中性, 不撞那堵墙。本模块只回答一个问题:
  『在 OKX 单账户、扣真实两腿手续费后, 资金费 carry 到底有没有可捕捉的正 spread?』

跨所价差不在范围: 本机无法访问 Binance(地区受限), 且需第二账户+跨境(用户安全约束禁止VPN绕过)。

—— 金融模型 (delta 中性 cash-and-carry, 收正资金费) ——
  开仓: 卖空永续 1 份名义 + 买入等额现货 → delta≈0。
  资金费: realizedRate>0 时【多头付空头】, 故【空永续】每 8h 收到 +rate×名义。
  持有 t0→实际exit单份净(对一腿名义, 一阶/恒名义近似, bps量级足够):
     net = Σ已实现资金费(t0,exit]  −  (perp_exit/perp_t0 − spot_exit/spot_t0)  −  (两腿往返费+滑点)
         = 收到的资金费  −  对冲价差残差(basis 漂移, 均值回复, 挤压时会爆=主要风险)  −  成本
  费率(OKX 非VIP): 现货 maker0.08%/taker0.10% ; 永续 maker0.02%/taker0.05% → 全maker往返20bps, 全taker30bps。
  选币: 按【每币自身】过去 trail 期均资金费选高正费 top_k (因果, 仅用 <t0 信息)。
  显著性: 每个 rebalance 先聚成 1 个篮子收益再算 NW-t (避免跨币相关灌大 n)。

—— 经 4-Agent 对抗审计加固 (v2): 经济与现金流被独立验证【正确】(符号/无重复计/费用一次/篮子层面t)。 ——
  修掉 4 个会【假阳性】的判据缺陷:
   1) 选币不再要求"exit时刻有数据"(原会静默丢掉持有期内退市/熔断的高费小币=最不利尾部) →
      改为仅用 <t0 信息选币; exit 用 (t0,t1] 内最后可得价【强制平仓建模】, 无价则平在入场(残差0但照付费)。
   2) 判决门要求 n_reb≥20 ∧ 持有≥7天 (carry是低频, 短样本t/Sharpe是噪声); 短样本只展示不判决。
   3) 现货腿【流动性门】(spot ADV) + 报告实际被选中的币 → 看 edge 是否只活在不可成交的小币; 净值为"未计滑点"。
   4) Newey-West lag 设地板(≥4)抗 carry 自身的自相关; Sharpe 退出判决门, 只靠 NW-t; 多重检验只认长持有格。
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

FUND_PER_DAY = 3            # 每日 3 次资金费 (8h)
FUND_PER_YEAR = 3 * 365
EIGHT_H_MS = 8 * 3_600_000
MIN_VERDICT_N = 20          # 判决门: 少于这么多次 rebalance 不下"可做"结论
MIN_VERDICT_HOLD_DAYS = 7   # 判决门: carry 宜长持; 只认长持有格 (短持有费拖累大且噪声多)


def _log(m: str) -> None:
    print(m, flush=True)


def _nw_t(x: np.ndarray, lag: Optional[int] = None) -> tuple[float, int]:
    """均值=0 的 Newey-West(HAC) t 值, 返回 (t, lag)。lag 设地板≥4 抗 carry 序列自相关
    (n**0.25 是渐近式, 小样本会塌到1-2, 欠修正→t虚高)。"""
    x = np.asarray(x, float)
    n = len(x)
    if n < 5:
        return np.nan, 0
    if lag is None:
        lag = max(4, int(n ** 0.25))
    lag = min(lag, n - 1)
    u = x - x.mean()
    s = float((u * u).mean())
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1.0)
        s += 2.0 * w * float((u[l:] * u[:-l]).mean())
    s = max(s, 1e-30)
    se = float(np.sqrt(s / n))
    return (float(x.mean() / se) if se > 0 else np.nan), lag


def _ar1(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan
    a, b = x[1:], x[:-1]
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-15 else np.nan


# ----------------------------- 对齐到 8h 资金费网格 -----------------------------

def align_funding_panel(perp_df: pd.DataFrame, spot_df: pd.DataFrame,
                        funding_df: pd.DataFrame) -> pd.DataFrame:
    """把 perp/spot 收盘价对齐到每个资金费时刻(取该时刻或之前最近的 K线收盘)。
    返回 DataFrame[ts, perp, spot, funding, basis], 按 ts 升序。"""
    if len(funding_df) < 5 or len(perp_df) < 5 or len(spot_df) < 5:
        return pd.DataFrame(columns=["ts", "perp", "spot", "funding", "basis"])
    f = funding_df.sort_values("ts").reset_index(drop=True)
    p = perp_df[["ts", "c"]].sort_values("ts").rename(columns={"c": "perp"})
    s = spot_df[["ts", "c"]].sort_values("ts").rename(columns={"c": "spot"})
    out = pd.merge_asof(f, p, on="ts", direction="backward", tolerance=EIGHT_H_MS)
    out = pd.merge_asof(out, s, on="ts", direction="backward", tolerance=EIGHT_H_MS)
    out = out.dropna(subset=["perp", "spot"])
    out = out[(out["perp"] > 0) & (out["spot"] > 0)].reset_index(drop=True)
    out["basis"] = out["perp"] / out["spot"] - 1.0
    return out


def build_carry_panel(perp_dfs: dict, spot_dfs: dict, funding_dfs: dict,
                      log: Callable[[str], None] = _log) -> dict[str, pd.DataFrame]:
    panels = {}
    for inst, pdf in perp_dfs.items():
        sdf = spot_dfs.get(inst)
        fdf = funding_dfs.get(inst)
        if sdf is None or fdf is None:
            continue
        pan = align_funding_panel(pdf, sdf, fdf)
        if len(pan) >= 20:
            panels[inst] = pan
    log(f"对齐资金费面板: {len(panels)} 标的 (需 perp+spot+funding 三者齐全且≥20期)")
    return panels


def spot_adv_table(spot_dfs: dict) -> dict[str, float]:
    """每个现货标的的日均报价额(USDT) —— 现货腿流动性, 用于剔除不可成交的小币。
    spot_dfs 的 key 已是永续 instId (见 CLI 映射)。"""
    adv = {}
    for inst, df in spot_dfs.items():
        if len(df) < 5 or "volquote" not in df:
            adv[inst] = 0.0
            continue
        dt = float(np.median(np.diff(df["ts"].values))) if len(df) > 2 else EIGHT_H_MS
        bars_per_day = 86_400_000 / max(dt, 1.0)
        adv[inst] = float(df["volquote"].mean() * bars_per_day)
    return adv


# ----------------------------- 资金费统计 -----------------------------

def funding_stats(panels: dict[str, pd.DataFrame], min_per8h_bps: float) -> pd.DataFrame:
    rows = []
    for inst, pan in panels.items():
        fnd = pan["funding"].values
        per8h = float(np.mean(fnd) * 1e4)
        rows.append(dict(
            inst=inst.split("-")[0], n=len(fnd),
            ann_pct=float(np.mean(fnd) * FUND_PER_YEAR * 100),
            pos_share=float(np.mean(fnd > 0)),
            per8h_bps=per8h,
            harvest=("✓" if per8h > min_per8h_bps else " "),   # 是否过正费门(可短永续harvest)
            ar1=_ar1(fnd),
            basis_bps=float(np.mean(pan["basis"]) * 1e4),
        ))
    # 按【带符号】每8h费率降序 → 顶部是真正可harvest的高正费币 (不再按 |年化| 把负费币顶上来)
    return pd.DataFrame(rows).sort_values("per8h_bps", ascending=False)


def funding_persistence(panels: dict[str, pd.DataFrame]) -> dict:
    """harvest 根本假设: 过去资金费能否预测下一期? 横截面 当期→下期 秩相关。"""
    parts = []
    for inst, pan in panels.items():
        d = pan[["ts", "funding"]].copy()
        d["next"] = d["funding"].shift(-1)
        parts.append(d.dropna())
    if not parts:
        return {"xs_ic": np.nan, "ir": np.nan, "n": 0}
    big = pd.concat(parts, ignore_index=True)
    ics = []
    for ts, g in big.groupby("ts"):
        if len(g) >= 5 and g["funding"].std() > 0 and g["next"].std() > 0:
            fr = g["funding"].rank() - g["funding"].rank().mean()
            nr = g["next"].rank() - g["next"].rank().mean()
            d = np.sqrt((fr * fr).sum() * (nr * nr).sum())
            if d > 1e-12:
                ics.append(float((fr * nr).sum() / d))
    if not ics:
        return {"xs_ic": np.nan, "ir": np.nan, "n": 0}
    ics = np.array(ics)
    t, _ = _nw_t(ics)
    return {"xs_ic": float(ics.mean()), "ir": float(t), "n": len(ics)}


# ----------------------------- carry 回测 (因果选币 · 强制平仓建模 · 篮子层面 · 扣费) -----------------------------

def carry_backtest(panels: dict[str, pd.DataFrame], *, hold_days: float,
                   spot_adv: dict, top_k: int = 8, min_per8h_bps: float = 0.0,
                   trail_days: float = 3.0, fees_rt_bps: float = 20.0, slippage_rt_bps: float = 0.0,
                   min_spot_adv: float = 3_000_000.0) -> Optional[dict]:
    """delta 中性 carry 篮子回测, 按【真实时间】rebalance(非重叠), 兼容 4h/8h 混合资金费周期。
      选币只用 <t0 信息(每币自身过去 trail_days 均费>min, 现货ADV够); exit 用 (t0,t1] 内最后可得价
      【强制平仓建模】(退市/熔断不静默丢弃); 单币净=Σ未来已实现资金费−对冲残差−(往返费+滑点); 篮子=等权。"""
    hold_ms = int(hold_days * 86_400_000)
    trail_ms = int(trail_days * 86_400_000)
    cost = (fees_rt_bps + slippage_rt_bps) / 1e4
    P = {inst: pan.set_index("ts").sort_index() for inst, pan in panels.items()}
    lo = min(int(p.index[0]) for p in P.values())
    hi = max(int(p.index[-1]) for p in P.values())
    if hi - lo < trail_ms + hold_ms + 5 * EIGHT_H_MS:
        return None

    basket, gross, fundonly = [], [], []
    sel_counter: dict[str, int] = {}
    n_names_log = []
    t0 = lo + trail_ms
    while t0 + hold_ms <= hi:
        t1 = t0 + hold_ms
        cands = []
        for inst, pan in P.items():
            if spot_adv.get(inst, 0.0) < min_spot_adv:        # 现货腿流动性门
                continue
            idx = pan.index
            before = idx[idx <= t0]
            if len(before) == 0 or t0 - int(before[-1]) > 2 * EIGHT_H_MS:   # 入场附近无报价→跳过
                continue
            past = pan.loc[(idx < t0) & (idx >= t0 - trail_ms), "funding"]  # 每币自身、严格<t0
            if len(past) < 2:
                continue
            tf = float(past.mean())
            if tf * 1e4 > min_per8h_bps:                       # 只收正费 (可短永续, 不需现货融券)
                cands.append((tf, inst, int(before[-1])))
        if not cands:
            t0 = t1
            continue
        cands.sort(reverse=True)
        sel = cands[:top_k]
        rets, grs, fos = [], [], []
        for _, inst, ts0 in sel:
            pan = P[inst]; idx = pan.index
            p0 = pan.at[ts0, "perp"]; s0 = pan.at[ts0, "spot"]
            win = idx[(idx > ts0) & (idx <= t1)]
            if len(win) == 0:                                 # 持有期内无任何数据→强制平在入场(只亏费)
                fund = 0.0; residual = 0.0
            else:
                exit_ts = win[-1]
                fund = float(pan.loc[(idx > ts0) & (idx <= exit_ts), "funding"].sum())
                residual = (pan.at[exit_ts, "perp"] / p0 - 1.0) - (pan.at[exit_ts, "spot"] / s0 - 1.0)
            g = fund - residual
            rets.append(g - cost); grs.append(g); fos.append(fund)
            sel_counter[inst] = sel_counter.get(inst, 0) + 1
        if rets:
            basket.append(float(np.mean(rets)))
            gross.append(float(np.mean(grs)))
            fundonly.append(float(np.mean(fos)))
            n_names_log.append(len(rets))
        t0 = t1                                               # 非重叠
    if len(basket) < 8:
        return None
    b = np.array(basket); g = np.array(gross); fo = np.array(fundonly)
    rb_per_year = 365.0 / hold_days
    nw_t, nw_lag = _nw_t(b)
    top_names = sorted(sel_counter.items(), key=lambda kv: kv[1], reverse=True)[:8]
    return dict(
        n_reb=len(b), avg_names=float(np.mean(n_names_log)),
        net_per_trade_bps=float(b.mean() * 1e4), gross_per_trade_bps=float(g.mean() * 1e4),
        fundonly_per_trade_bps=float(fo.mean() * 1e4),
        residual_per_trade_bps=float((g - fo).mean() * 1e4),
        net_ann_pct=float(b.mean() * rb_per_year * 100),
        sharpe=float(b.mean() / b.std() * np.sqrt(rb_per_year)) if b.std() > 0 else np.nan,
        nw_t=nw_t, nw_lag=nw_lag, pos=float((b > 0).mean()),
        maxdd_bps=float(_max_drawdown(b) * 1e4),
        fees_rt_bps=fees_rt_bps, slippage_rt_bps=slippage_rt_bps,
        hold_days=hold_days,
        top_names=[(k.split("-")[0], v) for k, v in top_names],
    )


def _max_drawdown(per_trade: np.ndarray) -> float:
    eq = np.cumsum(per_trade)        # 等权 per-trade net bps 的累加(非复利)权益曲线
    peak = np.maximum.accumulate(eq)
    return float((eq - peak).min())


def _edge_ok(s: Optional[dict]) -> bool:
    """判决门: 净正 ∧ NW-t>2 ∧ n_reb≥20 ∧ 持有≥7天 (Sharpe 退出判决门, 仅展示)。"""
    return bool(s and s["net_ann_pct"] > 0 and np.isfinite(s["nw_t"]) and s["nw_t"] > 2
                and s["n_reb"] >= MIN_VERDICT_N and s["hold_days"] >= MIN_VERDICT_HOLD_DAYS)


# ----------------------------- 编排 + 报告 -----------------------------

def run(perp_dfs: dict, spot_dfs: dict, funding_dfs: dict, *,
        holds_days=(3, 7, 14, 30), top_k: int = 8, min_per8h_bps: float = 0.0,
        fees_maker_bps: float = 20.0, fees_taker_bps: float = 30.0,
        slippage_rt_bps: float = 0.0, min_spot_adv: float = 3_000_000.0,
        log: Callable[[str], None] = _log) -> dict:
    panels = build_carry_panel(perp_dfs, spot_dfs, funding_dfs, log=log)
    if len(panels) < 5:
        return {"error": f"可用标的太少({len(panels)})"}
    adv = spot_adv_table(spot_dfs)
    fstats = funding_stats(panels, min_per8h_bps)
    persist = funding_persistence(panels)
    bt = {}
    for rd in holds_days:
        bt[rd] = {
            "maker": carry_backtest(panels, hold_days=rd, spot_adv=adv, top_k=top_k,
                                    min_per8h_bps=min_per8h_bps, fees_rt_bps=fees_maker_bps,
                                    slippage_rt_bps=slippage_rt_bps, min_spot_adv=min_spot_adv),
            "taker": carry_backtest(panels, hold_days=rd, spot_adv=adv, top_k=top_k,
                                    min_per8h_bps=min_per8h_bps, fees_rt_bps=fees_taker_bps,
                                    slippage_rt_bps=slippage_rt_bps, min_spot_adv=min_spot_adv),
        }
    n_liq = sum(1 for v in adv.values() if v >= min_spot_adv)
    return dict(n_inst=len(panels), n_liq=n_liq, fstats=fstats, persist=persist, bt=bt,
                holds_days=list(holds_days), top_k=top_k, min_per8h_bps=min_per8h_bps,
                fees_maker=fees_maker_bps, fees_taker=fees_taker_bps,
                slippage=slippage_rt_bps, min_spot_adv=min_spot_adv)


def format_report(meta: dict, res: dict) -> str:
    if res.get("error"):
        return f"carry 验证失败: {res['error']}"
    L = ["=" * 76,
         "结构性 carry 验证 (资金费套利 + perp-spot basis · delta中性 · 扣费 · 已过对抗审计v2)",
         "=" * 76,
         f"数据: {res['n_inst']} 标的(perp+spot+funding齐全), 其中现货ADV≥${res['min_spot_adv']/1e6:.0f}M 的 "
         f"{res['n_liq']} 个可做对冲 / 跨 {meta.get('span_days','?')} 天",
         f"费率: 全maker往返 {res['fees_maker']:.0f}bps | 全taker往返 {res['fees_taker']:.0f}bps "
         f"+ 滑点 {res['slippage']:.0f}bps (现货0.08-0.10%+永续0.02-0.05%, 两腿各进出)",
         f"harvest: 每币过去均资金费>{res['min_per8h_bps']:.1f}bps/8h 的 top{res['top_k']} 正费币, "
         f"卖空永续+买现货(delta中性, 现货ADV门)", ""]

    f: pd.DataFrame = res["fstats"]
    L.append("① 资金费分布 (按每8h费率降序; harvest✓=过正费门; 年化=若持续单边, 真实看 AR1):")
    L.append("   [币 | 每8h(bps) | 年化% | 正费占比 | 资金费AR1 | basis(bps) | harvest | n期]")
    for _, r in f.head(15).iterrows():
        L.append(f"   {r['inst']:>8}: {r['per8h_bps']:+6.2f} | {r['ann_pct']:+7.1f} | "
                 f"{r['pos_share']:4.0%} | AR1{r['ar1']:+.2f} | {r['basis_bps']:+7.1f} | "
                 f"  {r['harvest']}    | {int(r['n'])}")
    if len(f) > 15:
        L.append(f"   ... 另 {len(f)-15} 个 (完整见返回对象)")
    p = res["persist"]
    can = (p.get("xs_ic") or 0) > 0.05 and (p.get("ir") or 0) > 2 and (p.get("n") or 0) > 30
    L.append(f"\n   资金费持续性 (harvest根本假设, 当期→下期): IC={p.get('xs_ic', float('nan')):+.3f} "
             f"(NW-t={p.get('ir', float('nan')):+.1f}, n={p.get('n', 0)})  "
             f"{'→ 高费币下期大概率仍高费, 可瞄准' if can else '→ 持续性不足'}")
    L.append("")

    L.append("② ★carry 篮子回测 (因果选币·篮子层面·扣两腿费·非重叠·强制平仓建模):")
    L.append("   [持有 费档 | 净/笔bps | 毛 | 仅资金费 | 残差 | 年化净% | Sharpe | NW-t(lag) | 正占 | 回撤bps | 笔数]")
    any_edge = False
    for rd in res["holds_days"]:
        for cn in ("maker", "taker"):
            s = res["bt"][rd][cn]
            if s is None:
                L.append(f"   {rd:g}天 {cn:5}: 样本不足")
                continue
            ok = _edge_ok(s)
            any_edge = any_edge or ok
            nflag = "" if s["n_reb"] >= MIN_VERDICT_N else " ⚠n不足不判决"
            fl = "可做✓" if ok else ("弱正" if s["net_ann_pct"] > 0 else "净负✗")
            L.append(f"   {rd:g}天 {cn:5}: {s['net_per_trade_bps']:+6.1f} | {s['gross_per_trade_bps']:+5.1f} | "
                     f"{s['fundonly_per_trade_bps']:+5.1f} | {s['residual_per_trade_bps']:+5.1f} | "
                     f"{s['net_ann_pct']:+6.1f}% | SR{s['sharpe']:+.2f} | t{s['nw_t']:+.2f}({s['nw_lag']}) | "
                     f"{s['pos']:3.0%} | {s['maxdd_bps']:+.0f} | n{s['n_reb']}{nflag} {fl}")
    # 展示长持有格实际被选中的币(看 edge 是否只活在不可成交小币)
    longest = max(res["holds_days"])
    sl = res["bt"][longest]["maker"]
    if sl and sl.get("top_names"):
        names = ", ".join(f"{k}×{v}" for k, v in sl["top_names"])
        L.append(f"   [{longest:g}天最常被选中的币]: {names}")
    L.append("")

    L.append("◆ 判决 (扣费后 · 年化净正 ∧ NW-t>2 ∧ n_reb≥20 ∧ 持有≥7天 才算可做; Sharpe不进判决门):")
    if any_edge:
        L.append("  ✅ 存在扣费后稳健(长持有·够样本·抗自相关)的资金费 carry edge → 值得建 delta中性 carry 系统(需现货腿)。")
        L.append("     下一步: 加现货滑点/借币成本/挤压尾部压力 → 容量核算(被选币现货深度) → 小资金影子盘。")
    else:
        L.append("  ❌ 无扣费后稳健 carry edge: 主流币资金费太薄(两腿20-30bps费碾压), 高费多在小币(已被流动性门/容量限制)。")
        L.append("     资金费虽真实且持续(可瞄准), 但对1000U散户、扣两腿费后净负。出路: 更高VIP降费/更长持有/或承认不可行。")
    L.append("")
    L.append("诚实边界: 净值【未计现货滑点/借币费/挤压尾部】(默认滑点0); 高费多为小币(容量/流动性受限);")
    L.append("  负费harvest需现货融券(更难), 只测正费可执行侧; 持有/费档/top_k为样本内网格(选币因果)。必要非充分。")
    L.append("=" * 76)
    return "\n".join(L)
