"""出场 / 盈亏不对称(triple-barrier)搜索 —— 方程里最后一个未穷尽的杠杆 (W/L)。

思路: 入场用逐周期方向模型(horizon_model)的【高置信信号】, 出场改为【路径感知三重障碍】:
  止盈 TP = k_tp×ATR, 止损 SL = k_sl×ATR, 时间止 = h_max 根。哪个先碰用哪个(同根内 TP/SL 都碰→保守取SL)。
  网格搜 (k_tp,k_sl) → 训练段锁定最优 → 测试段样本外、扣费、按入场日聚类 Newey-West t。

回答: 给一个 55% 胜率/AUC0.52 的方向模型配"让利润跑、快砍亏损"的出场, 能否把扣费后做正(靠拉大 W/L)?
诚实先验: 最优停时定理 —— 零漂移下任何出场都造不出正期望(扣费后为负); 我们的漂移极小(AUC0.52),
  故这是低成功率的最后一搏, 但它确实是 E=pW−(1−p)L−cost 里没动过的那一项。

防前视: 入场信号来自净化 walk-forward 的 OOS 预测; 障碍只用入场后的 high/low 路径(未来价格但属"持仓中"realized,
  非用于预测); 网格只在训练段(入场ts≤cut)选, 测试段(入场ts>cut)评估; 同标的非重叠(平仓后才再入)。
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from . import candle_research as cr
from . import horizon_model as hm


def _log(m: str) -> None:
    print(m, flush=True)


def _barrier(high, low, close, i, direction, tp, sl, hmax):
    """从入场 i 起模拟三重障碍, 返回 (实现收益分数, 持有根数)。tp/sl 为价格分数(已=k×ATR)。"""
    e = close[i]
    if not np.isfinite(e) or e <= 0:
        return None
    n = len(close)
    end = min(i + hmax, n - 1)
    if direction > 0:
        tpx, slx = e * (1 + tp), e * (1 - sl)
        for j in range(i + 1, end + 1):
            hj, lj = high[j], low[j]
            if not np.isfinite(hj):
                continue
            ht, hs = hj >= tpx, lj <= slx
            if ht and hs:
                return (-sl, j - i)         # 同根歧义 → 保守取止损
            if ht:
                return (tp, j - i)
            if hs:
                return (-sl, j - i)
    else:
        tpx, slx = e * (1 - tp), e * (1 + sl)
        for j in range(i + 1, end + 1):
            hj, lj = high[j], low[j]
            if not np.isfinite(hj):
                continue
            ht, hs = lj <= tpx, hj >= slx
            if ht and hs:
                return (-sl, j - i)
            if ht:
                return (tp, j - i)
            if hs:
                return (-sl, j - i)
    cj = close[end]
    if not np.isfinite(cj):
        return None
    ret = (cj / e - 1.0) if direction > 0 else -(cj / e - 1.0)   # 时间止: 收盘了结
    return (ret, end - i)


def _trades_for_grid(paths, entries, k_tp, k_sl, hmax, cost_bps):
    """对每个标的、非重叠地按高置信入场点模拟障碍。entries[inst]=(idx[], dir[], atr[])。
    返回 DataFrame[entry_ts, inst, net]。"""
    rows = []
    for inst, (idx, ts_arr, sdir, atr) in entries.items():
        high, low, close = paths[inst][1], paths[inst][2], paths[inst][3]
        next_ok = -1
        for k in range(len(idx)):
            i = idx[k]
            if i <= next_ok or not np.isfinite(atr[k]) or atr[k] <= 0:
                continue
            r = _barrier(high, low, close, i, sdir[k], k_tp * atr[k], k_sl * atr[k], hmax)
            if r is None:
                continue
            ret, off = r
            rows.append((ts_arr[k], inst, ret - cost_bps / 1e4))
            next_ok = i + off
    return pd.DataFrame(rows, columns=["ts", "inst", "net"])


def _stats(trades: pd.DataFrame):
    if len(trades) < 20:
        return None
    net = trades["net"].values
    wins = net[net > 0]; losses = net[net < 0]
    daily = trades.assign(d=(trades["ts"] // 86_400_000)).groupby("d")["net"].mean().values  # 按日聚类
    return dict(n=len(net), net_bps=float(net.mean() * 1e4), winrate=float((net > 0).mean()),
                avg_w_bps=float(wins.mean() * 1e4) if len(wins) else 0.0,
                avg_l_bps=float(losses.mean() * 1e4) if len(losses) else 0.0,
                wl=float(wins.mean() / -losses.mean()) if len(losses) and losses.mean() < 0 else np.nan,
                nw_t=cr._nw_t(daily), n_days=len(daily))


def run(dfs: dict, bar: str = "5m", horizons_min=(10, 15, 30, 60),
        conf_frac: float = 0.10, cost_maker_bps: float = 4.0, cost_taker_bps: float = 10.0,
        k_tps=(1.0, 1.5, 2.0, 3.0), k_sls=(0.75, 1.0, 1.5, 2.0), hmax_mult: int = 3,
        n_folds: int = 4, log: Callable[[str], None] = _log) -> dict:
    if not hm._HAS_SK:
        return {"error": "sklearn 未安装"}
    bar_min = cr.BAR_MIN[bar]
    res = {}
    for H in horizons_min:
        log(f"  H={H}min: 方向模型OOS + 三重障碍网格搜 ...")
        panel, bar_ms = hm.build_panel(dfs, bar, H)
        oos, tconf = hm.purged_oos_predict(panel, H, bar_ms, n_folds=n_folds)
        if len(oos) < 500:
            res[H] = {"skip": True}; continue
        oos = oos.merge(panel[["ts", "inst", "atr"]], on=["ts", "inst"], how="left")
        hb = max(1, round(H / bar_min)); hmax = hb * hmax_mult
        thr = float(np.quantile(np.abs(tconf - 0.0), 1 - conf_frac)) if len(tconf) > 50 else \
            float((oos["p"] - 0.5).abs().quantile(1 - conf_frac))
        oos["conf"] = (oos["p"] - 0.5).abs()
        ent = oos[(oos["conf"] >= thr) & oos["atr"].notna()].copy()
        ent["dir"] = np.sign(ent["p"].values - 0.5)
        # 每标的入场点(grid index) + 方向 + ATR
        paths = {}
        for inst, df in dfs.items():
            d = hm._grid_ohlcv(df, bar_ms)
            paths[inst] = (d.index.values, d["h"].values, d["l"].values, d["c"].values)
        pos = {inst: {int(t): k for k, t in enumerate(paths[inst][0])} for inst in paths}
        entries = {}
        for inst, g in ent.groupby("inst"):
            if inst not in pos:
                continue
            g = g.sort_values("ts")
            idx = np.array([pos[inst].get(int(t), -1) for t in g["ts"].values])
            m = idx >= 0
            entries[inst] = (idx[m], g["ts"].values[m], g["dir"].values[m], g["atr"].values[m])
        ts_all = np.sort(ent["ts"].unique())
        cut = int(ts_all[int(len(ts_all) * 0.7)])
        # 网格: 训练段锁最优(按训练净), 测试段评估
        best = None
        grid_rows = []
        for ktp in k_tps:
            for ksl in k_sls:
                tr = _trades_for_grid(paths, entries, ktp, ksl, hmax, cost_maker_bps)
                if len(tr) < 40:
                    continue
                tr_tr = tr[tr["ts"] <= cut]; tr_te = tr[tr["ts"] > cut]
                str_ = _stats(tr_tr); ste = _stats(tr_te)
                if str_ is None or ste is None:
                    continue
                grid_rows.append((ktp, ksl, str_["net_bps"], ste))
                if best is None or str_["net_bps"] > best[2]:
                    best = (ktp, ksl, str_["net_bps"], ste, tr_te)
        # 用最优网格的测试段, 再算 taker 成本
        best_te_taker = None
        if best is not None:
            ktp, ksl = best[0], best[1]
            tr_te_tk = _trades_for_grid(paths, entries, ktp, ksl, hmax, cost_taker_bps)
            best_te_taker = _stats(tr_te_tk[tr_te_tk["ts"] > cut])
        res[H] = {"hb": hb, "hmax": hmax, "n_entries": len(ent), "thr": thr,
                  "best": best[:4] if best else None, "best_te_taker": best_te_taker,
                  "grid": grid_rows}
    return {"bar": bar, "n_inst": len(dfs), "horizons": list(horizons_min),
            "conf_frac": conf_frac, "by_h": res}


def format_report(meta: dict, res: dict) -> str:
    if res.get("error"):
        return f"出场搜索失败: {res['error']}"
    L = ["=" * 80, "出场/盈亏不对称(triple-barrier)搜索 (方向模型入场 + TP/SL/时间止 网格 · 训练锁定 · OOS扣费)", "=" * 80,
         f"数据: {res['n_inst']} 标的 / bar={res['bar']} / 跨 {meta.get('span_days','?')} 天 / "
         f"入场=方向模型置信前 {res['conf_frac']*100:g}%", ""]
    any_edge = False
    for H in res["horizons"]:
        d = res["by_h"][H]
        if d.get("skip") or not d.get("best"):
            L.append(f"━ H={H}min ━ 样本不足"); continue
        ktp, ksl, tr_net, ste = d["best"]
        tk = d["best_te_taker"]
        ok = (ste["net_bps"] > 0 and np.isfinite(ste["nw_t"]) and ste["nw_t"] > 2 and
              tk and tk["net_bps"] > 0 and (tk["nw_t"] or 0) > 2)
        any_edge = any_edge or ok
        L.append(f"━ 未来 {H}min ━ (最优 TP={ktp}×ATR / SL={ksl}×ATR / 时间止 {d['hmax']}根, 入场{d['n_entries']})")
        L.append(f"   训练净={tr_net:+.1f}bps → 测试OOS(maker): 净{ste['net_bps']:+.1f}bps "
                 f"(NW-t={ste['nw_t']:+.2f}, 胜率{ste['winrate']:.0%}, "
                 f"W={ste['avg_w_bps']:+.0f}/L={ste['avg_l_bps']:+.0f} W/L={ste['wl']:.2f}, "
                 f"n={ste['n']}, 天{ste['n_days']})")
        if tk:
            L.append(f"   测试OOS(taker): 净{tk['net_bps']:+.1f}bps (NW-t={tk['nw_t']:+.2f}, 胜率{tk['winrate']:.0%})")
        L.append(f"   判定: {'净正✓(显著)' if ok else ('弱正' if ste['net_bps'] > 0 else '净负✗')}")
        L.append("")
    L.append("◆ 判决:")
    if any_edge:
        L.append("  ⚠ 有 H 在测试段双费净正且 NW-t>2 → 这是首个候选! 须 DSR/PBO + 跨regime + 滑点 + 影子盘 再确认。")
    else:
        L.append("  ❌ 即便给方向模型配最优'让利跑/快砍亏'的三重障碍出场, 测试段扣费后仍无稳健净正。")
        L.append("     印证最优停时定理: 漂移(AUC0.52)太弱, 出场只能重分布、造不出盖过成本的正期望。")
    L.append("")
    L.append("诚实边界: 障碍用bar的high/low(同根TP/SL歧义保守取SL); 网格训练段锁定测试段评估; 未计滑点;")
    L.append("  入场=方向模型OOS高置信; 多H多格多重检验→任何正格上线前须 DSR/PBO + 可成交。")
    L.append("=" * 80)
    return "\n".join(L)
