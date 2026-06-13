"""US-stock-perp 日内日历结构 试点研究 (EXPLORATORY, 非冻结/非交易信号)。

假设(预登记式, 但样本仅~100天 -> 低置信, 明确标注): OKX 股票永续 24/7 交易, 而底层股票只在
美东 9:30-16:00(RTH)交易。由此产生加密里【不存在】的结构性效应, 且【无需任何外部数据】:
  - 收盘后(隔夜/周末)永续相对"最近一次RTH收盘"的漂移, 可能向收盘价【均值回归】;
  - 开盘前定位、隔夜/周末跳空、按市场时段(RTH/盘前/盘后)切换的动量↔回归差异。

特征【全部免费, 纯时间戳+永续K线】; 用与 BTC/ETH 研究【同一套严谨机器】(purged walk-forward +
样本外净edge闸门 + 应力)评估。诚实先验: 仍大概率无过成本边际, 但这是数据最足、结构唯一的一拳。

跑法:  python run_stock_calendar_research.py [--days 100] [--preset deep]
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from okxb.research import candle_data as cd          # noqa: E402
from okxb.research import pro_model_workflow as pmw   # noqa: E402

OUT = ROOT / "stock_calendar_research"
DATA = OUT / "data"
REPORTS = OUT / "reports"
BAR = "5m"
ET = "America/New_York"
RTH_OPEN = 9 * 60 + 30          # 570  (美东 9:30)
RTH_CLOSE = 16 * 60             # 960  (美东 16:00)

# 预登记候选: 流动性最好的 4 只(深度可交易); 周期: 日内 3h + 隔夜 12h。固定, 不事后挑。
SYMBOLS = ["MU-USDT-SWAP", "SOXL-USDT-SWAP", "NVDA-USDT-SWAP", "TSLA-USDT-SWAP"]
HORIZONS = (180, 720)


def log(m: str) -> None:
    print(m, flush=True)


def _load(symbol: str, days: int, force: bool) -> pd.DataFrame:
    return cd.get_candles(symbol, BAR, days, DATA / "candles", force=force, update=not force, log=log)


def build_calendar_panel(symbol: str, df: pd.DataFrame, H: int, bar_ms: int) -> pd.DataFrame:
    """单标的 5m K线 -> [ts, inst, fwd, y, 日历/微观特征]。全部因果(只用过去/当前)。"""
    d = df.sort_values("ts").reset_index(drop=True)
    ts = d["ts"].astype("int64")
    c = pd.to_numeric(d["c"], errors="coerce")
    et = pd.to_datetime(ts, unit="ms", utc=True).dt.tz_convert(ET)
    mod = (et.dt.hour * 60 + et.dt.minute).to_numpy()          # 美东当日分钟
    dow = et.dt.dayofweek.to_numpy()                           # 0=周一..6=周日
    weekday = dow < 5
    is_rth = (weekday & (mod >= RTH_OPEN) & (mod < RTH_CLOSE)).astype(float)
    is_pre = (weekday & (mod >= 4 * 60) & (mod < RTH_OPEN)).astype(float)
    is_after = (weekday & (mod >= RTH_CLOSE) & (mod < 20 * 60)).astype(float)
    is_weekend = (dow >= 5).astype(float)
    min_into_rth = np.where(is_rth > 0, mod - RTH_OPEN, 0.0)
    # 最近一次 RTH 收盘价(在RTH内的bar收盘, 向前填充) + 自该收盘以来的漂移(核心回归候选信号)
    last_rth_close = pd.Series(np.where(is_rth > 0, c.to_numpy(), np.nan)).ffill().to_numpy()
    ret_since_rth_close = c.to_numpy() / last_rth_close - 1.0
    # 微观控制项
    ret_1 = np.log(c).diff().to_numpy()
    roc = c.pct_change(12).to_numpy()                          # ~1h 动量
    rv = pd.Series(ret_1).rolling(60, min_periods=20).std().to_numpy()   # ~5h 已实现波动
    tod_sin = np.sin(2 * np.pi * mod / 1440.0)
    tod_cos = np.cos(2 * np.pi * mod / 1440.0)
    dow_sin = np.sin(2 * np.pi * dow / 7.0)
    dow_cos = np.cos(2 * np.pi * dow / 7.0)
    h = max(1, round(H / (bar_ms / 60_000)))
    fwd = (c.shift(-h) / c - 1.0).to_numpy()                   # H 期前向收益(标签)
    y = (fwd > 0).astype(float)
    out = pd.DataFrame({
        "ts": ts.to_numpy(), "inst": symbol, "fwd": fwd, "y": y,
        "is_rth": is_rth, "is_pre": is_pre, "is_after": is_after, "is_weekend": is_weekend,
        "min_into_rth": min_into_rth,
        "ret_since_rth_close": ret_since_rth_close,
        "ret_since_rth_close_abs": np.abs(ret_since_rth_close),
        "roc": roc, "rv": rv, "ret_1": ret_1,
        "tod_sin": tod_sin, "tod_cos": tod_cos, "dow_sin": dow_sin, "dow_cos": dow_cos,
    })
    out["y"] = out["y"].where(np.isfinite(out["fwd"]) & (out["fwd"] != 0.0), np.nan)
    return out


def _gate_for_H(panel: pd.DataFrame, H: int, bar_ms: int, preset: str, costs) -> dict:
    feats = pmw.feature_cols(panel)
    comp = pmw.purged_model_compare(panel, feats, H, bar_ms, n_folds=4, k_sel=12,
                                    min_train=4500, preset=preset, log=log)
    if comp.get("error"):
        return {"error": comp["error"]}
    best = None
    rows = []
    for name, (odf, train_abs) in comp["models"].items():
        mm = pmw.evaluate_scores(odf, H, bar_ms, train_abs, costs=costs, mode="single_asset") if len(odf) else {"skip": True}
        if mm.get("skip"):
            continue
        cells = [(fr, c) for fr, c in mm.get("curve", []) if c is not None]
        pass_cells = [(fr, c) for fr, c in cells if pmw._cell_pass(c, costs, min_ts=50)]
        stress_cells = [(fr, c) for fr, c in pass_cells if pmw._stress_pass(c, costs)]
        gate = (mm.get("primary_ic", 0.0) > 0.01 and len(pass_cells) >= 2 and len(stress_cells) >= 1)
        best_cell = max(cells, key=lambda fc: (fc[1].get("net_10", -1e9), fc[1].get("t_10", -1e9)),
                        default=(None, None))[1] or {}
        cand = {"model": name, "gate": gate, "primary_ic": mm.get("primary_ic"),
                "auc": mm.get("auc"), "net10": best_cell.get("net_10"), "t10": best_cell.get("t_10"),
                "net15": best_cell.get("net_15"), "n_ts": best_cell.get("n_ts"),
                "n_pass": len(pass_cells), "n_stress": len(stress_cells)}
        rows.append(cand)
        if best is None or (cand["gate"], cand.get("net10") or -1e9) > (best["gate"], best.get("net10") or -1e9):
            best = cand
    return {"best": best, "all": rows, "selected_freq": comp["selected_freq"][:12]}


def run(days: int, preset: str, force: bool) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    bar_ms = cd.BAR_MS[BAR]
    log(f"[stock-cal] 载入 {len(SYMBOLS)} 只股票永续 ({days}d {BAR}) ...")
    candles = {}
    for s in SYMBOLS:
        df = _load(s, days, force)
        if len(df) > 500:
            candles[s] = df
            span = (int(df["ts"].iloc[-1]) - int(df["ts"].iloc[0])) / 86_400_000
            log(f"  {s}: {len(df)} 根, 跨 {span:.0f} 天")
        else:
            log(f"  {s}: 数据不足({len(df)}), 跳过")
    if not candles:
        log("无可用股票永续数据(网络/未上线)。"); return
    costs = pmw.WorkflowCosts()
    lines = ["=" * 88, "US-stock-perp 日内日历结构 试点 (EXPLORATORY; 样本~100天 -> 低置信)",
             "=" * 88,
             "门控: OOS primary_ic>0.01 且 ≥2个置信桶过 maker+taker 净>0(NW-t>2) 且 ≥1桶过15bps应力。",
             f"标的(预登记): {', '.join(s.split('-')[0] for s in candles)}", ""]
    summary = []
    for H in HORIZONS:
        panels = [build_calendar_panel(s, candles[s], H, bar_ms) for s in candles]
        panel = pd.concat(panels, ignore_index=True)
        n = len(panel.dropna(subset=["fwd", "y"]))
        log(f"[H={H}] 面板 {n} 行, 评估中 ...")
        res = _gate_for_H(panel, H, bar_ms, preset, costs)
        if res.get("error"):
            lines.append(f"H={H}: 错误 {res['error']}"); continue
        b = res.get("best") or {}
        if not b:
            lines.append(f"H={H}min: 无有效模型(数据不足/折数不够)"); summary.append((H, {})); continue
        lines.append(f"H={H}min  best={b.get('model')}  gate={b.get('gate')}  "
                     f"primary_ic={b.get('primary_ic'):+.4f}  auc={b.get('auc'):.3f}  "
                     f"net10={b.get('net10')}bps(t{b.get('t10')})  net15={b.get('net15')}  n_ts={b.get('n_ts')}")
        topf = ", ".join(f"{k}×{v}" for k, v in res.get("selected_freq", [])[:8])
        lines.append(f"        常选特征: {topf}")
        summary.append((H, b))
    tradable = any((b or {}).get("gate") for _, b in summary)
    lines += ["", "结论:"]
    if tradable:
        lines.append("  候选(仅历史OOS过闸): 有日历结构模型过了成本闸门。鉴于样本仅~100天, 置信低, "
                     "下一步需【预登记冻结 + forward 实测】+ 在更长历史上复核, 当前不可交易。")
    else:
        lines.append("  NO EDGE: 日历结构特征未在成本调整后给出稳健样本外净edge。"
                     "(符合先验; 也可能样本太短/效应太弱。research-only, 仓位=0。)")
    lines.append("=" * 88)
    report = "\n".join(lines)
    (REPORTS / "stock_calendar_report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)


def main() -> None:
    ap = argparse.ArgumentParser(description="US-stock-perp 日内日历结构 试点研究。")
    ap.add_argument("--days", type=int, default=100)
    ap.add_argument("--preset", default="deep", choices=["fast", "deep"])
    ap.add_argument("--force", action="store_true", help="强制全量重拉K线(默认增量/缓存)")
    args = ap.parse_args()
    run(args.days, args.preset, args.force)


if __name__ == "__main__":
    main()
