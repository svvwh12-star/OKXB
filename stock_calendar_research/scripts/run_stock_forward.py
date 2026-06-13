"""US-stock-perp 日历结构 候选 forward 实测 (预登记冻结 + 前向裁决)。

冻结候选(预登记; 来自 run_stock_calendar_research 的 100d/deep + --robust 稳健性电池):
  ST720: H=720min, hist_gbm, 池化 MU/SOXL/NVDA/TSLA, top_frac=0.10, oos_ic_sign=+1
         机制: 永续相对"最近一次RTH(美东9:30-16:00)收盘"的隔夜/周末漂移 -> 均值回归。
         in-sample: 过净edge闸门 + 4只逐标的IC全正 + 前后半都正 + 复现一致; 但门控借线(t~2)、
         样本仅~100天 -> 历史过闸【必须】由 forward 样本外裁决, 现不可交易。
诚实先验: 短样本历史过闸常是假阳; forward 大概率仍 PENDING/KILL。demo-only, 过 PASS 前不实盘。

freeze  : 在【钉死的100天窗口+固定种子】上训练, 存 model/feats/median/scaler/tau + 元数据。
evaluate: 复用冻结件(不refit/不重选), 只评 official_forward_start 之后【已结算】的 bar,
          net of taker + x1.5 应力, 检 PASS/KILL/PENDING(与加密同协议)。
snapshot: 当前打分(监控)。

前向观测持久化: evaluate 把【已结算且过tau】的观测 (ts,inst,score,fwd) 增量写入
   forward/ST720_obs.csv(按 ts+inst 去重、不可变), 裁决从【全部累积观测】算 ->
   跨 OKX ~100天保留期【无限期累积】(只要每 <90 天至少评估一次, 就不丢任何已结算bar)。
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPTS))

import run_stock_calendar_research as sc              # noqa: E402  (复用同一套面板/特征/缓存)
from okxb.research import candle_data as cd           # noqa: E402
from okxb.research import candle_research as cr        # noqa: E402
from okxb.research import feature_lab as fl            # noqa: E402
from okxb.research import pro_model_workflow as pmw    # noqa: E402
from sklearn.preprocessing import StandardScaler       # noqa: E402

OUT = ROOT / "stock_calendar_research"
DATA = OUT / "data"
FROZEN = OUT / "frozen"
FWD = OUT / "forward"
BAR = sc.BAR

# 预登记冻结候选(钉死, 不事后改)
CANDIDATE = {"code": "ST720", "H": 720, "model": "hist_gbm",
             "symbols": sc.SYMBOLS, "top_frac": 0.10, "oos_ic_sign": +1}
FREEZE_DAYS = 100
EVAL_DAYS = 110
GEN_DAYS = 100
# 留出泛化检验用: 训练4只之外、有一定流动性的其它 OKX 股票永续(模型从未见过)
HELDOUT = ["INTC-USDT-SWAP", "MRVL-USDT-SWAP", "MSTR-USDT-SWAP", "AMD-USDT-SWAP", "COIN-USDT-SWAP",
           "QQQ-USDT-SWAP", "GOOGL-USDT-SWAP", "AAPL-USDT-SWAP", "AMZN-USDT-SWAP", "MSFT-USDT-SWAP",
           "SPY-USDT-SWAP", "META-USDT-SWAP", "NFLX-USDT-SWAP", "PLTR-USDT-SWAP", "AVGO-USDT-SWAP"]
K_SEL = 12
PRESET = "deep"
TAKER_BPS = 10.0
STRESS_MULT = 1.5
T_PASS = 2.13               # 单候选保守阈(沿用加密协议)
N_MIN = 50
N_KILL = 30
WEEK_MS = 7 * 86_400_000


def log(m: str) -> None:
    print(m, flush=True)


def _pooled_panel(symbols, days, H, bar_ms, *, force=False, update=True):
    panels = []
    for s in symbols:
        df = cd.get_candles(s, BAR, days, DATA / "candles", force=force, update=update, log=log)
        if len(df) > 500:
            panels.append(sc.build_calendar_panel(s, df, H, bar_ms))
    return pd.concat(panels, ignore_index=True) if panels else pd.DataFrame()


def _net(port: np.ndarray, cost_bps: float):
    st = cr._stats_from_gross(port, cost_bps, 1) if len(port) >= 8 else None
    return (st["net_bps"], st["nw_t"]) if st else (np.nan, np.nan)


def forward_verdict(net15, t15, net10, n_ts, fwd_ic_sign, train_ic_sign, weeks) -> str:
    """与加密 forward 协议同逻辑(纯函数)。"""
    if n_ts >= 8:
        if (net10 is not None and net10 <= 0) or (fwd_ic_sign != 0 and fwd_ic_sign != train_ic_sign):
            return "KILL"
    if weeks >= 8 and n_ts < N_KILL:
        return "KILL"
    if (net15 is not None and net15 > 0 and (t15 or 0) > T_PASS and n_ts >= N_MIN
            and fwd_ic_sign == train_ic_sign):
        return "PASS"
    return "PENDING"


def freeze() -> None:
    FROZEN.mkdir(parents=True, exist_ok=True)
    bar_ms = cd.BAR_MS[BAR]
    stamp = int(time.time() * 1000)
    H = CANDIDATE["H"]
    panel = _pooled_panel(CANDIDATE["symbols"], FREEZE_DAYS, H, bar_ms, force=False, update=True)
    if not len(panel):
        log("[freeze] 无可用面板(候选股票永续数据不足)。"); return
    feats = pmw.feature_cols(panel)
    data = panel[["ts", "inst", "fwd", "y"] + feats].replace([np.inf, -np.inf], np.nan).dropna(subset=["fwd", "y"])
    sel, _ = fl.select_features(data, feats, k=K_SEL)
    med = data[sel].median()
    X = data[sel].fillna(med).values
    kind, factory = pmw._model_factories(preset=PRESET)[CANDIDATE["model"]]
    scaler = StandardScaler().fit(X) if kind.endswith("scale") else None
    Xf = scaler.transform(X) if scaler is not None else X
    target = data["y"].values.astype(int) if kind.startswith("clf") else data["fwd"].values.astype(float)
    model = factory().fit(Xf, target)
    train_score = pmw._score_from_model(kind, model, Xf)
    tau = float(np.nanquantile(np.abs(train_score), 1 - CANDIDATE["top_frac"]))
    insample_ic_sign = int(np.sign(cr._spearman(train_score, data["fwd"].values) or 0.0))
    cutoff = int(data["ts"].max())
    d = FROZEN / CANDIDATE["code"]
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "model.pkl", "wb") as fh:
        pickle.dump({"model": model, "kind": kind, "scaler": scaler}, fh)
    (d / "feature_list.json").write_text(json.dumps(sel), encoding="utf-8")
    (d / "median.json").write_text(json.dumps(med.to_dict()), encoding="utf-8")
    (d / "meta.json").write_text(json.dumps({
        "code": CANDIDATE["code"], "H": H, "model": CANDIDATE["model"], "symbols": CANDIDATE["symbols"],
        "top_frac": CANDIDATE["top_frac"], "bar_ms": bar_ms, "tau": tau,
        "ic_sign": CANDIDATE["oos_ic_sign"], "insample_ic_sign": insample_ic_sign, "n_sel": len(sel),
        "freeze_cutoff_ts": cutoff, "frozen_at_ms": stamp, "official_forward_start_ts": stamp,
    }, indent=2), encoding="utf-8")
    log(f"[freeze {CANDIDATE['code']}] H={H} {CANDIDATE['model']} 池化{len(CANDIDATE['symbols'])}只 "
        f"sel={len(sel)} tau={tau:.5f} oos_ic_sign={CANDIDATE['oos_ic_sign']:+d} "
        f"(in-sample {insample_ic_sign:+d}) cutoff={sc and pd.to_datetime(cutoff, unit='ms', utc=True)}")


def evaluate() -> None:
    FWD.mkdir(parents=True, exist_ok=True)
    now = int(time.time() * 1000)
    d = FROZEN / CANDIDATE["code"]
    if not (d / "meta.json").exists():
        log(f"[{CANDIDATE['code']}] 尚未冻结 — 先 --mode freeze"); return
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    H, bar_ms, tau = meta["H"], meta["bar_ms"], meta["tau"]
    official_start = max(int(meta["freeze_cutoff_ts"]), int(meta["official_forward_start_ts"]))
    train_ic_sign = meta["ic_sign"]
    weeks = (now - meta["frozen_at_ms"]) / WEEK_MS
    blob = pickle.load(open(d / "model.pkl", "rb"))
    model, kind, scaler = blob["model"], blob["kind"], blob["scaler"]
    sel = json.loads((d / "feature_list.json").read_text(encoding="utf-8"))
    med = pd.Series(json.loads((d / "median.json").read_text(encoding="utf-8")))
    panel = _pooled_panel(meta["symbols"], EVAL_DAYS, H, bar_ms, force=False, update=True)
    row = {"asof_utc": pd.to_datetime(now, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M"),
           "code": CANDIDATE["code"], "H": H, "weeks": round(weeks, 2), "n_signals": 0,
           "n_ts": 0, "verdict": "PENDING", "start_utc": pd.to_datetime(official_start, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M")}
    # 1) 当前窗口内【已结算且过tau】的观测 (ts,inst,score,fwd)
    new_obs = pd.DataFrame(columns=["ts", "inst", "score", "fwd"])
    if len(panel):
        for c in sel:
            if c not in panel.columns:
                panel[c] = np.nan
        settle_hi = now - H * 60_000
        sub = panel[(panel["ts"] > official_start) & (panel["ts"] <= settle_hi)]
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=["fwd", "y"])
        if len(sub):
            X = sub[sel].fillna(med).values
            Xf = scaler.transform(X) if scaler is not None else X
            sub = sub.assign(score=pmw._score_from_model(kind, model, Xf))
            new_obs = sub.loc[np.abs(sub["score"].values) >= tau, ["ts", "inst", "score", "fwd"]].copy()
    # 2) 合并到【持久化观测库】(去重 ts+inst, 留旧 -> 前向记录不可变), 存盘。
    #    已结算bar的 (score,fwd) 是定值; 一旦记下即永久, 不再受滚动窗口/OKX~100天保留期限制。
    obs_path = FWD / "ST720_obs.csv"
    if obs_path.exists():
        try:
            old = pd.read_csv(obs_path)
        except Exception:  # noqa: BLE001
            old = pd.DataFrame(columns=["ts", "inst", "score", "fwd"])
        allobs = pd.concat([old, new_obs], ignore_index=True)
    else:
        allobs = new_obs
    if len(allobs):
        allobs["ts"] = allobs["ts"].astype("int64")
        allobs = allobs.drop_duplicates(["ts", "inst"], keep="first").sort_values("ts").reset_index(drop=True)
        allobs.to_csv(obs_path, index=False)
    # 3) 裁决从【全部累积观测】算(净edge闸门), 不只看当前窗口
    if len(allobs):
        h = max(1, round(H / (bar_ms / 60_000)))
        reb = set(cr._rebalance_ts(np.sort(allobs["ts"].unique()), bar_ms, h).tolist())
        tr = allobs[allobs["ts"].isin(reb)]
        if len(tr):
            signed = np.sign(tr["score"].values) * tr["fwd"].values
            port = pd.Series(signed, index=tr["ts"].values).groupby(level=0).mean().values
            net10, t10 = _net(port, TAKER_BPS)
            net15, t15 = _net(port, TAKER_BPS * STRESS_MULT)
            fic = cr._spearman(tr["score"].values, tr["fwd"].values) if len(tr) > 5 else 0.0
            fic_sign = int(np.sign(fic or 0.0))
            verdict = forward_verdict(net15, t15, net10, len(port), fic_sign, train_ic_sign, weeks)
            row.update({"n_signals": len(tr), "n_ts": len(port),
                        "net10_bps": round(net10, 1), "t10": round(t10, 2),
                        "net15_bps": round(net15, 1), "t15": round(t15, 2),
                        "fwd_ic": round(float(fic), 4), "fwd_ic_sign": fic_sign,
                        "train_ic_sign": train_ic_sign, "verdict": verdict})
    hist = FWD / "stock_forward_status.csv"
    df = pd.DataFrame([row])
    if hist.exists():
        try:
            if pd.read_csv(hist, nrows=0).columns.tolist() != df.columns.tolist():
                hist.replace(FWD / f"stock_forward_status_archived_{int(time.time())}.csv")
        except Exception:  # noqa: BLE001
            hist.replace(FWD / f"stock_forward_status_archived_{int(time.time())}.csv")
    df.to_csv(hist, mode="a", header=not hist.exists(), index=False)
    log(f"[{CANDIDATE['code']}] H={H} weeks={weeks:.1f} 累积观测={len(allobs)} n_ts={row.get('n_ts')} "
        f"net10={row.get('net10_bps', '-')}(t{row.get('t10', '-')}) "
        f"net15={row.get('net15_bps', '-')}(t{row.get('t15', '-')}) "
        f"fwd_ic={row.get('fwd_ic', '-')} -> {row['verdict']}")
    log("决策: 全期 0 PASS -> 干净关闭(候选证伪); 任一 PASS -> 第二段独立 forward + demo 微量成交验证。")


def snapshot() -> None:
    d = FROZEN / CANDIDATE["code"]
    if not (d / "meta.json").exists():
        log("尚未冻结。"); return
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    H, bar_ms, tau = meta["H"], meta["bar_ms"], meta["tau"]
    blob = pickle.load(open(d / "model.pkl", "rb"))
    model, kind, scaler = blob["model"], blob["kind"], blob["scaler"]
    sel = json.loads((d / "feature_list.json").read_text(encoding="utf-8"))
    med = pd.Series(json.loads((d / "median.json").read_text(encoding="utf-8")))
    panel = _pooled_panel(meta["symbols"], 35, H, bar_ms, force=False, update=True)
    scores = {}
    if len(panel):
        for c in sel:
            if c not in panel.columns:
                panel[c] = np.nan
        last = panel.sort_values("ts").groupby("inst").tail(1)
        X = last[sel].fillna(med).values
        Xf = scaler.transform(X) if scaler is not None else X
        sccol = pmw._score_from_model(kind, model, Xf)
        for inst, scv in zip(last["inst"].values, sccol):
            scores[str(inst).split("-")[0]] = {"score": round(float(scv), 4),
                                               "hit": bool(abs(scv) >= tau),
                                               "side": "buy" if scv > 0 else "sell"}
    print("SNAPSHOT_JSON " + json.dumps({"tau": round(tau, 4), "scores": scores}, ensure_ascii=False))


def generalize() -> None:
    """留出泛化检验: 用冻结的 ST720 模型给【从未训练过的】其它股票永续打分, 看样本外IC。
    结构性通性 -> 留出名字也多数正IC; 过拟合这4只 -> 留出~0/负。(同期, 跨标的OOS; 纯历史, 不交易)"""
    d = FROZEN / CANDIDATE["code"]
    if not (d / "meta.json").exists():
        log("尚未冻结 — 先 --mode freeze"); return
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    H, bar_ms = meta["H"], meta["bar_ms"]
    blob = pickle.load(open(d / "model.pkl", "rb"))
    model, kind, scaler = blob["model"], blob["kind"], blob["scaler"]
    sel = json.loads((d / "feature_list.json").read_text(encoding="utf-8"))
    med = pd.Series(json.loads((d / "median.json").read_text(encoding="utf-8")))
    train_syms = list(meta["symbols"])

    def _ic(symbol: str):
        df = cd.get_candles(symbol, BAR, GEN_DAYS, DATA / "candles", force=False, update=True, log=log)
        if len(df) < 1000:
            return None
        panel = sc.build_calendar_panel(symbol, df, H, bar_ms)
        for c in sel:
            if c not in panel.columns:
                panel[c] = np.nan
        cl = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=["fwd"])
        if len(cl) < 500:
            return None
        X = cl[sel].fillna(med).values
        Xf = scaler.transform(X) if scaler is not None else X
        ic = cr._spearman(pmw._score_from_model(kind, model, Xf), cl["fwd"].values)
        span = (int(df["ts"].iloc[-1]) - int(df["ts"].iloc[0])) / 86_400_000
        return (round(float(ic or 0.0), 4), len(cl), round(span, 0))

    out = ["=" * 80, "ST720 留出泛化检验: 冻结模型在【未训练的】股票永续上的样本外IC", "=" * 80,
           f"训练于: {', '.join(s.split('-')[0] for s in train_syms)} | 读法: 训练名字都强正; "
           "留出名字若多数正 -> 结构性通性; 若~0/负 -> 过拟合这4只。", "",
           "训练名字(参考, 同模型 in-sample 偏乐观):"]
    for s in train_syms:
        r = _ic(s)
        out.append(f"  {s.split('-')[0]:6} IC={r[0]:+.4f} (n{r[1]}, {r[2]:.0f}d)" if r else f"  {s.split('-')[0]:6} 数据不足")
    out.append("\n留出名字(从未训练, 真泛化):")
    ics = []
    for s in [x for x in HELDOUT if x not in train_syms]:
        r = _ic(s)
        if r:
            ics.append(r[0]); out.append(f"  {s.split('-')[0]:6} IC={r[0]:+.4f} (n{r[1]}, {r[2]:.0f}d)")
        else:
            out.append(f"  {s.split('-')[0]:6} 数据不足, 跳过")
    if ics:
        pos = sum(1 for x in ics if x > 0)
        med_ic, mean_ic = float(np.median(ics)), float(np.mean(ics))
        out.append(f"\n留出汇总: {len(ics)}只, 正IC {pos}只, 中位IC={med_ic:+.4f}, 均值={mean_ic:+.4f}")
        if pos >= 0.7 * len(ics) and med_ic > 0.02:
            out.append("=> 倾向【结构性通性】: 多数未见名字也正IC, 不只是这4只的脾气。")
        elif pos <= 0.5 * len(ics) or med_ic < 0.005:
            out.append("=> 倾向【过拟合/弱泛化】: 留出名字IC普遍~0或负, 效应不通用。")
        else:
            out.append("=> 混合/边界: 部分泛化, 谨慎对待。")
    out.append("=" * 80)
    rep = "\n".join(out)
    (OUT / "reports").mkdir(parents=True, exist_ok=True)
    (OUT / "reports" / "stock_generalize.txt").write_text(rep, encoding="utf-8")
    print("\n" + rep)


POOLS = {
    "半导体OOS": ["AMD-USDT-SWAP", "MRVL-USDT-SWAP", "INTC-USDT-SWAP", "AVGO-USDT-SWAP"],
    "指数OOS": ["SPY-USDT-SWAP", "QQQ-USDT-SWAP"],
    "巨头OOS": ["AAPL-USDT-SWAP", "MSFT-USDT-SWAP", "GOOGL-USDT-SWAP", "AMZN-USDT-SWAP", "META-USDT-SWAP", "NFLX-USDT-SWAP"],
    "高波动OOS": ["COIN-USDT-SWAP", "MSTR-USDT-SWAP", "PLTR-USDT-SWAP"],
    "训练池(参考)": ["MU-USDT-SWAP", "SOXL-USDT-SWAP", "NVDA-USDT-SWAP", "TSLA-USDT-SWAP"],
}


def costgate() -> None:
    """按板块池化跑【净edge成本闸门】(不只IC), 用冻结的 ST720 模型。跨标的样本外(同期), 纯历史不交易。
    回答: 哪个板块的信号真过成本墙(taker+x1.5应力+t>2), 而不只是方向对。"""
    d = FROZEN / CANDIDATE["code"]
    if not (d / "meta.json").exists():
        log("尚未冻结 — 先 --mode freeze"); return
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    H, bar_ms = meta["H"], meta["bar_ms"]
    blob = pickle.load(open(d / "model.pkl", "rb"))
    model, kind, scaler = blob["model"], blob["kind"], blob["scaler"]
    sel = json.loads((d / "feature_list.json").read_text(encoding="utf-8"))
    med = pd.Series(json.loads((d / "median.json").read_text(encoding="utf-8")))
    costs = pmw.WorkflowCosts()

    def _score(panel):
        for c in sel:
            if c not in panel.columns:
                panel[c] = np.nan
        cl = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=["fwd", "y"])
        if len(cl) < 500:
            return None
        X = cl[sel].fillna(med).values
        Xf = scaler.transform(X) if scaler is not None else X
        oos = cl[["ts", "inst", "fwd", "y"]].copy()
        oos["score"] = pmw._score_from_model(kind, model, Xf)
        return oos

    # bucket 阈值用冻结模型在训练池上的 |train score|(部署一致)
    tp = _score(_pooled_panel(meta["symbols"], FREEZE_DAYS, H, bar_ms, update=True))
    train_abs = np.abs(tp["score"].values) if tp is not None else np.array([])

    def _pool_gate(syms):
        oos = _score(_pooled_panel(syms, GEN_DAYS, H, bar_ms, update=True))
        if oos is None or not len(train_abs):
            return None
        mm = pmw.evaluate_scores(oos, H, bar_ms, train_abs, costs=costs, mode="single_asset")
        if mm.get("skip"):
            return {"skip": True}
        cells = [(fr, c) for fr, c in mm.get("curve", []) if c is not None]
        pass_cells = [(fr, c) for fr, c in cells if pmw._cell_pass(c, costs, min_ts=50)]
        stress = [(fr, c) for fr, c in pass_cells if pmw._stress_pass(c, costs)]
        gate = mm.get("primary_ic", 0.0) > 0.01 and len(pass_cells) >= 2 and len(stress) >= 1
        bc = max(cells, key=lambda fc: (fc[1].get("net_10", -1e9), fc[1].get("t_10", -1e9)),
                 default=(None, None))[1] or {}
        return {"gate": gate, "ic": mm.get("primary_ic"), "auc": mm.get("auc"),
                "net10": bc.get("net_10"), "t10": bc.get("t_10"), "net15": bc.get("net_15"),
                "n_ts": bc.get("n_ts"), "n_pass": len(pass_cells), "n_stress": len(stress)}

    out = ["=" * 92, "ST720 按板块 净edge成本闸门 (冻结模型; 跨标的OOS同期; 纯历史不交易)", "=" * 92,
           "门控: ic>0.01 且 ≥2桶过 maker+taker净>0(t>2) 且 ≥1桶过15bps应力。看哪个板块真过成本墙。", ""]
    for name, syms in POOLS.items():
        r = _pool_gate(syms)
        if r is None:
            out.append(f"{name:14} 数据不足/跳过"); continue
        if r.get("skip"):
            out.append(f"{name:14} 信号过少, evaluate skip"); continue
        out.append(f"{name:14} gate={str(r['gate']):5} ic={r['ic']:+.4f} auc={r['auc']:.3f} "
                   f"net10={r['net10']}bps(t{r['t10']}) net15={r['net15']} n_ts={r['n_ts']} "
                   f"(pass桶{r['n_pass']}/应力{r['n_stress']})")
    out.append("=" * 92)
    rep = "\n".join(out)
    (OUT / "reports").mkdir(parents=True, exist_ok=True)
    (OUT / "reports" / "stock_costgate.txt").write_text(rep, encoding="utf-8")
    print("\n" + rep)


def selftest() -> None:
    assert forward_verdict(5.0, 2.5, 6.0, 60, 1, 1, 2.0) == "PASS"
    assert forward_verdict(5.0, 2.5, 6.0, 40, 1, 1, 2.0) == "PENDING"     # n<50
    assert forward_verdict(5.0, 1.8, 6.0, 60, 1, 1, 2.0) == "PENDING"     # t<2.13
    assert forward_verdict(5.0, 2.5, -1.0, 60, 1, 1, 2.0) == "KILL"       # net10<=0
    assert forward_verdict(5.0, 2.5, 6.0, 60, -1, 1, 2.0) == "KILL"       # ic 反向
    assert forward_verdict(5.0, 2.5, 6.0, 20, 1, 1, 9.0) == "KILL"        # 8wk 且 n<30
    log("selftest OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="US-stock 日历候选 forward 实测。")
    ap.add_argument("--mode", choices=["freeze", "evaluate", "snapshot", "selftest", "generalize", "costgate"],
                    required=True)
    args = ap.parse_args()
    {"freeze": freeze, "evaluate": evaluate, "snapshot": snapshot, "selftest": selftest,
     "generalize": generalize, "costgate": costgate}[args.mode]()


if __name__ == "__main__":
    main()
