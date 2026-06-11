"""阈值/出场策略网格校准器 (在录制的逐拍数据上做三重障碍事件回测)。

输入: recordings/calib_*.jsonl (由 TickRecorder 录制)。
做法 (Lopez de Prado triple-barrier 思想):
  1. 入场重放: 对每组 (入场分S, 确认C, 持续N), 在每个标的的逐拍序列上生成做多/做空入场事件
     (与实盘 HFM80 同构: 取较强方向, 达标且连续N拍才触发, 触发后消费)。
  2. 出场重放: 对每笔入场, 按 (持仓上限H, 止盈盈亏比tp_rr, 出场模式mode) 模拟先触者:
       止盈(tp_rr*SL) / 止损(SL) / 超时(H) / 反转(对向达标连续2拍) / 移动止盈(trailing)。
       SL 由入场时 ATR 推导 (与 scorer.sl_tp 同公式); 每笔净收益 = 实现收益 - 往返成本(可加保守系数)。
  3. 汇总: 笔数 / 胜率 / PF / 平均盈亏比 / 每笔Sharpe / 最大回撤 / 总收益, 并给出做多vs做空、
     各出场原因 的拆分。按"总收益最高"与"最稳(Sharpe最高)"分别选出推荐配置。

诚实声明: 任何回测都高估成交 (maker 假设按中价成交、无队列/无逆选)。把 cost_haircut_mult 调大、
或用 maker_fill=false (按吃价) 可更保守。结论是相对比较与方向校准, 不是实盘收益承诺。
"""
from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import labeling as lb

ARR_KEYS = ("t", "m", "L", "S", "fd", "td", "tr", "a", "sp", "c", "rv")

# 与实盘 composite/hfm80 去噪一致的默认参数 (单一真相; controller 从 config 覆盖)
DEFAULT_SM = {
    "hl_f": 1.0, "hl_t": 2.0, "hl_s": 0.5, "enter": 0.12, "exit": 0.04,
    "miss_grace": 1, "rv_lo": 5e-5, "rv_hi": 1.2e-3, "hv_scale": 0.6,
    "persist_bonus": 1, "dt": 0.5, "w_flow": 35.0, "w_trend": 15.0,
    "min_trad": 0.5,         # 可交易性独立门槛 (与实盘 signal.min_tradability 一致)
    "edge_k": 1.5, "edge_horizon_s": 30.0,   # 期望净edge: 期望幅度 = k×方向强度×持有期波动 (与实盘一致)
}


def _a(dt, hl):
    return 1.0 if hl <= 0 else 1.0 - 0.5 ** (dt / hl)


# ----------------------------- 数据加载 -----------------------------

def find_recordings(rec_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(rec_dir, "calib_*.jsonl")))


def load_calib(paths: list[str], max_rows: int = 1_500_000) -> tuple[dict, dict]:
    """读取 JSONL -> {inst: {key: [..]}} (按 t 升序)。返回 (by_inst, meta)。
    超过 max_rows 则按比例抽稀 (隔点取样), 保证响应。"""
    raw: dict[str, list[dict]] = {}
    n_total = 0
    for p in paths:
        try:
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        d = json.loads(line)
                    except ValueError:
                        continue
                    raw.setdefault(d["i"], []).append(d)
                    n_total += 1
        except OSError:
            continue
    stride = 1
    if n_total > max_rows:
        stride = math.ceil(n_total / max_rows)
    by_inst: dict[str, dict] = {}
    kept = 0
    for inst, rows in raw.items():
        rows.sort(key=lambda r: r["t"])
        if stride > 1:
            rows = rows[::stride]
        if len(rows) < 30:
            continue
        cols = {k: [] for k in ARR_KEYS}
        for r in rows:
            for k in ARR_KEYS:
                cols[k].append(r.get(k))
        by_inst[inst] = cols
        kept += len(rows)
    meta = {"n_rows": n_total, "n_kept": kept, "n_inst": len(by_inst),
            "stride": stride, "files": len(paths)}
    return by_inst, meta


# ----------------------------- 网格定义 -----------------------------

@dataclass
class Grid:
    S: list = field(default_factory=lambda: [60, 64, 68, 72, 76])   # 对称尺度(多+空=100,50中性): 入场门槛网格
    C: list = field(default_factory=lambda: [0.20, 0.30, 0.40])
    N: list = field(default_factory=lambda: [2, 3, 4])
    H: list = field(default_factory=lambda: [30, 60, 120])           # 持仓上限(秒)
    tp_rr: list = field(default_factory=lambda: [1.2, 1.6, 2.2])     # 止盈盈亏比
    modes: list = field(default_factory=lambda: ["barrier", "barrier+rev", "barrier+rev+trail"])


MODE_LABEL = {
    "barrier": "三重障碍(止盈/止损/超时)",
    "barrier+rev": "障碍+反转平仓",
    "barrier+trail": "障碍+移动止盈",
    "barrier+rev+trail": "障碍+反转+移动止盈",
}


# ----------------------------- 入场事件 -----------------------------

def _conf(fd: float, td: float, d: int) -> float:
    return 0.5 * max(0.0, d * (fd or 0.0)) + 0.5 * max(0.0, d * (td or 0.0))


def _sl_pct(atr: Optional[float], sp_bps: Optional[float]) -> float:
    atr = atr if atr is not None else 0.002
    spread = (sp_bps if sp_bps is not None else 5.0) / 1e4
    sl = max(1.2 * atr, 3.0 * spread, 0.0010)
    return min(sl, 0.012)


def gen_entries(s: dict, S_thr: float, C_thr: float, N: int,
                sm: Optional[dict] = None) -> list[tuple[int, int]]:
    """返回 [(idx, side)]。【关键】对录制的原始 fd/td/L/S 重放与实盘一致的去噪:
    EMA平滑 + 双阈值闩锁 + 波动自适应(死盘抑制/剧烈多平滑) + 单拍容错, 再套 N拍持续。
    这样回测测的就是实盘真正会做的那批入场 (否则校准跑偏)。"""
    sm = {**DEFAULT_SM, **(sm or {})}      # 合并: 调用方传部分键也不会缺键 (修复 KeyError)
    L, Sh, fd, td, tr, rv = s["L"], s["S"], s["fd"], s["td"], s["tr"], s["rv"]
    dt = sm["dt"]
    a_f, a_t, a_s = _a(dt, sm["hl_f"]), _a(dt, sm["hl_t"]), _a(dt, sm["hl_s"])
    wf, wt = sm["w_flow"], sm["w_trend"]
    wdir = (wf + wt) or 1.0
    ema_f = ema_t = ema_d = None
    latch = cur_side = run = misses = 0
    out: list[tuple[int, int]] = []
    for i in range(len(L)):
        fdi, tdi = (fd[i] or 0.0), (td[i] or 0.0)
        tri = tr[i] if (i < len(tr) and tr[i] is not None) else 1.0
        rvi = rv[i] if i < len(rv) else None
        scale = sm["hv_scale"] if (rvi is not None and rvi > sm["rv_hi"]) else 1.0
        ema_f = fdi if ema_f is None else ema_f + a_f * scale * (fdi - ema_f)
        ema_t = tdi if ema_t is None else ema_t + a_t * scale * (tdi - ema_t)
        dir_s = max(-1.0, min(1.0, (wf * ema_f + wt * ema_t) / wdir))
        ema_d = dir_s if ema_d is None else ema_d + a_s * (dir_s - ema_d)
        if latch == 0:
            latch = 1 if ema_d >= sm["enter"] else (-1 if ema_d <= -sm["enter"] else 0)
        elif latch > 0:
            latch = -1 if ema_d <= -sm["enter"] else (0 if ema_d < sm["exit"] else 1)
        else:
            latch = 1 if ema_d >= sm["enter"] else (0 if ema_d > -sm["exit"] else -1)
        if rvi is not None and rvi < sm["rv_lo"]:       # 死盘: 不入场
            cur_side = run = misses = 0
            continue
        eff_N = N + (sm["persist_bonus"] if (rvi is not None and rvi > sm["rv_hi"]) else 0)
        # 对称尺度分数 = 50×(1 + 顺向×方向) (与实盘一致); 可交易性独立门槛(不乘进分数)
        cand = 1 if ema_d >= 0 else -1
        score = 50.0 * (1.0 + cand * ema_d)
        conf = 0.5 * max(0.0, cand * ema_f) + 0.5 * max(0.0, cand * ema_t)
        ok = score >= S_thr and tri >= sm["min_trad"] and conf >= C_thr and latch == cand
        if ok:
            if cur_side == cand:
                run += 1
                misses = 0
            else:
                cur_side, run, misses = cand, 1, 0
        else:
            if cur_side != 0 and misses < sm["miss_grace"]:
                misses += 1
                continue
            cur_side = run = misses = 0
            continue
        if run >= eff_N:
            out.append((i, cand))
            run = misses = 0           # 消费触发
    return out


# ----------------------------- 出场模拟 -----------------------------

def sim_one(s: dict, idx: int, side: int, H_ms: int, tp_rr: float,
            mode: str, cost_mult: float, maker_fill: bool,
            rev_gap: float, S_thr: float, C_thr: float,
            sl_cost_mult: float = 2.5, min_edge_cost: float = 1.2,
            sm: Optional[dict] = None) -> tuple[float, str, int]:
    """从 idx 入场, 返回 (净收益小数, 出场原因, 出场时间ms)。
    SL/TP 含成本地板; 入场施加【期望净edge】门槛 (与实盘 scorer.expected_edge 一致, 取代旧 tp/cost 恒成立门)。"""
    sm = {**DEFAULT_SM, **(sm or {})}
    m, t, a, sp, c = s["m"], s["t"], s["a"], s["sp"], s["c"]
    entry = m[idx]
    if not entry or entry <= 0:
        return 0.0, "skip", t[idx]
    cost = (c[idx] or 0.0011) * cost_mult
    if not maker_fill:                        # 吃价入场: 额外付半价差(更保守)
        cost += (sp[idx] if sp[idx] is not None else 5.0) / 1e4 * 0.5
    # 与 scorer.sl_tp 同公式: 成本感知止损地板 + tp 双地板
    sl_pct = max(_sl_pct(a[idx], sp[idx]), sl_cost_mult * cost)
    tp_pct = max(tp_rr * sl_pct, 3.0 * cost)
    # 期望净edge门槛 (与实盘 expected_edge 一致): 期望幅度=k×方向强度×持有期波动; 净edge/cost >= 门槛
    dir_strength = abs((s["L"][idx] if s["L"][idx] is not None else 50.0) - 50.0) / 50.0
    rv_i = s["rv"][idx] if (idx < len(s["rv"]) and s["rv"][idx] is not None) else 0.0
    h_steps = max(1.0, sm["edge_horizon_s"] / sm["dt"])
    exp_move = min(sm["edge_k"] * dir_strength * (rv_i * math.sqrt(h_steps)), tp_pct)
    net_edge = exp_move - cost
    if cost <= 0 or (net_edge / cost) < min_edge_cost:
        return 0.0, "filtered", t[idx]        # 实盘也不会进 -> 不计入
    use_rev = "rev" in mode
    use_trail = "trail" in mode
    rev_thr = S_thr - rev_gap
    n = len(m)
    hwm = entry
    rev_run = 0
    entry_ts = t[idx]
    for j in range(idx + 1, n):
        px = m[j]
        if px is None or px <= 0:
            continue
        elapsed = t[j] - entry_ts
        # 1) 止盈/止损 (先触者)
        if side > 0:
            if px >= entry * (1 + tp_pct):
                return tp_pct - cost, "tp", t[j]
            if px <= entry * (1 - sl_pct):
                return -sl_pct - cost, "sl", t[j]
        else:
            if px <= entry * (1 - tp_pct):
                return tp_pct - cost, "tp", t[j]
            if px >= entry * (1 + sl_pct):
                return -sl_pct - cost, "sl", t[j]
        moved = side * (px / entry - 1.0)
        # 2) 移动止盈
        if use_trail and moved >= 0.0015:
            hwm = max(hwm, px) if side > 0 else min(hwm, px)
            trail_dist = max(0.5 * sl_pct, 0.0008) * entry
            breached = (px <= hwm - trail_dist) if side > 0 else (px >= hwm + trail_dist)
            if breached and moved >= cost:
                return moved - cost, "trail", t[j]
        # 3) 反转平仓 (对向达标连续2拍)
        if use_rev:
            opp = -side
            opp_score = (s["S"][j] if side > 0 else s["L"][j]) or 0.0
            opp_conf = _conf(s["fd"][j], s["td"][j], opp)
            if opp_score >= rev_thr and opp_conf >= C_thr:
                rev_run += 1
            else:
                rev_run = 0
            if rev_run >= 2 and sl_pct >= cost:
                return moved - cost, "rev", t[j]
        # 4) 超时
        if elapsed > H_ms:
            return moved - cost, "time", t[j]
    # 数据末尾未触
    last = m[-1]
    return side * (last / entry - 1.0) - cost, "eod", t[-1]


# ----------------------------- 汇总指标 -----------------------------

@dataclass
class Result:
    S: float; C: float; N: int; H: int; tp_rr: float; mode: str
    n: int; total: float; pf: float; win: float; rr: float
    sharpe: float; maxdd: float
    long_n: int; long_total: float; short_n: int; short_total: float
    reasons: dict
    nets: list = field(default_factory=list)   # 按时间序的每笔净收益 (供 DSR/PBO)
    dsr: Optional[float] = None                # 紧缩夏普概率 (样本外)

    @property
    def avg_bps(self) -> float:
        return (self.total / self.n * 1e4) if self.n else 0.0


def _cap(v: float) -> float:
    return 99.9 if v == float("inf") else round(v, 3)


def _aggregate(trades: list, combo: dict) -> Result:
    """trades: [(net, side, reason, exit_ts)]。maxDD 用全局按出场时间排序的真实权益曲线。"""
    chrono_trades = sorted(trades, key=lambda z: z[3])             # 真实时间序
    nets = [x[0] for x in chrono_trades]                           # 按时间序的每笔净收益
    longs = [x[0] for x in trades if x[1] > 0]
    shorts = [x[0] for x in trades if x[1] < 0]
    reasons: dict[str, int] = {}
    for x in trades:
        reasons[x[2]] = reasons.get(x[2], 0) + 1
    return Result(
        S=combo["S"], C=combo["C"], N=combo["N"], H=combo["H"],
        tp_rr=combo["tp_rr"], mode=combo["mode"],
        n=len(nets), total=sum(nets), pf=_cap(lb.profit_factor(nets)),
        win=lb.win_rate(nets), rr=_cap(lb.avg_rr(nets)),
        sharpe=lb.sharpe(nets), maxdd=lb.max_drawdown(nets),
        long_n=len(longs), long_total=sum(longs),
        short_n=len(shorts), short_total=sum(shorts), reasons=reasons, nets=nets,
    )


# ----------------------------- 主流程 -----------------------------

def _sim_combo(by_inst: dict, S, C, N, H, tp_rr, mode, *, cooldown_ms, cost_mult,
               maker_fill, rev_gap, sl_cost_mult, min_edge_cost, sm=None, events=None) -> Result:
    """对一个完整配置在给定数据集上模拟, 返回 Result。"""
    trades: list = []
    for inst, s in by_inst.items():
        evs = events[inst] if events is not None else gen_entries(s, S, C, N, sm=sm)
        if not evs:
            continue
        ts = s["t"]
        free_after = -1.0
        for (idx, side) in evs:
            if ts[idx] < free_after:
                continue
            net, reason, exit_t = sim_one(s, idx, side, H * 1000, tp_rr, mode,
                                          cost_mult, maker_fill, rev_gap, S, C,
                                          sl_cost_mult, min_edge_cost, sm=sm)
            if reason in ("skip", "filtered"):
                continue
            trades.append((net, side, reason, exit_t))
            free_after = exit_t + cooldown_ms
    return _aggregate(trades, {"S": S, "C": C, "N": N, "H": H, "tp_rr": tp_rr, "mode": mode})


def run_grid(by_inst: dict, grid: Grid, cooldown_s: float = 20.0,
             cost_mult: float = 1.0, maker_fill: bool = True, rev_gap: float = 22.0,
             sl_cost_mult: float = 2.5, min_edge_cost: float = 1.2, sm: Optional[dict] = None,
             progress: Optional[Callable[[int, int], None]] = None) -> list[Result]:
    cooldown_ms = cooldown_s * 1000.0
    entry_combos = [(S, C, N) for S in grid.S for C in grid.C for N in grid.N]
    total_steps = len(entry_combos)
    results: list[Result] = []
    for ci, (S, C, N) in enumerate(entry_combos):
        events = {inst: gen_entries(s, S, C, N, sm=sm) for inst, s in by_inst.items()}
        for H in grid.H:
            for tp_rr in grid.tp_rr:
                for mode in grid.modes:
                    r = _sim_combo(by_inst, S, C, N, H, tp_rr, mode, cooldown_ms=cooldown_ms,
                                   cost_mult=cost_mult, maker_fill=maker_fill, rev_gap=rev_gap,
                                   sl_cost_mult=sl_cost_mult, min_edge_cost=min_edge_cost,
                                   sm=sm, events=events)
                    if r.n > 0:
                        results.append(r)
        if progress:
            progress(ci + 1, total_steps)
    return results


def _split_series(by_inst: dict, train_frac: float = 0.7):
    """按时间(索引)切训练/测试; 太短则全归训练、无测试。"""
    train, test = {}, {}
    for inst, s in by_inst.items():
        n = len(s["t"])
        k = int(n * train_frac)
        if k < 20 or (n - k) < 20:
            train[inst] = s
            continue
        train[inst] = {key: s[key][:k] for key in ARR_KEYS}
        test[inst] = {key: s[key][k:] for key in ARR_KEYS}
    return train, test


def _slice_inst(by_inst: dict, lo_frac: float, hi_frac: float) -> dict:
    """按【每个标的各自】的比例切片(避免跨标的时间错位); 仅保留>=20行的标的。"""
    out = {}
    for inst, s in by_inst.items():
        n = len(s["t"])
        lo, hi = int(n * lo_frac), int(n * hi_frac)
        if hi - lo >= 20:
            out[inst] = {k: s[k][lo:hi] for k in ARR_KEYS}
    return out


def walk_forward(by_inst: dict, grid: Grid, min_trades: int, k_folds: int = 4,
                 anchored: bool = True, **kw) -> Optional[dict]:
    """滚动/锚定前进检验: 多段"训练→紧邻未见测试段"拼成真实样本外。
    kw 透传给 run_grid/_sim_combo (cooldown_s/cost_mult/maker_fill/rev_gap/sl_cost_mult/min_edge_cost/sm)。"""
    seg = 1.0 / (k_folds + 1)
    sim_kw = dict(cooldown_ms=kw.get("cooldown_s", 20.0) * 1000.0,
                  cost_mult=kw.get("cost_mult", 1.0), maker_fill=kw.get("maker_fill", True),
                  rev_gap=kw.get("rev_gap", 22.0), sl_cost_mult=kw.get("sl_cost_mult", 2.5),
                  min_edge_cost=kw.get("min_edge_cost", 1.2), sm=kw.get("sm"))
    grid_kw = {k: kw[k] for k in ("cooldown_s", "cost_mult", "maker_fill", "rev_gap",
                                  "sl_cost_mult", "min_edge_cost", "sm") if k in kw}
    folds, stitched = [], []
    for i in range(1, k_folds + 1):
        te_lo, te_hi = i * seg, (i + 1) * seg
        tr_lo = 0.0 if anchored else (i - 1) * seg
        tr = _slice_inst(by_inst, tr_lo, te_lo)
        te = _slice_inst(by_inst, te_lo, te_hi)
        if not tr or not te:
            continue
        tr_res = run_grid(tr, grid, **grid_kw)
        pick = pick_best(tr_res, max(5, min_trades // 2)).get("best_stable")
        if not pick:
            continue
        oos = _sim_combo(te, pick.S, pick.C, pick.N, pick.H, pick.tp_rr, pick.mode, **sim_kw)
        folds.append({"is_sharpe": pick.sharpe, "is_total": pick.total,
                      "oos_sharpe": oos.sharpe, "oos_total": oos.total, "oos_n": oos.n,
                      "cfg": (pick.S, pick.C, pick.N, pick.H, pick.tp_rr, pick.mode)})
        stitched.extend(oos.nets)
    if not folds:
        return None
    is_mu = sum(f["is_sharpe"] for f in folds) / len(folds)
    oos_mu = sum(f["oos_sharpe"] for f in folds) / len(folds)
    degr = (is_mu - oos_mu) / abs(is_mu) if abs(is_mu) > 1e-9 else None
    return {"folds": folds, "stitched": stitched,
            "stitched_total": sum(stitched), "stitched_n": len(stitched),
            "stitched_pf": _cap(lb.profit_factor(stitched)),
            "stitched_win": lb.win_rate(stitched), "stitched_maxdd": lb.max_drawdown(stitched),
            "degradation": degr, "dsr": lb.deflated_sharpe(stitched, len(folds))}


def run_calibration(by_inst: dict, grid: Grid, min_trades: int, cooldown_s: float = 20.0,
                    cost_mult: float = 1.0, maker_fill: bool = True, rev_gap: float = 22.0,
                    sl_cost_mult: float = 2.5, min_edge_cost: float = 1.2, sm: Optional[dict] = None,
                    progress: Optional[Callable[[int, int], None]] = None) -> dict:
    """走步/留出校准 + 机构级稳健性: 训练段网格选优 → 测试段样本外复评 + 紧缩夏普(DSR)
    + 滚动前进检验(walk-forward) + 过拟合概率(PBO)。"""
    train, test = _split_series(by_inst)
    train_results = run_grid(train, grid, cooldown_s, cost_mult, maker_fill, rev_gap,
                             sl_cost_mult, min_edge_cost, sm, progress)
    picks = pick_best(train_results, min_trades)
    n_trials = len(train_results)
    kw = dict(cooldown_ms=cooldown_s * 1000.0, cost_mult=cost_mult, maker_fill=maker_fill,
              rev_gap=rev_gap, sl_cost_mult=sl_cost_mult, min_edge_cost=min_edge_cost, sm=sm)
    oos = {}
    if test:
        for key in ("best_profit", "best_stable"):
            r = picks.get(key)
            if r:
                o = _sim_combo(test, r.S, r.C, r.N, r.H, r.tp_rr, r.mode, **kw)
                o.dsr = lb.deflated_sharpe(o.nets, n_trials)   # 样本外紧缩夏普
                oos[key] = o
    # 过拟合概率 PBO: 仅在收益前K个配置上 (控制 C(8,4)=70 splits 的成本)
    pbo = None
    topk = picks.get("top", [])[:12]
    config_nets = {f"c{i}": r.nets for i, r in enumerate(topk) if r.nets}
    if len(config_nets) >= 2:
        res = lb.pbo_cscv(config_nets, s_blocks=8)
        pbo = res[0] if res else None
    # 滚动前进检验 (数据够长才做)
    wf = None
    try:
        min_len = min((len(s["t"]) for s in by_inst.values()), default=0)
        if min_len >= 5 * (len(by_inst) > 0) and min_len >= 200:
            wf = walk_forward(by_inst, grid, min_trades, k_folds=4, anchored=True,
                              cooldown_s=cooldown_s, cost_mult=cost_mult, maker_fill=maker_fill,
                              rev_gap=rev_gap, sl_cost_mult=sl_cost_mult,
                              min_edge_cost=min_edge_cost, sm=sm)
    except Exception:
        wf = None
    return {"train_results": train_results, "picks": picks, "oos": oos,
            "has_test": bool(test), "n_trials": n_trials, "pbo": pbo, "walk_forward": wf}


def _stable_score(r: Result) -> float:
    """稳健度: 要求正收益 + 足够样本; 用每笔Sharpe * sqrt(n) 衡量一致性, 回撤惩罚。"""
    if r.total <= 0 or r.n < 5:
        return -1e9
    dd_pen = 1.0 + abs(r.maxdd) / (abs(r.total) + 1e-9)
    return r.sharpe * math.sqrt(r.n) / dd_pen


def pick_best(results: list[Result], min_trades: int) -> dict:
    valid = [r for r in results if r.n >= min_trades]
    profitable = [r for r in valid if r.total > 0]
    best_profit = max(profitable, key=lambda r: r.total, default=None)
    best_stable = max(profitable, key=_stable_score, default=None)
    top = sorted(valid, key=lambda r: r.total, reverse=True)[:12]
    return {"valid": valid, "profitable": profitable,
            "best_profit": best_profit, "best_stable": best_stable, "top": top}


# ----------------------------- 配置映射 / 报告 -----------------------------

def result_to_signal_cfg(r: Result) -> dict:
    """把一个回测配置映射为可应用的 config 键值 (点路径)。"""
    cfg = {
        "signal.min_composite_score": int(r.S),
        "signal.confirm_min": round(r.C, 2),
        "signal.persist_ticks": int(r.N),
        "signal.tp_rr": round(r.tp_rr, 2),
        # 不再强行改 min_edge_to_cost_ratio: 回测就用当前配置值跑, 保持训练/部署一致
        "execution.max_hold_seconds": int(r.H),
        # 出场模式开关: 用迟滞gap/移动止盈启动阈值 来开/关
        "signal.reversal_hyst_gap": 22 if "rev" in r.mode else 99,
        "signal.trail_arm_pct": 0.0015 if "trail" in r.mode else 9.0,
    }
    return cfg


def format_report_oos(meta: dict, calib: dict, min_trades: int) -> str:
    """走步/样本外校准报告 (诚实版): 训练段选优 + 样本外复评 + 过拟合警示。"""
    picks, oos = calib["picks"], calib["oos"]
    lines = ["=" * 64,
             "OKXB 策略校准报告 (训练段选优 → 样本外复评 · 三重障碍 · 与实盘门槛对齐)",
             "=" * 64,
             f"数据: {meta['n_rows']} 行 / {meta['n_inst']} 标的 / {meta['files']} 文件"
             + (f" (抽稀 1/{meta['stride']})" if meta.get('stride', 1) > 1 else ""),
             f"网格试验数 N={calib['n_trials']} —— 试得越多, 胜出者越可能只是运气, 务必看『样本外』。",
             ("样本外: 已切训练70%/测试30%, 下方给出样本外复评。"
              if calib["has_test"] else
              "⚠ 数据太短未能切出测试段 —— 当前仅样本内, 极易过拟合, 请多录一段再校准!"),
             ""]
    if not picks["valid"]:
        lines.append("⚠ 没有配置达到最小样本量(每配置≥%d笔)。让虚拟盘多跑一段再校准。" % min_trades)
        return "\n".join(lines)
    for tag, key in (("◆ 最稳健 (推荐)", "best_stable"), ("★ 收益最高", "best_profit")):
        r = picks.get(key)
        if not r:
            continue
        lines.append(f"{tag}  [训练段/样本内]:")
        lines.append(_fmt_result(r))
        o = oos.get(key)
        if o and o.n > 0:
            lines.append(f"   └─ 样本外复评: 笔数={o.n} 总收益={o.total*1e4:.1f}bps "
                         f"均值={o.avg_bps:.2f}bps 胜率={o.win:.1%} PF={o.pf:.2f} "
                         f"最大回撤={o.maxdd*1e4:.1f}bps")
            verdict = ("✅ 样本外仍为正且接近, 相对可信" if o.total > 0 and o.avg_bps >= 0.5 * r.avg_bps
                       else ("⚠ 样本外明显退化" if o.total > 0 else "❌ 样本外转负 → 大概率过拟合, 不要用"))
            lines.append(f"      判定: {verdict}")
            if o.dsr is not None:
                flag = "✅显著" if o.dsr >= 0.95 else "⚠ 多重检验下不显著(可能是网格运气)"
                lines.append(f"      紧缩夏普 DSR={o.dsr:.3f} (扣除试了{calib['n_trials']}组的运气) {flag}")
        elif calib["has_test"]:
            lines.append("   └─ 样本外: 该配置在测试段没有触发交易(样本不足, 结论存疑)")
        lines.append("")
    # 滚动前进检验
    wf = calib.get("walk_forward")
    if wf:
        lines.append("— 滚动前进检验 (多段训练→紧邻未见测试, 拼接=真实样本外) —")
        for i, f in enumerate(wf["folds"], 1):
            lines.append(f"  折{i}: IS Sharpe {f['is_sharpe']:.2f}/{f['is_total']*1e4:.0f}bps "
                         f"→ OOS Sharpe {f['oos_sharpe']:.2f}/{f['oos_total']*1e4:.0f}bps (n={f['oos_n']})")
        lines.append(f"  拼接OOS: 笔数={wf['stitched_n']} 总收益={wf['stitched_total']*1e4:.1f}bps "
                     f"PF={wf['stitched_pf']:.2f} 胜率={wf['stitched_win']:.1%} "
                     f"回撤={wf['stitched_maxdd']*1e4:.1f}bps")
        if wf.get("degradation") is not None:
            dg = wf["degradation"]
            lines.append(f"  IS→OOS Sharpe 衰减={dg:.0%} " + ("✅尚可" if dg <= 0.5 else "⚠ 衰减大(过拟合迹象)"))
        if wf.get("dsr") is not None:
            lines.append(f"  拼接OOS 紧缩夏普 DSR={wf['dsr']:.3f} " + ("✅显著" if wf['dsr'] >= 0.95 else "⚠ 不显著"))
        lines.append("")
    if calib.get("pbo") is not None:
        pbo = calib["pbo"]
        lines.append(f"过拟合概率 PBO={pbo:.2f} (前12配置) " +
                     ("✅低, 网格优胜较可信" if pbo <= 0.5 else "❌高, 网格优胜大概率过拟合"))
        lines.append("")
    lines.append("— 训练段收益前若干 (仅供观察方向, 非样本外) —")
    for r in picks["top"][:8]:
        lines.append(_fmt_result(r))
    lines.append("")
    lines.append("解读: 『做多/做空』各自笔数与净收益(做多/做空决策点); 『出场』各原因分布(哪种卖出最有效); "
                 "DSR≥0.95 且 PBO≤0.5 且 走步OOS为正 才算稳; 否则只是运气。")
    lines.append("⚠ 回测仍对成交乐观(maker按中价成交); 应用后务必再跑一段虚拟盘, "
                 "满足 go_live_gate(样本外≥300笔, PF≥1.25) 再考虑实盘。非收益承诺。")
    return "\n".join(lines)


def _fmt_result(r: Result) -> str:
    return (f"S={r.S:.0f} C={r.C:.2f} N={r.N} | 持仓<={r.H}s tp_rr={r.tp_rr} "
            f"{MODE_LABEL.get(r.mode, r.mode)}\n"
            f"    笔数={r.n} 总收益={r.total*1e4:.1f}bps 均值={r.avg_bps:.2f}bps "
            f"胜率={r.win:.1%} PF={r.pf:.2f} 盈亏比={r.rr:.2f} "
            f"Sharpe={r.sharpe:.3f} 最大回撤={r.maxdd*1e4:.1f}bps\n"
            f"    做多 {r.long_n}笔/{r.long_total*1e4:+.1f}bps · "
            f"做空 {r.short_n}笔/{r.short_total*1e4:+.1f}bps · "
            f"出场: {r.reasons}")


def format_report(by_inst: dict, meta: dict, results: list[Result],
                  picks: dict, min_trades: int) -> str:
    lines = []
    lines.append("=" * 64)
    lines.append("OKXB 策略校准报告 (录制逐拍数据 · 三重障碍事件回测)")
    lines.append("=" * 64)
    lines.append(f"数据: {meta['n_rows']} 行 / {meta['n_inst']} 标的 / {meta['files']} 个录制文件"
                 + (f" (抽稀 1/{meta['stride']})" if meta.get("stride", 1) > 1 else ""))
    lines.append(f"有效样本门槛: 每配置 >= {min_trades} 笔; 有效配置 {len(picks['valid'])} 个, "
                 f"其中盈利 {len(picks['profitable'])} 个。")
    lines.append("")
    if not picks["valid"]:
        lines.append("⚠ 没有任何配置达到最小样本量。说明录制时长不足或市场太淡。")
        lines.append("  建议: 让虚拟盘多跑一段(累积更多 calib_*.jsonl 行)再校准。")
        return "\n".join(lines)

    bp, bs = picks["best_profit"], picks["best_stable"]
    if bp:
        lines.append("★ 收益最高配置:")
        lines.append(_fmt_result(bp))
        lines.append("")
    if bs and bs is not bp:
        lines.append("◆ 最稳健配置 (推荐, Sharpe优先):")
        lines.append(_fmt_result(bs))
        lines.append("")
    elif bs:
        lines.append("(最稳健配置与收益最高配置相同)")
        lines.append("")

    lines.append("— 收益前若干配置 —")
    for r in picks["top"]:
        lines.append(_fmt_result(r))
    lines.append("")
    lines.append("解读: 『做多/做空』分别给出各自笔数与净收益, 即你问的"
                 "『做多/做空决策点是否成立』; 『出场』给出各原因(tp止盈/sl止损/time超时/"
                 "rev反转/trail移动止盈)的成交分布, 即"
                 "『哪种卖出策略最有效』。")
    lines.append("⚠ 回测对成交乐观(maker按中价成交), 实盘通常更差; 务必先用推荐配置再跑一段"
                 "虚拟盘复核, 满足 go_live_gate(样本外>=300笔, PF>=1.25) 再考虑实盘。")
    return "\n".join(lines)
