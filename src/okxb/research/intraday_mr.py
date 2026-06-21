"""预登记的 15/30 分钟日内均值回归研究候选 (intraday mean-reversion, IMR)。

定位: 纯【前向验证研究】——只采集、不实盘; 过【全部闸门】才考虑 demo (大概率不会)。
诚实预期: 在 15m/30m 上扣 15bps 往返成本去 fade, 净收益多半为负 -> 大概率 KILL/PENDING。
这正是要测的: 用前向数据如实证伪, 而不是样本内调参调出"看着能赚"。

预登记(冻结, 看到前向数据后【不得】修改):
  universe = BTC/ETH/SOL-USDT-SWAP; 周期 = 15m 与 30m; 每个 (标的×周期) 为一个候选 (共6, 族级=6)。
  特征   = 最近一根 bar 的收益相对【过去 96 根】的 z 分。
  规则   = |z|>1 时【反向 fade】: z>1 做空、z<-1 做多 (无自由参数: 96/1.0/1 均为预先固定的常规值)。
  持有   = 1 根 bar; 成本 = 15bps(应力)/10bps(温和); PASS 需 >=100 独立前向时间戳 + 全部闸门。

纯标准库 + 项目内 forward_integrity / labeling 原语; 本模块【无网络】(取数在 runner 脚本)。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Optional

from .forward_integrity import bonferroni_t
from .labeling import deflated_sharpe, profit_factor

UNIVERSE = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"]
HORIZONS = {"15m": 15, "30m": 30}     # bar 周期(分钟); 持有 = 1 根 bar
Z_WINDOW = 96                         # 计算 z 分的回溯 bar 数 (96*15m=24h, 96*30m=48h)
Z_ENTER = 1.0                         # |z|>1 -> fade
HOLD_BARS = 1
COST_BPS_STRESS = 15.0
COST_BPS_MILD = 10.0
MIN_FWD_TS = 100                      # PASS 需 >=100 独立前向时间戳
FAMILY_TRIALS = len(UNIVERSE) * len(HORIZONS)   # 6 个并行候选 (族级多重检验)


def code_of(inst: str, horizon: str) -> str:
    return f"IMR_{inst.split('-')[0]}_{horizon}"


def normalize_candles(raw: list) -> list:
    """OKX candles (新->旧, 每条 [ts,o,h,l,c,...,confirm]) -> [(ts:int, close:float)] 旧->新, 仅【已确认】。
    丢弃未确认的当前 bar (confirm!='1'), 防 look-ahead。"""
    out = []
    for c in raw or []:
        try:
            if len(c) >= 9 and str(c[8]) != "1":
                continue
            out.append((int(c[0]), float(c[4])))
        except (ValueError, TypeError, IndexError):
            continue
    out.sort(key=lambda x: x[0])
    return out


def _zscore(returns: list, window: int) -> Optional[float]:
    if len(returns) < window:
        return None
    w = returns[-window:]
    mu = sum(w) / len(w)
    var = sum((x - mu) ** 2 for x in w) / (len(w) - 1)
    sd = math.sqrt(var)
    return (w[-1] - mu) / sd if sd > 1e-12 else None


def signal_direction(z: Optional[float], enter: float = Z_ENTER) -> int:
    """均值回归 fade: 涨过头(z>enter)->做空(-1); 跌过头(z<-enter)->做多(+1); 否则不交易(0)。"""
    if z is None:
        return 0
    if z > enter:
        return -1
    if z < -enter:
        return 1
    return 0


@dataclass(slots=True)
class Label:
    bar_ts: int
    direction: int
    net15_bps: float
    net10_bps: float
    z: float
    entry_px: float
    exit_px: float


def iter_labels(candles: list, *, window: int = Z_WINDOW, enter: float = Z_ENTER,
                hold: int = HOLD_BARS) -> Iterator[Label]:
    """对【已实现】的决策 bar 逐一产出 look-ahead-safe 标签:
    z 只用到 <=i 的收益, 前向只用 i->i+hold; 最后 hold 根 bar (前向未实现) 不产出。"""
    closes = [c[1] for c in candles]
    ts = [c[0] for c in candles]
    if len(closes) < 2:
        return
    rets = [0.0] + [closes[j] / closes[j - 1] - 1.0 for j in range(1, len(closes))]
    n = len(closes)
    for i in range(window, n - hold):
        z = _zscore(rets[: i + 1], window)
        d = signal_direction(z, enter)
        if d == 0 or z is None:
            continue
        entry, exit_ = closes[i], closes[i + hold]
        if entry <= 0:
            continue
        fwd = exit_ / entry - 1.0
        yield Label(bar_ts=ts[i], direction=d,
                    net15_bps=d * fwd * 1e4 - COST_BPS_STRESS,
                    net10_bps=d * fwd * 1e4 - COST_BPS_MILD,
                    z=z, entry_px=entry, exit_px=exit_)


@dataclass(slots=True)
class Verdict:
    verdict: str       # PASS / PENDING / KILL
    reason: str
    metrics: dict


def evaluate_candidate(net15: list, net10: list, *, train_net15: Optional[float],
                       train_ic_sign: int, pbo: Optional[float] = None,
                       family_trials: int = FAMILY_TRIALS,
                       already_dead: bool = False) -> Verdict:
    """按 RESEARCH_INTEGRITY §1.6 全部闸门判决 (sticky KILL)。"""
    n = len(net15)
    m15 = (sum(net15) / n) if n else 0.0
    m10 = (sum(net10) / n) if n else 0.0
    metrics = {"n_ts": n, "net15_mean_bps": round(m15, 3), "net10_mean_bps": round(m10, 3)}
    if already_dead:
        return Verdict("KILL", "already_dead_sticky", metrics)

    t = None
    if n >= 2:
        var = sum((x - m15) ** 2 for x in net15) / (n - 1)
        sd = math.sqrt(var)
        t = (m15 / (sd / math.sqrt(n))) if sd > 1e-12 else None
    thr = bonferroni_t(family_trials)
    dsr = deflated_sharpe(net15, family_trials) if n >= 10 else None
    pf = profit_factor(net15) if n else 0.0
    fwd_sign = 1 if m15 > 0 else (-1 if m15 < 0 else 0)
    metrics.update({
        "t": (round(t, 3) if t is not None else None), "t_threshold": round(thr, 3),
        "dsr": (round(dsr, 4) if dsr is not None else None),
        "pbo": (round(pbo, 4) if pbo is not None else None),
        "pf": (round(pf, 3) if pf != float("inf") else None),
        "fwd_ic_sign": fwd_sign, "train_ic_sign": train_ic_sign,
    })

    # sticky KILL: 连温和成本(10bps)都为负且样本够 -> 判死, 永不复活
    if n >= 30 and m10 <= 0.0:
        return Verdict("KILL", "net10<=0 after >=30 fwd ts (无edge, even at 10bps)", metrics)
    if n < MIN_FWD_TS:
        return Verdict("PENDING", f"insufficient forward ts ({n}/{MIN_FWD_TS})", metrics)

    gates = []
    if m15 <= 0:
        gates.append("net15<=0")
    if t is None or t < thr:
        gates.append("t<bonferroni")
    if dsr is None or dsr < 0.95:
        gates.append("dsr<0.95")
    if pbo is not None and pbo > 0.20:
        gates.append("pbo>0.20")
    if train_ic_sign and fwd_sign != train_ic_sign:
        gates.append("ic_sign_flip")
    if train_net15 is not None and train_net15 > 0 and m15 < 0.5 * train_net15:
        gates.append("decayed")
    if pf < 1.2:
        gates.append("pf<1.2")
    if gates:
        return Verdict("PENDING", "failed gates: " + ",".join(gates), metrics)
    return Verdict("PASS", "all gates passed (forward-validated)", metrics)
