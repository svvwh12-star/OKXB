"""Triple-barrier 标注 + 前瞻收益 + 回测指标 (RESEARCH_BRIEF §7)。

triple_barrier: 给定 mid 序列与入场点, 判定先触 TP(+1)/先触 SL(-1)/超时(0) 及实现收益。
训练目标不是"涨跌", 而是"在当前成本与止盈止损结构下这笔是否值得做"。
"""
from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from itertools import combinations

_EULER = 0.5772156649015329


@dataclass
class BarrierResult:
    label: int            # +1 TP / -1 SL / 0 超时
    exit_ts: int
    ret_pct: float        # 有方向的实现收益 (小数分数)


def _mid_at_or_after(ts_list: list[int], mids: list[float], target_ts: int) -> float | None:
    i = bisect_left(ts_list, target_ts)
    return mids[i] if i < len(ts_list) else None


def triple_barrier(ts_list: list[int], mids: list[float], entry_idx: int,
                   side: int, tp_pct: float, sl_pct: float,
                   max_hold_ms: int) -> BarrierResult:
    """side: +1 多 / -1 空。ts_list 升序, 与 mids 等长。"""
    entry = mids[entry_idx]
    entry_ts = ts_list[entry_idx]
    tp_px = entry * (1 + side * tp_pct)
    sl_px = entry * (1 - side * sl_pct)
    for j in range(entry_idx + 1, len(ts_list)):
        if ts_list[j] - entry_ts > max_hold_ms:
            ret = side * (mids[j] / entry - 1)
            return BarrierResult(0, ts_list[j], ret)
        px = mids[j]
        if side > 0:
            if px >= tp_px:
                return BarrierResult(1, ts_list[j], tp_pct)
            if px <= sl_px:
                return BarrierResult(-1, ts_list[j], -sl_pct)
        else:
            if px <= tp_px:
                return BarrierResult(1, ts_list[j], tp_pct)
            if px >= sl_px:
                return BarrierResult(-1, ts_list[j], -sl_pct)
    # 数据末尾未触: 用最后价
    last = mids[-1]
    return BarrierResult(0, ts_list[-1], side * (last / entry - 1))


def forward_return(ts_list: list[int], mids: list[float], entry_ts: int,
                   entry_mid: float, horizon_ms: int, side: int) -> float | None:
    """入场后 horizon 的有方向收益 (小数分数)。"""
    fut = _mid_at_or_after(ts_list, mids, entry_ts + horizon_ms)
    if fut is None or entry_mid <= 0:
        return None
    return side * (fut / entry_mid - 1)


# ----------------------------- 回测指标 -----------------------------

def profit_factor(returns: list[float]) -> float:
    gains = sum(r for r in returns if r > 0)
    losses = -sum(r for r in returns if r < 0)
    return gains / losses if losses > 0 else float("inf")


def win_rate(returns: list[float]) -> float:
    if not returns:
        return 0.0
    return sum(1 for r in returns if r > 0) / len(returns)


def avg_rr(returns: list[float]) -> float:
    wins = [r for r in returns if r > 0]
    losses = [-r for r in returns if r < 0]
    aw = sum(wins) / len(wins) if wins else 0.0
    al = sum(losses) / len(losses) if losses else 0.0
    return aw / al if al > 0 else float("inf")


def sharpe(returns: list[float]) -> float:
    """每笔收益的简单 Sharpe (非年化)。样本少时仅供参考。"""
    if len(returns) < 2:
        return 0.0
    mu = sum(returns) / len(returns)
    var = sum((r - mu) ** 2 for r in returns) / (len(returns) - 1)
    sd = math.sqrt(var)
    return mu / sd if sd > 1e-12 else 0.0


def max_drawdown(returns: list[float]) -> float:
    """按顺序累加收益的最大回撤 (小数分数)。"""
    peak = cum = 0.0
    mdd = 0.0
    for r in returns:
        cum += r
        peak = max(peak, cum)
        mdd = min(mdd, cum - peak)
    return mdd


# ----------------------------- 机构级稳健性统计 -----------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """标准正态分位 (Acklam 有理逼近, 绝对误差~1e-9)。"""
    if p <= 0.0:
        return -1e9
    if p >= 1.0:
        return 1e9
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _moments(x: list[float]):
    """(mean, sd, skew g1, kurtosis g2非超额)。sd≈0 时返回 (mu,0,0,3)。"""
    n = len(x)
    if n < 2:
        return (x[0] if x else 0.0), 0.0, 0.0, 3.0
    mu = sum(x) / n
    m2 = sum((v - mu) ** 2 for v in x) / n
    sd = math.sqrt(m2)
    if sd < 1e-15:
        return mu, 0.0, 0.0, 3.0
    m3 = sum((v - mu) ** 3 for v in x) / n
    m4 = sum((v - mu) ** 4 for v in x) / n
    return mu, sd, m3 / sd ** 3, m4 / sd ** 4


def deflated_sharpe(returns: list[float], n_trials: int, sr_benchmark: float = 0.0):
    """紧缩夏普概率 (Bailey & Lopez de Prado): 在试过 n_trials 个配置后, 该夏普是真实而非运气的概率。
    返回 P(真实SR>基准), >=0.95 视为通过; 样本太少返回 None。"""
    T = len(returns)
    if T < 10 or n_trials < 1:
        return None
    mu, sd, g1, g2 = _moments(returns)
    if sd < 1e-15:
        return None
    sr = mu / sd                                  # 每笔夏普 (与 sharpe() 一致)
    n = max(2, int(n_trials))
    sd_sr0 = math.sqrt(1.0 / T)
    e_max = sd_sr0 * ((1 - _EULER) * _norm_ppf(1 - 1.0 / n)
                      + _EULER * _norm_ppf(1 - 1.0 / (n * math.e)))
    sr_star = sr_benchmark + e_max
    denom = 1.0 - g1 * sr + ((g2 - 1.0) / 4.0) * sr * sr
    if denom <= 1e-12:
        return None
    z = (sr - sr_star) * math.sqrt(T - 1) / math.sqrt(denom)
    return _norm_cdf(z)


def _perf(nets: list[float]) -> float:
    """简单表现度量 (每笔均值/标准差), 供 PBO 排名。"""
    if len(nets) < 2:
        return 0.0
    mu = sum(nets) / len(nets)
    var = sum((r - mu) ** 2 for r in nets) / (len(nets) - 1)
    sd = math.sqrt(var)
    return mu / sd if sd > 1e-12 else 0.0


def pbo_cscv(config_nets: dict, s_blocks: int = 8):
    """过拟合概率 (CSCV, Bailey et al.): 训练段最优配置在样本外的排名是否随机。
    config_nets: {cfg_id: 按时间序的每笔净收益}; 返回 (pbo, mean_logit) 或 None。"""
    ids = [k for k, v in config_nets.items() if v and len(v) >= s_blocks * 2]
    if len(ids) < 2:
        return None
    L = min(len(config_nets[k]) for k in ids)
    L -= L % s_blocks
    if L < s_blocks * 2:
        return None
    blocks = {k: [config_nets[k][:L][b * (L // s_blocks):(b + 1) * (L // s_blocks)]
                  for b in range(s_blocks)] for k in ids}
    half = s_blocks // 2
    logits = []
    for is_idx in combinations(range(s_blocks), half):
        oos_idx = [b for b in range(s_blocks) if b not in is_idx]
        is_perf, oos_perf = {}, {}
        for k in ids:
            is_perf[k] = _perf([x for b in is_idx for x in blocks[k][b]])
            oos_perf[k] = _perf([x for b in oos_idx for x in blocks[k][b]])
        best = max(ids, key=lambda k: is_perf[k])
        ranked = sorted(ids, key=lambda k: oos_perf[k])   # 升序
        rank = ranked.index(best)
        w = (rank + 1) / (len(ids) + 1)
        w = min(max(w, 1e-6), 1 - 1e-6)
        logits.append(math.log(w / (1 - w)))
    if not logits:
        return None
    pbo = sum(1 for x in logits if x <= 0) / len(logits)
    return pbo, sum(logits) / len(logits)
