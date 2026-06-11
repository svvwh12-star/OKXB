"""信号有效性检验 (只读, 不改信号)。

用录制的逐拍数据, 实测综合分到底能不能预测【未来真实涨跌】:
  - 信息系数 IC: signed_score(=多分-空分) 与 未来收益 的相关性 (Pearson + Spearman秩相关)
  - 分桶曲线: 把 signed_score 按分位分成若干桶, 看每桶对应的【未来平均收益】是否随分数单调上升
  - 方向命中率: 强信号时 sign(分数)==sign(未来收益) 的比例
  - 单调性: 桶序号 与 桶均值收益 的秩相关 (~+1 = 分数越高未来越涨)

判定基准 (高频量化经验): |IC|>=0.05 较好; 0.02~0.05 弱但真实; <0.02 基本是噪声(别用它做方向)。
诚实: 这衡量"分数是否含未来信息", 是 edge 的必要非充分条件 (还要扣费、能成交)。
"""
from __future__ import annotations

from bisect import bisect_left
from typing import Optional

from . import calibrator as cal


def _pearson(xs, ys) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    d = (sxx * syy) ** 0.5
    return sxy / d if d > 1e-15 else 0.0


def _rank(vals) -> list:
    idx = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and vals[idx[j + 1]] == vals[idx[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[idx[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs, ys) -> float:
    return _pearson(_rank(xs), _rank(ys))


def _pairs(by_inst: dict, h_ms: int, lag_ms: int = 0):
    """返回 [(signed_score, forward_return)]。lag_ms>0 模拟成交延迟:
    信号在 i 观察到, 但入场价用 i 之后 lag_ms 的价, 收益再从入场点往前 h_ms (检验信号是否吃得到)。"""
    out = []
    for s in by_inst.values():
        t, m, L, Sh = s["t"], s["m"], s["L"], s["S"]
        n = len(t)
        for i in range(n):
            sg = (L[i] or 0.0) - (Sh[i] or 0.0)
            e = bisect_left(t, t[i] + lag_ms, i) if lag_ms else i   # 实际可成交点
            if e >= n:
                continue
            me = m[e]
            if not me or me <= 0:
                continue
            j = bisect_left(t, t[e] + h_ms, e + 1)
            if j >= n:
                continue
            mj = m[j]
            if not mj or mj <= 0:
                continue
            if t[j] - t[e] > h_ms + 2000:        # 跨录制会话断点的错配(实际跨度远超h)→丢弃
                continue
            out.append((sg, mj / me - 1.0))
    return out


def _regime_at(by_inst: dict, h_ms: int):
    """按入场时刻的波动(rv)分 低/中/高波 三档, 各自算 强信号方向幅度 vs 成本。
    回答关键问题: '只在高波动时做'能否让方向幅度盖过成本?"""
    rows = []   # (band, signed, fwd, cost) —— band 在【每个标的自身】rv 分布内分档(否则只是按标的大小分类)
    for s in by_inst.values():
        t, m, L, Sh = s["t"], s["m"], s["L"], s["S"]
        rv, c = s.get("rv", []), s.get("c", [])
        rvv = sorted(x for x in rv if x is not None)
        if len(rvv) < 30:
            continue
        lo, hi = rvv[len(rvv) // 3], rvv[2 * len(rvv) // 3]   # 该标的自身的波动三分位
        n = len(t)
        for i in range(n):
            rvi = rv[i] if i < len(rv) else None
            mi = m[i]
            if rvi is None or not mi or mi <= 0:
                continue
            j = bisect_left(t, t[i] + h_ms, i + 1)
            if j >= n:
                continue
            mj = m[j]
            if not mj or mj <= 0 or t[j] - t[i] > h_ms + 2000:   # 跨会话错配丢弃
                continue
            band = "低波" if rvi <= lo else ("高波" if rvi > hi else "中波")
            ci = c[i] if (i < len(c) and c[i] is not None) else 0.0011
            rows.append((band, (L[i] or 0.0) - (Sh[i] or 0.0), mj / mi - 1.0, ci))
    if len(rows) < 150:
        return None
    out = {}
    for band in ("低波", "中波", "高波"):
        part = [(sg, fr, cc) for b, sg, fr, cc in rows if b == band]
        if len(part) < 50:
            continue
        ic = _spearman([r[0] for r in part], [r[1] for r in part])
        strong = sorted(part, key=lambda r: abs(r[0]), reverse=True)[:max(1, len(part) // 2)]
        cap = (sum((1.0 if r[0] > 0 else -1.0) * r[1] for r in strong) / len(strong)) * 1e4
        cost = (sorted(r[2] for r in part)[len(part) // 2]) * 1e4
        out[band] = {"n": len(part), "ic": ic, "cap_bps": cap, "cost_bps": cost, "net_bps": cap - cost}
    return out


def validate(by_inst: dict, horizons_s=(5, 15, 60, 300, 900), n_buckets: int = 8) -> dict:
    # 代表性往返成本(bps): 所有行 c 的中位数 (用于"该周期强信号方向幅度是否盖过成本")
    all_c = sorted(c for s in by_inst.values() for c in s.get("c", []) if c is not None)
    cost_bps = (all_c[len(all_c) // 2] * 1e4) if all_c else 11.0
    res = {}
    for h in horizons_s:
        pairs = _pairs(by_inst, h * 1000)
        if len(pairs) < 50:
            res[h] = {"n": len(pairs)}
            continue
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        ic_p = _pearson(xs, ys)
        ic_s = _spearman(xs, ys)
        # 多档成交延迟后的 IC: 若随延迟明显衰减, 说明信号太快, 实盘吃不到 edge
        lags = {}
        for lag_s in (0.5, 1.0, 2.0):
            pl = _pairs(by_inst, h * 1000, lag_ms=int(lag_s * 1000))
            lags[lag_s] = _spearman([p[0] for p in pl], [p[1] for p in pl]) if len(pl) >= 50 else 0.0
        # 方向命中率 (取 |signed| 前50%; 排除"未动"的平拍, 否则把 fr==0 算作miss会严重低估真实命中)
        strong = sorted(pairs, key=lambda p: abs(p[0]), reverse=True)[:max(1, len(pairs) // 2)]
        nonflat = [(sg, fr) for sg, fr in strong if fr != 0.0]
        hits = sum(1 for sg, fr in nonflat if (sg > 0 and fr > 0) or (sg < 0 and fr < 0))
        dir_acc = hits / len(nonflat) if nonflat else 0.0
        # 分位分桶
        srt = sorted(pairs, key=lambda p: p[0])
        buckets = []
        per = len(srt) // n_buckets
        for b in range(n_buckets):
            chunk = srt[b * per:(b + 1) * per] if b < n_buckets - 1 else srt[b * per:]
            if not chunk:
                continue
            sg = sum(c[0] for c in chunk) / len(chunk)
            fr = sum(c[1] for c in chunk) / len(chunk)
            up = sum(1 for c in chunk if c[1] > 0) / len(chunk)
            buckets.append((round(sg, 1), fr * 1e4, up, len(chunk)))
        mono = _spearman(list(range(len(buckets))), [b[1] for b in buckets]) if len(buckets) >= 3 else 0.0
        # 强信号"方向幅度": 在预测方向上的平均位移(bps); 与往返成本比 -> 该周期是否赚得回手续费
        cap_bps = (sum((1.0 if sg > 0 else -1.0) * fr for sg, fr in strong) / len(strong)) * 1e4
        res[h] = {"n": len(pairs), "ic_p": ic_p, "ic_s": ic_s, "ic_lag": lags,
                  "dir_acc": dir_acc, "buckets": buckets, "mono": mono,
                  "cap_bps": cap_bps, "cost_bps": cost_bps, "net_bps": cap_bps - cost_bps}
        if h <= 60:                          # 只在有(弱)预测力的短周期看 regime 分档
            res[h]["regime"] = _regime_at(by_inst, h * 1000)
    return res


def _verdict(ic: float, mono: float) -> str:
    a = abs(ic)
    if a >= 0.05 and mono >= 0.6:
        return "✅ 有预测力 (可用)"
    if a >= 0.02 and mono >= 0.4:
        return "🟡 弱预测力 (真实但偏弱, 需严格扣费+择优)"
    return "❌ 基本是噪声 (别用它做方向, 先改信号)"


def format_report(meta: dict, res: dict) -> str:
    lines = ["=" * 60, "信号有效性检验 (综合分能否预测未来真实涨跌 · 只读不改信号)", "=" * 60,
             f"数据: {meta['n_rows']} 行 / {meta['n_inst']} 标的"
             + (f" (抽稀1/{meta['stride']})" if meta.get('stride', 1) > 1 else ""), ""]
    any_ok = False
    for h, d in res.items():
        if d.get("n", 0) < 50:
            lines.append(f"horizon {h}s: 样本太少({d.get('n', 0)}), 跳过。多录一段再测。")
            continue
        ic = d["ic_s"]   # 用秩相关 IC 作主判据(对异常值稳健)
        hlabel = f"{h}s" if h < 120 else f"{h // 60}min"
        lines.append(f"━ 未来 {hlabel} ━  样本={d['n']}  "
                     f"IC(Pearson)={d['ic_p']:+.3f}  IC(Spearman)={d['ic_s']:+.3f}  "
                     f"方向命中率={d['dir_acc']:.1%}  分桶单调性={d['mono']:+.2f}")
        lines.append(f"   判定: {_verdict(ic, d['mono'])}")
        lines.append(f"   ★周期经济性: 强信号方向幅度 {d['cap_bps']:+.1f}bps − 往返成本 {d['cost_bps']:.1f}bps "
                     f"= 净 {d['net_bps']:+.1f}bps  [{'够本✓' if d['net_bps'] > 0 else '不够本✗(此周期赚不回手续费)'}]")
        reg = d.get("regime")
        if reg:
            lines.append("   分波动regime(低/中/高波) 强信号方向幅度 vs 成本:")
            for name, r in reg.items():
                flag = "够本✓" if r["net_bps"] > 0 else "✗"
                lines.append(f"     {name}: 幅度{r['cap_bps']:+.1f} − 成本{r['cost_bps']:.1f} = 净{r['net_bps']:+.1f}bps  "
                             f"IC={r['ic']:+.3f} (n={r['n']}) {flag}")
        lg = d.get("ic_lag", {})
        if lg:
            lagstr = "  ".join(f"+{k}s={v:+.3f}" for k, v in sorted(lg.items()))
            icl = lg.get(1.0, 0.0)
            keep = "仍稳健" if (abs(ic) > 1e-9 and abs(icl) >= 0.6 * abs(ic)) else "随延迟明显衰减→信号偏快, 实盘可能吃不到"
            lines.append(f"   成交延迟后 IC(Spearman): {lagstr}  ({keep})")
        lines.append("   分数分桶(signed=多分−空分, 低→高) → 未来均值收益(扣费前; 实际扣费由『校准』计):")
        for sg, frbps, up, n in d["buckets"]:
            bar = "█" * min(20, int(abs(frbps)))
            lines.append(f"     signed{sg:+6.1f} → {frbps:+7.2f}bps  上涨率{up:5.1%}  (n={n}) {bar}")
        lines.append("")
        if abs(ic) >= 0.02 and d["mono"] >= 0.4:
            any_ok = True
    # 周期匹配诊断 (直接回答用户: 分数预测哪个时间窗? 该窗能否盖过成本? 与交易频率匹配吗?)
    valid_h = {h: d for h, d in res.items() if d.get("n", 0) >= 50}
    if valid_h:
        best_h = max(valid_h, key=lambda h: abs(valid_h[h]["ic_s"]))
        prof = sorted(h for h, d in valid_h.items() if d["net_bps"] > 0)
        lines.append("")
        lines.append("◆ 周期匹配诊断 (回答: 分数预测哪个时间窗? 该窗能否盖过成本? 与交易频率匹配吗?):")
        lines.append(f"  · 信号预测力最强的周期 ≈ {best_h}s (|IC|={abs(valid_h[best_h]['ic_s']):.3f})。")
        if prof:
            pl = ", ".join((f"{p}s" if p < 120 else f"{p // 60}min") for p in prof)
            lines.append(f"  · 强信号方向幅度【盖过】往返成本的周期: {pl} → 持仓时长 / edge_horizon 应对齐这些周期。")
        else:
            lines.append("  · ⚠ 没有任何周期 强信号方向幅度盖过往返成本 → 当前行情+成本下【无可做edge】。")
            lines.append("    出路: 只在更大波动时做(成本占比下降) / 降成本(maker出场) / 信号无效→Stage B重拟合权重。")
        lines.append("  · 对照实盘: edge_horizon=30s、持仓 min20/max180s。若最强周期 ≠ 这些, 即"
                     "『预测窗与交易频率不匹配』, 应把 edge_horizon 与持仓时长调到最强周期。")
    lines.append("")
    lines.append("解读: 分桶应『从上到下、收益由负到正单调上升』, 且 IC 离0越远越好。")
    lines.append("IC基准: |IC|≥0.05好, 0.02~0.05弱但真实, <0.02噪声。这是"
                 "『分数是否含未来信息』的直接证据 —— 比任何理论都实在。")
    if not any_ok:
        lines.append("⚠ 当前各周期 IC 都接近0 → 现版打分对你的数据几乎没有方向预测力, "
                     "建议先做核心信号重构(方向/质量分离+修OFI+micro-price)再校准/交易。")
    else:
        lines.append("✅ 至少一个周期有(弱)预测力 → 分数含真实信息; 可进一步做分数→概率校准与阈值校准。")
    lines.append("注: 这是必要非充分条件(还要扣费后为正、且能成交)。")
    return "\n".join(lines)
