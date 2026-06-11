"""仓位计算 (RESEARCH_BRIEF §6 / 用户方案 §7.5)。

核心: 真实风险 = 名义价值 × (止损% + 费用% + 滑点%); 杠杆只影响保证金占用, 不放大风险。
所有 *_pct 为小数分数 (0.002 = 0.2%)。
"""
from __future__ import annotations

from decimal import ROUND_DOWN, Decimal


def max_notional_by_risk(risk_usdt: float, sl_pct: float, cost_pct: float) -> float:
    """单笔最大名义价值 = 单笔风险USDT / (止损% + 成本%)。"""
    denom = sl_pct + cost_pct
    return risk_usdt / denom if denom > 0 else 0.0


def final_notional(*, risk_usdt: float, sl_pct: float, cost_pct: float,
                   equity_usdt: float, single_symbol_cap_usdt: float,
                   depth_notional: float, depth_use_frac: float = 0.03,
                   margin_avail_usdt: float | None = None,
                   leverage: float = 1.0, margin_use_frac: float = 0.70) -> float:
    """取各约束最小值: 风险上限 / 单标的上限 / 盘口深度占比 / 可用保证金×杠杆。"""
    candidates = [
        max_notional_by_risk(risk_usdt, sl_pct, cost_pct),
        single_symbol_cap_usdt,
        depth_notional * depth_use_frac,
    ]
    if margin_avail_usdt is not None:
        candidates.append(margin_avail_usdt * leverage * margin_use_frac)
    return max(0.0, min(c for c in candidates if c is not None))


def notional_to_contracts(notional_usdt: float, price: float, ct_val: float,
                          lot_sz: str, min_sz: str) -> Decimal:
    """名义价值 -> 合约张数 (按 lotSz 向下取整, 须 >= minSz)。
    USDT 本位 SWAP: 每张名义 = ct_val(基础币数量) × price。"""
    if price <= 0 or ct_val <= 0:
        return Decimal("0")
    raw = Decimal(str(notional_usdt)) / (Decimal(str(ct_val)) * Decimal(str(price)))
    step = Decimal(lot_sz)
    contracts = (raw / step).to_integral_value(rounding=ROUND_DOWN) * step
    if contracts < Decimal(min_sz):
        return Decimal("0")
    return contracts


def round_price(price: float, tick_sz: str, side_up: bool = False) -> Decimal:
    """按 tickSz 取整价格。side_up=True 向上取整 (卖/挂高), 否则向下。"""
    from decimal import ROUND_CEILING, ROUND_FLOOR
    step = Decimal(tick_sz)
    p = Decimal(str(price))
    rounding = ROUND_CEILING if side_up else ROUND_FLOOR
    return (p / step).to_integral_value(rounding=rounding) * step
