"""P1 / C-6: position sizing math — the conversion from risk budget to notional/contracts.
A wrong result here is a direct money error (oversized position or rejected order)."""
from decimal import Decimal

from okxb.risk import sizing


def test_each_constraint_can_bind():
    base = dict(sl_pct=0.01, cost_pct=0.001, equity_usdt=1000.0, leverage=10.0)
    # risk budget binds: 2 / (0.01+0.001) ≈ 181.8
    n = sizing.final_notional(risk_usdt=2.0, single_symbol_cap_usdt=1e9,
                              depth_notional=1e12, margin_avail_usdt=1e9, **base)
    assert abs(n - 2.0 / 0.011) < 1.0
    # single-symbol cap binds
    n = sizing.final_notional(risk_usdt=1e9, single_symbol_cap_usdt=500.0,
                              depth_notional=1e12, margin_avail_usdt=1e9, **base)
    assert n == 500.0
    # order-book depth binds (depth * 0.03)
    n = sizing.final_notional(risk_usdt=1e9, single_symbol_cap_usdt=1e9,
                              depth_notional=1000.0, margin_avail_usdt=1e9, **base)
    assert abs(n - 30.0) < 1e-9
    # available margin binds (margin * leverage * 0.70)
    n = sizing.final_notional(risk_usdt=1e9, single_symbol_cap_usdt=1e9,
                              depth_notional=1e12, margin_avail_usdt=100.0,
                              sl_pct=0.01, cost_pct=0.001, equity_usdt=1000.0, leverage=2.0)
    assert abs(n - 140.0) < 1e-9


def test_tiny_sl_pct_inflates_notional_and_zero_denom_guarded():
    # headline C-6 risk: a degenerate tiny sl_pct silently produces a huge notional
    assert sizing.max_notional_by_risk(2.0, 0.0001, 0.0) == 2.0 / 0.0001     # = 20000
    assert sizing.max_notional_by_risk(2.0, 0.0, 0.0) == 0.0                 # zero denom -> 0, not crash


def test_final_notional_never_negative():
    n = sizing.final_notional(risk_usdt=2.0, sl_pct=0.01, cost_pct=0.001, equity_usdt=1000.0,
                              single_symbol_cap_usdt=-100.0, depth_notional=1e12)
    assert n == 0.0                                                          # negative cap clamps to 0


def test_notional_to_contracts_rounding_and_guards():
    # price 100, ctVal 0.01 -> 1 USDT/contract; 50 USDT -> 50 contracts (lotSz 1)
    assert sizing.notional_to_contracts(50.0, 100.0, 0.01, "1", "1") == Decimal("50")
    # below minSz -> 0
    assert sizing.notional_to_contracts(0.5, 100.0, 0.01, "1", "1") == Decimal("0")
    # zero/neg price or ctVal -> 0 (no division blow-up)
    assert sizing.notional_to_contracts(50.0, 0.0, 0.01, "1", "1") == Decimal("0")
    assert sizing.notional_to_contracts(50.0, 100.0, 0.0, "1", "1") == Decimal("0")
    # lotSz floor: 47 / (1*1)=47 -> floor to multiple of 5 -> 45
    assert sizing.notional_to_contracts(47.0, 1.0, 1.0, "5", "1") == Decimal("45")


def test_round_price_direction():
    assert sizing.round_price(100.27, "0.1", side_up=False) == Decimal("100.2")   # floor (buy/passive)
    assert sizing.round_price(100.21, "0.1", side_up=True) == Decimal("100.3")    # ceil (sell/passive)
