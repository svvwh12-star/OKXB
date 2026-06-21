"""P1: RiskEngine.evaluate — the single money gate. Table-driven branch coverage.

Also pins the RiskDecision field name (`approved_notional_usdt`) — the contract that app.py
violated by reading `.approved_notional` (silently dropping every approved auto order).
"""
from decimal import Decimal

from okxb.config import Config
from okxb.core.enums import (EventAction, OrderType, PosSide, RiskAction, Side,
                             StrategyId, SystemState)
from okxb.core.models import MarketEvent, OrderIntent, Position, RiskDecision
from okxb.risk.engine import RiskEngine, set_stock_symbols

EVAL_KW = dict(sl_pct=0.01, cost_pct=0.0015, price=100.0, depth_notional=1e6,
               leverage=5.0, margin_avail_usdt=1000.0)


def _eng() -> RiskEngine:
    return RiskEngine(Config.load())


def _intent(reduce_only=False, side=Side.BUY, inst="BTC-USDT-SWAP") -> OrderIntent:
    return OrderIntent(inst_id=inst, side=side, pos_side=PosSide.NET,
                       order_type=OrderType.POST_ONLY, notional_usdt=Decimal("100"), px=None,
                       reduce_only=reduce_only, strategy=StrategyId.HFM80, signal_id="s1",
                       sl_pct=0.01, tp_pct=0.02, max_loss_usdt=Decimal("2"), ttl_ms=1000)


def _pos(inst, notional="100") -> Position:
    return Position(inst_id=inst, pos_side=PosSide.NET, size=Decimal("1"),
                    avg_px=Decimal("100"), notional_usdt=Decimal(notional), upl=Decimal("0"))


def test_approve_happy_path_uses_correct_field():
    d = _eng().evaluate(_intent(), **EVAL_KW, is_strong_signal=True)
    assert d.action == RiskAction.APPROVE
    assert d.approved_notional_usdt > 0                  # the field app.py must read
    assert not hasattr(d, "approved_notional")           # the wrong name that silently broke trading


def test_halted_returns_halt():
    re = _eng()
    re._halted_permanently = True
    re._update_system_state()
    assert re.evaluate(_intent(), **EVAL_KW, is_strong_signal=True).action == RiskAction.HALT


def test_stale_data_rejected():
    re = _eng()
    re.set_data_age_ms(10 ** 6)
    assert re.evaluate(_intent(), **EVAL_KW, is_strong_signal=True).action == RiskAction.REJECT


def test_close_only_blocks_open_but_allows_reduce():
    # total cumulative loss in (-50,-40] with a modest day_pnl -> CLOSE_ONLY (not the daily HALT rung)
    re = _eng()
    re.total_pnl, re.day_pnl = -42.0, 0.0
    re._update_system_state()
    assert re.system_state == SystemState.CLOSE_ONLY
    assert re.evaluate(_intent(reduce_only=False), **EVAL_KW, is_strong_signal=True).action == RiskAction.REDUCE_ONLY
    assert re.evaluate(_intent(reduce_only=True), **EVAL_KW, is_strong_signal=True).action == RiskAction.APPROVE


def test_strong_only_rejects_weak():
    re = _eng()
    re.total_pnl, re.day_pnl = -32.0, 0.0                 # total <= -30 -> STRONG_ONLY
    re._update_system_state()
    assert re.system_state == SystemState.STRONG_ONLY
    assert re.evaluate(_intent(), **EVAL_KW, is_strong_signal=False).action == RiskAction.REJECT


def test_taker_weak_rejected():
    d = _eng().evaluate(_intent(), **EVAL_KW, is_strong_signal=False, is_taker=True)
    assert d.action == RiskAction.REJECT


def test_funding_window_blocks_weak():
    d = _eng().evaluate(_intent(), **EVAL_KW, is_strong_signal=False, seconds_to_funding=60.0)
    assert d.action == RiskAction.REJECT


def test_concurrency_limit():
    re = _eng()
    re.open_positions = {"ETH-USDT-SWAP": _pos("ETH-USDT-SWAP"), "SOL-USDT-SWAP": _pos("SOL-USDT-SWAP")}
    d = re.evaluate(_intent(inst="BTC-USDT-SWAP"), **EVAL_KW, is_strong_signal=True)
    assert d.action == RiskAction.REJECT and "并发" in d.reason


def test_total_notional_cap():
    re = _eng()
    re.open_positions = {"ETH-USDT-SWAP": _pos("ETH-USDT-SWAP", notional="3000")}   # > 2500 cap
    d = re.evaluate(_intent(inst="BTC-USDT-SWAP"), **EVAL_KW, is_strong_signal=True)
    assert d.action == RiskAction.REJECT and "名义" in d.reason


def test_event_veto_block_long():
    ev = MarketEvent(ticker="BTC", event_type="8k", sentiment=-1.0, confidence=0.9,
                     severity="high", time_horizon="short", source_quality="official",
                     action=EventAction.BLOCK_LONG, valid_until_ms=10 ** 18)
    re = RiskEngine(Config.load(), event_provider=lambda inst, side: ev)
    assert re.evaluate(_intent(side=Side.BUY), **EVAL_KW, is_strong_signal=True).action == RiskAction.REJECT


def test_stock_perp_concurrency_separate_cap():
    set_stock_symbols(["AAPL", "SPY", "BTC"])   # ensure recognized; restore after
    try:
        re = _eng()
        re.open_positions = {"AAPL-USDT-SWAP": _pos("AAPL-USDT-SWAP")}   # 1 stock perp = cap
        d = re.evaluate(_intent(inst="SPY-USDT-SWAP"), **EVAL_KW, is_strong_signal=True)
        assert d.action == RiskAction.REJECT and "股票永续" in d.reason
    finally:
        set_stock_symbols(Config.load().get("universe.stock_symbols", []))
