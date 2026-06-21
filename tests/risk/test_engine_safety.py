"""P0 capital-safety regression tests (audit 2026-06: C-3 / C-4 / killswitch / C-2).

These pin the behaviours the audit found missing on the money path:
  C-3  equity peak-drawdown hard stop (includes unrealized PnL via real equity)
  C-4  killswitch/ladder state survives a process restart (persist -> reload)
  C-2  stock-classification drift detection (dist config missing stock_symbols)
"""
import asyncio

from okxb.config import Config
from okxb.core.enums import SystemState
from okxb.risk.engine import (RiskEngine, set_stock_symbols,
                              stock_classification_mismatches)
from okxb.state.store import StateStore


def _engine() -> RiskEngine:
    return RiskEngine(Config.load())   # initial_equity=1000, total_drawdown_hard_stop_pct=5.0


# ----------------- C-3: equity peak-drawdown hard stop (incl. unrealized) -----------------

def test_equity_drawdown_below_threshold_stays_normal():
    re = _engine()
    re.update_equity(960.0)            # 4% drawdown from peak 1000 < 5%
    assert not re._halted_permanently
    assert re.system_state == SystemState.NORMAL


def test_equity_drawdown_hard_stop_trips_on_unrealized():
    re = _engine()
    # No realized loss registered at all (total_pnl == 0) -> proves it is equity-driven,
    # i.e. an open unrealized loss reflected in real totalEq now trips the stop.
    re.update_equity(949.0)            # 5.1% drawdown from peak 1000 >= 5%
    assert re.total_pnl == 0.0
    assert re._halted_permanently
    assert re.system_state == SystemState.HALTED


def test_peak_tracks_real_equity_then_drawdown():
    re = _engine()
    re.update_equity(2000.0)           # peak rises to 2000
    assert re.peak_equity == 2000.0
    re.update_equity(1901.0)           # 4.95% < 5% -> ok
    assert not re._halted_permanently
    re.update_equity(1899.0)           # 5.05% >= 5% -> halt
    assert re._halted_permanently


def test_realized_total_killswitch_still_trips():
    re = _engine()
    re.register_close(-60.0)           # total_pnl -60 <= total_killswitch_at (-50)
    assert re._halted_permanently
    assert re.system_state == SystemState.HALTED


def test_clear_halt_rebaselines_and_recovers():
    re = _engine()
    re.update_equity(900.0)
    assert re._halted_permanently
    re.clear_halt()                    # operator override = rebaseline, not silent un-halt
    assert not re._halted_permanently
    assert re.system_state == SystemState.NORMAL
    assert re.peak_equity == 900.0     # peak rebased to current equity (won't instantly re-trip)
    assert re.total_pnl == 0.0


# ----------------- C-4: state survives restart -----------------

def test_permanent_halt_survives_restart():
    re1 = _engine()
    re1.update_equity(900.0)           # trip
    assert re1._halted_permanently
    snapshot = re1.to_state()

    re2 = _engine()                    # fresh process
    assert re2.system_state == SystemState.NORMAL
    re2.load_state(snapshot)           # reload persisted state
    assert re2._halted_permanently
    assert re2.system_state == SystemState.HALTED


def test_day_pnl_resets_next_day_total_persists():
    re = _engine()
    snap = {
        "halted_permanently": False, "halt_reason": "", "halt_date": "",
        "peak_equity": 1000.0, "equity": 1000.0,
        "total_pnl": -12.0, "day_pnl": -7.0, "day_date": "2000-01-01",
        "consecutive_losses": 3,
    }
    re.load_state(snap)
    assert re.day_pnl == 0.0           # different UTC day -> daily counter reset
    assert re.total_pnl == -12.0       # cumulative persists
    assert re.consecutive_losses == 3


# ----------------- C-2: stock classification drift -----------------

def test_stock_classification_mismatch_detects_missing_config():
    saved = set(stock_classification_mismatches.__globals__["_STOCK_SYMBOLS"])
    try:
        set_stock_symbols([])          # simulate dist config missing stock_symbols
        miss = stock_classification_mismatches(["AAPL-USDT-SWAP", "BTC-USDT-SWAP"])
        assert "AAPL-USDT-SWAP" in miss
        assert "BTC-USDT-SWAP" not in miss
        set_stock_symbols(["AAPL"])    # correctly configured -> no mismatch
        assert stock_classification_mismatches(["AAPL-USDT-SWAP"]) == []
    finally:
        set_stock_symbols(saved)


# ----------------- C-4 storage layer: kv round-trip -----------------

def test_state_store_kv_roundtrip(tmp_path):
    async def go():
        st = StateStore(str(tmp_path / "s.sqlite"))
        await st.open()
        assert await st.get_kv("risk_state") is None
        await st.set_kv("risk_state", {"halted_permanently": True, "total_pnl": -1.5})
        v = await st.get_kv("risk_state")
        await st.set_kv("risk_state", {"halted_permanently": False})   # upsert
        v2 = await st.get_kv("risk_state")
        await st.close()
        return v, v2

    v, v2 = asyncio.run(go())
    assert v["halted_permanently"] is True and v["total_pnl"] == -1.5
    assert v2["halted_permanently"] is False
