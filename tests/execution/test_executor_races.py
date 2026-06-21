"""P1 / H-7: executor order-lifecycle race reconciliation (deterministic, fake exchange).

Covers the four races the audit flagged:
  H-7a WS place errors -> REST resend with same clOrdId -> exchange 51016 dedup -> adopt, no double position
  H-7b TTL cancel races with a fill -> re-query authoritative fill -> open the position (no hanging naked fill)
  H-7c IOC accFillSz settles late (filled-but-0) -> bounded retry catches the fill
  H-7d close IOC unfilled/partial -> position kept/updated, NOT fake-closed (no untracked naked position)
"""
import asyncio
import time
from decimal import Decimal
from types import SimpleNamespace

from okxb.config import Config
from okxb.core.enums import Side, StrategyId
from okxb.exchange.okx_rest import OkxError
from okxb.execution.executor import Executor
from okxb.execution.order_manager import ManagedPosition, OrderManager
from okxb.risk.engine import RiskEngine


class FakeRest:
    def __init__(self, *, place_exc=None, get_default=None):
        self.placed, self.algos, self.canceled, self.canceled_algos = [], [], [], []
        self.place_exc = place_exc
        self.get_default = get_default or {"state": "canceled", "accFillSz": "0"}

    async def place_order(self, **kw):
        self.placed.append(kw)
        if self.place_exc is not None:
            raise self.place_exc
        return {"ordId": "O1", "sCode": "0"}

    async def place_algo_order(self, **kw):
        self.algos.append(kw)
        return {"algoId": "A1"}

    async def get_order(self, inst, *, ord_id=None, cl_ord_id=None):
        return dict(self.get_default)

    async def cancel_order(self, inst, *, ord_id=None, cl_ord_id=None):
        self.canceled.append(cl_ord_id or ord_id)
        return {}

    async def cancel_algos(self, orders):
        self.canceled_algos.append(orders)
        return []


class WSDown:
    logged_in = True

    async def place_order(self, **kw):
        raise OkxError("60012", "ws timeout", "ws")


class FakeInst:
    def tick_sz(self, i): return "0.1"
    def ct_val(self, i): return 0.01
    def lot_sz(self, i): return "1"
    def min_sz(self, i): return "1"
    def is_tradable(self, i): return True


class FakeStore:
    def __init__(self): self.audits = []
    async def audit(self, k, m): self.audits.append((k, m))
    async def record_pnl(self, *a, **k): pass
    async def record_signal(self, *a, **k): pass
    async def upsert_order(self, *a, **k): pass
    async def set_kv(self, *a, **k): pass


class FakeGw:
    def get_bbo(self, i): return SimpleNamespace(bid_px=100.0, ask_px=100.1, mid=100.05)
    def get_book(self, i): return SimpleNamespace(bbo=lambda: self.get_bbo(i))


def _ex(rest, *, ws=None):
    cfg = Config.load()
    return Executor(rest=rest, gateway=FakeGw(), instruments=FakeInst(), order_manager=OrderManager(),
                    risk=RiskEngine(cfg), store=FakeStore(), config=cfg, dry_run=False,
                    alert_fn=lambda m: None, ws=ws)


def _pos(contracts="5", **kw):
    base = dict(inst_id="BTC-USDT-SWAP", side=Side.BUY, contracts=Decimal(contracts),
                entry_px=Decimal("100"), sl_px=Decimal("99"), tp_px=Decimal("102"),
                strategy=StrategyId.HFM80, signal_id="sig1", entry_ms=0, max_loss_usdt=Decimal("5"),
                tp_order_oid="TP1", sl_algo_oid="SL1")
    base.update(kw)
    return ManagedPosition(**base)


def _has(audits, kind):
    return any(k == kind for k, _ in audits)


# ----------------- H-7a: WS->REST dedup via clOrdId idempotency -----------------

def test_ws_fail_probe_finds_existing_no_resend():
    # WS errored but the order DID land -> pre-probe finds it -> adopt, do NOT resend via REST
    rest = FakeRest(get_default={"ordId": "EXIST", "state": "live", "accFillSz": "0"})
    ex = _ex(rest, ws=WSDown())
    res = asyncio.run(ex._place(inst_id="BTC-USDT-SWAP", td_mode="isolated", side="buy",
                                ord_type="post_only", sz="5", px="100", pos_side="net",
                                reduce_only=False, cl_ord_id="okxbsig1"))
    assert res.get("_deduped") is True and res.get("ordId") == "EXIST"
    assert len(rest.placed) == 0                     # never resent -> no double order


def test_rest_duplicate_51016_adopted():
    # no WS; REST resend hits exact 51016 -> adopt existing instead of treating as failure
    rest = FakeRest(place_exc=OkxError("51016", "Duplicated clOrdId", "trade/order"),
                    get_default={"ordId": "EXIST", "state": "filled", "accFillSz": "5", "avgPx": "100"})
    ex = _ex(rest, ws=None)
    res = asyncio.run(ex._place(inst_id="BTC-USDT-SWAP", td_mode="isolated", side="buy",
                                ord_type="post_only", sz="5", px="100", pos_side="net",
                                reduce_only=False, cl_ord_id="okxbsig1"))
    assert res.get("_deduped") is True and res.get("ordId") == "EXIST"


def test_rest_non_dedup_error_propagates():
    rest = FakeRest(place_exc=OkxError("50001", "server error", "trade/order"))
    ex = _ex(rest, ws=None)
    try:
        asyncio.run(ex._place(inst_id="BTC-USDT-SWAP", td_mode="isolated", side="buy",
                              ord_type="post_only", sz="5", px="100", pos_side="net",
                              reduce_only=False, cl_ord_id="okxbsig1"))
        assert False, "should have raised"
    except OkxError as e:
        assert e.code == "50001"                     # non-dedup errors are NOT swallowed


# ----------------- H-7b: TTL cancel races a fill -----------------

class RaceRest(FakeRest):
    """live until cancel; cancel 'races' a fill so subsequent get_order shows filled."""
    def __init__(self):
        super().__init__()
        self._filled = False

    async def get_order(self, inst, *, ord_id=None, cl_ord_id=None):
        if self._filled:
            return {"state": "filled", "accFillSz": "5", "avgPx": "100"}
        return {"state": "live", "accFillSz": "0"}

    async def cancel_order(self, inst, *, ord_id=None, cl_ord_id=None):
        self.canceled.append(cl_ord_id or ord_id)
        self._filled = True
        return {}


def test_ttl_cancel_then_raced_fill_opens_position():
    rest = RaceRest()
    ex = _ex(rest)
    sig = SimpleNamespace(inst_id="BTC-USDT-SWAP", side=Side.BUY, sl_pct=0.01, tp_pct=0.02,
                          strategy=StrategyId.HFM80, signal_id="sig1")
    asyncio.run(ex._ttl_watch("BTC-USDT-SWAP", "okxbsig1", sig, Decimal("5"), Decimal("100"), 0.0))
    pos = ex._om.get("BTC-USDT-SWAP")
    assert pos is not None and pos.contracts == Decimal("5")   # raced fill captured, not a naked hang
    assert "okxbsig1" in rest.canceled


# ----------------- H-7c: IOC accFillSz settles late -----------------

class SeqRest(FakeRest):
    """get_order returns a queued sequence (then falls back to get_default)."""
    def __init__(self, seq):
        super().__init__()
        self._seq = list(seq)

    async def get_order(self, inst, *, ord_id=None, cl_ord_id=None):
        return self._seq.pop(0) if self._seq else dict(self.get_default)


def test_resolve_order_retries_filled_but_zero():
    rest = SeqRest([{"state": "filled", "accFillSz": "0"},                  # settle lag: inconsistent
                    {"state": "filled", "accFillSz": "5", "avgPx": "100", "fee": "-0.3"}])  # then populated
    ex = _ex(rest)
    state, filled, avg, fee = asyncio.run(ex._resolve_order("BTC-USDT-SWAP", "x", retries=4, delay_s=0.0))
    assert state == "filled" and filled == Decimal("5") and avg == Decimal("100") and fee == Decimal("-0.3")


def test_resolve_order_canceled_is_terminal_no_fill():
    rest = FakeRest(get_default={"state": "canceled", "accFillSz": "0"})
    ex = _ex(rest)
    state, filled, avg, fee = asyncio.run(ex._resolve_order("BTC-USDT-SWAP", "x", retries=4, delay_s=0.0))
    assert state == "canceled" and filled == Decimal("0") and avg is None


# ----------------- H-7d: close reconciliation -----------------

def test_close_unfilled_keeps_position_and_protection():
    rest = FakeRest(get_default={"state": "canceled", "accFillSz": "0"})   # IOC didn't fill
    ex = _ex(rest)
    pos = _pos("5")
    ex._om.add_position(pos)
    asyncio.run(ex.close_position(pos, "止损", Decimal("100")))
    assert ex._om.get("BTC-USDT-SWAP") is not None         # NOT fake-closed
    assert pos.closing is False                             # released for retry next tick
    assert ex._risk.total_pnl == 0.0                        # no PnL booked for a non-fill
    assert _has(ex._store.audits, "close_unfilled")
    assert not rest.canceled and not rest.canceled_algos    # protection NOT cancelled -> still guarded


def test_close_partial_keeps_remainder_with_protection():
    rest = FakeRest(get_default={"state": "filled", "accFillSz": "3", "avgPx": "100"})
    ex = _ex(rest)
    pos = _pos("5")
    ex._om.add_position(pos)
    before_consec = ex._risk.consecutive_losses
    asyncio.run(ex.close_position(pos, "止损", Decimal("100")))
    kept = ex._om.get("BTC-USDT-SWAP")
    assert kept is not None and kept.contracts == Decimal("2")   # 5 - 3 remaining
    assert pos.closing is False
    assert _has(ex._store.audits, "close_partial")
    assert not rest.canceled and not rest.canceled_algos    # remainder's SL/TP still live (H-7-A fix)
    assert ex._risk.consecutive_losses == before_consec     # partial slice must NOT bump consec losses


def test_close_full_removes_position():
    rest = FakeRest(get_default={"state": "filled", "accFillSz": "5", "avgPx": "100"})
    ex = _ex(rest)
    pos = _pos("5")
    ex._om.add_position(pos)
    asyncio.run(ex.close_position(pos, "止损", Decimal("100")))
    assert ex._om.get("BTC-USDT-SWAP") is None              # fully closed -> removed
    assert "TP1" in rest.canceled                           # TP order canceled (only after full close)
    assert rest.canceled_algos                              # SL algo canceled


def test_close_uses_real_fee_when_present():
    # exit fee from get_order.fee + entry fee per contract -> net PnL (not the rate estimate)
    rest = FakeRest(get_default={"state": "filled", "accFillSz": "5", "avgPx": "100", "fee": "-0.25"})
    ex = _ex(rest)
    pos = _pos("5", entry_fee_per_contract=Decimal("-0.02"))
    ex._om.add_position(pos)
    asyncio.run(ex.close_position(pos, "止损", Decimal("100")))
    # gross=(100-100)*5*0.01=0; entry_fee_slice=-0.02*5=-0.10; exit_fee=-0.25 -> net=-0.35
    assert abs(ex._risk.total_pnl - (-0.35)) < 1e-9


def test_close_noop_when_not_managed():
    rest = FakeRest(get_default={"state": "filled", "accFillSz": "5", "avgPx": "100"})
    ex = _ex(rest)
    pos = _pos("5")                                         # NOT added to _om
    asyncio.run(ex.close_position(pos, "止损", Decimal("100")))
    assert not rest.placed                                  # lock-wrapper guard: unmanaged -> no order


def test_close_reject_keeps_position():
    rest = FakeRest(place_exc=OkxError("50001", "server busy", "trade/order"))
    ex = _ex(rest)
    pos = _pos("5")
    ex._om.add_position(pos)
    asyncio.run(ex.close_position(pos, "止损", Decimal("100")))
    assert ex._om.get("BTC-USDT-SWAP") is not None          # place failed -> keep, don't fake-close
    assert pos.closing is False
    assert _has(ex._store.audits, "close_reject")


# ----------------- taker IOC must never silently drop a possibly-filled order -----------------

def _taker_sig():
    return SimpleNamespace(inst_id="BTC-USDT-SWAP", side=Side.BUY, taker=True,
                           sl_pct=0.01, tp_pct=0.02, strategy=StrategyId.HFM80, signal_id="sigT",
                           composite_score=80.0, edge_to_cost=2.0, model_prob=0.6)


def test_taker_opens_on_fill():
    rest = FakeRest(get_default={"state": "filled", "accFillSz": "5", "avgPx": "100"})
    ex = _ex(rest)
    asyncio.run(ex.submit(_taker_sig(), Decimal("50"), True))
    assert ex._om.has("BTC-USDT-SWAP")


def test_taker_filled_state_zero_accfill_still_opens():
    rest = FakeRest(get_default={"state": "filled", "accFillSz": "0"})   # state terminal, accFillSz lags
    ex = _ex(rest)
    asyncio.run(ex.submit(_taker_sig(), Decimal("50"), True))
    assert ex._om.has("BTC-USDT-SWAP")                   # opened with requested size, not dropped


def test_taker_unknown_flags_reconcile():
    rest = FakeRest()

    async def _raise(inst, **kw):
        raise OkxError("50001", "down", "get")
    rest.get_order = _raise
    ex = _ex(rest)
    asyncio.run(ex.submit(_taker_sig(), Decimal("50"), True))
    assert not ex._om.has("BTC-USDT-SWAP")
    assert _has(ex._store.audits, "taker_reconcile_needed")


# ----------------- reconcile_positions: close the orphan/ghost loop (review CRITICAL) -----------------

def test_reconcile_arms_orphan_stop_idempotent():
    rest = FakeRest()
    ex = _ex(rest)
    real = [{"instId": "BTC-USDT-SWAP", "pos": "5", "avgPx": "100"}]
    asyncio.run(ex.reconcile_positions(real))
    assert len(rest.algos) == 1                          # protective reduce-only SL placed
    assert ex._om.has("BTC-USDT-SWAP")                   # now also managed
    assert _has(ex._store.audits, "orphan_protected")
    asyncio.run(ex.reconcile_positions(real))            # idempotent: now managed -> no second arm
    assert len(rest.algos) == 1


def test_reconcile_rebuilds_managed_orphan():
    rest = FakeRest()
    ex = _ex(rest)
    asyncio.run(ex.reconcile_positions([{"instId": "BTC-USDT-SWAP", "pos": "5", "avgPx": "100"}]))
    pos = ex._om.get("BTC-USDT-SWAP")
    assert pos is not None
    assert pos.strategy == StrategyId.RECONCILED         # rebuilt lifecycle, not just a bare stop
    assert pos.contracts == Decimal("5") and pos.entry_px == Decimal("100")
    assert pos.sl_algo_oid == "A1"                       # armed protective stop recorded on the position


def test_reconcile_skips_managed():
    rest = FakeRest()
    ex = _ex(rest)
    ex._om.add_position(_pos("5"))
    asyncio.run(ex.reconcile_positions([{"instId": "BTC-USDT-SWAP", "pos": "5", "avgPx": "100"}]))
    assert len(rest.algos) == 0                          # already tracked -> not an orphan


def test_reconcile_removes_old_ghost_keeps_fresh():
    rest = FakeRest()
    ex = _ex(rest)
    old = _pos("5", entry_ms=0)                           # ancient -> real ghost if absent
    fresh = _pos("3", inst_id="ETH-USDT-SWAP", entry_ms=int(time.time() * 1000))
    ex._om.add_position(old)
    ex._om.add_position(fresh)
    asyncio.run(ex.reconcile_positions([]))              # exchange reports no positions
    assert ex._om.get("BTC-USDT-SWAP") is None            # old ghost cleaned up
    assert ex._om.get("ETH-USDT-SWAP") is not None        # fresh kept (snapshot-lag age guard)
    assert _has(ex._store.audits, "ghost_removed")
