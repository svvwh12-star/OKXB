"""P1 / H-7: executor order-lifecycle race reconciliation (deterministic, fake exchange).

Covers the four races the audit flagged:
  H-7a WS place errors -> REST resend with same clOrdId -> exchange 51016 dedup -> adopt, no double position
  H-7b TTL cancel races with a fill -> re-query authoritative fill -> open the position (no hanging naked fill)
  H-7c IOC accFillSz settles late (filled-but-0) -> bounded retry catches the fill
  H-7d close IOC unfilled/partial -> position kept/updated, NOT fake-closed (no untracked naked position)
"""
import asyncio
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

def test_ws_fail_rest_duplicate_adopts_existing_no_double():
    rest = FakeRest(place_exc=OkxError("51016", "Duplicated clOrdId", "trade/order"),
                    get_default={"ordId": "EXIST", "state": "live", "accFillSz": "0"})
    ex = _ex(rest, ws=WSDown())
    res = asyncio.run(ex._place(inst_id="BTC-USDT-SWAP", td_mode="isolated", side="buy",
                                ord_type="post_only", sz="5", px="100", pos_side="net",
                                reduce_only=False, cl_ord_id="okxbsig1"))
    assert res.get("_deduped") is True and res.get("ordId") == "EXIST"
    assert len(rest.placed) == 1                     # exactly one REST resend, not a loop


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
                    {"state": "filled", "accFillSz": "5", "avgPx": "100"}])  # then populated
    ex = _ex(rest)
    state, filled, avg = asyncio.run(ex._resolve_order("BTC-USDT-SWAP", "x", retries=4, delay_s=0.0))
    assert state == "filled" and filled == Decimal("5") and avg == Decimal("100")


def test_resolve_order_canceled_is_terminal_no_fill():
    rest = FakeRest(get_default={"state": "canceled", "accFillSz": "0"})
    ex = _ex(rest)
    state, filled, avg = asyncio.run(ex._resolve_order("BTC-USDT-SWAP", "x", retries=4, delay_s=0.0))
    assert state == "canceled" and filled == Decimal("0") and avg is None


# ----------------- H-7d: close reconciliation -----------------

def test_close_unfilled_keeps_position():
    rest = FakeRest(get_default={"state": "canceled", "accFillSz": "0"})   # IOC didn't fill
    ex = _ex(rest)
    pos = _pos("5")
    ex._om.add_position(pos)
    asyncio.run(ex.close_position(pos, "止损", Decimal("100")))
    assert ex._om.get("BTC-USDT-SWAP") is not None         # NOT fake-closed
    assert pos.closing is False                             # released for retry next tick
    assert ex._risk.total_pnl == 0.0                        # no PnL booked for a non-fill
    assert _has(ex._store.audits, "close_unfilled")


def test_close_partial_keeps_remainder():
    rest = FakeRest(get_default={"state": "filled", "accFillSz": "3", "avgPx": "100"})
    ex = _ex(rest)
    pos = _pos("5")
    ex._om.add_position(pos)
    asyncio.run(ex.close_position(pos, "止损", Decimal("100")))
    kept = ex._om.get("BTC-USDT-SWAP")
    assert kept is not None and kept.contracts == Decimal("2")   # 5 - 3 remaining
    assert pos.closing is False
    assert _has(ex._store.audits, "close_partial")


def test_close_full_removes_position():
    rest = FakeRest(get_default={"state": "filled", "accFillSz": "5", "avgPx": "100"})
    ex = _ex(rest)
    pos = _pos("5")
    ex._om.add_position(pos)
    asyncio.run(ex.close_position(pos, "止损", Decimal("100")))
    assert ex._om.get("BTC-USDT-SWAP") is None              # fully closed -> removed
    assert "TP1" in rest.canceled                           # TP order canceled
    assert rest.canceled_algos                              # SL algo canceled


def test_close_reject_keeps_position():
    rest = FakeRest(place_exc=OkxError("50001", "server busy", "trade/order"))
    ex = _ex(rest)
    pos = _pos("5")
    ex._om.add_position(pos)
    asyncio.run(ex.close_position(pos, "止损", Decimal("100")))
    assert ex._om.get("BTC-USDT-SWAP") is not None          # place failed -> keep, don't fake-close
    assert pos.closing is False
    assert _has(ex._store.audits, "close_reject")
