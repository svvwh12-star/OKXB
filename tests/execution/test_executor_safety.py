"""P0 executor regression tests (audit 2026-06: C-5 exchange-resident stop, C-7 dry_run safety).

Uses fake exchange/store clients so the money-path behaviour is verified deterministically
without touching OKX — far more reliable than fishing for a live auto-fill.
"""
import asyncio
from decimal import Decimal
from types import SimpleNamespace

from okxb.config import Config
from okxb.core.enums import Side, StrategyId
from okxb.execution.executor import Executor
from okxb.execution.order_manager import OrderManager
from okxb.risk.engine import RiskEngine


class FakeRest:
    def __init__(self):
        self.orders = []
        self.algos = []

    async def place_order(self, **kw):
        self.orders.append(kw)
        return {"ordId": "TP1"}

    async def place_algo_order(self, **kw):
        self.algos.append(kw)
        return {"algoId": "SL1"}

    async def get_order(self, *a, **k):
        return {"state": "live", "accFillSz": "0"}


class FakeInst:
    def tick_sz(self, i): return "0.1"
    def ct_val(self, i): return 0.01
    def lot_sz(self, i): return "1"
    def min_sz(self, i): return "1"
    def is_tradable(self, i): return True


class FakeStore:
    async def audit(self, *a, **k): pass
    async def record_pnl(self, *a, **k): pass
    async def record_signal(self, *a, **k): pass
    async def set_kv(self, *a, **k): pass


class FakeGw:
    def get_bbo(self, i):
        return SimpleNamespace(bid_px=100.0, ask_px=100.1, mid=100.05)

    def get_book(self, i):
        return SimpleNamespace(bbo=lambda: self.get_bbo(i))


def _executor(dry_run: bool, rest: FakeRest) -> Executor:
    cfg = Config.load()
    return Executor(
        rest=rest, gateway=FakeGw(), instruments=FakeInst(),
        order_manager=OrderManager(), risk=RiskEngine(cfg), store=FakeStore(),
        config=cfg, dry_run=dry_run, alert_fn=lambda m: None, ws=None,
    )


def _sig():
    return SimpleNamespace(
        inst_id="BTC-USDT-SWAP", side=Side.BUY, taker=False,
        sl_pct=0.01, tp_pct=0.02, strategy=StrategyId.HFM80, signal_id="sigtest123",
        composite_score=80.0, edge_to_cost=2.0, model_prob=0.6,
    )


# ----------------- C-5: exchange-resident reduce-only stop on open -----------------

def test_open_managed_places_exchange_resident_stop():
    rest = FakeRest()
    ex = _executor(dry_run=False, rest=rest)
    asyncio.run(ex._open_managed("BTC-USDT-SWAP", _sig(), Decimal("5"), Decimal("100")))

    assert len(rest.algos) == 1, "应在交易所端挂且仅挂一张止损 algo"
    algo = rest.algos[0]
    assert algo["side"] == Side.SELL.value          # 多仓的离场方向是卖
    assert algo["reduce_only"] is True              # 必须 reduce-only (net_mode)
    assert algo.get("sl_trigger_px")                # 设了止损触发价
    assert algo["sl_ord_px"] == "-1"                # 默认市价兜底 (死手场景优先保证离场)

    pos = ex._om.get("BTC-USDT-SWAP")
    assert pos is not None and pos.sl_algo_oid == "SL1", "止损 algoId 应记录到受管持仓上"


def test_exchange_stop_can_be_disabled_by_flag():
    rest = FakeRest()
    ex = _executor(dry_run=False, rest=rest)
    ex._exch_stop = False                           # 关掉开关
    asyncio.run(ex._open_managed("BTC-USDT-SWAP", _sig(), Decimal("5"), Decimal("100")))
    assert rest.algos == []                         # 不挂 algo


# ----------------- C-7: dry_run must never place real orders -----------------

def test_dry_run_places_no_real_orders():
    rest = FakeRest()
    ex = _executor(dry_run=True, rest=rest)
    asyncio.run(ex.submit(_sig(), Decimal("50"), False))
    assert rest.orders == [], "dry_run 不得调用 place_order"
    assert rest.algos == [], "dry_run 不得调用 place_algo_order"
