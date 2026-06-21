"""P1: CompositeScorer numeric invariants — long+short symmetry, sl/tp bounds, cost, edge.
A miscomputed cost makes the edge_to_cost entry gate over-permissive (lets negative-expectancy
trades through); a too-small sl_pct inflates notional. Both are direct money paths."""
from okxb.config import Config
from okxb.core.enums import Side, StrategyId
from okxb.core.models import FeatureSet
from okxb.signal.composite import CompositeScorer


def _fs(**kw):
    base = dict(inst_id="BTC-USDT-SWAP", ts=1)
    base.update(kw)
    return FeatureSet(**base)


def _scorer():
    return CompositeScorer(Config.load())


def test_long_short_sum_to_100_and_direction():
    r = _scorer().score(_fs(obi_5_z=2.0, ofi_z=1.5, trade_imbalance_3s=0.5,
                            mid_return_5s=0.001, mid_return_15s=0.002,
                            realized_vol_60s=1e-3, spread_bps=5.0, atr_1m=0.002))
    assert abs(r.long_score + r.short_score - 100.0) <= 0.11
    assert abs(r.long_score_s + r.short_score_s - 100.0) <= 0.11
    assert r.long_score >= r.short_score          # bullish order flow -> long favored


def test_bearish_flips_direction():
    r = _scorer().score(_fs(obi_5_z=-2.0, ofi_z=-1.5, trade_imbalance_3s=-0.5,
                            mid_return_5s=-0.001, mid_return_15s=-0.002,
                            realized_vol_60s=1e-3, spread_bps=5.0))
    assert r.short_score >= r.long_score


def test_sl_tp_bounds():
    sc = _scorer()
    fs = _fs(atr_1m=0.002, spread_bps=5.0)
    sl, tp = sc.sl_tp(fs)
    assert 0.0010 <= sl <= 0.012                  # floor and 1.2% cap enforced
    assert tp >= sl                               # tp_rr >= 1


def test_cost_taker_exceeds_maker():
    sc = _scorer()
    fs = _fs(spread_bps=5.0)
    maker = sc.total_cost_pct(fs, taker=False)
    taker = sc.total_cost_pct(fs, taker=True)
    assert maker > 0 and taker > maker            # taker entry strictly costlier


def test_build_signal_edge_to_cost_consistent():
    sc = _scorer()
    fs = _fs(atr_1m=0.002, spread_bps=5.0, realized_vol_60s=1e-3)
    sig = sc.build_signal(fs, StrategyId.HFM80, Side.BUY, 80.0)
    assert sig.total_cost_pct > 0 and sig.sl_pct > 0 and sig.tp_pct >= sig.sl_pct
    assert abs(sig.edge_to_cost - sig.expected_edge_pct / sig.total_cost_pct) < 1e-9
