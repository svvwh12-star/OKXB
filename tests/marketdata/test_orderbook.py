"""P1: local order book seqId integrity + reconstruction. A corrupt book silently sizes
and directs real trades (depth feeds sizing; bbo feeds direction), so gaps must be caught."""
from okxb.marketdata.orderbook import OrderBook


def _snap(ob):
    return ob.apply_snapshot({"bids": [["100", "2"], ["99.9", "3"]],
                              "asks": [["100.1", "1"], ["100.2", "4"]],
                              "seqId": 1, "ts": 1})


def test_snapshot_then_inorder_update():
    ob = OrderBook("X")
    assert _snap(ob)
    bbo = ob.bbo()
    assert bbo.bid_px == 100.0 and bbo.ask_px == 100.1 and bbo.bid_sz == 2.0
    assert ob.apply_update({"bids": [["100", "5"]], "asks": [], "seqId": 2, "prevSeqId": 1, "ts": 2})
    assert ob.best_bid() == (100.0, 5.0)


def test_gap_detected_and_heartbeat():
    ob = OrderBook("X")
    _snap(ob)
    # prevSeqId != last_seq and seq != last -> gap -> False (caller must re-subscribe)
    assert ob.apply_update({"bids": [["100", "9"]], "seqId": 5, "prevSeqId": 3}) is False
    # heartbeat: prevSeqId != last but seqId == last -> no change, True
    assert ob.apply_update({"seqId": 1, "prevSeqId": 3}) is True


def test_update_before_snapshot_returns_false():
    assert OrderBook("X").apply_update({"seqId": 2, "prevSeqId": 1}) is False


def test_zero_size_removes_level():
    ob = OrderBook("X")
    _snap(ob)
    ob.apply_update({"bids": [["100", "0"]], "seqId": 2, "prevSeqId": 1})   # remove top bid
    assert ob.best_bid() == (99.9, 3.0)


def test_depth_notional_within_bps():
    ob = OrderBook("X")
    _snap(ob)                                    # mid = 100.05
    d = ob.depth_notional("bid", 50)             # lo = mid*(1-0.005) ≈ 99.55 -> both bids count
    assert abs(d - (100.0 * 2 + 99.9 * 3)) < 1e-6


def test_crossed_or_empty_book_bbo_none():
    ob = OrderBook("X")
    assert ob.bbo() is None                      # empty -> no bbo (won't trade on a missing book)
