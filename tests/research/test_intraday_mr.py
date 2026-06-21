"""Pre-registered 15/30-min intraday mean-reversion candidate — signal/labeling/verdict.
Guards: no look-ahead in labeling, correct net-of-cost math, and the forward gates (incl. sticky KILL)."""
import random

from okxb.research import intraday_mr as imr


def test_normalize_drops_unconfirmed_and_sorts():
    raw = [["3", "0", "0", "0", "102", "0", "0", "0", "1"],   # newest first
           ["2", "0", "0", "0", "101", "0", "0", "0", "1"],
           ["1", "0", "0", "0", "100", "0", "0", "0", "0"]]   # confirm=0 -> dropped
    assert imr.normalize_candles(raw) == [(2, 101.0), (3, 102.0)]


def test_signal_direction_is_mean_reversion():
    assert imr.signal_direction(2.0) == -1     # up too much -> short (fade)
    assert imr.signal_direction(-2.0) == 1     # down too much -> long (fade)
    assert imr.signal_direction(0.5) == 0
    assert imr.signal_direction(None) == 0


def test_iter_labels_no_lookahead_and_net_math():
    rng = random.Random(0)
    closes = [100.0]
    for _ in range(140):
        closes.append(closes[-1] * (1 + rng.gauss(0, 0.01)))
    candles = list(enumerate(closes))                 # (ts=i, close)
    labels = list(imr.iter_labels(candles, window=20, enter=0.5, hold=1))
    assert labels                                     # some signals fire
    # the last `hold` bars have no realized forward -> never labeled (look-ahead-safe)
    assert max(lb.bar_ts for lb in labels) <= candles[-1 - 1][0]
    # internal consistency of net-of-cost (15/10 bps) vs entry/exit/direction
    for lb in labels:
        exp15 = lb.direction * (lb.exit_px / lb.entry_px - 1) * 1e4 - imr.COST_BPS_STRESS
        exp10 = lb.direction * (lb.exit_px / lb.entry_px - 1) * 1e4 - imr.COST_BPS_MILD
        assert abs(lb.net15_bps - exp15) < 1e-6
        assert abs(lb.net10_bps - exp10) < 1e-6
        assert lb.direction in (-1, 1)


def test_verdict_pending_when_too_few_samples():
    net15 = [4.0, 6.0] * 25      # n=50 (<100), positive
    net10 = [9.0, 11.0] * 25
    v = imr.evaluate_candidate(net15, net10, train_net15=5.0, train_ic_sign=1)
    assert v.verdict == "PENDING" and "insufficient" in v.reason


def test_verdict_sticky_kill_when_no_edge_at_mild_cost():
    net15 = [-2.0] * 40
    net10 = [-1.0] * 40          # m10<=0, n>=30 -> KILL
    v = imr.evaluate_candidate(net15, net10, train_net15=5.0, train_ic_sign=1)
    assert v.verdict == "KILL"


def test_verdict_already_dead_is_sticky():
    v = imr.evaluate_candidate([5.0] * 120, [10.0] * 120, train_net15=5.0,
                               train_ic_sign=1, already_dead=True)
    assert v.verdict == "KILL" and "sticky" in v.reason


def test_verdict_pass_on_strong_forward_series():
    net15 = [4.0, 6.0] * 60      # n=120, mean=5, low variance -> big t & DSR
    net10 = [9.0, 11.0] * 60     # mean=10 > 0
    v = imr.evaluate_candidate(net15, net10, train_net15=5.0, train_ic_sign=1, pbo=0.1)
    assert v.verdict == "PASS", v.reason


def test_family_trials_is_six():
    assert imr.FAMILY_TRIALS == 6
    assert imr.code_of("BTC-USDT-SWAP", "15m") == "IMR_BTC_15m"
