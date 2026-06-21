"""RV-5: the active gate now requires Deflated Sharpe + PBO. Pin the statistical machinery
it relies on, and that evaluate_scores exposes dsr/pbo.
"""
import numpy as np

from okxb.research import pro_model_workflow as pmw
from okxb.research.labeling import deflated_sharpe, pbo_cscv

DAY = 86_400_000


def test_deflated_sharpe_penalizes_more_trials():
    rng = np.random.default_rng(1)
    r = rng.normal(0.5, 1.0, 200).tolist()        # positive-mean series
    d1 = deflated_sharpe(r, n_trials=1)
    d200 = deflated_sharpe(r, n_trials=200)
    assert d1 is not None and d200 is not None
    assert d1 > d200                               # correcting for more trials lowers confidence


def test_deflated_sharpe_noise_not_significant():
    rng = np.random.default_rng(7)
    r = rng.normal(0.0, 1.0, 300).tolist()        # zero-mean noise -> no real edge
    d = deflated_sharpe(r, n_trials=50)
    assert d is None or d < 0.95                   # must NOT clear the 0.95 gate


def test_deflated_sharpe_tiny_sample_returns_none():
    assert deflated_sharpe([0.1, 0.2, 0.3], n_trials=5) is None   # T<10 -> can't validate


def test_pbo_cscv_returns_probability():
    rng = np.random.default_rng(3)
    cfgs = {i: rng.normal(0.0, 1.0, 240).tolist() for i in range(6)}
    res = pbo_cscv(cfgs)
    assert res is not None
    pbo, _logit = res
    assert 0.0 <= pbo <= 1.0


def test_evaluate_scores_exposes_dsr_and_pbo():
    rng = np.random.default_rng(0)
    n = 800
    ts = (np.arange(n) * DAY).astype(np.int64)
    score = rng.normal(0.0, 1.0, n)
    fwd = 0.0008 * np.sign(score) + rng.normal(0.0, 0.01, n)   # mild signal + noise
    import pandas as pd
    oos = pd.DataFrame({"ts": ts, "inst": "BTC-USDT-SWAP", "score": score,
                        "fwd": fwd, "y": (fwd > 0).astype(float)})
    mm = pmw.evaluate_scores(oos, 1440, DAY, np.array([]), mode="single_asset", n_trials=35)
    assert "dsr" in mm and "pbo" in mm             # wiring present
    # dsr is either None (insufficient) or a probability in [0,1]
    assert mm["dsr"] is None or (0.0 <= mm["dsr"] <= 1.0)
    assert mm["pbo"] is None or (0.0 <= mm["pbo"] <= 1.0)
