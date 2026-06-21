"""P4: manual-order risk gate. A miss here means a manual order can exceed notional/leverage
caps with RiskEngine out of the loop — a direct capital-safety regression."""
from okxb.risk import manual_gate as mg


class FakeCfg:
    def __init__(self, d):
        self._d = d

    def get(self, dotted, default=None):
        return self._d.get(dotted, default)


def _isolate(tmp_path):
    mg.set_ledger_path_for_test(tmp_path / "ledger.json")


def teardown_function(_):
    mg.set_ledger_path_for_test(None)


def test_per_trade_cap_blocks(tmp_path):
    _isolate(tmp_path)
    cfg = FakeCfg({"risk.manual_max_notional_per_trade_usdt": 1000.0})
    assert mg.check_open(900.0, cfg) is None
    blocked = mg.check_open(1500.0, cfg)
    assert blocked and "单笔" in blocked


def test_day_notional_cap_accumulates(tmp_path):
    _isolate(tmp_path)
    cfg = FakeCfg({"risk.manual_max_notional_per_trade_usdt": 1e9,
                   "risk.manual_max_notional_per_day_usdt": 1000.0,
                   "risk.manual_max_trades_per_day": 999})
    assert mg.check_open(600.0, cfg) is None
    mg.record_open(600.0)
    assert mg.check_open(300.0, cfg) is None           # 600+300 <= 1000
    mg.record_open(300.0)
    blocked = mg.check_open(200.0, cfg)                 # 900+200 > 1000
    assert blocked and "单日" in blocked


def test_trades_per_day_cap(tmp_path):
    _isolate(tmp_path)
    cfg = FakeCfg({"risk.manual_max_notional_per_trade_usdt": 1e9,
                   "risk.manual_max_notional_per_day_usdt": 1e9,
                   "risk.manual_max_trades_per_day": 2})
    assert mg.check_open(10.0, cfg) is None
    mg.record_open(10.0)
    mg.record_open(10.0)
    blocked = mg.check_open(10.0, cfg)                  # would be the 3rd
    assert blocked and "笔数" in blocked


def test_leverage_cap(tmp_path):
    _isolate(tmp_path)
    cfg = FakeCfg({"risk.manual_max_leverage": 20})
    assert mg.check_leverage(10, cfg) is None
    blocked = mg.check_leverage(50, cfg)
    assert blocked and "杠杆" in blocked


def test_defaults_used_without_cfg(tmp_path):
    _isolate(tmp_path)
    # default per-trade cap is 2500
    assert mg.check_open(2000.0, None) is None
    assert mg.check_open(3000.0, None) is not None


def test_today_usage_reflects_records(tmp_path):
    _isolate(tmp_path)
    mg.record_open(123.0)
    u = mg.today_usage()
    assert u["count"] == 1 and abs(u["notional"] - 123.0) < 1e-6


def test_malformed_ledger_does_not_raise(tmp_path):
    """A hand-edited / corrupt ledger bucket must not crash the gate (it would otherwise be
    swallowed by the fail-CLOSED controller wrapper and reject — but the gate itself must be robust)."""
    import json
    _isolate(tmp_path)
    p = tmp_path / "ledger.json"
    p.write_text(json.dumps({mg._today(): {"oops": 1}}), encoding="utf-8")   # missing notional/count
    assert mg.check_open(100.0, None) is None          # treats malformed bucket as zero-used
    mg.record_open(50.0)                                # must not raise, rebuilds a clean bucket
    u = mg.today_usage()
    assert u["count"] == 1 and abs(u["notional"] - 50.0) < 1e-6

    p.write_text(json.dumps({mg._today(): "garbage"}), encoding="utf-8")     # non-dict bucket
    assert mg.check_open(100.0, None) is None
