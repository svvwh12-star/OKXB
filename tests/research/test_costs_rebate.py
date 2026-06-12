from okxb.research.pro_model_workflow import WorkflowCosts


class _Cfg:
    """Minimal config stub exposing .get(key, default) like the real Config."""
    def __init__(self, maker, taker, rebate):
        self._d = {"fees.crypto_maker_pct": maker, "fees.crypto_taker_pct": taker,
                   "fees.fee_rebate_frac": rebate}

    def get(self, key, default=None):
        return self._d.get(key, default)


def test_rebate_applied_to_round_trip_fees():
    c = WorkflowCosts.from_config(_Cfg(0.02, 0.05, 0.20))
    assert abs(c.maker_bps - 3.2) < 1e-6     # 0.02% one side -> 4bps RT -> x0.8
    assert abs(c.taker_bps - 8.0) < 1e-6     # 0.05% -> 10bps RT -> x0.8
    assert abs(c.stress_bps - 15.0) < 1e-6   # stress = 1.5x RT taker, NOT rebated


def test_no_rebate_matches_legacy_defaults():
    c = WorkflowCosts.from_config(_Cfg(0.02, 0.05, 0.0))
    assert abs(c.maker_bps - 4.0) < 1e-6
    assert abs(c.taker_bps - 10.0) < 1e-6


def test_rebate_is_clamped():
    c = WorkflowCosts.from_config(_Cfg(0.02, 0.05, 5.0))   # absurd rebate clamps to 0.9
    assert abs(c.taker_bps - 1.0) < 1e-6                   # 10 * (1-0.9)
