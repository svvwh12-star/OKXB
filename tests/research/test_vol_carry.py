from okxb.research.vol_carry import bs_price, iron_condor_pnl


def test_bs_atm_call_equals_put_at_zero_rate():
    c = bs_price(100, 100, 0.1, 0.6, True)
    p = bs_price(100, 100, 0.1, 0.6, False)
    assert abs(c - p) < 1e-9            # put-call parity at r=0


def test_bs_increasing_in_vol():
    assert bs_price(100, 100, 0.1, 0.8, True) > bs_price(100, 100, 0.1, 0.4, True)


def test_iron_condor_collects_premium_when_calm():
    pnl = iron_condor_pnl(100, 100, 0.6, 7 / 365, 1.0, 2.0, fee_rate=0.0, fee_cap_frac=0.0)
    assert pnl > 0                       # no move -> keep the credit


def test_iron_condor_loss_is_bounded_on_crash():
    calm = iron_condor_pnl(100, 100, 0.6, 7 / 365, 1.0, 2.0, fee_rate=0.0, fee_cap_frac=0.0)
    crash = iron_condor_pnl(100, 55, 0.6, 7 / 365, 1.0, 2.0, fee_rate=0.0, fee_cap_frac=0.0)
    assert crash < calm                  # a big move loses
    # defined risk: loss bounded by the put-spread width (Kp - Kpw), well under full notional
    m = 0.6 * (7 / 365) ** 0.5
    width = 100 * (2.718281828 ** (-1.0 * m)) - 100 * (2.718281828 ** (-2.0 * m))
    assert crash > -(width + 1.0)        # cannot lose more than the defined spread


def test_fees_reduce_pnl():
    no_fee = iron_condor_pnl(100, 100, 0.6, 7 / 365, 1.0, 2.0, fee_rate=0.0, fee_cap_frac=0.0)
    with_fee = iron_condor_pnl(100, 100, 0.6, 7 / 365, 1.0, 2.0, fee_rate=0.0003, fee_cap_frac=0.125)
    assert with_fee < no_fee
