import pandas as pd

from okxb.research.deribit_data import parse_dvol, expiry_days_from_name


def _ms(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def test_parse_dvol_to_ascending_close():
    res = {"data": [[1700000000000, 50, 55, 49, 52], [1700043200000, 52, 58, 51, 57]]}
    df = parse_dvol(res)
    assert list(df.columns) == ["ts", "dvol"]
    assert df["ts"].is_monotonic_increasing
    assert df["dvol"].iloc[-1] == 57.0


def test_parse_dvol_dedups():
    res = {"data": [[1700000000000, 50, 55, 49, 52], [1700000000000, 50, 55, 49, 52]]}
    assert len(parse_dvol(res)) == 1


def test_expiry_days_from_instrument_name():
    # BTC-27JUN25-100000-C expires 2025-06-27 08:00 UTC; asof 2025-06-20 00:00 UTC -> 7 whole days
    assert expiry_days_from_name("BTC-27JUN25-100000-C", asof_ms=_ms("2025-06-20")) == 7
