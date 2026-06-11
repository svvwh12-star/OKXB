from okxb.research.candle_data import parse_oi_rows


def test_parse_oi_rows_sorts_ascending_and_floats():
    # OKX returns newest-first [ts, oi, oiCcy, oiUsd]
    raw = [["1700086400000", "5100", "51", "1.1e9"],
           ["1700000000000", "5000", "50", "1.0e9"]]
    df = parse_oi_rows(raw)
    assert list(df.columns) == ["ts", "oi", "oi_usd"]
    assert df["ts"].is_monotonic_increasing
    assert df["oi"].iloc[0] == 5000.0
    assert df["oi_usd"].iloc[0] == 1.0e9


def test_parse_oi_rows_skips_malformed():
    raw = [["1700000000000", "5000", "50", "1.0e9"], ["bad", "x"], []]
    df = parse_oi_rows(raw)
    assert len(df) == 1
