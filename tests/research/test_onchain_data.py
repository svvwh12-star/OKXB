import pandas as pd

from okxb.research.onchain_data import to_tidy_ms


def test_to_tidy_ms_converts_time_to_ms_and_renames():
    raw = pd.DataFrame({"asset": ["btc", "btc"],
                        "time": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
                        "AdrActCnt": ["100", "110"]})
    out = to_tidy_ms(raw, "AdrActCnt")
    assert list(out.columns) == ["ts", "value"]
    assert out["ts"].iloc[0] == int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)
    assert out["value"].iloc[1] == 110.0
    assert out["ts"].is_monotonic_increasing
