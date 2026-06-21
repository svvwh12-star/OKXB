import pandas as pd

from okxb.research.macro_data import stooq_to_tidy_ms


def test_stooq_to_tidy_ms_drops_nan_and_sorts():
    idx = pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"])
    raw = pd.DataFrame({"Close": [13.5, 13.2, None]}, index=idx)
    out = stooq_to_tidy_ms(raw)
    assert list(out.columns) == ["ts", "value"]
    assert len(out) == 2  # NaN row dropped
    assert out["ts"].is_monotonic_increasing
    assert out["value"].iloc[0] == 13.2  # 2024-01-01 sorts first
