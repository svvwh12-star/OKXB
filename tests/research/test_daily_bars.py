from okxb.research import candle_data as cd
from okxb.research.daily_panel import daily_grid


def test_daily_and_weekly_bars_registered():
    assert cd.BAR_MS["1D"] == 86_400_000
    assert cd.BAR_MS["12H"] == 43_200_000
    assert cd.BAR_MS["1W"] == 604_800_000


def test_daily_grid_is_contiguous_utc_midnights():
    # 2024-01-01..2024-01-04 inclusive (UTC midnights, ms)
    start = 1_704_067_200_000  # 2024-01-01T00:00:00Z
    end = start + 3 * 86_400_000
    grid = daily_grid(start, end)
    assert list(grid) == [start + i * 86_400_000 for i in range(4)]
    assert (grid % 86_400_000 == 0).all()  # every point is a UTC midnight
