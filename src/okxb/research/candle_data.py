"""OKX 历史 K 线离线拉取 (公共数据, 免鉴权) —— 供分钟~小时级方向预测研究使用。

为什么单独写 / 不走 OkxRestClient:
  研究拉取是【离线、纯公共行情】, 直接用 httpx 同步客户端可避免耦合 .env/Config/
  限速器与交易事件循环; 不接触任何密钥; 落盘缓存, 重跑秒级加载。
  (生产路径若要实时分钟特征, 用 okx_rest.get_history_candles, 那是异步交易客户端的家。)

数据口径:
  OKX /api/v5/market/history-candles 每条 = [ts, o, h, l, c, vol(张), volCcy(基币),
  volCcyQuote(USDT), confirm]; 返回【新→旧】; after=返回早于该 ts 的更旧数据(向过去翻页)。
  只保留 confirm=='1'(已收盘)的 K 线, 丢弃正在形成的最新一根。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

import httpx
import pandas as pd

HOSTS = ["https://www.okx.com", "https://us.okx.com", "https://eea.okx.com"]

# 各 bar 周期的毫秒数 (用于覆盖范围判断)
BAR_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1H": 3_600_000, "2H": 7_200_000, "4H": 14_400_000,
    "6H": 21_600_000, "12H": 43_200_000,
    "1D": 86_400_000, "1W": 604_800_000,
}
_COLS = ["ts", "o", "h", "l", "c", "vol", "volccy", "volquote", "confirm"]


def _log(msg: str) -> None:
    print(msg, flush=True)


def fetch_history(inst_id: str, bar: str, days: float, *,
                  host: Optional[str] = None, throttle: float = 0.11,
                  max_pages: int = 4000, log: Callable[[str], None] = _log) -> pd.DataFrame:
    """向过去回填 `days` 天的 K 线。返回按 ts 升序、float 列、ts 为 int(ms) 的 DataFrame。

    只含已收盘 K 线; 跨 host 自动故障转移; 节流避免触发公共接口频率上限(~20/2s)。"""
    if bar not in BAR_MS:
        raise ValueError(f"未知 bar={bar}, 支持 {list(BAR_MS)}")
    bar_ms = BAR_MS[bar]
    now_ms = int(time.time() * 1000)
    start_target = now_ms - int(days * 86_400_000)
    hosts = [host] if host else list(HOSTS)
    rows: dict[int, list[str]] = {}
    chosen: Optional[str] = None
    after: Optional[int] = None
    pages = 0
    with httpx.Client(timeout=15.0, headers={"User-Agent": "okxb-research/1"}) as cli:
        while pages < max_pages:
            data = None
            for h in (hosts if chosen is None else [chosen]):
                try:
                    params = {"instId": inst_id, "bar": bar, "limit": "100"}
                    if after is not None:
                        params["after"] = str(after)
                    r = cli.get(h + "/api/v5/market/history-candles", params=params)
                    j = r.json()
                    if j.get("code") == "0":
                        data = j.get("data", [])
                        chosen = h
                        break
                    else:
                        log(f"  {inst_id} {bar}: code={j.get('code')} msg={j.get('msg')}")
                except Exception as e:  # noqa: BLE001 - 容错: 换 host 重试
                    log(f"  {inst_id} {bar}: {type(e).__name__} {e} (换host)")
                    continue
            if not data:
                break
            for row in data:
                if len(row) >= 9 and row[-1] != "1":   # 未收盘, 跳过
                    continue
                rows[int(row[0])] = row
            oldest = min(int(row[0]) for row in data)
            pages += 1
            if oldest <= start_target:
                break
            after = oldest
            time.sleep(throttle)
    kept = [rows[ts] for ts in sorted(rows) if ts >= start_target]
    if not kept:
        return pd.DataFrame(columns=_COLS).astype({"ts": "int64"})
    df = pd.DataFrame([r[:9] for r in kept], columns=_COLS)
    df["ts"] = df["ts"].astype("int64")
    for c in ("o", "h", "l", "c", "vol", "volccy", "volquote"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop(columns=["confirm"]).dropna().reset_index(drop=True)
    return df


def _cache_file(root: Path, inst_id: str, bar: str) -> Path:
    return root / f"{inst_id}_{bar}.csv"


INCR_MAX_GAP_DAYS = 10.0      # 缓存比这更旧 -> 直接全量重拉(增量翻页太多, 且恐有洞)
INCR_BUFFER_DAYS = 0.15       # 增量多拉一点与缓存重叠, 保证拼接无缝(无空洞)


def get_candles(inst_id: str, bar: str, days: float, root: Path, *,
                force: bool = False, update: bool = False,
                log: Callable[[str], None] = _log) -> pd.DataFrame:
    """带磁盘缓存的拉取。
      force=True : 全量重拉并覆盖缓存。
      update=True: 增量更新——保留缓存, 只拉"缓存最新ts之后"的新 bar 拼接(几次调用而非几百次);
                   缓存回看不足 / 间隔过大(>INCR_MAX_GAP_DAYS)时自动回退全量; 增量拉空则沿用缓存。
      默认       : 缓存覆盖请求范围则直接用, 否则全量重拉。"""
    root.mkdir(parents=True, exist_ok=True)
    f = _cache_file(root, inst_id, bar)
    now_ms = int(time.time() * 1000)
    need_start = now_ms - int(days * 86_400_000)
    bar_ms = BAR_MS[bar]

    def _full() -> pd.DataFrame:
        df = fetch_history(inst_id, bar, days, log=log)
        if len(df):
            df.to_csv(f, index=False)
        return df

    if force:
        return _full()

    cache = None
    if f.exists():
        try:
            cache = pd.read_csv(f)
            cache["ts"] = cache["ts"].astype("int64")
        except Exception as e:  # noqa: BLE001
            log(f"  {inst_id} {bar}: 缓存读取失败 ({e}), 重拉")
            cache = None
    if cache is None or len(cache) <= 10:
        return _full()

    cache_start = int(cache["ts"].iloc[0])
    cache_max = int(cache["ts"].iloc[-1])
    covers_start = cache_start <= need_start + bar_ms * 5

    if not update:
        if covers_start:
            return cache[cache["ts"] >= need_start].reset_index(drop=True)
        log(f"  {inst_id} {bar}: 缓存覆盖不足, 重拉")
        return _full()

    # --- update 增量路径 ---
    gap_ms = now_ms - cache_max
    if not covers_start or gap_ms > INCR_MAX_GAP_DAYS * 86_400_000:
        log(f"  {inst_id} {bar}: 缓存回看不足或间隔过大, 回退全量")
        return _full()
    if gap_ms < bar_ms:                                 # 已最新(不足一根新bar)
        return cache[cache["ts"] >= need_start].reset_index(drop=True)
    new = fetch_history(inst_id, bar, gap_ms / 86_400_000 + INCR_BUFFER_DAYS, log=log)
    if not len(new):                                    # 网络没取到新 bar -> 沿用缓存, 不破坏
        log(f"  {inst_id} {bar}: 增量拉取为空, 沿用现有缓存")
        return cache[cache["ts"] >= need_start].reset_index(drop=True)
    merged = (pd.concat([cache, new], ignore_index=True)
              .drop_duplicates("ts").sort_values("ts").reset_index(drop=True))
    merged = merged[merged["ts"] >= need_start].reset_index(drop=True)
    merged.to_csv(f, index=False)
    return merged


def fetch_universe(insts: list[str], bar: str, days: float, root: Path, *,
                   force: bool = False, log: Callable[[str], None] = _log) -> dict[str, pd.DataFrame]:
    """逐标的拉取(带缓存)。返回 {inst_id: DataFrame}; 只保留非空。"""
    out: dict[str, pd.DataFrame] = {}
    for i, inst in enumerate(insts, 1):
        df = get_candles(inst, bar, days, root, force=force, log=log)
        if len(df) >= 50:
            out[inst] = df
            span_d = (df["ts"].iloc[-1] - df["ts"].iloc[0]) / 86_400_000
            log(f"[{i}/{len(insts)}] {inst} {bar}: {len(df)} 根, 跨 {span_d:.1f} 天")
        else:
            log(f"[{i}/{len(insts)}] {inst} {bar}: 数据不足({len(df)}), 跳过")
    return out


# ----------------------- 选标的 (复用 universe 口径, 同步公共接口) -----------------------

def top_crypto(n: int, *, min_quote_vol_usd: float = 8_000_000,
               host: Optional[str] = None, log: Callable[[str], None] = _log) -> list[str]:
    """按 24h 报价额取前 n 的 USDT 本位加密永续 (排除股票永续代码)。"""
    stock_syms = {
        "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "META", "GOOGL", "AMZN", "GOOG",
        "NFLX", "AMD", "INTC", "AVGO", "COIN", "MSTR", "PLTR", "ORCL", "MU", "QCOM",
        "ARM", "ADBE", "CRM", "SMCI", "UBER", "ABNB", "SHOP", "DIS", "BA", "JPM",
        "V", "MA", "WMT", "PYPL",
    }
    hosts = [host] if host else list(HOSTS)
    tickers = []
    with httpx.Client(timeout=15.0, headers={"User-Agent": "okxb-research/1"}) as cli:
        for h in hosts:
            try:
                r = cli.get(h + "/api/v5/market/tickers", params={"instType": "SWAP"})
                j = r.json()
                if j.get("code") == "0":
                    tickers = j.get("data", [])
                    break
            except Exception:  # noqa: BLE001
                continue

    def qvol(t: dict) -> float:
        try:
            return float(t.get("volCcy24h", 0) or 0) * float(t.get("last", 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    cand = [
        t for t in tickers
        if t.get("instId", "").endswith("-USDT-SWAP")
        and t["instId"].split("-")[0].upper() not in stock_syms
        and qvol(t) >= min_quote_vol_usd
    ]
    cand.sort(key=qvol, reverse=True)
    out = [t["instId"] for t in cand[:n]]
    log(f"选出 {len(out)} 个加密永续 (按24h报价额): {', '.join(s.split('-')[0] for s in out)}")
    return out


def fetch_funding_series(inst_id: str, days: float, *, host: Optional[str] = None,
                         log: Callable[[str], None] = _log) -> pd.DataFrame:
    """近 `days` 天的【已实现】资金费率时间序列 (每 8h 一次), 按 ts 升序。
    列: ts(=fundingTime, ms), funding(realizedRate, 带符号小数: >0 多头付空头)。"""
    now_ms = int(time.time() * 1000)
    start = now_ms - int(days * 86_400_000)
    hosts = [host] if host else list(HOSTS)
    rows: list[tuple[int, float]] = []
    chosen: Optional[str] = None
    after: Optional[int] = None
    with httpx.Client(timeout=15.0, headers={"User-Agent": "okxb-research/1"}) as cli:
        for _ in range(60):
            data = None
            for h in (hosts if chosen is None else [chosen]):
                try:
                    params = {"instId": inst_id, "limit": "100"}
                    if after is not None:
                        params["after"] = str(after)
                    r = cli.get(h + "/api/v5/public/funding-rate-history", params=params)
                    j = r.json()
                    if j.get("code") == "0":
                        data = j.get("data", [])
                        chosen = h
                        break
                except Exception:  # noqa: BLE001
                    continue
            if not data:
                break
            stop = False
            for row in data:
                ft = int(row.get("fundingTime", 0))
                # realizedRate 优先 (该期实际结算费率); 显式判空, 不靠 '0' 的真值性 → 回退 fundingRate
                rr = row.get("realizedRate")
                rate = rr if rr not in (None, "") else row.get("fundingRate", 0)
                if ft < start:
                    stop = True
                    continue
                try:
                    rows.append((ft, float(rate)))
                except (TypeError, ValueError):
                    pass
            if stop:
                break
            after = min(int(row.get("fundingTime", 0)) for row in data)
            time.sleep(0.08)
    if not rows:
        return pd.DataFrame(columns=["ts", "funding"]).astype({"ts": "int64"})
    df = pd.DataFrame(sorted(set(rows)), columns=["ts", "funding"])
    df["ts"] = df["ts"].astype("int64")
    return df


def fetch_funding_mean(inst_id: str, days: float, *, host: Optional[str] = None,
                       log: Callable[[str], None] = _log) -> Optional[float]:
    """近 `days` 天的【带符号】平均资金费率 (每 8h)。返回 None=取不到。"""
    df = fetch_funding_series(inst_id, days, host=host, log=log)
    return float(df["funding"].mean()) if len(df) else None


def parse_oi_rows(rows: list) -> pd.DataFrame:
    """Parse OKX open-interest-history rows [ts, oi, oiCcy, oiUsd] -> ascending df(ts, oi, oi_usd)."""
    out = []
    for r in rows:
        try:
            out.append((int(r[0]), float(r[1]), float(r[3]) if len(r) > 3 else float("nan")))
        except (TypeError, ValueError, IndexError):
            continue
    df = pd.DataFrame(sorted(set(out)), columns=["ts", "oi", "oi_usd"])
    if len(df):
        df["ts"] = df["ts"].astype("int64")
    return df


def fetch_oi_history(inst_id: str, period: str, days: float, *,
                     host: Optional[str] = None, log: Callable[[str], None] = _log) -> pd.DataFrame:
    """Open-interest history for a perp. period in OKX codes (e.g. '1H','1D'). Ascending df(ts, oi, oi_usd).

    Endpoint: /api/v5/rubik/stat/contracts/open-interest-history (rows newest-first
    [ts, oi(contracts), oiCcy(base), oiUsd]). Pages backward via `after`.
    """
    now_ms = int(time.time() * 1000)
    start = now_ms - int(days * 86_400_000)
    hosts = [host] if host else list(HOSTS)
    url_path = "/api/v5/rubik/stat/contracts/open-interest-history"
    rows: list = []
    after: Optional[int] = None
    last_oldest: Optional[int] = None
    with httpx.Client(timeout=25.0, headers={"User-Agent": "okxb-research/1"}) as cli:
        for _ in range(200):
            data = None
            for _attempt in range(3):                      # retry a page before giving up (transient net)
                for h in hosts:
                    try:
                        params = {"instId": inst_id, "period": period, "limit": "100"}
                        if after is not None:
                            params["after"] = str(after)
                        j = cli.get(h + url_path, params=params).json()
                        if j.get("code") == "0":
                            data = j.get("data", [])
                            break
                    except Exception:  # noqa: BLE001
                        continue
                if data is not None:
                    break
                time.sleep(0.3)
            if not data:
                break
            rows.extend(data)
            oldest = min(int(x[0]) for x in data)
            if oldest <= start or len(data) < 2:
                break
            # OKX rubik OI-history retains only the recent window and ignores `after`/`before`;
            # if paging no longer reaches older data, stop instead of re-fetching the same page.
            if last_oldest is not None and oldest >= last_oldest:
                break
            last_oldest = oldest
            after = oldest
            time.sleep(0.1)
    df = parse_oi_rows(rows)
    return df[df["ts"] >= start].reset_index(drop=True)


def perp_to_spot(inst_id: str) -> str:
    """永续 instId → 现货 instId。BTC-USDT-SWAP → BTC-USDT。"""
    return inst_id[:-5] if inst_id.endswith("-SWAP") else inst_id


if __name__ == "__main__":   # 快速自检: 拉 BTC 近 2 天 5m
    root = Path(__file__).resolve().parents[3] / "dist" / "candles"
    d = get_candles("BTC-USDT-SWAP", "5m", 2, root, force=True)
    print(d.tail())
    print("rows", len(d), "span_days",
          (d["ts"].iloc[-1] - d["ts"].iloc[0]) / 86_400_000 if len(d) else 0)
