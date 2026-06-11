"""自动选标的 (按流动性)。

- 加密永续: USDT 本位 SWAP, 按 24h 报价额 (volCcy24h × last) 排序取 Top-N, 过最低成交量阈值。
- 股票永续: OKX 无专门字段区分, 用股票代码表识别; 成交量本就低, 不按量卡 (由盘口准入门控过滤)。
更多(流动)标的 = 超短线更多出手机会; 但只纳入活跃的, 避免低流动山寨拖累。
"""
from __future__ import annotations

from ..config import Config
from ..exchange.okx_rest import OkxRestClient


def _quote_vol(t: dict) -> float:
    try:
        return float(t.get("volCcy24h", 0) or 0) * float(t.get("last", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _base(inst_id: str) -> str:
    return inst_id.split("-")[0].upper()


async def build_universe(rest: OkxRestClient, config: Config) -> tuple[list[str], dict]:
    """返回 (标的列表, 诊断信息)。mode=fixed 时用配置的优先列表。"""
    u = config.section("universe")
    if str(u.get("mode", "auto")).lower() == "fixed":
        uni = list(u.get("crypto_priority", []) or []) + list(u.get("stock_perp_priority", []) or [])
        return uni, {"mode": "fixed", "n": len(uni)}

    insts = await rest.get_instruments("SWAP")
    tickers = await rest.get_tickers("SWAP")
    vol = {t["instId"]: _quote_vol(t) for t in tickers}
    stock_syms = {s.upper() for s in (u.get("stock_symbols", []) or [])}
    # 用 "-USDT-SWAP 结尾 + live" 作为筛选 (与『验证』里已验证可行的判据一致);
    # 不再要求 settleCcy=='USDT' —— OKX 股票/TradFi 永续的 settleCcy 字段可能不同, 会误删全部股票永续。
    live = [i for i in insts if i.get("state") == "live" and i["instId"].endswith("-USDT-SWAP")]

    # 股票永续: 代码命中 + live; 按量排序取前 max_stock_perp (股票量本就低, min 默认0)
    stocks = [i["instId"] for i in live if _base(i["instId"]) in stock_syms]
    min_stock = float(u.get("min_stock_quote_vol_usd", 0))
    stocks = [s for s in stocks if vol.get(s, 0) >= min_stock]
    stocks = sorted(stocks, key=lambda x: vol.get(x, 0), reverse=True)[:int(u.get("max_stock_perp", 25))]

    # 加密: 排除股票代码, 过最低量阈值, 按量取前 max_crypto
    min_crypto = float(u.get("min_crypto_quote_vol_usd", 20_000_000))
    crypto = [i["instId"] for i in live if _base(i["instId"]) not in stock_syms]
    crypto = [c for c in crypto if vol.get(c, 0) >= min_crypto]
    crypto = sorted(crypto, key=lambda x: vol.get(x, 0), reverse=True)[:int(u.get("max_crypto", 30))]

    uni = crypto + stocks
    diag = {"mode": "auto", "crypto": len(crypto), "stock": len(stocks),
            "matched_stock": len([i for i in live if _base(i["instId"]) in stock_syms]),
            "top_crypto": crypto[:5], "stock_list": stocks}
    return uni, diag
