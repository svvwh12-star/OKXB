"""微观结构因子 (纯函数)。

重要 (见 RESEARCH_BRIEF §7): OBI / OFI / TradeImbalance 高度相关, 是同一个
"流动性压力"因子的不同侧面。SignalService 必须按"一个因子"对待, 不可叠加放大信心。
"""
from __future__ import annotations

from ..core.models import BBO, Trade
from ..marketdata.orderbook import OrderBook


def obi(book: OrderBook, levels: int = 5) -> float | None:
    """盘口失衡 OBI = (bid量 - ask量)/(bid量 + ask量), 取前 levels 档。范围 [-1, 1]。"""
    snap = book.snapshot(levels)
    if not snap or not snap.bids or not snap.asks:
        return None
    bid_sz = sum(l.sz for l in snap.bids)
    ask_sz = sum(l.sz for l in snap.asks)
    tot = bid_sz + ask_sz
    return (bid_sz - ask_sz) / tot if tot else None


def spread_bps(bbo: BBO) -> float | None:
    return bbo.spread_bps if bbo and bbo.mid else None


def depth_5bps(book: OrderBook) -> float | None:
    """双边 5bps 内名义深度之和 (USDT 近似)。"""
    b = book.depth_notional("bid", 5.0)
    a = book.depth_notional("ask", 5.0)
    return b + a if (b or a) else None


def trade_imbalance(trades: list[Trade], notional_floor: float = 0.0) -> float | None:
    """主动成交失衡 = (主买量 - 主卖量)/总量。范围 [-1, 1]。
    notional_floor>0: 按本窗成交名义额做 James-Stein 式收缩 ti×notion/(notion+floor),
    防薄量单笔把 TI 钉死±1 (淡市一两笔小单不该给满信心方向)。默认0=不收缩(行为不变), 待 Stage B 校准。"""
    if not trades:
        return None
    buy = sum(t.sz for t in trades if t.side.value == "buy")
    sell = sum(t.sz for t in trades if t.side.value == "sell")
    tot = buy + sell
    if not tot:
        return None
    ti = (buy - sell) / tot
    if notional_floor > 0.0:
        notion = sum(t.sz * t.px for t in trades)        # 近似名义额(报价币)
        ti *= notion / (notion + notional_floor)
    return ti


def ofi_tick(prev: BBO | None, cur: BBO | None) -> float | None:
    """经典 L1 订单流失衡 OFI (Cont-Kukanov-Stoikov 2014), 按当前 L1 深度归一化。
       e = [价升→+新买量 / 价降→−旧买量 / 价平→Δ买量] − [价降→+新卖量 / 价升→−旧卖量 / 价平→Δ卖量]
    不再把主动成交量加进来 (市价单冲击已被最优队列变化捕捉, 避免重复计入);
    主动成交失衡由独立的 trade_imbalance 因子表达。"""
    if prev is None or cur is None:
        return None
    if cur.bid_px > prev.bid_px:
        d_bid = cur.bid_sz
    elif cur.bid_px < prev.bid_px:
        d_bid = -prev.bid_sz
    else:
        d_bid = cur.bid_sz - prev.bid_sz
    if cur.ask_px < prev.ask_px:
        d_ask = cur.ask_sz
    elif cur.ask_px > prev.ask_px:
        d_ask = -prev.ask_sz
    else:
        d_ask = cur.ask_sz - prev.ask_sz
    e = d_bid - d_ask
    depth = (cur.bid_sz + cur.ask_sz) / 2.0       # 按 L1 平均深度归一化(OFI 斜率与深度成反比)
    return e / depth if depth > 0 else e
