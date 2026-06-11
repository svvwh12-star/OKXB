"""因子引擎: 维护每标的滚动状态, 产出 FeatureSet。

设计: 由扫描循环按节流频率 (如 250-500ms) 调用 compute(inst_id); 引擎从 gateway
读当前盘口/成交, 计算原始因子 + 相对滚动窗口的 z-score, 并把原始值压入历史。
"""
from __future__ import annotations

import math
import time
from collections import deque

from ..config import Config
from ..core.models import BBO, FeatureSet
from ..marketdata.gateway import MarketDataGateway
from ..risk.engine import is_stock_perp
from . import microstructure as ms
from . import volatility as vol


def _median(vals: list[float]) -> float:
    n = len(vals)
    return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0


def _zscore(history: deque, x: float | None, sd_floor: float = 0.0) -> float | None:
    """稳健 z-score: 中位数/MAD (对非平稳HF数据和异常值更稳), 缩尾到 |z|<=4。
    MAD≈0 时回退普通均值/标准差。
    sd_floor: 绝对离散度地板 —— 淡市里 MAD 极小会把微小变化放大成大 z(clamp4→tanh2≈0.96 假满信号);
    给一个因子量级的下限, 使"要真有 sd_floor 量级的绝对变化"才算强信号 (与趋势路径的 σ 地板对称)。"""
    if x is None or len(history) < 10:
        return None
    vals = sorted(history)
    med = _median(vals)
    mad = _median(sorted(abs(v - med) for v in vals))
    sd = 1.4826 * mad
    if sd > 1e-12:
        sd = max(sd, sd_floor)                          # 绝对地板, 防淡市 z 爆表
        z = (x - med) / sd
    else:
        n = len(history)
        mu = sum(history) / n
        var = sum((v - mu) ** 2 for v in history) / (n - 1)
        sd2 = max(math.sqrt(var), sd_floor)
        if sd2 <= 1e-12:
            return 0.0
        z = (x - mu) / sd2
    return max(-4.0, min(4.0, z))


class FeatureEngine:
    def __init__(self, gateway: MarketDataGateway, config: Config,
                 mid_window: int = 600, factor_window: int = 200):
        self._gw = gateway
        self._cfg = config
        self._mids: dict[str, deque] = {}          # (ts_ms, micro-price) 实盘消费
        self._mids_plain: dict[str, deque] = {}    # (ts_ms, 纯中间价) 仅供录制 rv_mid 对比
        self._obi_hist: dict[str, deque] = {}
        self._ofi_hist: dict[str, deque] = {}
        self._last_bbo: dict[str, BBO] = {}
        self._last_trade_ts: dict[str, int] = {}
        self._marks: dict[str, float] = {}          # 标记/指数价 (供 basis); 由 app 周期注入
        self._basis_hist: dict[str, deque] = {}
        self._mid_window = mid_window
        self._fw = factor_window
        # 因子 z-score 绝对离散度地板 (防淡市 z 爆表; OFI/basis 量级未定→默认0=不变, 待Stage B校准)
        sg = config.section("signal")
        self._obi_sd_floor = float(sg.get("obi_sd_floor", 0.05))
        self._ofi_sd_floor = float(sg.get("ofi_sd_floor", 0.0))
        self._basis_sd_floor = float(sg.get("basis_sd_floor", 0.0))
        self._ti_notional_floor = float(sg.get("ti_notional_floor", 0.0))  # 0=不变

    def set_marks(self, marks: dict[str, float]) -> None:
        self._marks = marks or {}

    def _hist(self, store: dict, inst_id: str, maxlen: int) -> deque:
        d = store.get(inst_id)
        if d is None:
            d = deque(maxlen=maxlen)
            store[inst_id] = d
        return d

    def _bbo(self, inst_id: str) -> BBO | None:
        return self._gw.get_bbo(inst_id) or self._gw.get_book(inst_id).bbo()

    def _mid_return(self, inst_id: str, horizon_ms: int) -> float | None:
        hist = self._mids.get(inst_id)
        if not hist or len(hist) < 2:
            return None
        now_ts, now_mid = hist[-1]
        target = now_ts - horizon_ms
        past_mid = None
        for ts, mid in reversed(hist):
            if ts <= target:
                past_mid = mid
                break
        if past_mid is None or past_mid <= 0:
            return None
        return now_mid / past_mid - 1.0

    def compute(self, inst_id: str) -> FeatureSet:
        now = int(time.time() * 1000)
        book = self._gw.get_book(inst_id)
        bbo = self._bbo(inst_id)
        fs = FeatureSet(inst_id=inst_id, ts=now)
        if bbo is None or not book.ready:
            return fs

        # 价格历史用 micro-price (size-weighted mid): 趋势/波动更少受单边报价抖动污染
        mids = self._hist(self._mids, inst_id, self._mid_window)
        mids.append((now, bbo.microprice))
        mid_vals = [m for _, m in mids]
        pmids = self._hist(self._mids_plain, inst_id, self._mid_window)   # 纯中间价序列(供录制 rv_mid 对比)
        pmids.append((now, bbo.mid))

        # 微观结构
        fs.obi_5 = ms.obi(book, 5)
        obi_h = self._hist(self._obi_hist, inst_id, self._fw)
        if fs.obi_5 is not None:
            fs.obi_5_z = _zscore(obi_h, fs.obi_5, self._obi_sd_floor)
            obi_h.append(fs.obi_5)

        fs.spread_bps = ms.spread_bps(bbo)
        fs.depth_5bps = ms.depth_5bps(book)

        recent = self._gw.recent_trades(inst_id, 3000)
        fs.trade_imbalance_3s = ms.trade_imbalance(recent, self._ti_notional_floor)

        # OFI: 经典 L1 (Cont), 逐事件累计 (网关在每次盘口更新累加, 已深度归一化), 取走本拍窗内总流;
        # 不再混入主动成交量(避免重复计), 也不再是"仅0.5s首尾两点之差"。
        ofi = self._gw.take_ofi(inst_id)
        ofi_h = self._hist(self._ofi_hist, inst_id, self._fw)
        if ofi is not None:
            fs.ofi = ofi
            fs.ofi_z = _zscore(ofi_h, ofi, self._ofi_sd_floor)
            ofi_h.append(ofi)

        # 趋势
        fs.mid_return_5s = self._mid_return(inst_id, 5000)
        fs.mid_return_15s = self._mid_return(inst_id, 15000)
        fs.mid_return_60s = self._mid_return(inst_id, 60000)

        # 波动
        fs.realized_vol_60s = vol.realized_vol(mid_vals[-120:])
        fs.rv_mid = vol.realized_vol([m for _, m in pmids][-120:])   # 纯中间价波动(对比micro-price是否虚高)
        fs.atr_1m = vol.atr_proxy(mid_vals, 120)

        # 股票永续 basis: perp 相对标记/指数价的偏离 + 滚动 z-score
        if is_stock_perp(inst_id):
            mark = self._marks.get(inst_id)
            if mark and mark > 0:
                basis = bbo.mid / mark - 1.0
                fs.basis_index = basis
                bh = self._hist(self._basis_hist, inst_id, self._fw)
                fs.basis_z = _zscore(bh, basis, self._basis_sd_floor)
                bh.append(basis)

        return fs
