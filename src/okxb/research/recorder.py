"""逐拍校准录制器 (calibration tick recorder)。

每个运行会话(演练或实盘)把每个标的每一拍的【中间价 + 双向综合分 + 流向/趋势分量 +
波动/价差/成本】落盘为 JSONL。这是校准器 (backtest.py) 的唯一输入。

为何录这几列就够 (而非全量盘口):
  - 入场可对【任意阈值/确认/持续】重放: 用 L/S(双向分) 与 fd/td(流向/趋势分量, 可重算 conf)。
  - 出场可对【任意 止盈RR / 持仓时长 / 反转 / 移动止盈】重放: 用 m(中间价路径) + a(ATR 推 SL) + c(成本)。
  - 既录做多分又录做空分 -> 做多/做空决策点可分别评估。

行格式 (短键, 紧凑):
  t  ts(ms)        i  instId      m  mid
  L  long_score    S  short_score
  fd order_flow_dir td trend_dir  q quality(0-1)
  sp spread_bps    a  atr_1m(小数) c 往返成本(小数, maker入+taker出+半价差+滑点)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


class TickRecorder:
    def __init__(self, enabled: bool, path: Path, flush_every: int = 200):
        self.enabled = bool(enabled)
        self.path = Path(path)
        self.count = 0
        self._fh = None
        self._since_flush = 0
        self._flush_every = flush_every

    def _ensure(self) -> None:
        if self.enabled and self._fh is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8")

    def record(self, inst: str, mid: Optional[float], comp, fs, cost: float) -> None:
        """comp = CompositeResult, fs = FeatureSet。mid 为 None 则跳过(盘口未就绪)。"""
        if not self.enabled or mid is None:
            return
        try:
            self._ensure()
            row = {
                "t": int(fs.ts), "i": inst, "m": round(float(mid), 8),
                "L": comp.long_score, "S": comp.short_score,         # 原始分(诚实记录)
                "fd": comp.order_flow_dir_raw, "td": comp.trend_dir_raw,  # 原始方向分量(供校准重放去噪)
                "tr": comp.tradability,                              # 可交易性[0,1] 已平滑(与实盘门一致)
                "trr": comp.tradability_raw,                         # 可交易性(未平滑, 供重调EMA)
                "sp": round(fs.spread_bps, 3) if fs.spread_bps is not None else None,
                "a": round(fs.atr_1m, 8) if fs.atr_1m is not None else None,
                "c": round(float(cost), 8),
                "rv": round(fs.realized_vol_60s, 8) if fs.realized_vol_60s is not None else None,
                "rvm": round(fs.rv_mid, 8) if getattr(fs, "rv_mid", None) is not None else None,  # 纯中间价波动(对比micro-price虚高)
                # --- Stage B 逐因子(供"底层因子 → 未来收益"重拟合权重; 现在就开始积累) ---
                "ob": round(fs.obi_5_z, 4) if fs.obi_5_z is not None else None,        # OBI z
                "of": round(fs.ofi_z, 4) if fs.ofi_z is not None else None,            # OFI z (逐事件)
                "ti": round(fs.trade_imbalance_3s, 4) if fs.trade_imbalance_3s is not None else None,
                "r5": round(fs.mid_return_5s, 8) if fs.mid_return_5s is not None else None,
                "r15": round(fs.mid_return_15s, 8) if fs.mid_return_15s is not None else None,
                "r60": round(fs.mid_return_60s, 8) if fs.mid_return_60s is not None else None,
                "dp": round(fs.depth_5bps, 2) if fs.depth_5bps is not None else None,  # 5bps名义深度
                "bz": round(fs.basis_z, 4) if getattr(fs, "basis_z", None) is not None else None,  # 股票basis z
            }
            self._fh.write(json.dumps(row, separators=(",", ":")) + "\n")
            self.count += 1
            self._since_flush += 1
            if self._since_flush >= self._flush_every:
                self._fh.flush()
                self._since_flush = 0
        except Exception:
            # 录制绝不能影响交易主循环
            pass

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:
                pass
            self._fh = None
