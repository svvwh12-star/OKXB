"""meta-labeling 训练集构建。

输入: run_phase1 --collect-features 产出的 phase1_features_*.jsonl + phase1_ticks_*.jsonl
标签: 对每根 bar 按其方向做 triple-barrier, TP 先于 SL = 1 (值得做), 否则 0。
特征: 原始因子 + 触发分 (composite)。
meta-labeling 思想 (Lopez de Prado): 主信号给方向, 二级模型判"该不该做 + 多大注"。
"""
from __future__ import annotations

import json
from bisect import bisect_left
from pathlib import Path
from typing import Optional

from . import labeling as lb

# 特征列的规范顺序 (训练与推理必须一致)
FEATURE_COLS = ["obi_5_z", "ofi_z", "trade_imb_3s", "ret_5s", "ret_15s", "ret_60s",
                "spread_bps", "rvol_60s", "atr_1m", "composite"]
MAX_HOLD_MS = 180_000


def vec_from_logged(row: dict) -> Optional[list[float]]:
    """从 phase1_features 行抽取特征向量 (缺关键因子返回 None)。"""
    f = row.get("f", {})
    if f.get("obi_5_z") is None or f.get("ofi_z") is None:
        return None
    vals = {**f, "composite": row.get("composite", 0.0)}
    return [float(vals.get(c) if vals.get(c) is not None else 0.0) for c in FEATURE_COLS]


def vec_from_featureset(fs, composite: float) -> list[float]:
    """推理期: 从 FeatureSet 抽取同序特征向量。"""
    m = {
        "obi_5_z": fs.obi_5_z, "ofi_z": fs.ofi_z, "trade_imb_3s": fs.trade_imbalance_3s,
        "ret_5s": fs.mid_return_5s, "ret_15s": fs.mid_return_15s, "ret_60s": fs.mid_return_60s,
        "spread_bps": fs.spread_bps, "rvol_60s": fs.realized_vol_60s, "atr_1m": fs.atr_1m,
        "composite": composite,
    }
    return [float(m.get(c) if m.get(c) is not None else 0.0) for c in FEATURE_COLS]


def _load_ticks(path: Path) -> dict[str, tuple[list[int], list[float]]]:
    by_inst: dict[str, tuple[list[int], list[float]]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            ts_list, mids = by_inst.setdefault(d["inst"], ([], []))
            ts_list.append(int(d["ts"]))
            mids.append(float(d["mid"]))
    return by_inst


def build_dataset(features_path: Path, ticks_path: Path
                  ) -> tuple[list[list[float]], list[int], list[dict]]:
    """返回 (X, y, rows)。y=1 表示该方向 TP 先于 SL。按 ts 升序。"""
    ticks = _load_ticks(ticks_path)
    rows: list[dict] = []
    with open(features_path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    rows.sort(key=lambda r: r["ts"])

    X: list[list[float]] = []
    y: list[int] = []
    kept: list[dict] = []
    for r in rows:
        inst = r["inst"]
        if inst not in ticks:
            continue
        vec = vec_from_logged(r)
        if vec is None:
            continue
        ts_list, mids = ticks[inst]
        idx = bisect_left(ts_list, int(r["ts"]))
        if idx >= len(ts_list) - 2:
            continue
        side = 1 if r.get("side") == "buy" else -1
        br = lb.triple_barrier(ts_list, mids, idx, side,
                               float(r.get("tp_pct", 0.004)),
                               float(r.get("sl_pct", 0.002)), MAX_HOLD_MS)
        X.append(vec)
        y.append(1 if br.label == 1 else 0)
        kept.append(r)
    return X, y, kept
