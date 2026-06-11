#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
策略校准 (命令行)。
================================================
在录制的逐拍数据 (recordings/calib_*.jsonl) 上做三重障碍事件回测, 网格搜索
入场决策点(分/确认/持续) x 出场策略(持仓时长/止盈盈亏比/反转/移动止盈),
按"收益最高""最稳健"给出推荐配置 (做多/做空 分别评估)。

用法:
  python scripts/calibrate.py                 # 用最近的录制文件, 打印报告
  python scripts/calibrate.py --all           # 合并所有 calib_*.jsonl
  python scripts/calibrate.py --apply stable   # 把"最稳健"配置写入 config.yaml
  python scripts/calibrate.py --apply profit   # 把"收益最高"配置写入 config.yaml

诚实提醒: 回测对成交乐观, 实盘更差; 应用后请再跑一段虚拟盘复核。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from okxb.config import Config                                  # noqa: E402
from okxb.research import calibrator as cal                     # noqa: E402
from okxb.research import config_patch                          # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="OKXB 策略校准")
    ap.add_argument("--all", action="store_true", help="合并所有录制文件 (默认仅最近一个)")
    ap.add_argument("--apply", choices=["stable", "profit"], help="把推荐配置写入 config.yaml")
    args = ap.parse_args()

    cfg = Config.load()
    rec_dir = str(ROOT / cfg.get("paths.recordings_dir", "recordings"))
    files = cal.find_recordings(rec_dir)
    if not files:
        print(f"未找到录制文件: {rec_dir}/calib_*.jsonl\n请先在 GUI 或 python -m okxb.app 跑一段虚拟盘。")
        return
    if not args.all:
        files = files[-1:]
    print(f"读取 {len(files)} 个文件 ...")
    by_inst, meta = cal.load_calib(files)
    print(f"  {meta['n_rows']} 行 / {meta['n_inst']} 标的"
          + (f" (抽稀 1/{meta['stride']})" if meta["stride"] > 1 else ""))
    if not by_inst:
        print("数据太少, 无法校准。")
        return

    ccfg = cfg.section("research").get("calibrate", {}) if cfg.section("research") else {}
    min_trades = int(ccfg.get("min_trades", 25))
    cost_mult = float(ccfg.get("cost_haircut_mult", 1.0))
    maker_fill = bool(ccfg.get("maker_fill", True))
    cooldown = float(cfg.get("signal.cooldown_seconds", 20))

    sl_cost_mult = float(cfg.get("signal.sl_min_cost_mult", 2.5))
    min_edge = float(cfg.get("signal.min_edge_to_cost_ratio", 1.2))
    sm = {"hl_f": float(cfg.get("signal.ema_flow_half_life_s", 1.0)),
          "hl_t": float(cfg.get("signal.ema_trend_half_life_s", 2.0)),
          "hl_s": float(cfg.get("signal.ema_score_half_life_s", 0.5)),
          "enter": float(cfg.get("signal.dir_hyst_enter", 0.12)),
          "exit": float(cfg.get("signal.dir_hyst_exit", 0.04)),
          "miss_grace": int(cfg.get("signal.persist_miss_grace", 1)),
          "rv_lo": float(cfg.get("signal.regime_rv_lo", 2e-4)),
          "rv_hi": float(cfg.get("signal.regime_rv_hi", 1.2e-3)),
          "hv_scale": float(cfg.get("signal.regime_hv_alpha_scale", 0.6)),
          "persist_bonus": int(cfg.get("signal.regime_persist_bonus", 1)), "dt": 0.5}

    def prog(i, n):
        print(f"  网格 {i}/{n} ...", end="\r")
    print("回测中 (训练段网格 + 样本外 + 走步 + DSR/PBO) ...")
    calib = cal.run_calibration(by_inst, cal.Grid(), min_trades, cooldown_s=cooldown,
                                cost_mult=cost_mult, maker_fill=maker_fill,
                                sl_cost_mult=sl_cost_mult, min_edge_cost=min_edge, sm=sm, progress=prog)
    print()
    picks = calib["picks"]
    print(cal.format_report_oos(meta, calib, min_trades))

    if args.apply:
        chosen = picks["best_stable"] if args.apply == "stable" else picks["best_profit"]
        if not chosen:
            print("\n没有可应用的盈利配置。")
            return
        updates = cal.result_to_signal_cfg(chosen)
        path = ROOT / "config" / "config.yaml"
        summary = config_patch.apply_updates(path, updates)
        print(f"\n✓ 已写入 {path}:")
        for s in summary:
            print(f"    {s}")
        print("重启引擎生效。建议再跑一段虚拟盘复核。")


if __name__ == "__main__":
    main()
