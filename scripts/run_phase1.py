#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 1 只读扫描器 (不下单)。
================================================
连接 OKX 公共行情 WS, 实时计算因子与综合评分, 把高分【假设信号】记入
recordings/, 并周期打印面板。用于在投入任何资金前, 验证:
  - 每个标的的盘口/价差/深度分布
  - 综合信号在真实数据上的触发频率与质量
  - (配合 research/labeling) 信号后 30s/60s/180s 的前瞻净收益

运行:  python scripts/run_phase1.py
停止:  Ctrl+C
"""
from __future__ import annotations

import argparse
import asyncio
import json
import signal as sigmod
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from okxb.config import Config, Secrets                       # noqa: E402
from okxb.core.enums import Side, StrategyId                  # noqa: E402
from okxb.features.engine import FeatureEngine                # noqa: E402
from okxb.marketdata.gateway import MarketDataGateway         # noqa: E402
from okxb.signal.composite import CompositeScorer            # noqa: E402

COMPUTE_INTERVAL_S = 0.5
DASHBOARD_EVERY_S = 5.0


def build_universe(cfg: Config) -> list[str]:
    crypto = cfg.get("universe.crypto_priority", []) or []
    stocks = cfg.get("universe.stock_perp_priority", []) or []
    # Phase1 只读: 默认加密 majors + 前两个股票永续 (公共数据无需交易资格)
    return list(crypto) + list(stocks[:2])


async def main(collect_features: bool = False) -> None:
    cfg = Config.load()
    secrets = Secrets()
    universe = build_universe(cfg)
    min_score = float(cfg.get("signal.min_composite_score", 80))

    rec_dir = ROOT / (cfg.get("paths.recordings_dir", "recordings"))
    rec_dir.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time())
    sig_file = rec_dir / f"phase1_signals_{stamp}.jsonl"
    tick_file = rec_dir / f"phase1_ticks_{stamp}.jsonl"
    feat_file = rec_dir / f"phase1_features_{stamp}.jsonl"
    if collect_features:
        print(f"[phase1] 特征采集开启 -> {feat_file} (用于 meta-labeling 训练)")

    print(f"[phase1] 模式={secrets.mode.value} 区域={secrets.region} 标的={universe}")
    print(f"[phase1] 信号阈值 composite>={min_score}; 记录 -> {sig_file}")
    print("[phase1] 连接行情 WS ... (等待快照, 数十秒内填充)\n")

    gw = MarketDataGateway(cfg, secrets, universe)
    fe = FeatureEngine(gw, cfg)
    scorer = CompositeScorer(cfg)

    stop = asyncio.Event()

    def _on_signal(_s=None):
        stop.set()
    try:
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(sigmod.SIGINT, _on_signal)
    except (NotImplementedError, RuntimeError):
        pass  # Windows 上 SIGINT 处理器可能不可用, 退而依赖 KeyboardInterrupt

    gw_task = asyncio.create_task(gw.run())
    n_signals = 0
    last_dash = 0.0

    try:
        while not stop.is_set():
            await asyncio.sleep(COMPUTE_INTERVAL_S)
            rows = []
            ticks = []
            feats = []
            for inst in universe:
                fs = fe.compute(inst)
                res = scorer.score(fs)
                rows.append((inst, fs, res))
                bbo0 = gw.get_bbo(inst) or gw.get_book(inst).bbo()
                if bbo0:
                    ticks.append({"ts": int(time.time() * 1000), "inst": inst,
                                  "mid": bbo0.mid, "spread_bps": round(bbo0.spread_bps, 3)})
                    if collect_features and fs.obi_5_z is not None:
                        side = "buy" if res.long_score >= res.short_score else "sell"
                        sl, tp = scorer.sl_tp(fs)
                        feats.append({
                            "ts": int(time.time() * 1000), "inst": inst, "mid": bbo0.mid,
                            "side": side, "composite": max(res.long_score, res.short_score),
                            "sl_pct": sl, "tp_pct": tp, "cost_pct": scorer.total_cost_pct(fs),
                            "f": {"obi_5_z": fs.obi_5_z, "ofi_z": fs.ofi_z,
                                  "trade_imb_3s": fs.trade_imbalance_3s,
                                  "ret_5s": fs.mid_return_5s, "ret_15s": fs.mid_return_15s,
                                  "ret_60s": fs.mid_return_60s, "spread_bps": fs.spread_bps,
                                  "rvol_60s": fs.realized_vol_60s, "atr_1m": fs.atr_1m}})
                # 记录高分假设信号
                for side, sc in ((Side.BUY, res.long_score), (Side.SELL, res.short_score)):
                    if sc >= min_score:
                        sig = scorer.build_signal(fs, StrategyId.HFM80, side, sc)
                        bbo = gw.get_bbo(inst) or gw.get_book(inst).bbo()
                        rec = {
                            "ts": sig.ts, "inst": inst, "side": side.value,
                            "composite": sc, "model_prob": sig.model_prob,
                            "edge_pct": sig.expected_edge_pct, "cost_pct": sig.total_cost_pct,
                            "edge_to_cost": round(sig.edge_to_cost, 3),
                            "sl_pct": sig.sl_pct, "tp_pct": sig.tp_pct,
                            "mid": bbo.mid if bbo else None,
                            "components": res.components,
                        }
                        with open(sig_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n_signals += 1

            if ticks:
                with open(tick_file, "a", encoding="utf-8") as f:
                    for t in ticks:
                        f.write(json.dumps(t, ensure_ascii=False) + "\n")
            if feats:
                with open(feat_file, "a", encoding="utf-8") as f:
                    for rec in feats:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            now = time.monotonic()
            if now - last_dash >= DASHBOARD_EVERY_S:
                last_dash = now
                _print_dashboard(rows, gw, n_signals)
    finally:
        await gw.stop()
        gw_task.cancel()
        print(f"\n[phase1] 已停止。假设信号 {n_signals} 条 -> {sig_file}")
        print(f"[phase1] tick 记录 -> {tick_file}")
        print(f"[phase1] 用 python scripts/analyze_phase1.py 分析前瞻净收益")


def _print_dashboard(rows, gw, n_signals) -> None:
    age = gw.data_age_ms()
    print(f"--- 面板 (数据老化 {age}ms, 累计信号 {n_signals}) ---")
    print(f"{'标的':<16}{'mid':>12}{'价差bps':>9}{'OBI_z':>8}{'OFI_z':>8}{'多':>7}{'空':>7}")
    for inst, fs, res in rows:
        mid = ""
        bbo = gw.get_bbo(inst) or gw.get_book(inst).bbo()
        if bbo:
            mid = f"{bbo.mid:.4f}"
        sp = f"{fs.spread_bps:.1f}" if fs.spread_bps is not None else "-"
        oz = f"{fs.obi_5_z:.2f}" if fs.obi_5_z is not None else "-"
        fz = f"{fs.ofi_z:.2f}" if fs.ofi_z is not None else "-"
        print(f"{inst:<16}{mid:>12}{sp:>9}{oz:>8}{fz:>8}{res.long_score:>7.0f}{res.short_score:>7.0f}")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase1 只读扫描器")
    ap.add_argument("--collect-features", action="store_true",
                    help="记录每根 bar 的因子, 用于 meta-labeling 训练")
    args = ap.parse_args()
    try:
        asyncio.run(main(args.collect_features))
    except KeyboardInterrupt:
        print("\n[phase1] 中断退出。")
