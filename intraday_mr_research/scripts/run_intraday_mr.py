"""预登记 15/30 分钟日内均值回归候选 (IMR) 的前向采集 + 评估 runner。

用法 (开发态; 公共行情免密钥):
  python run_intraday_mr.py --mode collect     # 抓最新bar -> 追加 look-ahead-safe 标签到哈希链 (首次自动冻结)
  python run_intraday_mr.py --mode evaluate    # 仅用冻结后样本判决 (>=100 独立前向ts + 全部闸门; sticky KILL)
  python run_intraday_mr.py --mode status      # 看进度

纪律: 冻结后【绝不】改规则; 只采集 + 周期评估。最可能长期 PENDING 或 KILL —— 那就是诚实答案。
打包 .exe 下 sys.executable 不是 python, 无法跑本脚本; 请用开发态 python。
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]            # intraday_mr_research/
SRC = ROOT.parent / "src"
sys.path.insert(0, str(SRC))

from okxb.config import Config, Secrets                                  # noqa: E402
from okxb.exchange.okx_rest import OkxRestClient                         # noqa: E402
from okxb.research import intraday_mr as imr                             # noqa: E402
from okxb.research.forward_integrity import (append_rows_hashchain,      # noqa: E402
                                             read_dead, verify_hashchain,
                                             write_dead, write_manifest)
from okxb.research.labeling import pbo_cscv                              # noqa: E402

FROZEN = ROOT / "frozen"
META = FROZEN / "meta.json"
LEDGER = ROOT / "data" / "forward_append_only" / "imr_ledger_hashchain.csv"
STATUS = ROOT / "data" / "forward_append_only" / "imr_status_hashchain.csv"
REPORT = ROOT / "reports" / "imr_status.md"


def _now_ms() -> int:
    return int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)


async def _fetch_recent(rest, inst: str, bar: str, limit: int = 300) -> list:
    try:
        return imr.normalize_candles(await rest.get_candles(inst, bar=bar, limit=limit))
    except Exception as e:  # noqa: BLE001
        print(f"  [{inst} {bar}] 取最新bar失败: {e!r}")
        return []


async def _fetch_history(rest, inst: str, bar: str, want: int = 1000) -> list:
    """分页拉历史 (get_history_candles, 每页<=100), 返回 [(ts,close)] 旧->新。"""
    acc: dict[int, float] = {}
    after = None
    try:
        for _ in range(want // 100 + 1):
            page = await rest.get_history_candles(inst, bar=bar, after=after, limit=100)
            norm = imr.normalize_candles(page)
            if not norm:
                break
            for ts, c in norm:
                acc[ts] = c
            after = str(min(t for t, _ in norm))      # 继续向更早翻
            if len(acc) >= want:
                break
    except Exception as e:  # noqa: BLE001
        print(f"  [{inst} {bar}] 取历史失败(用已得 {len(acc)} 根): {e!r}")
    return sorted(acc.items())


def _load_meta() -> dict | None:
    if META.exists():
        try:
            return json.loads(META.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


async def _freeze(rest) -> dict:
    """首次冻结: 记录 official_forward_start=now, 并用历史估出每候选的训练期 net15/ic (供衰减闸门)。"""
    print("[freeze] 首次冻结候选 (记录 official_forward_start + 训练期基准)...")
    now = _now_ms()
    cands = {}
    for inst in imr.UNIVERSE:
        for hz in imr.HORIZONS:
            code = imr.code_of(inst, hz)
            hist = await _fetch_history(rest, inst, hz, want=1000)
            labs = [lb for lb in imr.iter_labels(hist) if lb.bar_ts <= now]
            nets = [lb.net15_bps for lb in labs]
            m = (sum(nets) / len(nets)) if nets else 0.0
            cands[code] = {"inst": inst, "horizon": hz, "train_n": len(nets),
                           "train_net15_mean": round(m, 4),
                           "train_ic_sign": 1 if m > 0 else (-1 if m < 0 else 0)}
            print(f"  {code}: 训练 {len(nets)} 笔, net15均值≈{m:.2f}bps")
    meta = {"official_forward_start_ts_ms": now,
            "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "params": {"window": imr.Z_WINDOW, "enter": imr.Z_ENTER, "hold": imr.HOLD_BARS,
                       "cost_stress_bps": imr.COST_BPS_STRESS, "cost_mild_bps": imr.COST_BPS_MILD,
                       "universe": imr.UNIVERSE, "horizons": list(imr.HORIZONS),
                       "family_trials": imr.FAMILY_TRIALS, "min_fwd_ts": imr.MIN_FWD_TS},
            "candidates": cands}
    FROZEN.mkdir(parents=True, exist_ok=True)
    META.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    write_manifest(FROZEN, artifacts=("meta.json",))    # 不可变性: 改 meta 即断 manifest
    print(f"[freeze] 完成, official_forward_start={meta['frozen_utc']}")
    return meta


def _existing_ts() -> set:
    seen = set()
    if LEDGER.exists():
        with LEDGER.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                seen.add((r.get("code"), r.get("bar_ts")))
    return seen


async def _collect(rest, meta: dict) -> None:
    start = int(meta["official_forward_start_ts_ms"])
    seen = _existing_ts()
    rows = []
    for inst in imr.UNIVERSE:
        for hz in imr.HORIZONS:
            code = imr.code_of(inst, hz)
            candles = await _fetch_recent(rest, inst, hz, limit=300)
            new = 0
            for lb in imr.iter_labels(candles):
                if lb.bar_ts <= start or (code, str(lb.bar_ts)) in seen:
                    continue
                rows.append({"code": code, "inst": inst, "horizon": hz, "bar_ts": lb.bar_ts,
                             "direction": lb.direction, "z": round(lb.z, 4),
                             "net15_bps": round(lb.net15_bps, 4), "net10_bps": round(lb.net10_bps, 4),
                             "entry_px": lb.entry_px, "exit_px": lb.exit_px, "asof_ms": _now_ms()})
                seen.add((code, str(lb.bar_ts)))
                new += 1
            print(f"  {code}: 新增 {new} 笔前向标签")
    if rows:
        rows.sort(key=lambda r: r["bar_ts"])
        append_rows_hashchain(LEDGER, rows)
    print(f"[collect] 本轮新增 {len(rows)} 笔 -> {LEDGER.name} (链校验: {verify_hashchain(LEDGER) or 'OK'})")


def _read_forward(meta: dict) -> dict:
    """读哈希链, 按 code 分组冻结后(direction!=0)的 net 序列 (按时间)。"""
    start = int(meta["official_forward_start_ts_ms"])
    by_code: dict[str, list] = {}
    if not LEDGER.exists():
        return by_code
    with LEDGER.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                if int(r["bar_ts"]) <= start or int(r["direction"]) == 0:
                    continue
                by_code.setdefault(r["code"], []).append(
                    (int(r["bar_ts"]), float(r["net15_bps"]), float(r["net10_bps"])))
            except (ValueError, TypeError, KeyError):
                continue
    for code in by_code:
        by_code[code].sort(key=lambda x: x[0])
    return by_code


def _evaluate(meta: dict) -> None:
    by_code = _read_forward(meta)
    nets15 = {c: [x[1] for x in rows] for c, rows in by_code.items()}
    pbo_res = pbo_cscv({c: v for c, v in nets15.items() if len(v) >= 16})
    pbo = pbo_res[0] if pbo_res else None
    cands = meta.get("candidates", {})
    lines = ["# IMR 15/30min 前向状态", "",
             f"official_forward_start: {meta.get('frozen_utc')}",
             f"family_trials={imr.FAMILY_TRIALS}  PASS需≥{imr.MIN_FWD_TS}独立前向ts  "
             f"pbo(跨候选)={'' if pbo is None else round(pbo,3)}", "",
             "| code | n_ts | net15 | net10 | t | DSR | PF | verdict | reason |",
             "|---|---:|---:|---:|---:|---:|---:|---|---|"]
    status_rows = []
    for code, meta_c in cands.items():
        rows = by_code.get(code, [])
        net15 = [x[1] for x in rows]
        net10 = [x[2] for x in rows]
        dead = read_dead(FROZEN, code) is not None
        v = imr.evaluate_candidate(net15, net10, train_net15=meta_c.get("train_net15_mean"),
                                   train_ic_sign=int(meta_c.get("train_ic_sign", 0)),
                                   pbo=pbo, already_dead=dead)
        if v.verdict == "KILL" and not dead:
            write_dead(FROZEN, code, v.reason, v.metrics)      # sticky
        m = v.metrics

        def s(x):
            return "" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))
        lines.append(f"| {code} | {m['n_ts']} | {s(m['net15_mean_bps'])} | {s(m['net10_mean_bps'])} | "
                     f"{s(m.get('t'))} | {s(m.get('dsr'))} | {s(m.get('pf'))} | {v.verdict} | {v.reason} |")
        status_rows.append({"code": code, "asof_ms": _now_ms(), "verdict": v.verdict,
                            "reason": v.reason, **{k: m.get(k) for k in
                            ("n_ts", "net15_mean_bps", "net10_mean_bps", "t", "dsr", "pf")}})
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if status_rows:
        append_rows_hashchain(STATUS, status_rows)
    print("\n".join(lines))
    print(f"\n[evaluate] -> {REPORT}")


async def _amain() -> int:
    ap = argparse.ArgumentParser(description="IMR 15/30min 前向采集/评估")
    ap.add_argument("--mode", choices=["collect", "evaluate", "status"], default="collect")
    args = ap.parse_args()
    meta = _load_meta()
    if args.mode == "status":
        if not meta:
            print("尚未冻结 (先跑一次 --mode collect)。")
            return 0
        by_code = _read_forward(meta)
        print(f"official_forward_start={meta.get('frozen_utc')}")
        for code in meta.get("candidates", {}):
            n = len(by_code.get(code, []))
            dead = "DEAD" if read_dead(FROZEN, code) else ""
            print(f"  {code}: 已采集 {n}/{imr.MIN_FWD_TS} 独立前向ts {dead}")
        print(f"链校验: {verify_hashchain(LEDGER) or 'OK'}")
        return 0
    rest = OkxRestClient(Secrets(), Config.load())
    try:
        if meta is None and args.mode == "collect":
            meta = await _freeze(rest)
        if meta is None:
            print("尚未冻结, 请先 --mode collect。")
            return 2
        if args.mode == "collect":
            await _collect(rest, meta)
        else:
            _evaluate(meta)
    finally:
        await rest.aclose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_amain()))
