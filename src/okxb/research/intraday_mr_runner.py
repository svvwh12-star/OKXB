"""IMR 15/30min 候选的【进程内】采集/评估/状态 —— 供 GUI 按钮与 CLI 共用。

为何进程内: 打包成 .exe 后 sys.executable 不是 python, 无法 subprocess 跑脚本(多周期页就有此限制);
本模块直接在 app 进程里跑, 按钮即可用。工件写到 paths.APP_DIR/data/intraday_mr (打包态=exe 同级
data/, 开发态=项目 data/, 均已被 .gitignore 忽略)。OKX 公共行情免密钥。
"""
from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path
from typing import Optional

from .. import paths
from ..config import Config, Secrets
from ..exchange.okx_rest import OkxRestClient
from . import intraday_mr as imr
from .forward_integrity import (append_rows_hashchain, read_dead, verify_hashchain,
                                write_dead, write_manifest)
from .labeling import pbo_cscv


def _base() -> Path:
    b = paths.APP_DIR / "data" / "intraday_mr"
    b.mkdir(parents=True, exist_ok=True)
    return b


def _frozen() -> Path:
    return _base() / "frozen"


def _meta_path() -> Path:
    return _frozen() / "meta.json"


def _ledger() -> Path:
    return _base() / "imr_ledger_hashchain.csv"


def _status_chain() -> Path:
    return _base() / "imr_status_hashchain.csv"


def _report() -> Path:
    return _base() / "imr_status.md"


def _now_ms() -> int:
    return int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)


def _new_rest():
    return OkxRestClient(Secrets(), Config.load())


def _load_meta() -> Optional[dict]:
    p = _meta_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


async def _fetch_recent(rest, inst: str, bar: str, limit: int = 300) -> list:
    try:
        return imr.normalize_candles(await rest.get_candles(inst, bar=bar, limit=limit))
    except Exception as e:  # noqa: BLE001
        print(f"[imr] {inst} {bar} 取最新bar失败: {e!r}")
        return []


async def _fetch_history(rest, inst: str, bar: str, want: int = 1000) -> list:
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
            after = str(min(t for t, _ in norm))
            if len(acc) >= want:
                break
    except Exception as e:  # noqa: BLE001
        print(f"[imr] {inst} {bar} 取历史失败(用已得 {len(acc)} 根): {e!r}")
    return sorted(acc.items())


async def _freeze(rest) -> dict:
    now = _now_ms()
    cands = {}
    for inst in imr.UNIVERSE:
        for hz in imr.HORIZONS:
            code = imr.code_of(inst, hz)
            hist = await _fetch_history(rest, inst, hz, want=1000)
            nets = [lb.net15_bps for lb in imr.iter_labels(hist) if lb.bar_ts <= now]
            m = (sum(nets) / len(nets)) if nets else 0.0
            cands[code] = {"inst": inst, "horizon": hz, "train_n": len(nets),
                           "train_net15_mean": round(m, 4),
                           "train_ic_sign": 1 if m > 0 else (-1 if m < 0 else 0)}
    meta = {"official_forward_start_ts_ms": now,
            "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "params": {"window": imr.Z_WINDOW, "enter": imr.Z_ENTER, "hold": imr.HOLD_BARS,
                       "cost_stress_bps": imr.COST_BPS_STRESS, "cost_mild_bps": imr.COST_BPS_MILD,
                       "universe": imr.UNIVERSE, "horizons": list(imr.HORIZONS),
                       "family_trials": imr.FAMILY_TRIALS, "min_fwd_ts": imr.MIN_FWD_TS},
            "candidates": cands}
    _frozen().mkdir(parents=True, exist_ok=True)
    _meta_path().write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    write_manifest(_frozen(), artifacts=("meta.json",))
    return meta


def _existing_ts() -> set:
    seen = set()
    p = _ledger()
    if p.exists():
        with p.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                seen.add((r.get("code"), r.get("bar_ts")))
    return seen


async def collect() -> str:
    """抓最新 bar -> 追加 look-ahead-safe 前向标签到哈希链 (首次自动冻结)。返回可读摘要。"""
    rest = _new_rest()
    out = []
    try:
        meta = _load_meta()
        if meta is None:
            out.append("首次运行: 冻结候选 (记录 official_forward_start + 训练基准)...")
            meta = await _freeze(rest)
            out.append(f"已冻结 official_forward_start = {meta['frozen_utc']}")
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
                out.append(f"  {code}: 新增 {new} 笔")
        if rows:
            rows.sort(key=lambda r: r["bar_ts"])
            append_rows_hashchain(_ledger(), rows)
        out.append(f"本轮新增 {len(rows)} 笔前向标签 (链校验: {verify_hashchain(_ledger()) or 'OK'})")
        out.append("提示: 15m/30m fade 扣成本多半为负; 攒满 ≥100 独立ts 再点『评估判决』。")
    finally:
        await rest.aclose()
    return "\n".join(out)


def _read_forward(meta: dict) -> dict:
    start = int(meta["official_forward_start_ts_ms"])
    by_code: dict[str, list] = {}
    p = _ledger()
    if not p.exists():
        return by_code
    with p.open(newline="", encoding="utf-8") as f:
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


def evaluate() -> str:
    """仅用冻结后样本判决 (>=100 独立前向ts + 全部闸门; sticky KILL)。返回报告文本。"""
    meta = _load_meta()
    if meta is None:
        return "尚未冻结 — 请先点『采集一次』(会自动冻结)。"
    by_code = _read_forward(meta)
    nets15 = {c: [x[1] for x in rows] for c, rows in by_code.items()}
    pbo_res = pbo_cscv({c: v for c, v in nets15.items() if len(v) >= 16})
    pbo = pbo_res[0] if pbo_res else None
    lines = ["IMR 15/30min 前向判决",
             f"official_forward_start: {meta.get('frozen_utc')}",
             f"family_trials={imr.FAMILY_TRIALS}  PASS需≥{imr.MIN_FWD_TS}独立前向ts  "
             f"pbo(跨候选)={'-' if pbo is None else round(pbo, 3)}", ""]
    status_rows = []
    for code, meta_c in meta.get("candidates", {}).items():
        rows = by_code.get(code, [])
        net15 = [x[1] for x in rows]
        net10 = [x[2] for x in rows]
        dead = read_dead(_frozen(), code) is not None
        v = imr.evaluate_candidate(net15, net10, train_net15=meta_c.get("train_net15_mean"),
                                   train_ic_sign=int(meta_c.get("train_ic_sign", 0)),
                                   pbo=pbo, already_dead=dead)
        if v.verdict == "KILL" and not dead:
            write_dead(_frozen(), code, v.reason, v.metrics)
        m = v.metrics
        lines.append(f"[{v.verdict}] {code}  n={m['n_ts']}  net15={m['net15_mean_bps']}  "
                     f"net10={m['net10_mean_bps']}  DSR={m.get('dsr')}  PF={m.get('pf')}  — {v.reason}")
        status_rows.append({"code": code, "asof_ms": _now_ms(), "verdict": v.verdict,
                            "reason": v.reason, **{k: m.get(k) for k in
                            ("n_ts", "net15_mean_bps", "net10_mean_bps", "t", "dsr", "pf")}})
    try:
        _report().write_text("\n".join(lines) + "\n", encoding="utf-8")
        if status_rows:
            append_rows_hashchain(_status_chain(), status_rows)
    except Exception as e:  # noqa: BLE001
        lines.append(f"(写报告失败: {e!r})")
    return "\n".join(lines)


def status() -> str:
    """看进度 (无网络): 各候选已采集多少独立前向ts。"""
    meta = _load_meta()
    if meta is None:
        return "尚未冻结 — 点『采集一次』开始 (首次会自动冻结 official_forward_start)。"
    by_code = _read_forward(meta)
    lines = [f"official_forward_start: {meta.get('frozen_utc')}",
             f"工件目录: {_base()}", ""]
    for code in meta.get("candidates", {}):
        n = len(by_code.get(code, []))
        dead = " · DEAD(已判死)" if read_dead(_frozen(), code) else ""
        lines.append(f"  {code}: {n}/{imr.MIN_FWD_TS} 独立前向ts{dead}")
    lines.append(f"\n链校验: {verify_hashchain(_ledger()) or 'OK'}")
    lines.append("纪律: 冻结后不改规则; 攒满再判决。最可能长期 PENDING/KILL —— 那就是诚实答案。")
    return "\n".join(lines)
