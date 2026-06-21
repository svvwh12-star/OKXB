"""AI 前向验证 runner (进程内): 周期记录 AI 方向判断, HORIZON 分钟后用实价结算, 累计判 AI 前向 edge。

工件 -> research_base()/ai_forward/。token 控制: 每轮最多 len(WATCHLIST) 次 AI 调用, 且【每标的每
HORIZON 至多记录一次】(冷却), 即便狂点也封顶 ~watchlist/小时。AI 未配置则一次不调、只提示。
打包 exe 内可用 (进程内, 复用 GUI 同一条 AI 分析路径)。
"""
from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path
from typing import Optional

from ..config import Config, Secrets
from ..events.llm_classifier import LLMClassifier
from ..exchange.okx_rest import OkxRestClient
from . import ai_forward as af
from .datadir import research_base
from .forward_integrity import (append_rows_hashchain, read_dead, verify_hashchain,
                                write_dead, write_manifest)


def _base() -> Path:
    b = research_base() / "ai_forward"
    b.mkdir(parents=True, exist_ok=True)
    return b


def _frozen() -> Path:
    return _base() / "frozen"


def _meta_path() -> Path:
    return _frozen() / "meta.json"


def _open_ledger() -> Path:
    return _base() / "ai_open_hashchain.csv"


def _resolved_ledger() -> Path:
    return _base() / "ai_resolved_hashchain.csv"


def _report() -> Path:
    return _base() / "ai_forward_status.md"


def _now_ms() -> int:
    return int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)


def truthy(v) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "on")


def auto_enabled() -> bool:
    """无人值守是否自动记录 AI 前向 (AI_FORWARD_AUTO)。"""
    return truthy(Secrets().ai_forward_auto)


def _load_meta() -> Optional[dict]:
    p = _meta_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _freeze() -> dict:
    meta = {"official_forward_start_ts_ms": _now_ms(),
            "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "params": {"horizon_min": af.HORIZON_MIN, "cost_stress_bps": af.COST_BPS_STRESS,
                       "cost_mild_bps": af.COST_BPS_MILD, "watchlist": af.WATCHLIST,
                       "family_trials": af.FAMILY_TRIALS, "min_fwd_ts": af.MIN_FWD_TS},
            "note": "记录 AI 方向判断, 60min 后实价结算; 测 AI 是否有前向 edge。只记录不实盘。"}
    _frozen().mkdir(parents=True, exist_ok=True)
    _meta_path().write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    write_manifest(_frozen(), artifacts=("meta.json",))
    return meta


def _read_opens() -> list:
    p = _open_ledger()
    out = []
    if p.exists():
        with p.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    out.append({"ts": int(r["ts"]), "inst": r["inst"],
                                "direction": int(r["direction"]), "entry_px": float(r["entry_px"])})
                except (ValueError, KeyError, TypeError):
                    continue
    return out


def _resolved_keys() -> set:
    p = _resolved_ledger()
    keys = set()
    if p.exists():
        with p.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                keys.add((r.get("inst"), r.get("open_ts")))
    return keys


def _last_open_ts(opens: list, inst: str) -> int:
    ts = [o["ts"] for o in opens if o["inst"] == inst]
    return max(ts) if ts else 0


async def _ticker_mid(rest, inst: str) -> Optional[float]:
    try:
        t = await rest.get_ticker(inst)
        bid = float(t.get("bidPx", 0) or 0)
        ask = float(t.get("askPx", 0) or 0)
        if bid > 0 and ask > 0:
            return (bid + ask) / 2.0
        last = float(t.get("last", 0) or 0)
        return last or None
    except Exception:
        return None


async def _resolve(rest) -> int:
    """结算已到期(>=HORIZON)的开仓记录, 追加到 resolved 哈希链。返回新结算数。"""
    opens = _read_opens()
    done = _resolved_keys()
    now = _now_ms()
    horizon_ms = af.HORIZON_MIN * 60_000
    rows = []
    for o in opens:
        if (o["inst"], str(o["ts"])) in done:
            continue
        if now - o["ts"] < horizon_ms:
            continue
        exit_px = await _ticker_mid(rest, o["inst"])
        if not exit_px or o["entry_px"] <= 0:
            continue
        d, e = o["direction"], o["entry_px"]
        rows.append({"open_ts": o["ts"], "inst": o["inst"], "direction": d,
                     "entry_px": e, "exit_px": exit_px,
                     "net15_bps": round(af.net_bps(d, e, exit_px, af.COST_BPS_STRESS), 4),
                     "net10_bps": round(af.net_bps(d, e, exit_px, af.COST_BPS_MILD), 4),
                     "dir_ret_bps": round(d * (exit_px / e - 1.0) * 1e4, 4),
                     "resolved_ts": now})
        done.add((o["inst"], str(o["ts"])))
    if rows:
        rows.sort(key=lambda r: r["open_ts"])
        append_rows_hashchain(_resolved_ledger(), rows)
    return len(rows)


async def collect(record: bool = True) -> str:
    """结算到期记录; record=True 时再记录本轮 AI 方向判断(每标的每 HORIZON 至多一次)。返回摘要。
    record=False 用于无人值守仅【免费结算】历史记录而不调用 AI(不烧 token)。"""
    out = []
    meta = _load_meta()
    if meta is None:
        meta = _freeze()
        out.append(f"首次运行: 已冻结 official_forward_start = {meta['frozen_utc']}")
    clf = LLMClassifier.from_secrets(Secrets())
    if record and not clf.enabled:
        out.append("⚠ AI 未启用(规则/未填 key) → 本轮不记录(AI前向验证需要真实 AI 方向)。"
                   "请到『账户与密钥』选 DeepSeek 填 key 后再用。")
    if not record:
        out.append("(自动记录关闭: 仅免费结算到期记录, 不调用 AI)")
    opens = _read_opens()
    now = _now_ms()
    horizon_ms = af.HORIZON_MIN * 60_000
    new_rows = []
    if record and clf.enabled:
        from ..gui.controller import _ai_analyze   # 复用 GUI 同一条 AI 分析路径 (无 tkinter 依赖)
        for inst in af.WATCHLIST:
            if now - _last_open_ts(opens, inst) < horizon_ms:
                out.append(f"  {inst}: 冷却中(每{af.HORIZON_MIN}min至多记一次), 跳过")
                continue
            try:
                res = await _ai_analyze(inst, {})
            except Exception as e:  # noqa: BLE001
                out.append(f"  {inst}: AI 调用失败 {e!r}")
                continue
            struct = res.get("struct") or {}
            d = struct.get("direction")
            mid = (res.get("qctx") or {}).get("mid")
            if not res.get("ok") or d not in ("long", "short") or not mid:
                out.append(f"  {inst}: AI 未给明确方向/无价, 跳过")
                continue
            dirn = 1 if d == "long" else -1
            new_rows.append({"ts": now, "inst": inst, "direction": dirn,
                             "confidence": struct.get("confidence", 0.5), "entry_px": float(mid)})
            out.append(f"  {inst}: 记录 AI {d} @ {mid}")
    if new_rows:
        append_rows_hashchain(_open_ledger(), new_rows)
    rest = OkxRestClient(Secrets(), Config.load())
    try:
        resolved = await _resolve(rest)
    finally:
        await rest.aclose()
    out.append(f"本轮: 新记录 {len(new_rows)} 条 AI 判断, 结算 {resolved} 条到期 "
               f"(链校验 open={verify_hashchain(_open_ledger()) or 'OK'} "
               f"resolved={verify_hashchain(_resolved_ledger()) or 'OK'})")
    out.append("攒满 ≥100 条已结算样本再点『评估判决』。诚实预期: AI 方向多半无前向 edge -> PENDING/KILL。")
    return "\n".join(out)


def _read_resolved(meta: dict) -> list:
    start = int(meta["official_forward_start_ts_ms"])
    p = _resolved_ledger()
    out = []
    if not p.exists():
        return out
    with p.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                if int(r["open_ts"]) <= start:
                    continue
                out.append((int(r["open_ts"]), float(r["net15_bps"]),
                            float(r["net10_bps"]), float(r["dir_ret_bps"])))
            except (ValueError, KeyError, TypeError):
                continue
    out.sort(key=lambda x: x[0])
    return out


def evaluate() -> str:
    meta = _load_meta()
    if meta is None:
        return "尚未冻结 — 请先点『采集一次』(会自动冻结并开始记录 AI 判断)。"
    rows = _read_resolved(meta)
    net15 = [x[1] for x in rows]
    net10 = [x[2] for x in rows]
    dir_ret = [x[3] for x in rows]
    dead = read_dead(_frozen(), af.CODE) is not None
    v = af.evaluate(net15, net10, dir_ret, already_dead=dead)
    if v.verdict == "KILL" and not dead:
        write_dead(_frozen(), af.CODE, v.reason, v.metrics)
    m = v.metrics
    lines = ["AI 前向验证 · 判决",
             f"official_forward_start: {meta.get('frozen_utc')}",
             f"horizon={af.HORIZON_MIN}min  PASS需≥{af.MIN_FWD_TS}已结算样本",
             "",
             f"[{v.verdict}] n={m['n_ts']}  net15={m['net15_mean_bps']}bps  net10={m['net10_mean_bps']}bps  "
             f"AI_IC={m['ai_ic_bps']}bps  DSR={m.get('dsr')}  PF={m.get('pf')}",
             f"— {v.reason}"]
    try:
        _report().write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        pass
    return "\n".join(lines)


def status() -> str:
    meta = _load_meta()
    if meta is None:
        return "尚未冻结 — 点『采集一次』开始 (会自动冻结 official_forward_start)。"
    opens = _read_opens()
    resolved = len(_read_resolved(meta))
    dead = " · DEAD(已判死)" if read_dead(_frozen(), af.CODE) else ""
    lines = [f"official_forward_start: {meta.get('frozen_utc')}",
             f"工件目录: {_base()}",
             f"已记录 AI 判断: {len(opens)} 条; 已结算: {resolved}/{af.MIN_FWD_TS}{dead}",
             f"观察名单: {', '.join(af.WATCHLIST)}  horizon={af.HORIZON_MIN}min",
             "纪律: 冻结后不改; 攒满再判。最可能长期 PENDING/KILL —— 那就是诚实答案。"]
    return "\n".join(lines)
