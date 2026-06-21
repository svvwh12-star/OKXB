"""手动下单风控闸门 (P4): 让额度/频次纪律成为每一笔手动【开仓】的唯一关卡。

背景 (审计 H-1 残留): 此前手动开仓/套单/加杠杆完全绕过 RiskEngine 的名义额度约束,
只在引擎熔断态做只读拦截 —— 实盘可手动开出任意大仓位, 风控不在回路。本模块提供一个
【独立、引擎在不在跑都生效】的闸门:
  - 单笔名义上限     risk.manual_max_notional_per_trade_usdt
  - 单日名义累计上限 risk.manual_max_notional_per_day_usdt
  - 单日开仓笔数上限 risk.manual_max_trades_per_day
  - 单笔杠杆上限     risk.manual_max_leverage
日内计数持久化到 data/manual_trade_ledger.json (按【本地日期】分桶), 跨重启/跨进程一致。
平仓/撤单永不受限; demo 与 live 同样生效 (纪律即护栏, 防"虚假信心 -> 重仓")。
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Optional

from .. import paths

# 默认上限 (~1000U 账户 + 杠杆后的合理暴露; 用户可在 config.yaml 的 risk: 段收紧)
DEFAULTS = {
    "manual_max_notional_per_trade_usdt": 2500.0,
    "manual_max_notional_per_day_usdt": 10000.0,
    "manual_max_trades_per_day": 30,
    "manual_max_leverage": 50,
}

_LEDGER_OVERRIDE: Optional[Path] = None      # 仅供测试注入


def set_ledger_path_for_test(p: Optional[Path]) -> None:
    global _LEDGER_OVERRIDE
    _LEDGER_OVERRIDE = p


def _ledger_path() -> Path:
    return _LEDGER_OVERRIDE if _LEDGER_OVERRIDE is not None else paths.data_path("manual_trade_ledger.json")


def _today() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d")     # 本地日期 (与用户作息一致)


def _load() -> dict:
    p = _ledger_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save(data: dict) -> None:
    p = _ledger_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _bucket(data: dict) -> dict:
    b = data.get(_today())
    return b if isinstance(b, dict) else {"notional": 0.0, "count": 0}


def caps(cfg) -> dict:
    """从 config 读上限 (缺省用 DEFAULTS); cfg 可为 None。"""
    out = dict(DEFAULTS)
    if cfg is not None:
        for k, dv in DEFAULTS.items():
            v = cfg.get(f"risk.{k}")
            if v is not None:
                out[k] = type(dv)(v)
    return out


def check_leverage(lever, cfg=None) -> Optional[str]:
    """加杠杆闸门: 超过上限则拒绝。"""
    try:
        lv = float(lever)
    except (TypeError, ValueError):
        return None
    cap = caps(cfg)["manual_max_leverage"]
    if lv > cap:
        return (f"⛔ 杠杆 {int(lv)}x 超过手动上限 {int(cap)}x (risk.manual_max_leverage)。"
                "高杠杆放大爆仓风险, 已拒绝。如确需更高请在参数里上调上限并自负风险。")
    return None


def check_open(notional_usdt: float, cfg=None) -> Optional[str]:
    """开仓闸门: 返回阻断原因 (str) 或 None(放行)。仅校验【新增名义】, 不落账(成交后再 record_open)。"""
    try:
        n = float(notional_usdt or 0.0)
    except (TypeError, ValueError):
        n = 0.0
    c = caps(cfg)
    per = float(c["manual_max_notional_per_trade_usdt"])
    if n > per:
        return (f"⛔ 单笔名义 ≈{n:,.0f}U 超过单笔上限 {per:,.0f}U "
                "(risk.manual_max_notional_per_trade_usdt)。已拒绝, 请减小金额/杠杆。")
    b = _bucket(_load())
    used = float(b.get("notional", 0.0) or 0.0)
    cnt = int(b.get("count", 0) or 0)
    day_cap = float(c["manual_max_notional_per_day_usdt"])
    if used + n > day_cap:
        return (f"⛔ 今日手动开仓名义累计将达 ≈{used + n:,.0f}U, 超过单日上限 {day_cap:,.0f}U "
                f"(risk.manual_max_notional_per_day_usdt)。今日已用 ≈{used:,.0f}U。已拒绝。")
    max_trades = int(c["manual_max_trades_per_day"])
    if cnt + 1 > max_trades:
        return (f"⛔ 今日手动开仓已达 {cnt} 笔, 达单日笔数上限 {max_trades} "
                "(risk.manual_max_trades_per_day)。防过度交易, 已拒绝。")
    return None


def record_open(notional_usdt: float) -> None:
    """成交后调用: 把本笔名义与计数累加到今日桶。"""
    try:
        n = float(notional_usdt or 0.0)
    except (TypeError, ValueError):
        n = 0.0
    data = _load()
    b = _bucket(data)
    b["notional"] = round(float(b.get("notional", 0.0) or 0.0) + n, 2)
    b["count"] = int(b.get("count", 0) or 0) + 1
    data[_today()] = b
    # 顺手清理 14 天前的旧桶, 不让文件无限增长
    cutoff = (dt.datetime.now() - dt.timedelta(days=14)).strftime("%Y-%m-%d")
    for k in [k for k in data if k < cutoff]:
        data.pop(k, None)
    _save(data)


def today_usage() -> dict:
    """供 UI 展示: 今日已用名义/笔数。"""
    return _bucket(_load())
