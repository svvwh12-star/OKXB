"""多周期研究监控 tab 的后端(只读 + 影子触发)。

定位(已确认): 只读研究监控 + demo 影子执行验证。**不替换控制台方向源、不驱动实盘下单。**
亮灯由 forward 状态(PASS/PENDING/KILL)驱动, 不是历史 gate。现状 A/B/C 全 PENDING -> 无绿灯。

依赖项目根的 btc_single_asset_research/(forward_status.csv + run_forward_shadow.py)。
开发态(python okxb_gui.py)有效; 打包 .exe 需另把该研究目录 + 一个可跑脚本的 python 随附。
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

_MODEL = {"A": "hist_gbm(180min)", "B": "lightgbm(6h)", "C": "mlp(9h)"}


def _root() -> Path:
    return Path(__file__).resolve().parents[3]            # src/okxb/gui -> <project root>


def _research() -> Path:
    return _root() / "btc_single_asset_research"


def forward_status() -> dict:
    """读 forward_status.csv 最新一批(A/B/C) + shadow_trades 摘要。纯文件, 无需引擎/密钥。"""
    out: dict = {"ok": False, "rows": [], "shadow": [], "note": ""}
    fpath = _research() / "forward" / "forward_status.csv"
    if not fpath.exists():
        out["note"] = "未找到 forward_status.csv — 先在 scripts 跑 run_forward_shadow.py --mode evaluate"
        return out
    try:
        import pandas as pd
        df = pd.read_csv(fpath)
        if "asof_utc" in df.columns and len(df):
            latest = df[df["asof_utc"] == df["asof_utc"].iloc[-1]]
        else:
            latest = df.tail(3)
        out["rows"] = latest.fillna("").to_dict("records")
        out["ok"] = True
        sp = _research() / "forward" / "shadow_trades.csv"
        if sp.exists():
            out["shadow"] = pd.read_csv(sp).fillna("").tail(8).to_dict("records")
    except Exception as e:  # noqa: BLE001
        out["note"] = f"读取失败: {e!r}"
    return out


def shadow_run(armed: bool) -> str:
    """subprocess 跑 run_forward_shadow.py --mode shadow [--arm], 返回精简输出。
    armed=False: dry-run(公共数据, 不下单); armed=True: 在 demo 真下单(需 OKXB_MODE=demo + demo 密钥)。"""
    script = _research() / "scripts" / "run_forward_shadow.py"
    if not script.exists():
        return (f"未找到 {script}\n(多周期研究依赖 btc_single_asset_research/; "
                "仅开发态或随附研究目录+python时可用)")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_root() / "src")
    cmd = [sys.executable, str(script), "--mode", "shadow"] + (["--arm"] if armed else [])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                           env=env, cwd=str(_research() / "scripts"))
    except subprocess.TimeoutExpired:
        return "影子运行超时(>300s; 多周期模型拉数据较慢)。"
    except Exception as e:  # noqa: BLE001
        return f"运行失败: {e!r}\n(打包 .exe 下 sys.executable 不是 python, 无法跑脚本; 请用开发态)"
    txt = r.stdout or ""
    keys = ("shadow", "score", "入场", "跳过", "不入场", "拒绝", "demo", "Error", "失败", "Traceback")
    lines = [ln for ln in txt.splitlines() if any(k in ln for k in keys)]
    body = "\n".join(lines[-30:]) if lines else txt[-1500:]
    if r.returncode != 0 and r.stderr:
        body += "\n[stderr]\n" + r.stderr[-600:]
    return body or "(无输出)"


def _parse_scores(shadow_out: str) -> dict:
    """从 shadow dry-run 输出解析每个候选当前的 score/tau。"""
    out: dict = {}
    for m in re.finditer(r"\[shadow (\w+)\]\s*score=([+-]?[\d.]+)\s*tau=([\d.]+)", shadow_out):
        try:
            out[m.group(1)] = {"score": float(m.group(2)), "tau": float(m.group(3))}
        except ValueError:
            pass
    return out


def ai_compare() -> str:
    """组装多周期量化结论(forward 状态 + 当前打分) -> 喂 AI 做【只读】对照解读。
    慢: 含一次影子 dry-run(拉数据+打分) + 一次 AI 调用。AI 被硬约束不得把 PENDING/KILL 说成机会。"""
    fs = forward_status()
    if not fs.get("ok"):
        return fs.get("note", "无 forward 状态; 先在 scripts 跑 run_forward_shadow.py --mode evaluate")
    scores = _parse_scores(shadow_run(False))               # 当前 A/B/C 打分(dry-run, 不下单)
    lines = []
    for r in fs.get("rows", []):
        code = str(r.get("code", "?"))
        sc = scores.get(code, {})
        hit = "是" if sc and abs(sc.get("score", 0.0)) >= sc.get("tau", 1e9) else "否"
        lines.append(
            f"候选{code} {_MODEL.get(code, '')}: 当前打分={sc.get('score', '?')} tau={sc.get('tau', '?')} "
            f"超tau={hit} | forward: n_ts={r.get('n_ts', '-')} net10={r.get('net10_bps', '-')}bps "
            f"net15={r.get('net15_bps', '-')}bps fwd_ic={r.get('fwd_ic', '-')} verdict={r.get('verdict', 'PENDING')}")
    quant_text = "\n".join(lines) or "(无候选数据)"
    try:
        import asyncio

        from ..config import Secrets
        from ..events.llm_classifier import LLMClassifier
        ai = asyncio.run(LLMClassifier.from_secrets(Secrets()).explain_quant_monitor(quant_text))
    except Exception as e:  # noqa: BLE001
        ai = f"AI 解读失败: {e!r}"
    return "【量化结论(喂给 AI 的客观数据)】\n" + quant_text + "\n\n" + ai
