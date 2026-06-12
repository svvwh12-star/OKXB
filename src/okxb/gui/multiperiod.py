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


def auto_step(live: bool) -> str:
    """跑一轮 BTC 多周期自动交易: --mode auto。
    demo(live=False): 直接影子下单(执行验证); live=True: 先 evaluate 刷新 verdict, 再【仅对 PASS 候选】实盘下单。
    现状 A/B/C 全 PENDING -> 实盘这步不会下任何单(门控按预期工作)。subprocess, 仅开发态/随附研究目录可用。"""
    script = _research() / "scripts" / "run_forward_shadow.py"
    if not script.exists():
        return (f"未找到 {script}\n(多周期自动交易依赖 btc_single_asset_research/; 仅开发态可用)")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_root() / "src")
    cmd = [sys.executable, str(script), "--mode", "auto"] + (["--live"] if live else ["--arm"])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                           env=env, cwd=str(_research() / "scripts"))
    except subprocess.TimeoutExpired:
        return "自动一轮超时(>600s; 实盘会先 evaluate 拉 150d 数据, 较慢)。"
    except Exception as e:  # noqa: BLE001
        return f"运行失败: {e!r}\n(打包 .exe 无法跑脚本; 请用开发态 python okxb_gui.py)"
    txt = r.stdout or ""
    keys = ("shadow", "门控", "入场", "跳过", "不入场", "verdict", "PASS", "KILL", "demo", "live",
            "Error", "失败", "拒绝", "Traceback")
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


def _get_snapshot() -> dict:
    """subprocess 跑 --mode snapshot, 解析 {features(客观因子), scores(A/B/C 打分)}。"""
    script = _research() / "scripts" / "run_forward_shadow.py"
    if not script.exists():
        return {}
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_root() / "src")
    try:
        r = subprocess.run([sys.executable, str(script), "--mode", "snapshot"],
                           capture_output=True, text=True, timeout=300,
                           env=env, cwd=str(_research() / "scripts"))
    except Exception:  # noqa: BLE001
        return {}
    import json
    for ln in (r.stdout or "").splitlines():
        if ln.startswith("SNAPSHOT_JSON "):
            try:
                return json.loads(ln[len("SNAPSHOT_JSON "):])
            except Exception:  # noqa: BLE001
                return {}
    return {}


def ai_compare() -> str:
    """AI 叙述 vs 量化测量 对照(你的方法论): 给 AI 【只】喂客观因子让它【独立盲分析】,
    程序【另行】并排量化测量 + 写死安全裁决。慢: 含一次 snapshot(拉数据+打分) + 一次 AI 调用。"""
    snap = _get_snapshot()
    feats = snap.get("features", {})
    scores = snap.get("scores", {})
    fs = forward_status()
    rows = {str(r.get("code")): r for r in fs.get("rows", [])} if fs.get("ok") else {}

    # ① AI 盲分析: 只给客观因子, 不给任何打分/结论(避免锚定)
    if feats:
        obj_text = "\n".join(f"  {k}: {v}" for k, v in feats.items())
        try:
            import asyncio

            from ..config import Secrets
            from ..events.llm_classifier import LLMClassifier
            ai = asyncio.run(LLMClassifier.from_secrets(Secrets()).analyze_market(obj_text))
        except Exception as e:  # noqa: BLE001
            ai = f"AI 分析失败: {e!r}"
    else:
        obj_text, ai = "(未取到客观因子; 多周期研究依赖 btc_single_asset_research/, 仅开发态可用)", \
            "无客观数据, 跳过 AI 分析。"

    # ② 量化测量(程序独立呈现, AI 看不到)
    qlines = []
    for code in ("A", "B", "C"):
        sc = scores.get(code, {})
        r = rows.get(code, {})
        qlines.append(f"  {code} {_MODEL.get(code, '')}: 当前打分={sc.get('score', '?')} tau={sc.get('tau', '?')} "
                      f"超tau={'是' if sc.get('hit') else '否'} | forward verdict={r.get('verdict', '?')} "
                      f"n_ts={r.get('n_ts', '-')} net10={r.get('net10_bps', '-')}bps fwd_ic={r.get('fwd_ic', '-')}")
    quant = "\n".join(qlines) or "(无量化数据)"

    note = ("【对照与裁决】① 是 AI 基于客观数据的【独立叙述视角】(它看不到量化打分/结论); "
            "③ 是严格的样本外净edge闸门 + forward 状态。二者可能给出不同方向——这正是"
            "'AI 叙述 vs 统计净edge'的区别。【最终交易裁决以量化为准】: A/B/C 现状均 PENDING"
            "(统计上无稳健净edge), 故即使 AI 给出明确方向, 也【不可交易】。")
    return ("【① AI 独立分析(只看客观数据, 不知量化结论)】\n" + ai +
            "\n\n【② 喂给 AI 的客观数据快照】\n" + obj_text +
            "\n\n【③ 量化测量(程序独立, AI 未见)】\n" + quant +
            "\n\n" + note)
