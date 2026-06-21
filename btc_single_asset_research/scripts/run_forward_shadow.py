"""Forward shadow test runner for the pre-registered 180min/6h/9h candidates.

Implements docs/FORWARD_SHADOW_PROTOCOL_h180_6h_9h.md exactly:
  freeze  -> train each candidate on the SAME 150d data, save model + selected features
             + median fill + scaler + the FROZEN absolute confidence threshold tau.
  evaluate-> reuse the IDENTICAL feature pipeline, score ONLY bars after the official shadow start
             that have already settled (ts <= now-H), using the frozen artifacts (NO refit,
             NO re-selection, NO re-bucketing), net of taker + funding-hold + x1.5 stress;
             check PASS/KILL.
  selftest-> assert the verdict logic.

Honest prior: 0 PASS (clean closure) is the expected, acceptable outcome. 9h is a
falsification target (train IC<0, AUC<0.5). Rules are frozen by the committed protocol.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# joblib/loky 在本机无 wmic 时探测物理核失败 -> 噪音 [WinError 2]。给定核数 + 兜底过滤该 warning。
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

ROOT = Path(__file__).resolve().parents[2]               # OKXB
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPTS))

import run_btc_enhanced_research as enh                  # noqa: E402  (reuse the exact pipeline)
from okxb.research import candle_data as cd              # noqa: E402
from okxb.research import candle_research as cr          # noqa: E402
from okxb.research import feature_lab as fl              # noqa: E402
from okxb.research import pro_model_workflow as pmw      # noqa: E402
from okxb.research import forward_integrity as fi        # noqa: E402  (RV-1/2/3 完整性工具)
from okxb.config import Config, Secrets                  # noqa: E402
from okxb.core.enums import Mode                         # noqa: E402
from sklearn.preprocessing import StandardScaler         # noqa: E402

OUT = ROOT / "btc_single_asset_research"
FROZEN = OUT / "frozen"
FWD = OUT / "forward"
PROTOCOL = ROOT / "docs" / "FORWARD_SHADOW_PROTOCOL_h180_6h_9h.md"

# ----- demo shadow auto-trade (EXECUTION-REALITY check only; DEMO-ONLY, fail-closed) -----
# Purpose: measure paper-net vs real-fill-net (fill rate / slippage / latency / funding /
# TP-SL triggers) for the pre-registered A/B/C candidates in the SIMULATED (demo) account.
# It does NOT touch real money, does NOT replace the console direction source, and does NOT
# affect the paper forward verdict. Expected outcome: demo loses/breaks even (signals are
# gate=False). A serial single-slot design avoids net-mode position merging across A/B/C.
SHADOW_RISK_PCT = 0.003        # demo per-trade max loss = equity x 0.3%
SHADOW_K_SL = 0.8             # SL = k_sl x H-window sigma
SHADOW_K_TP = 1.6            # TP = k_tp x H-window sigma (RR~2)
SHADOW_ORD = "post_only"     # maker entry (measures real maker fill rate); 'market' to force a taker fill
SHADOW_STATE = FWD / "shadow_state.json"
SHADOW_TRADES = FWD / "shadow_trades.csv"

# ----- pre-registered, frozen candidate sets, PER ASSET (must match the committed protocol) -----
# oos_ic_sign = the sign of the ORIGINAL OOS primary_ic (from {asset}_enhanced_summary.csv), NOT
# the in-sample fit. The forward "same-sign" check compares against THIS.
# BTC: C(9h) pre-registered as -1 (falsification target, OOS IC -0.0356).
# ETH (added 2026-06-12, from data/eth_enhanced_summary.csv; ALL gate=False / NO TRADE in-sample,
#      same as BTC — the forward test is the honest registration, expected to stay PENDING):
#   A H=180 rf_ret   top0.10  primary_ic=-0.0030 -> -1
#   B H=360 logit_l2 top0.005 primary_ic=+0.0476 -> +1
#   C H=540 hist_gbm top0.02  primary_ic=-0.0070 -> -1
ASSET_CFG = {
    "btc": {
        "perp": "BTC-USDT-SWAP", "spot": "BTC-USDT",
        "candidates": {
            "A": {"H": 180, "model": "hist_gbm", "top_frac": 0.01, "oos_ic_sign": +1},
            "B": {"H": 360, "model": "lightgbm", "top_frac": 0.02, "oos_ic_sign": +1},
            "C": {"H": 540, "model": "mlp",      "top_frac": 0.05, "oos_ic_sign": -1},
        },
    },
    "eth": {
        "perp": "ETH-USDT-SWAP", "spot": "ETH-USDT",
        "candidates": {
            "A": {"H": 180, "model": "rf_ret",   "top_frac": 0.1,   "oos_ic_sign": -1},
            "B": {"H": 360, "model": "logit_l2", "top_frac": 0.005, "oos_ic_sign": +1},
            "C": {"H": 540, "model": "hist_gbm", "top_frac": 0.02,  "oos_ic_sign": -1},
        },
    },
}
ASSET = "btc"
CANDIDATES = ASSET_CFG["btc"]["candidates"]
PERP, SPOT = ASSET_CFG["btc"]["perp"], ASSET_CFG["btc"]["spot"]


def _set_asset(asset: str) -> None:
    """Point module globals at one asset's frozen/forward dirs + candidate set.
    btc keeps the ORIGINAL unprefixed dirs (preserves the running BTC forward test);
    other assets nest under frozen/{asset} and forward/{asset}."""
    global ASSET, CANDIDATES, PERP, SPOT, FROZEN, FWD, SHADOW_STATE, SHADOW_TRADES
    if asset not in ASSET_CFG:
        raise SystemExit(f"unknown asset {asset!r}; choices: {list(ASSET_CFG)}")
    ASSET = asset
    cfg = ASSET_CFG[asset]
    CANDIDATES = cfg["candidates"]
    PERP, SPOT = cfg["perp"], cfg["spot"]
    FROZEN = (OUT / "frozen") if asset == "btc" else (OUT / "frozen" / asset)
    FWD = (OUT / "forward") if asset == "btc" else (OUT / "forward" / asset)
    SHADOW_STATE = FWD / "shadow_state.json"          # per-asset single-slot state
    SHADOW_TRADES = FWD / "shadow_trades.csv"


BAR = "5m"
K_SEL = 30
PRESET = "deep"
TAKER_BPS = 10.0
STRESS_MULT = 1.5            # protocol: net judged after cost x1.5  -> taker 10 -> 15 bps
# RV-3 多重检验: 当前并行前向测试的真实候选总数 (btc A/B/C + eth A/B/C + stock ST720 = 7)。
# 新增任何候选都必须同步此数, 否则族级假阳率被系统性低估。Bonferroni 阈值随之收紧。
N_PARALLEL_CANDIDATES = 7
bonferroni_t = fi.bonferroni_t
T_PASS = bonferroni_t(N_PARALLEL_CANDIDATES)   # 7 候选 -> ~2.46 (此前硬编码 2.13 仅校正了 3 个)
N_MIN = 50
N_KILL = 30
WEEK_MS = 7 * 86_400_000


def log(m: str) -> None:
    print(m, flush=True)


def _fetch_raw(days: int, *, source: str, force: bool, ext_force: "bool | None" = None,
               update: bool = False) -> dict:
    """拉取构建面板所需的全部原始输入(K线/资金费/外部源)。这是【唯一的网络密集步骤】; 多个 H 复用同一份。
      update=True : K线走增量(只拉缓存之后的新 bar, 几次调用而非百次翻页), 极大降低网络暴露。
      ext_force   : 日频外部源是否强制重取; 默认随 force。短周期外部源在 forward 路径(include_short=False)
                    根本没用到 -> 直接不拉, 省掉一整组 5m Rubik 调用。"""
    if ext_force is None:
        ext_force = force
    btc = enh.load_candles(PERP, BAR, days, source=source, force=force, update=update)
    spot = enh.load_candles(SPOT, BAR, days, source=source, force=force, update=update)
    funding = cd.fetch_funding_series(PERP, days + 5, log=log)
    daily_external, _ = enh.fetch_daily_external(days, force=ext_force, asset=ASSET)
    short_external = pd.DataFrame()        # forward 路径未用短周期外部源 -> 不拉
    return {"btc": btc, "spot": spot, "funding": funding,
            "daily_external": daily_external, "short_external": short_external}


def _raw_ok(raw: dict) -> bool:
    """原始输入可用于建面板的最低条件: 永续+现货 K线均非空(受限网络偶发空拉/被拒)。"""
    btc, spot = raw.get("btc"), raw.get("spot")
    return btc is not None and len(btc) > 0 and spot is not None and len(spot) > 0


def _panel_from_raw(raw: dict, H: int, *, apply_funding: bool):
    """从已拉取的原始输入构建某个 H 的面板(纯 CPU, 无网络)。"""
    btc, funding = raw["btc"], raw["funding"]
    panel, bar_ms = pmw.build_augmented_bank({PERP: btc}, {PERP: raw["spot"]}, {PERP: funding}, BAR, H)
    if apply_funding:
        panel = enh.apply_funding_hold_cost(panel, funding, H)
    panel = enh.enrich_panel(panel, btc, BAR, raw["daily_external"], raw["short_external"], include_short=False)
    return panel, bar_ms


def build_panel(H: int, *, days: int, source: str, force: bool, apply_funding: bool,
                ext_force: "bool | None" = None):
    """Single-H wrapper: fetch raw + build (kept for freeze / _score_now). evaluate fetches raw
    ONCE via _fetch_raw and reuses _panel_from_raw across all H (no per-H re-fetch)."""
    raw = _fetch_raw(days, source=source, force=force, ext_force=ext_force)
    return _panel_from_raw(raw, H, apply_funding=apply_funding)


# ----------------- RV-1/RV-2 预登记完整性 (复用 okxb.research.forward_integrity) -----------------

_write_manifest = fi.write_manifest
_verify_manifest = fi.verify_manifest


def _dead_path(code: str) -> Path:
    return fi.dead_path(FROZEN, code)


def _read_dead(code: str) -> Optional[dict]:
    return fi.read_dead(FROZEN, code)


def _write_dead(code: str, reason: str, stats: dict) -> None:
    fi.write_dead(FROZEN, code, reason, stats)


def _count_frozen_candidates() -> int:
    """跨所有族 (btc A/B/C + eth + stock) 统计已冻结候选数, 用于核对 N_PARALLEL_CANDIDATES 是否够大。"""
    n = 0
    for base in (OUT / "frozen", ROOT / "stock_calendar_research" / "frozen"):
        if base.exists():
            n += sum(1 for _ in base.rglob("meta.json"))
    return n


def freeze() -> None:
    FROZEN.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time() * 1000)
    for code, cfg in CANDIDATES.items():
        H, model_name, top_frac = cfg["H"], cfg["model"], cfg["top_frac"]
        # SAME 150d data + dist candles as the original research run (reproduce the candidate exactly)
        panel, bar_ms = build_panel(H, days=150, source="dist", force=False, apply_funding=(H >= 240))
        feats = pmw.feature_cols(panel)
        data = panel[["ts", "inst", "fwd", "y"] + feats].replace([np.inf, -np.inf], np.nan).dropna(subset=["fwd", "y"])
        sel, _ = fl.select_features(data, feats, k=K_SEL)
        med = data[sel].median()
        X = data[sel].fillna(med).values
        kind, factory = pmw._model_factories(preset=PRESET)[model_name]
        scaler = StandardScaler().fit(X) if kind.endswith("scale") else None
        Xf = scaler.transform(X) if scaler is not None else X
        target = data["y"].values.astype(int) if kind.startswith("clf") else data["fwd"].values.astype(float)
        model = factory().fit(Xf, target)
        train_score = pmw._score_from_model(kind, model, Xf)
        tau = float(np.nanquantile(np.abs(train_score), 1 - top_frac))      # FROZEN absolute threshold
        ic_sign = int(cfg["oos_ic_sign"])                                   # pre-registered OOS sign (NOT in-sample)
        insample_ic_sign = int(np.sign(cr._spearman(train_score, data["fwd"].values) or 0.0))
        d = FROZEN / code
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "model.pkl", "wb") as fh:
            pickle.dump({"model": model, "kind": kind, "scaler": scaler}, fh)
        (d / "feature_list.json").write_text(json.dumps(sel), encoding="utf-8")
        (d / "median.json").write_text(json.dumps(med.to_dict()), encoding="utf-8")
        (d / "meta.json").write_text(json.dumps({
            "code": code, "H": H, "model": model_name, "top_frac": top_frac, "bar_ms": bar_ms,
            "tau": tau, "ic_sign": ic_sign, "insample_ic_sign": insample_ic_sign, "n_sel": len(sel),
            "freeze_cutoff_ts": int(data["ts"].max()), "frozen_at_ms": stamp,
            "official_forward_start_ts": stamp,
            "apply_funding_train": bool(H >= 240),
        }, indent=2), encoding="utf-8")
        _write_manifest(d)                              # RV-1: 工件内容哈希清单 (证明此后未改)
        _dead_path(code).unlink(missing_ok=True)        # 重新冻结=全新登记, 清除旧的 KILL 死标
        log(f"[freeze {code}] H={H} {model_name} sel={len(sel)} tau={tau:.5f} "
            f"oos_ic_sign={ic_sign:+d} (in-sample {insample_ic_sign:+d}) cutoff={enh.ts_utc(int(data['ts'].max()))}")


def _net(port: np.ndarray, cost_bps: float):
    st = cr._stats_from_gross(port, cost_bps, 1) if len(port) >= 8 else None
    return (st["net_bps"], st["nw_t"]) if st else (np.nan, np.nan)


def forward_verdict(net15: float, t15: float, net10: float, n_ts: int,
                    fwd_ic_sign: int, train_ic_sign: int, weeks: float,
                    t_pass: Optional[float] = None) -> str:
    """Pure PASS/KILL logic per protocol section 4/5. t_pass 默认用族级 Bonferroni 阈值 T_PASS。"""
    tp = T_PASS if t_pass is None else t_pass
    if n_ts >= 8:
        if (net10 is not None and net10 <= 0) or (fwd_ic_sign != 0 and fwd_ic_sign != train_ic_sign):
            return "KILL"
    if weeks >= 6 and n_ts < N_KILL:
        return "KILL"
    if (net15 is not None and net15 > 0 and (t15 or 0) > tp and n_ts >= N_MIN
            and fwd_ic_sign == train_ic_sign):
        return "PASS"
    return "PENDING"


def evaluate() -> None:
    FWD.mkdir(parents=True, exist_ok=True)
    now_ms = int(time.time() * 1000)
    # Gather frozen candidates first, then fetch the raw inputs ONCE at the longest lookback any
    # candidate needs, and reuse across all H. The per-H forward window (settle filter below) is
    # identical to fetching per-candidate, since features are causal and warmed up by official_start.
    frozen = []
    for code in CANDIDATES:
        d = FROZEN / code
        if not (d / "meta.json").exists():
            log(f"[{code}] not frozen yet — run --mode freeze first")
            continue
        frozen.append((code, d, json.loads((d / "meta.json").read_text(encoding="utf-8"))))
    if not frozen:
        log("no frozen candidates — run --mode freeze first")
        return
    # RV-3: 用真实并行候选数做多重检验校正; 若磁盘上冻结的候选已超过申报数, 大声告警 (阈值偏松)
    on_disk = _count_frozen_candidates()
    log(f"[evaluate {ASSET}] 多重检验: T_PASS={T_PASS:.3f} (Bonferroni N={N_PARALLEL_CANDIDATES}; "
        f"磁盘上冻结候选共 {on_disk} 个)")
    if on_disk > N_PARALLEL_CANDIDATES:
        log(f"⚠ 冻结候选 {on_disk} > 申报 {N_PARALLEL_CANDIDATES}: 多重检验被低估, 请上调 N_PARALLEL_CANDIDATES!")
    days_max = max(150 + int((now_ms - m["freeze_cutoff_ts"]) / 86_400_000) + 10 for _, _, m in frozen)
    log(f"[evaluate {ASSET}] 增量拉取(只取新bar)+日缓存外部源, days={days_max}, {len(frozen)} 个周期复用")
    raw = _fetch_raw(days_max, source="refresh", force=False, update=True)
    if not _raw_ok(raw):            # 增量也没拿到可用K线 -> 退回纯缓存
        log(f"[evaluate {ASSET}] 增量 K线为空(网络拒绝/受限), 退回纯缓存...")
        raw = _fetch_raw(days_max, source="refresh", force=False)
    if not _raw_ok(raw):            # 缓存也空 -> 干净跳过, 绝不让空df崩成 IndexError(原 bug)
        log(f"[evaluate {ASSET}] K线仍为空, 本轮评估跳过(不报错)。稍后网络恢复再试; ETH 其它功能不受影响。")
        return

    rows = []
    for code, d, meta in frozen:
        weeks0 = round((now_ms - meta["frozen_at_ms"]) / WEEK_MS, 2)
        # RV-1: 冻结后被改动 -> 排除 (预登记失效, 不可再当作有效前向测试)
        tamper = _verify_manifest(d)
        if tamper and tamper != "no-manifest":
            log(f"[{code}] ⚠ 预登记完整性失败: {tamper} -> EXCLUDED")
            rows.append({"code": code, "H": meta.get("H"), "weeks_since_freeze": weeks0,
                         "n_signals": 0, "n_ts": 0, "verdict": "EXCLUDED", "start_utc": ""})
            continue
        # RV-2: KILL 粘性 -> 已判死的候选直接跳过, 永不复活成 PENDING/PASS
        dead = _read_dead(code)
        if dead:
            log(f"[{code}] 已判死(sticky, 不复活): {dead.get('reason')}")
            rows.append({"code": code, "H": meta.get("H"), "weeks_since_freeze": weeks0,
                         "n_signals": 0, "n_ts": 0, "verdict": "DEAD", "start_utc": ""})
            continue
        H, bar_ms, tau = meta["H"], meta["bar_ms"], meta["tau"]
        cutoff, train_ic_sign = meta["freeze_cutoff_ts"], meta["ic_sign"]
        official_start = max(int(cutoff), int(meta.get("official_forward_start_ts", meta["frozen_at_ms"])))
        weeks = (now_ms - meta["frozen_at_ms"]) / WEEK_MS
        blob = pickle.load(open(d / "model.pkl", "rb"))
        model, kind, scaler = blob["model"], blob["kind"], blob["scaler"]
        sel = json.loads((d / "feature_list.json").read_text(encoding="utf-8"))
        med = pd.Series(json.loads((d / "median.json").read_text(encoding="utf-8")))

        # evaluate on the TRUE net target (taker + funding always folded in), only settled forward bars
        panel, _ = _panel_from_raw(raw, H, apply_funding=True)
        for c in sel:                                   # tolerate a forward source that briefly drops a column
            if c not in panel.columns:
                panel[c] = np.nan
        settle_hi = now_ms - H * 60_000
        sub = panel[(panel["ts"] > official_start) & (panel["ts"] <= settle_hi)]
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=["fwd", "y"])
        row = {"code": code, "H": H, "weeks_since_freeze": round(weeks, 2), "n_signals": 0,
               "n_ts": 0, "verdict": "PENDING", "start_utc": enh.ts_utc(official_start)}
        if len(sub):
            X = sub[sel].fillna(med).values
            Xf = scaler.transform(X) if scaler is not None else X
            score = pmw._score_from_model(kind, model, Xf)
            hi = sub.assign(score=score)
            hi = hi[np.abs(hi["score"].values) >= tau]                      # FROZEN tau, not a forward quantile
            h = max(1, round(H / (bar_ms / 60_000)))
            reb = set(cr._rebalance_ts(np.sort(hi["ts"].unique()), bar_ms, h).tolist())
            tr = hi[hi["ts"].isin(reb)]
            if len(tr):
                signed = np.sign(tr["score"].values) * tr["fwd"].values
                port = pd.Series(signed, index=tr["ts"].values).groupby(level=0).mean().values
                net10, t10 = _net(port, TAKER_BPS)
                net15, t15 = _net(port, TAKER_BPS * STRESS_MULT)
                fic = cr._spearman(tr["score"].values, tr["fwd"].values) if len(tr) > 5 else 0.0
                fic_sign = int(np.sign(fic or 0.0))
                verdict = forward_verdict(net15, t15, net10, len(port), fic_sign, train_ic_sign, weeks)
                if verdict == "KILL":               # RV-2: 落盘死标, 此后不再复活
                    _write_dead(code, f"forward KILL: net10={net10:.1f} fwd_ic_sign={fic_sign} "
                                f"vs train={train_ic_sign} n_ts={len(port)} weeks={weeks:.1f}",
                                {"net10_bps": round(net10, 1), "n_ts": int(len(port)), "weeks": round(weeks, 1)})
                row.update({"n_signals": len(tr), "n_ts": len(port),
                            "net10_bps": round(net10, 1), "t10": round(t10, 2),
                            "net15_bps": round(net15, 1), "t15": round(t15, 2),
                            "fwd_ic": round(float(fic), 4), "fwd_ic_sign": fic_sign,
                            "train_ic_sign": train_ic_sign, "verdict": verdict})
        rows.append(row)
        log(f"[{code}] H={H} weeks={weeks:.1f} n_ts={row.get('n_ts')} "
            f"net10={row.get('net10_bps','-')}(t{row.get('t10','-')}) "
            f"net15={row.get('net15_bps','-')}(t{row.get('t15','-')}) "
            f"fwd_ic={row.get('fwd_ic','-')} -> {row['verdict']}")
    df = pd.DataFrame(rows)
    stamp = enh.ts_utc(now_ms)
    df.insert(0, "asof_utc", stamp)
    hist = FWD / "forward_status.csv"
    if hist.exists():
        try:
            old_cols = pd.read_csv(hist, nrows=0).columns.tolist()
            if old_cols != df.columns.tolist():
                archive = FWD / f"forward_status_archived_{int(time.time())}.csv"
                hist.replace(archive)
                log(f"archived incompatible status file to {archive}")
        except Exception as exc:  # noqa: BLE001
            archive = FWD / f"forward_status_archived_{int(time.time())}.csv"
            hist.replace(archive)
            log(f"archived unreadable status file to {archive}: {type(exc).__name__}: {exc}")
    df.to_csv(hist, mode="a", header=not hist.exists(), index=False)
    log(f"\nappended {len(rows)} rows to {hist}")
    log("decision: 0 PASS at window end -> clean closure; any PASS -> second independent forward + tiny live fill test.")


def guard_demo() -> Optional[str]:
    """Fail-closed: demo shadow auto-trade is allowed ONLY in the demo (simulated) account."""
    try:
        m = Secrets().mode
    except Exception as e:  # noqa: BLE001
        return f"无法读取交易模式(.env): {e!r}"
    if m != Mode.DEMO:
        return (f"拒绝: 当前 OKXB_MODE={getattr(m, 'value', m)}; 影子自动交易仅限模拟盘(demo)。"
                "请在 .env 设 OKXB_MODE=demo 并配 demo 密钥后再用 --arm。")
    return None


def guard_live() -> Optional[str]:
    """Fail-closed: LIVE auto-trade requires OKXB_MODE=live (a separate, deliberate switch)."""
    try:
        m = Secrets().mode
    except Exception as e:  # noqa: BLE001
        return f"无法读取交易模式(.env): {e!r}"
    if m != Mode.LIVE:
        return (f"拒绝: 当前 OKXB_MODE={getattr(m, 'value', m)}; 实盘多周期自动交易需 OKXB_MODE=live "
                "并配实盘密钥。")
    return None


def _latest_verdicts() -> dict:
    """最近一批 A/B/C 的 forward verdict ({} 表示尚未 evaluate)。实盘 PASS 门控的依据。"""
    f = FWD / "forward_status.csv"
    if not f.exists():
        return {}
    try:
        df = pd.read_csv(f)
        if "asof_utc" in df.columns and len(df):
            df = df[df["asof_utc"] == df["asof_utc"].iloc[-1]]
        return {str(r["code"]): str(r.get("verdict", "PENDING"))
                for _, r in df.iterrows() if pd.notna(r.get("code"))}
    except Exception:  # noqa: BLE001
        return {}


def _live_gated(frozen: list, verdicts: dict) -> list:
    """实盘可下单候选 = forward verdict 严格为 PASS 的。PENDING/KILL 一律不可实盘 —— 不可绕过的边际门控。"""
    return [c for c in frozen if verdicts.get(c) == "PASS"]


def shadow_size(equity: float, risk_pct: float, sl_pct: float, price: float) -> tuple[float, float]:
    """Risk-budget sizing: max_loss=equity*risk_pct; notional=max_loss/sl_pct; lever clamped [1,10]."""
    equity = max(float(equity or 0.0), 0.0)
    sl_pct = max(float(sl_pct or 0.0), 1e-4)
    notional = (equity * float(risk_pct)) / sl_pct
    lever = notional / equity if equity > 0 else 0.0
    return round(notional, 2), round(max(1.0, min(10.0, lever)), 2)


def tp_sl_pct(daily_vol_ann_pct: float, H_min: int,
              k_sl: float = SHADOW_K_SL, k_tp: float = SHADOW_K_TP) -> tuple[float, float]:
    """H-window SL/TP fractions from annualized daily vol (%). Floor SL at 0.2%."""
    dv = float(daily_vol_ann_pct or 35.0) / 100.0
    h_sigma = dv * (max(H_min, 1) / (365.0 * 24.0 * 60.0)) ** 0.5
    return round(max(0.002, k_sl * h_sigma), 5), round(max(0.003, k_tp * h_sigma), 5)


def _score_now(code: str) -> dict:
    """Score the CURRENT bar with the frozen model (no refit). Returns direction + tau hit + mid."""
    d = FROZEN / code
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    H, tau = meta["H"], meta["tau"]
    blob = pickle.load(open(d / "model.pkl", "rb"))
    model, kind, scaler = blob["model"], blob["kind"], blob["scaler"]
    sel = json.loads((d / "feature_list.json").read_text(encoding="utf-8"))
    med = pd.Series(json.loads((d / "median.json").read_text(encoding="utf-8")))
    # all-cache read: score direction on the most recent cached bar (5m bars lag anyway;
    # the actual demo order, when --arm, uses the live ticker via the manual trade path).
    # 30d lookback >> longest feature window (~3d for H=540) -> fast single-point scoring.
    panel, _ = build_panel(H, days=30, source="refresh", force=False, apply_funding=True, ext_force=False)
    for c in sel:
        if c not in panel.columns:
            panel[c] = np.nan
    last = panel.sort_values("ts").iloc[-1:]
    Xf = last[sel].fillna(med).values
    Xf = scaler.transform(Xf) if scaler is not None else Xf
    sc = float(pmw._score_from_model(kind, model, Xf)[0])
    dv = (float(last["btc_rv_1d_ann_pct"].iloc[0])
          if "btc_rv_1d_ann_pct" in last and pd.notna(last["btc_rv_1d_ann_pct"].iloc[0]) else 35.0)
    cand = enh.load_candles(PERP, BAR, 30, source="refresh", force=False)  # reuse cache build_panel just wrote
    mid = float(cand["c"].iloc[-1]) if len(cand) else float("nan")

    def _g(col):                                    # human-readable OBJECTIVE factor (no model output)
        try:
            v = last[col].iloc[0] if col in last.columns else None
            return round(float(v), 6) if v is not None and np.isfinite(float(v)) else None
        except Exception:  # noqa: BLE001
            return None
    feat = {k: v for k, v in {
        f"{PERP.split('-')[0]}现价": round(mid, 1) if np.isfinite(mid) else None,
        "近端动量roc": _g("roc"),
        "1日年化已实现波动%": round(dv, 1) if np.isfinite(dv) else None,
        "DVOL隐含波动": _g("dvol_daily"),
        "VRP(IV^2-RV^2)": _g("vrp_dvol2_minus_rv2_1d"),
        "DVOL-RV价差": _g("dvol_minus_rv_1d"),
        "资金费率(最近)": _g("funding_last"),
        "永续-现货基差": _g("basis"),
        "OI周转变化": _g("okx_oi_volume_1d_oi_volume_turnover_chg1"),
        "主动买卖不平衡z": _g("okx_taker_1d_taker_imb_z20"),
        "多空账户比": _g("okx_lsr_global_1d_lsr_global"),
        "交易所净流入z(链上)": _g("cm_FlowInExNtv_z20"),
        "MVRV(链上)": _g("cm_CapMVRVCur"),
        "活跃地址变化(链上)": _g("cm_AdrActCnt_chg1"),
    }.items() if v is not None}
    return {"code": code, "H": H, "tau": tau, "score": sc, "tau_hit": abs(sc) >= tau,
            "side": "buy" if sc > 0 else "sell", "ts": int(last["ts"].iloc[0]),
            "daily_vol": dv, "mid": mid, "feat": feat}


def _append_trade(rec: dict) -> None:
    FWD.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([rec]).to_csv(SHADOW_TRADES, mode="a", header=not SHADOW_TRADES.exists(), index=False)


def shadow(armed: bool = False, live: bool = False) -> None:
    """Serial single-slot multi-period auto-trade for BTC (one open slot across A/B/C).
      dry-run (default): show the decision only, place nothing.
      --arm  (demo)    : real DEMO orders on tau-hit for ALL frozen candidates (execution-reality test).
      --live (real $)  : real LIVE orders ONLY for candidates whose forward verdict==PASS (the edge gate);
                          all PENDING -> nothing placed. Requires OKXB_MODE=live.
    The PASS gate on the live path is non-negotiable: live never trades un-validated edge."""
    if live and (err := guard_live()):
        log(err)
        return
    if armed and not live and (err := guard_demo()):
        log(err)
        return
    place = armed or live
    inst = PERP
    now = int(time.time() * 1000)
    st = json.loads(SHADOW_STATE.read_text(encoding="utf-8")) if SHADOW_STATE.exists() else None

    if st:                                              # holding -> monitor exit
        due = now >= st["entry_ts"] + st["H"] * 60_000
        mins = (st["entry_ts"] + st["H"] * 60_000 - now) / 60000.0
        if not place:
            log(f"[shadow {st['code']}] (dry) 持仓中 {st['side']} 距H平仓 {mins:.0f}min" + (" → 将time-exit" if due else ""))
            return
        from okxb.gui import controller as gctl
        pan = gctl.manual_panel_sync(inst)
        poss = [p for p in pan.get("positions", []) if p.get("instId") == inst and float(p.get("pos", 0) or 0) != 0]
        if not poss:                                   # TP/SL already closed it
            _append_trade({**st, "event": "exit", "exit_reason": "tp_or_sl_filled", "exit_utc": enh.ts_utc(now)})
            SHADOW_STATE.unlink(missing_ok=True)
            log(f"[shadow {st['code']}] 仓已被止盈/止损平掉, 记录并清空槽。")
        elif due:
            res = gctl.manual_close_sync(inst)
            px = float((pan.get("ticker") or {}).get("last", 0) or 0) or None
            _append_trade({**st, "event": "exit", "exit_reason": "time_exit", "exit_px": px,
                           "exit_utc": enh.ts_utc(now), "order_result": res})
            SHADOW_STATE.unlink(missing_ok=True)
            log(f"[shadow {st['code']}] 到H time-exit 平仓: {res}")
        else:
            log(f"[shadow {st['code']}] 持仓中, 距H平仓还有 {mins:.0f}min。")
        return

    eq = float(Config.load().get("account.initial_equity_usdt", 1000) or 1000)   # flat -> consider entry
    if place:
        from okxb.gui import controller as gctl
        br = gctl.account_brief_sync()
        if br.get("ok"):
            eq = float(br.get("equity") or eq)
    frozen = [c for c in CANDIDATES if (FROZEN / c / "meta.json").exists()]
    if live:                                            # 实盘: 只对 forward verdict=PASS 的候选下单
        verdicts = _latest_verdicts()
        codes = _live_gated(frozen, verdicts)
        if not codes:
            log(f"[shadow] 实盘门控: 无 PASS 候选 (verdicts={verdicts or '未评估'}); 实盘不下单。"
                "这是 PASS 门控按预期工作, 非故障 —— 现状全 PENDING。")
            return
    else:                                               # demo / dry-run: 全部已冻结候选(执行真实性验证)
        codes = frozen
    for code in codes:
        sig = _score_now(code)
        if not sig["tau_hit"]:
            log(f"[shadow {code}] score={sig['score']:+.4f} tau={sig['tau']:.4f} 未达高置信, 跳过。")
            continue
        sl_pct, tp_pct = tp_sl_pct(sig["daily_vol"], sig["H"])
        notional, lever = shadow_size(eq, SHADOW_RISK_PCT, sl_pct, sig["mid"])
        d = 1 if sig["side"] == "buy" else -1
        tp_px = round(sig["mid"] * (1 + d * tp_pct), 1)
        sl_px = round(sig["mid"] * (1 - d * sl_pct), 1)
        plan = {"code": code, "H": sig["H"], "side": sig["side"], "entry_ts": now,
                "entry_utc": enh.ts_utc(now), "entry_mid": sig["mid"], "score": round(sig["score"], 4),
                "notional": notional, "lever": lever, "sl_pct": sl_pct, "tp_pct": tp_pct,
                "tp_px": tp_px, "sl_px": sl_px}
        if not place:
            log(f"[shadow {code}] (dry) 将入场 {sig['side']} 名义≈{notional}U 杠杆{lever}x "
                f"TP={tp_px} SL={sl_px} (现价{sig['mid']}) — 加 --arm(demo)/--live(实盘,仅PASS) 才真下单。")
            return
        acct = "live" if live else "demo"
        plan["account"] = acct
        from okxb.gui import controller as gctl
        gctl.manual_set_leverage_sync(inst, int(lever))
        res = gctl.manual_bracket_sync(inst, sig["side"], SHADOW_ORD, notional, "usdt", None, tp_px, sl_px)
        plan.update({"event": "entry", "order_result": res})
        SHADOW_STATE.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        _append_trade(plan)
        log(f"[shadow {code}] 入场({acct}): {res}")
        return
    log("[shadow] 当前无候选达高置信 tau, 不入场 (常态: 信号稀疏且 gate=False)。")


def auto(live: bool = False) -> None:
    """一轮自动交易: 实盘先 evaluate 刷新 forward verdict(PASS门控依据)再 shadow;
    demo 直接 shadow(执行真实性验证)。供 GUI / 计划任务每隔 N 分钟调用一次。"""
    if live:
        evaluate()                  # 实盘门控需要最新 verdict
    shadow(armed=not live, live=live)


def snapshot() -> None:
    """输出【客观因子】+ A/B/C 当前打分(JSON, 前缀 SNAPSHOT_JSON)。
    features 是给 AI 盲分析的客观数据(不含任何模型结论); scores 单独给程序做对照。"""
    feats: dict = {}
    scores: dict = {}
    for code in CANDIDATES:
        if not (FROZEN / code / "meta.json").exists():
            continue
        sig = _score_now(code)
        scores[code] = {"H": sig["H"], "score": round(sig["score"], 4), "tau": round(sig["tau"], 4),
                        "hit": bool(sig["tau_hit"]), "side": sig["side"]}
        if not feats:
            feats = sig.get("feat", {})
    print("SNAPSHOT_JSON " + json.dumps({"features": feats, "scores": scores}, ensure_ascii=False))


def selftest() -> None:
    # 显式 t_pass=2.13 保持这些断言稳定 (不随族大小变化); evaluate() 实际用族级 T_PASS。
    assert forward_verdict(5.0, 2.5, 6.0, 60, 1, 1, 2.0, t_pass=2.13) == "PASS"
    assert forward_verdict(5.0, 2.5, 6.0, 40, 1, 1, 2.0, t_pass=2.13) == "PENDING"   # n<50
    assert forward_verdict(5.0, 1.8, 6.0, 60, 1, 1, 2.0, t_pass=2.13) == "PENDING"   # t<2.13
    assert forward_verdict(5.0, 2.5, -1.0, 60, 1, 1, 2.0, t_pass=2.13) == "KILL"     # net10<=0
    assert forward_verdict(5.0, 2.5, 6.0, 60, -1, 1, 2.0, t_pass=2.13) == "KILL"     # IC flipped
    assert forward_verdict(5.0, 2.5, 6.0, 20, 1, 1, 6.5, t_pass=2.13) == "KILL"      # 6wk and n<30
    # RV-3: Bonferroni 阈值随候选数单调收紧; 默认 T_PASS 已按真实族 (>=7) 校正, 比旧的 2.13 更严
    assert bonferroni_t(3) < bonferroni_t(7) < bonferroni_t(20)
    assert abs(bonferroni_t(3) - 2.128) < 0.03
    assert T_PASS >= bonferroni_t(7) - 1e-9 and T_PASS > 2.13
    # RV-1: manifest 内容哈希能检出冻结后改动
    import shutil as _sh
    import tempfile as _tf
    _td = Path(_tf.mkdtemp())
    try:
        (_td / "model.pkl").write_bytes(b"frozen-bytes")
        (_td / "meta.json").write_text("{}", encoding="utf-8")
        _write_manifest(_td)
        assert _verify_manifest(_td) is None
        (_td / "meta.json").write_text('{"tampered": 1}', encoding="utf-8")
        assert _verify_manifest(_td) not in (None, "no-manifest")        # 检出篡改
    finally:
        _sh.rmtree(_td, ignore_errors=True)
    assert _read_dead("__nonexistent_code__") is None                    # RV-2: 无死标返回 None
    n, lv = shadow_size(1000.0, 0.003, 0.01, 60000.0)                  # max_loss=3 -> notional 300, lever clamps to 1
    assert abs(n - 300.0) < 1e-6 and lv == 1.0
    n2, lv2 = shadow_size(1000.0, 0.003, 0.001, 60000.0)               # tighter SL -> notional 3000, lever 3
    assert abs(n2 - 3000.0) < 1e-6 and abs(lv2 - 3.0) < 1e-6
    sl, tp = tp_sl_pct(35.0, 180)
    assert tp > sl > 0.0
    # live PASS-gate: only verdict==PASS candidates are tradable live; all-PENDING -> none
    assert _live_gated(["A", "B", "C"], {"A": "PENDING", "B": "PASS", "C": "KILL"}) == ["B"]
    assert _live_gated(["A", "B", "C"], {"A": "PENDING", "B": "PENDING", "C": "PENDING"}) == []
    assert _live_gated(["A", "B", "C"], {}) == []                       # not evaluated yet -> none
    log("selftest OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="Forward shadow test (pre-registered).")
    ap.add_argument("--mode", choices=["freeze", "evaluate", "selftest", "shadow", "snapshot", "auto"],
                    required=True)
    ap.add_argument("--asset", default="btc", choices=list(ASSET_CFG.keys()),
                    help="标的: btc(默认) 或 eth。各资产独立的 frozen/forward 目录与候选集。")
    ap.add_argument("--arm", action="store_true",
                    help="shadow/auto 下真在 demo 下单 (默认 dry-run: 只显示决策, 不下单)")
    ap.add_argument("--live", action="store_true",
                    help="实盘下单, 仅对 forward verdict=PASS 的候选 (需 OKXB_MODE=live); 全 PENDING 则不下单")
    args = ap.parse_args()
    _set_asset(args.asset)
    if args.mode == "shadow":
        shadow(armed=args.arm, live=args.live)
    elif args.mode == "auto":
        auto(live=args.live)
    else:
        {"freeze": freeze, "evaluate": evaluate, "selftest": selftest, "snapshot": snapshot}[args.mode]()


if __name__ == "__main__":
    main()
