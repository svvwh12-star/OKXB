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
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]               # OKXB
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPTS))

import run_btc_enhanced_research as enh                  # noqa: E402  (reuse the exact pipeline)
from okxb.research import candle_data as cd              # noqa: E402
from okxb.research import candle_research as cr          # noqa: E402
from okxb.research import feature_lab as fl              # noqa: E402
from okxb.research import pro_model_workflow as pmw      # noqa: E402
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

# ----- pre-registered, frozen candidate set (must match the committed protocol) -----
# oos_ic_sign = the sign of the ORIGINAL OOS primary_ic (from enhanced_summary.csv), NOT the
# in-sample fit (which is always +). The forward "same-sign" check compares against THIS.
# C (9h) is pre-registered as -1: it is a falsification target (OOS IC was -0.0356).
CANDIDATES = {
    "A": {"H": 180, "model": "hist_gbm", "top_frac": 0.01, "oos_ic_sign": +1},
    "B": {"H": 360, "model": "lightgbm", "top_frac": 0.02, "oos_ic_sign": +1},
    "C": {"H": 540, "model": "mlp",      "top_frac": 0.05, "oos_ic_sign": -1},
}
BAR = "5m"
K_SEL = 30
PRESET = "deep"
TAKER_BPS = 10.0
STRESS_MULT = 1.5            # protocol: net judged after cost x1.5  -> taker 10 -> 15 bps
T_PASS = 2.13               # Bonferroni for 3 candidates (alpha=0.05)
N_MIN = 50
N_KILL = 30
WEEK_MS = 7 * 86_400_000


def log(m: str) -> None:
    print(m, flush=True)


def build_panel(H: int, *, days: int, source: str, force: bool, apply_funding: bool):
    """Reproduce the enhanced pipeline (build_augmented_bank + funding-hold + enrich) for one H."""
    btc = enh.load_candles("BTC-USDT-SWAP", BAR, days, source=source, force=force)
    spot = enh.load_candles("BTC-USDT", BAR, days, source=source, force=force)
    funding = cd.fetch_funding_series("BTC-USDT-SWAP", days + 5, log=log)
    daily_external, _ = enh.fetch_daily_external(days, force=force)
    short_external, _ = enh.fetch_short_intraday_external(force=force)
    perp = {"BTC-USDT-SWAP": btc}
    panel, bar_ms = pmw.build_augmented_bank(perp, {"BTC-USDT-SWAP": spot}, {"BTC-USDT-SWAP": funding}, BAR, H)
    if apply_funding:
        panel = enh.apply_funding_hold_cost(panel, funding, H)
    panel = enh.enrich_panel(panel, btc, BAR, daily_external, short_external, include_short=False)
    return panel, bar_ms


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
        log(f"[freeze {code}] H={H} {model_name} sel={len(sel)} tau={tau:.5f} "
            f"oos_ic_sign={ic_sign:+d} (in-sample {insample_ic_sign:+d}) cutoff={enh.ts_utc(int(data['ts'].max()))}")


def _net(port: np.ndarray, cost_bps: float):
    st = cr._stats_from_gross(port, cost_bps, 1) if len(port) >= 8 else None
    return (st["net_bps"], st["nw_t"]) if st else (np.nan, np.nan)


def forward_verdict(net15: float, t15: float, net10: float, n_ts: int,
                    fwd_ic_sign: int, train_ic_sign: int, weeks: float) -> str:
    """Pure PASS/KILL logic per protocol section 4/5."""
    if n_ts >= 8:
        if (net10 is not None and net10 <= 0) or (fwd_ic_sign != 0 and fwd_ic_sign != train_ic_sign):
            return "KILL"
    if weeks >= 6 and n_ts < N_KILL:
        return "KILL"
    if (net15 is not None and net15 > 0 and (t15 or 0) > T_PASS and n_ts >= N_MIN
            and fwd_ic_sign == train_ic_sign):
        return "PASS"
    return "PENDING"


def evaluate() -> None:
    FWD.mkdir(parents=True, exist_ok=True)
    now_ms = int(time.time() * 1000)
    rows = []
    for code, cfg in CANDIDATES.items():
        d = FROZEN / code
        if not (d / "meta.json").exists():
            log(f"[{code}] not frozen yet — run --mode freeze first")
            continue
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        H, bar_ms, tau = meta["H"], meta["bar_ms"], meta["tau"]
        cutoff, train_ic_sign = meta["freeze_cutoff_ts"], meta["ic_sign"]
        official_start = max(int(cutoff), int(meta.get("official_forward_start_ts", meta["frozen_at_ms"])))
        weeks = (now_ms - meta["frozen_at_ms"]) / WEEK_MS
        blob = pickle.load(open(d / "model.pkl", "rb"))
        model, kind, scaler = blob["model"], blob["kind"], blob["scaler"]
        sel = json.loads((d / "feature_list.json").read_text(encoding="utf-8"))
        med = pd.Series(json.loads((d / "median.json").read_text(encoding="utf-8")))

        # evaluate on the TRUE net target (taker + funding always folded in), only settled forward bars
        days = 150 + int((now_ms - cutoff) / 86_400_000) + 10
        panel, _ = build_panel(H, days=days, source="refresh", force=True, apply_funding=True)
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
    """Fail-closed: shadow auto-trade is allowed ONLY in the demo (simulated) account."""
    try:
        m = Secrets().mode
    except Exception as e:  # noqa: BLE001
        return f"无法读取交易模式(.env): {e!r}"
    if m != Mode.DEMO:
        return (f"拒绝: 当前 OKXB_MODE={getattr(m, 'value', m)}; 影子自动交易仅限模拟盘(demo)。"
                "请在 .env 设 OKXB_MODE=demo 并配 demo 密钥后再用 --arm。")
    return None


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
    panel, _ = build_panel(H, days=200, source="refresh", force=True, apply_funding=True)
    for c in sel:
        if c not in panel.columns:
            panel[c] = np.nan
    last = panel.sort_values("ts").iloc[-1:]
    Xf = last[sel].fillna(med).values
    Xf = scaler.transform(Xf) if scaler is not None else Xf
    sc = float(pmw._score_from_model(kind, model, Xf)[0])
    dv = (float(last["btc_rv_1d_ann_pct"].iloc[0])
          if "btc_rv_1d_ann_pct" in last and pd.notna(last["btc_rv_1d_ann_pct"].iloc[0]) else 35.0)
    cand = enh.load_candles("BTC-USDT-SWAP", BAR, 200, source="refresh", force=False)  # reuse cache build_panel just wrote
    mid = float(cand["c"].iloc[-1]) if len(cand) else float("nan")
    return {"code": code, "H": H, "tau": tau, "score": sc, "tau_hit": abs(sc) >= tau,
            "side": "buy" if sc > 0 else "sell", "ts": int(last["ts"].iloc[0]), "daily_vol": dv, "mid": mid}


def _append_trade(rec: dict) -> None:
    FWD.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([rec]).to_csv(SHADOW_TRADES, mode="a", header=not SHADOW_TRADES.exists(), index=False)


def shadow(armed: bool = False) -> None:
    """DEMO-only serial single-slot shadow auto-trade. Default DRY-RUN (no orders);
    --arm places real demo orders via the existing manual trade path. Records paper-vs-fill."""
    if armed and (err := guard_demo()):
        log(err)
        return
    inst = "BTC-USDT-SWAP"
    now = int(time.time() * 1000)
    st = json.loads(SHADOW_STATE.read_text(encoding="utf-8")) if SHADOW_STATE.exists() else None

    if st:                                              # holding -> monitor exit
        due = now >= st["entry_ts"] + st["H"] * 60_000
        mins = (st["entry_ts"] + st["H"] * 60_000 - now) / 60000.0
        if not armed:
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
    if armed:
        from okxb.gui import controller as gctl
        br = gctl.account_brief_sync()
        if br.get("ok"):
            eq = float(br.get("equity") or eq)
    for code in CANDIDATES:
        if not (FROZEN / code / "meta.json").exists():
            continue
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
        if not armed:
            log(f"[shadow {code}] (dry) 将入场 {sig['side']} 名义≈{notional}U 杠杆{lever}x "
                f"TP={tp_px} SL={sl_px} (现价{sig['mid']}) — 加 --arm 才在 demo 真下单。")
            return
        from okxb.gui import controller as gctl
        gctl.manual_set_leverage_sync(inst, int(lever))
        res = gctl.manual_bracket_sync(inst, sig["side"], SHADOW_ORD, notional, "usdt", None, tp_px, sl_px)
        plan.update({"event": "entry", "order_result": res})
        SHADOW_STATE.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        _append_trade(plan)
        log(f"[shadow {code}] 入场(demo): {res}")
        return
    log("[shadow] 当前无候选达高置信 tau, 不入场 (常态: 信号稀疏且 gate=False)。")


def selftest() -> None:
    assert forward_verdict(5.0, 2.5, 6.0, 60, 1, 1, 2.0) == "PASS"
    assert forward_verdict(5.0, 2.5, 6.0, 40, 1, 1, 2.0) == "PENDING"   # n<50
    assert forward_verdict(5.0, 1.8, 6.0, 60, 1, 1, 2.0) == "PENDING"   # t<2.13
    assert forward_verdict(5.0, 2.5, -1.0, 60, 1, 1, 2.0) == "KILL"     # net10<=0
    assert forward_verdict(5.0, 2.5, 6.0, 60, -1, 1, 2.0) == "KILL"     # IC flipped
    assert forward_verdict(5.0, 2.5, 6.0, 20, 1, 1, 6.5) == "KILL"      # 6wk and n<30
    n, lv = shadow_size(1000.0, 0.003, 0.01, 60000.0)                  # max_loss=3 -> notional 300, lever clamps to 1
    assert abs(n - 300.0) < 1e-6 and lv == 1.0
    n2, lv2 = shadow_size(1000.0, 0.003, 0.001, 60000.0)               # tighter SL -> notional 3000, lever 3
    assert abs(n2 - 3000.0) < 1e-6 and abs(lv2 - 3.0) < 1e-6
    sl, tp = tp_sl_pct(35.0, 180)
    assert tp > sl > 0.0
    log("selftest OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="Forward shadow test (pre-registered).")
    ap.add_argument("--mode", choices=["freeze", "evaluate", "selftest", "shadow"], required=True)
    ap.add_argument("--arm", action="store_true",
                    help="shadow 模式下真在 demo 下单 (默认 dry-run: 只显示决策, 不下单)")
    args = ap.parse_args()
    if args.mode == "shadow":
        shadow(armed=args.arm)
    else:
        {"freeze": freeze, "evaluate": evaluate, "selftest": selftest}[args.mode]()


if __name__ == "__main__":
    main()
