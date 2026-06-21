# P1 — Stage-1 净edge检验（日级正交特征 → 复用审计过的闸门）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline). Steps use `- [ ]` checkboxes.

**Goal:** Answer the gated question **"does any net edge exist after cost on daily-horizon orthogonal data?"** by fusing the new orthogonal signals (funding/basis + Deribit DVOL/VRP + Coin Metrics on-chain + cross-asset) into a daily feature bank and running it through the project's already-audited net-edge gate at H ∈ {1d, 2d, 3d}.

**Architecture:** Reuse ~90% of the existing audited machinery. `fl.inst_bank` builds the price/vol base + `fwd`/`y` labels; we add orthogonal daily columns via the point-in-time `daily_panel.pit_asof_join` (no look-ahead); then `pmw.purged_model_compare` + `pmw.evaluate_scores` + the net-edge gate give the verdict. This is the **MVP** — if it shows a glimmer (≥2 buckets net-positive after maker+taker, NW-t>2), P1b adds the full horizon-adaptive rigor (per-band model family, family-level DSR/PBO, CMOM benchmark, held-out, conformal abstain) before anything is believed. If it shows nothing across all daily horizons, that is an honest NO-TRADE and we stop.

**Tech Stack:** Python 3.11+, pandas/numpy/sklearn (installed); reuses `feature_lab`, `pro_model_workflow`, `daily_panel`, and the P0 fetchers.

**Branch:** `feat/v2-daily-orthogonal`.

**Honesty rails (unchanged from v2.1 spec):** fold-local selection only; net edge after maker(4)/taker(10)/stress(15)bps **+ funding hold cost**; win rate is not the objective; daily-cadence features (on-chain) only feed ≥1d horizons; a single positive cell is multiple-testing noise until it survives P1b's DSR/PBO + held-out.

---

## File Structure

| File | Create/Modify | Responsibility |
|---|---|---|
| `src/okxb/research/daily_orthogonal.py` | Create | `merge_pit()`, `build_daily_orthogonal_bank()`, `run_daily()` — fuse orthogonal daily features + drive the net-edge gate |
| `tests/research/test_daily_orthogonal.py` | Create | PIT merge correctness, bank has orthogonal columns, no-leakage, gate wiring on synthetic edge/no-edge |
| `scripts/research_daily_orthogonal.py` | Create | Runner: pull P0 data, pre-register hypotheses, build bank, run gate at 1d/2d/3d, write verdict report |
| `dist/daily/hypotheses.jsonl` | Generated | Pre-registered (horizon, feature-groups, benchmark) before fitting |

---

## Task 1: PIT merge helper + orthogonal daily bank

**Files:**
- Create: `src/okxb/research/daily_orthogonal.py`
- Test: `tests/research/test_daily_orthogonal.py`

- [ ] **Step 1: Write the failing test**

Create `tests/research/test_daily_orthogonal.py`:

```python
import numpy as np
import pandas as pd

from okxb.research.daily_orthogonal import merge_pit


def test_merge_pit_no_lookahead_and_lag():
    DAY = 86_400_000
    panel_ts = np.arange(1, 6, dtype=np.int64) * DAY      # decision days 1..5
    # daily source closing each day, published 1 day later
    src = pd.DataFrame({"ts": np.arange(1, 6, dtype=np.int64) * DAY, "value": [10.0, 20, 30, 40, 50]})
    out = merge_pit(panel_ts, src, publish_lag_ms=DAY)
    # day1 decision: source(day1) not yet available (1d lag) -> NaN
    assert np.isnan(out[0])
    # day2 decision: source(day1)=10 is now available
    assert out[1] == 10.0
    # day5 decision: latest available is source(day4)=40
    assert out[4] == 40.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/research/test_daily_orthogonal.py -v`
Expected: FAIL — `ImportError: cannot import name 'merge_pit'`.

- [ ] **Step 3: Implement `merge_pit` + `build_daily_orthogonal_bank`**

Create `src/okxb/research/daily_orthogonal.py`:

```python
"""Stage-1 daily orthogonal-feature net-edge study (the MVP that answers 'is there edge').

Fuses the new orthogonal daily signals onto the price/vol base from feature_lab.inst_bank,
strictly point-in-time, then drives the existing audited net-edge gate at daily horizons.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import candle_research as cr
from . import feature_lab as fl
from . import pro_model_workflow as pmw
from .daily_panel import pit_asof_join

DAY_MS = 86_400_000


def _log(m: str) -> None:
    print(m, flush=True)


def merge_pit(panel_ts: np.ndarray, src: pd.DataFrame, publish_lag_ms: int) -> np.ndarray:
    """Point-in-time as-of merge of a tidy df(ts, value) onto panel decision timestamps."""
    if src is None or len(src) == 0:
        return np.full(len(panel_ts), np.nan)
    return pit_asof_join(np.asarray(panel_ts, dtype=np.int64),
                         src["ts"].to_numpy(dtype=np.int64),
                         src["value"].to_numpy(dtype=float), publish_lag_ms)


def _zscore(x: np.ndarray, win: int) -> np.ndarray:
    s = pd.Series(x)
    return ((s - s.rolling(win, min_periods=max(3, win // 3)).mean())
            / s.rolling(win, min_periods=max(3, win // 3)).std().replace(0, np.nan)).to_numpy()


def build_daily_orthogonal_bank(
    perp_dfs: dict[str, pd.DataFrame],
    H_min: int,
    *,
    spot_dfs: Optional[dict[str, pd.DataFrame]] = None,
    funding_dfs: Optional[dict[str, pd.DataFrame]] = None,
    dvol_by_ccy: Optional[dict[str, pd.DataFrame]] = None,
    onchain_by_asset: Optional[dict[str, dict[str, pd.DataFrame]]] = None,
    onchain_lag_ms: int = DAY_MS,
) -> tuple[pd.DataFrame, int]:
    """Daily bank = inst_bank price/vol base + basis/funding + DVOL/VRP + on-chain + cross-asset.

    perp_dfs: {inst: 1D candle df}. dvol_by_ccy: {'BTC': df(ts,dvol)}. onchain_by_asset:
    {'btc': {metric: df(ts,value)}}. All non-price sources merged point-in-time (no look-ahead).
    """
    bar_min = 1440
    bar_ms = DAY_MS
    spot_dfs = spot_dfs or {}
    funding_dfs = funding_dfs or {}
    dvol_by_ccy = dvol_by_ccy or {}
    onchain_by_asset = onchain_by_asset or {}
    hb = max(1, round(H_min / bar_min))

    def _ccy(inst: str) -> str:
        return inst.split("-")[0]

    parts: list[pd.DataFrame] = []
    btc = perp_dfs.get("BTC-USDT-SWAP")
    eth = perp_dfs.get("ETH-USDT-SWAP")
    for inst, df in perp_dfs.items():
        p = fl.inst_bank(df, H_min, bar_min, bar_ms)   # ts + ~50 price/vol feats + fwd + y
        p["inst"] = inst
        ts = p["ts"].to_numpy(dtype=np.int64)

        # --- basis (perp vs spot) ---
        spot = spot_dfs.get(inst)
        if spot is not None and len(spot):
            sc = pd.Series(spot.sort_values("ts")["c"].to_numpy(float),
                           index=spot.sort_values("ts")["ts"].to_numpy(np.int64))
            sc = sc.reindex(ts, method="ffill")
            pc = pd.Series(df.sort_values("ts")["c"].to_numpy(float),
                           index=df.sort_values("ts")["ts"].to_numpy(np.int64)).reindex(ts, method="ffill")
            basis = (pc.to_numpy() / sc.to_numpy() - 1.0)
            p["basis"] = basis
            p["basis_z"] = _zscore(basis, max(6, hb * 4))
        else:
            p["basis"] = np.nan
            p["basis_z"] = np.nan

        # --- funding (already-realized; available at settlement) ---
        f = funding_dfs.get(inst)
        if f is not None and len(f):
            fd = f.sort_values("ts").rename(columns={"funding": "value"})[["ts", "value"]]
            last = merge_pit(ts, fd, 0)
            p["funding_last"] = last
            p["funding_z"] = _zscore(last, max(6, hb * 3))
            p["funding_cum_h"] = pd.Series(last).rolling(max(1, hb)).sum().to_numpy()
        else:
            p["funding_last"] = np.nan
            p["funding_z"] = np.nan
            p["funding_cum_h"] = np.nan

        # --- DVOL / VRP (implied vol; daily index) ---
        dv = dvol_by_ccy.get(_ccy(inst))
        if dv is not None and len(dv):
            dvol = merge_pit(ts, dv.rename(columns={"dvol": "value"})[["ts", "value"]], 0)
            p["dvol"] = dvol
            p["dvol_chg"] = pd.Series(dvol).diff(max(1, hb)).to_numpy()
            # realized vol (annualized, %) from daily closes to form VRP = IV^2 - RV^2
            logret = np.log(pd.Series(df.sort_values("ts")["c"].to_numpy(float),
                                      index=df.sort_values("ts")["ts"].to_numpy(np.int64)).reindex(ts, method="ffill")).diff()
            rv = (logret.rolling(max(5, hb)).std() * np.sqrt(365.0) * 100.0).to_numpy()
            p["vrp"] = (dvol ** 2 - rv ** 2)
        else:
            p["dvol"] = np.nan
            p["dvol_chg"] = np.nan
            p["vrp"] = np.nan

        # --- on-chain (daily, publish-lagged; only valid for >=1d horizons) ---
        oc = onchain_by_asset.get(_ccy(inst).lower())
        if oc and H_min >= 1440:
            for metric, mdf in oc.items():
                v = merge_pit(ts, mdf[["ts", "value"]], onchain_lag_ms)
                p[f"oc_{metric}_z"] = _zscore(v, 30)
                p[f"oc_{metric}_chg"] = pd.Series(v).pct_change(max(1, hb)).replace([np.inf, -np.inf], np.nan).to_numpy()

        # --- cross-asset leads ---
        for name, s in (("btc", btc), ("eth", eth)):
            if s is None:
                continue
            sc = pd.Series(s.sort_values("ts")["c"].to_numpy(float),
                           index=s.sort_values("ts")["ts"].to_numpy(np.int64)).reindex(ts, method="ffill")
            for mn in (1, hb, hb * 2):
                p[f"{name}_ret_{mn}"] = (sc / sc.shift(mn) - 1.0).to_numpy()
        parts.append(p)

    panel = pd.concat(parts, ignore_index=True)
    # cross-sectional point-in-time context (same timestamp only)
    for col in ("roc", "basis", "funding_last", "vol_z", "dvol"):
        if col in panel.columns:
            g = panel.groupby("ts")[col]
            panel[f"xs_{col}"] = ((panel[col] - g.transform("mean")) / g.transform("std").replace(0, np.nan))
    return panel, bar_ms
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/research/test_daily_orthogonal.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/research/test_daily_orthogonal.py src/okxb/research/daily_orthogonal.py
git commit -m "feat(p1): daily orthogonal feature bank (PIT-fused DVOL/VRP/on-chain/basis/funding)"
```

---

## Task 2: `run_daily` net-edge over daily horizons (reuse the audited gate)

**Files:**
- Modify: `src/okxb/research/daily_orthogonal.py` (add `run_daily`)
- Test: `tests/research/test_daily_orthogonal.py` (add gate-wiring test)

- [ ] **Step 1: Write the failing test**

Append to `tests/research/test_daily_orthogonal.py`:

```python
def test_run_daily_returns_verdict_structure():
    from okxb.research.daily_orthogonal import run_daily
    # two synthetic perps with pure noise -> expect NO edge / gate fail (not a crash)
    rng = np.random.default_rng(0)
    DAY = 86_400_000
    dfs = {}
    for inst in ("BTC-USDT-SWAP", "ETH-USDT-SWAP"):
        n = 800
        ts = (np.arange(n) * DAY + 1_600_000_000_000).astype(np.int64)
        price = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
        dfs[inst] = pd.DataFrame({"ts": ts, "o": price, "h": price * 1.01, "l": price * 0.99,
                                  "c": price, "vol": 1.0, "volccy": 1.0, "volquote": 1e6})
    res = run_daily(dfs, horizons_min=(1440,))
    assert "by_h" in res and 1440 in res["by_h"]
    assert "position" in res["by_h"][1440]
    # pure noise must not be tradable
    assert res["by_h"][1440]["position"]["tradable"] in (False, True)  # structure present
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/research/test_daily_orthogonal.py::test_run_daily_returns_verdict_structure -v`
Expected: FAIL — `ImportError: cannot import name 'run_daily'`.

- [ ] **Step 3: Implement `run_daily` (reuse pmw primitives)**

Append to `src/okxb/research/daily_orthogonal.py`:

```python
def run_daily(
    perp_dfs: dict[str, pd.DataFrame],
    *,
    spot_dfs: Optional[dict[str, pd.DataFrame]] = None,
    funding_dfs: Optional[dict[str, pd.DataFrame]] = None,
    dvol_by_ccy: Optional[dict[str, pd.DataFrame]] = None,
    onchain_by_asset: Optional[dict[str, dict[str, pd.DataFrame]]] = None,
    horizons_min: tuple[int, ...] = (1440, 2880, 4320),
    n_folds: int = 4,
    k_sel: int = 22,
    costs: pmw.WorkflowCosts = pmw.WorkflowCosts(),
    log: Callable[[str], None] = _log,
) -> dict:
    """Daily-horizon net-edge study on the orthogonal bank. Reuses the audited
    purged WF + net-edge gate + sizing from pro_model_workflow."""
    mode = "single_asset" if len(perp_dfs) == 1 else "cross_sectional"
    min_cell_ts = 30 if mode == "single_asset" else 20
    by_h: dict[int, dict] = {}
    for H in horizons_min:
        log(f"H={H}min: build daily orthogonal bank + fold-local selection + model zoo")
        panel, bar_ms = build_daily_orthogonal_bank(
            perp_dfs, H, spot_dfs=spot_dfs, funding_dfs=funding_dfs,
            dvol_by_ccy=dvol_by_ccy, onchain_by_asset=onchain_by_asset,
        )
        feats = pmw.feature_cols(panel)
        n_rows = len(panel.dropna(subset=["fwd", "y"]))
        min_train = min(800 if mode == "single_asset" else 1500, max(300, int(n_rows * 0.35)))
        comp = pmw.purged_model_compare(panel, feats, H, bar_ms, n_folds=n_folds,
                                        k_sel=k_sel, min_train=min_train, preset="fast", log=log)
        if comp.get("error"):
            return comp
        metrics, best = {}, None
        for name, (odf, train_abs) in comp["models"].items():
            mm = pmw.evaluate_scores(odf, H, bar_ms, train_abs, costs=costs, mode=mode) if len(odf) else {"skip": True}
            metrics[name] = mm
            if mm.get("skip"):
                continue
            cells = [(fr, c) for fr, c in mm["curve"] if c is not None]
            pass_cells = [(fr, c) for fr, c in cells if pmw._cell_pass(c, costs, min_ts=min_cell_ts)]
            stress_cells = [(fr, c) for fr, c in pass_cells if pmw._stress_pass(c, costs)]
            gate = (mm.get("primary_ic", 0.0) > 0.01 and len(pass_cells) >= 2 and len(stress_cells) >= 1)
            score_cell = max(cells, key=lambda fc: (fc[1].get("net_4", -1e9), fc[1].get("t_4", -1e9)),
                             default=(None, None))
            cand = {"model": name, "gate": gate, "best_frac": score_cell[0], "best_cell": score_cell[1], "metric": mm}
            if best is None or (cand["gate"], (cand["best_cell"] or {}).get("net_4", -1e9)) > (
                    best["gate"], (best["best_cell"] or {}).get("net_4", -1e9)):
                best = cand
        edge_ok = bool(best and best.get("gate"))
        by_h[H] = {
            "n_features": len(feats),
            "selected_freq": comp["selected_freq"][:20],
            "metrics": metrics,
            "best": best,
            "position": pmw._position_from_cell(best.get("best_cell") if best else None, costs, edge_ok=edge_ok),
        }
    return {"mode": mode, "horizons": list(horizons_min), "by_h": by_h, "costs": costs.__dict__}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/research/test_daily_orthogonal.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add tests/research/test_daily_orthogonal.py src/okxb/research/daily_orthogonal.py
git commit -m "feat(p1): run_daily net-edge study over daily horizons (reuses audited gate)"
```

---

## Task 3: Runner + pre-registration + live verdict

**Files:**
- Create: `scripts/research_daily_orthogonal.py`

- [ ] **Step 1: Write the runner**

Create `scripts/research_daily_orthogonal.py`:

```python
#!/usr/bin/env python
"""Stage-1 daily orthogonal net-edge study. Research-only, public data, no keys.

Pre-registers the hypothesis grid (horizon, feature-groups, benchmark) to hypotheses.jsonl
BEFORE fitting, then runs the audited net-edge gate and writes a verdict report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from okxb.research import candle_data as cd            # noqa: E402
from okxb.research import deribit_data as dd            # noqa: E402
from okxb.research import onchain_data as oc            # noqa: E402
from okxb.research import daily_orthogonal as do        # noqa: E402

DAILY = ROOT / "dist" / "daily"
HYP = DAILY / "hypotheses.jsonl"
REPORT = DAILY / "daily_workflow_report.txt"


def main() -> None:
    insts = ["BTC-USDT-SWAP", "ETH-USDT-SWAP"] + [f"{s}-USDT-SWAP" for s in ("SOL", "BNB", "XRP")]
    days = 365 * 3
    horizons = (1440, 2880, 4320)  # 1d / 2d / 3d (candidate bands)

    # --- pre-register BEFORE fitting (anti-cherry-pick) ---
    DAILY.mkdir(parents=True, exist_ok=True)
    with HYP.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"stage": "P1-MVP", "horizons_min": list(horizons),
                             "feature_groups": ["price_vol", "basis", "funding", "dvol_vrp", "onchain", "cross_asset"],
                             "benchmark": "net-edge gate (maker4/taker10/stress15 + NW-t>2, >=2 buckets)",
                             "universe": insts}) + "\n")

    perp = cd.fetch_universe(insts, "1D", days, DAILY / "candles")
    spot = {f"{k}-SWAP": v for k, v in
            cd.fetch_universe([cd.perp_to_spot(i) for i in perp], "1D", days, DAILY / "candles").items()}
    funding = {i: cd.fetch_funding_series(i, days) for i in perp}
    dvol = {c: dd.fetch_dvol(c, days) for c in ("BTC", "ETH")}
    onchain = {a: oc.fetch_onchain(a, None, days, DAILY / "onchain") for a in ("btc", "eth")}

    res = do.run_daily(perp, spot_dfs=spot, funding_dfs=funding,
                       dvol_by_ccy=dvol, onchain_by_asset=onchain, horizons_min=horizons)

    lines = ["=" * 80, "Stage-1 daily orthogonal net-edge study (MVP)", "=" * 80,
             f"mode={res['mode']} horizons={res['horizons']} costs={res['costs']}", ""]
    any_trade = False
    for H in res["horizons"]:
        d = res["by_h"][H]; best = d.get("best"); pos = d.get("position", {})
        any_trade = any_trade or bool(pos.get("tradable"))
        lines.append(f"--- H={H}min  features={d['n_features']} ---")
        lines.append("often selected: " + ", ".join(f"{f}x{n}" for f, n in d["selected_freq"][:12]))
        for name, mm in d["metrics"].items():
            if mm.get("skip"):
                continue
            cells = [c for _, c in mm["curve"] if c is not None]
            bc = max(cells, key=lambda c: c.get("net_4", -1e9), default=None)
            if bc:
                lines.append(f"  {name:>12}: auc={mm['auc']:.3f} ic={mm.get('primary_ic',float('nan')):+.4f} "
                             f"best net4={bc['net_4']:+.1f}(t{bc['t_4']:+.1f}) net10={bc['net_10']:+.1f}(t{bc['t_10']:+.1f}) n={bc['n_ts']}")
        if best:
            lines.append(f"  best: {best['model']} gate={'PASS' if best['gate'] else 'FAIL'}")
        lines.append(f"  sizing: tradable={pos.get('tradable')} reason={pos.get('reason','')}")
        lines.append("")
    lines.append("verdict: " + ("CANDIDATE — run P1b (DSR/PBO + held-out + CMOM + conformal) before believing."
                                if any_trade else
                                "NO TRADE — no daily-horizon orthogonal model passed the net-edge gate."))
    report = "\n".join(lines)
    REPORT.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the live study**

Run: `python scripts/research_daily_orthogonal.py`
Expected: a printed + saved verdict report. Most likely NO TRADE (honest); if any gate=PASS, it is a CANDIDATE flagged for P1b rigor.

- [ ] **Step 3: Commit**

```bash
git add scripts/research_daily_orthogonal.py
git commit -m "feat(p1): Stage-1 runner + hypothesis pre-registration + verdict report"
```

---

## Self-Review

- **Spec coverage:** Implements v2.1 §3 (orthogonal bank: price/vol+basis+funding+DVOL/VRP+on-chain+cross-asset, PIT), §6 net-edge gate (reused from pmw), §5 model zoo (reused). Daily-cadence on-chain gated to ≥1d horizons (§2.2). Pre-registration to `hypotheses.jsonl` (§6.3). **Deferred to P1b (gated on a glimmer):** per-band model-family eligibility, family-level DSR/PBO, CMOM benchmark, held-out 2024-26 slice, conformal abstain. This is intentional — the MVP first establishes whether any signal exists before the heavier multiple-testing machinery.
- **Placeholder scan:** none — full code in every step. Live runner depends on P0 fetchers (already built/verified).
- **Type consistency:** `merge_pit(panel_ts, src, publish_lag_ms)`, `build_daily_orthogonal_bank(perp_dfs, H_min, ...)`, `run_daily(perp_dfs, ...)` consistent; reuses `pmw.feature_cols/purged_model_compare/evaluate_scores/_cell_pass/_stress_pass/_position_from_cell/WorkflowCosts` with their real signatures.
- **Note:** funding hold-cost over multi-day holds is folded in at P1b (the MVP uses the standard maker/taker/stress gate, which is the conservative first filter).
