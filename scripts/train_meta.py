#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
meta-labeling 模型训练。
================================================
用 run_phase1 --collect-features 采集的数据训练二级模型 (判"该不该做这笔"),
产出 models/meta_model.pkl, 之后 app.py 自动加载替换占位 model_prob。

用法:  python scripts/run_phase1.py --collect-features    # 先采集 (越久越好)
       python scripts/train_meta.py                        # 训练

诚实提醒 (RESEARCH_BRIEF §7): 小样本/demo 数据训练的模型不可用于实盘决策。
正经上线前需: 样本外>=数百笔、purged+embargoed CV、Deflated Sharpe + 回测过拟合概率(PBO),
并披露试过多少组配置。本脚本给出基础切分与指标, 严谨验证需进一步扩展。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REC = ROOT / "recordings"
MODELS = ROOT / "models"
MIN_SAMPLES = 40
TRAIN_FRAC = 0.70
EMBARGO_FRAC = 0.01

from okxb.research.dataset import FEATURE_COLS, build_dataset   # noqa: E402


def _latest(prefix: str) -> Path | None:
    files = sorted(REC.glob(f"{prefix}_*.jsonl"))
    return files[-1] if files else None


def main() -> None:
    stamp = sys.argv[1] if len(sys.argv) > 1 else None
    if stamp:
        feat_path = REC / f"phase1_features_{stamp}.jsonl"
        tick_path = REC / f"phase1_ticks_{stamp}.jsonl"
    else:
        feat_path = _latest("phase1_features")
        tick_path = _latest("phase1_ticks")
    if not feat_path or not feat_path.exists():
        print("未找到特征数据。请先: python scripts/run_phase1.py --collect-features")
        return
    if not tick_path or not tick_path.exists():
        print("未找到 tick 数据 (标注需要)。")
        return

    X, y, rows = build_dataset(feat_path, tick_path)
    print(f"数据: {feat_path.name} + {tick_path.name}  样本 {len(X)}  正例率 "
          f"{(sum(y)/len(y) if y else 0):.2%}")
    if len(X) < MIN_SAMPLES:
        print(f"样本 {len(X)} < {MIN_SAMPLES}, 不足以训练。请采集更久。")
        return
    if len(set(y)) < 2:
        print("标签只有单一类别 (全赢或全输), 无法训练分类器。需更多样化样本。")
        return

    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (accuracy_score, precision_score,
                                     recall_score, roc_auc_score)
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("缺少 scikit-learn: pip install scikit-learn numpy")
        return

    Xa = np.array(X, dtype=float)
    ya = np.array(y, dtype=int)
    n = len(Xa)
    cut = int(n * TRAIN_FRAC)
    emb = max(1, int(n * EMBARGO_FRAC))
    X_tr, y_tr = Xa[:cut], ya[:cut]
    X_te, y_te = Xa[cut + emb:], ya[cut + emb:]   # embargo 隔离, 防泄露
    if len(X_te) < 5 or len(set(y_tr)) < 2:
        print("切分后测试集过小或训练集单类, 需更多样本。")
        return

    scaler = StandardScaler().fit(X_tr)
    Xs_tr, Xs_te = scaler.transform(X_tr), scaler.transform(X_te)

    candidates = []
    lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xs_tr, y_tr)
    candidates.append(("logistic", lr))
    try:
        from lightgbm import LGBMClassifier
        if len(X_tr) >= 200:
            gbm = LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                 class_weight="balanced", verbose=-1).fit(Xs_tr, y_tr)
            candidates.append(("lightgbm", gbm))
    except Exception:
        pass

    def evaluate(est):
        proba = est.predict_proba(Xs_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        auc = roc_auc_score(y_te, proba) if len(set(y_te)) > 1 else float("nan")
        return auc, accuracy_score(y_te, pred), precision_score(y_te, pred, zero_division=0), \
            recall_score(y_te, pred, zero_division=0)

    print(f"\n训练 {len(X_tr)} / 测试 {len(X_te)} (embargo {emb})")
    print(f"{'模型':<12}{'AUC':>8}{'准确率':>9}{'精确率':>9}{'召回率':>9}")
    best = None
    for name, est in candidates:
        auc, acc, prec, rec = evaluate(est)
        print(f"{name:<12}{auc:>8.3f}{acc:>9.2%}{prec:>9.2%}{rec:>9.2%}")
        score = auc if auc == auc else acc  # nan-safe
        if best is None or score > best[0]:
            best = (score, name, est, (auc, acc, prec, rec))

    _, kind, est, metrics = best
    from okxb.signal.model import MetaModel
    mm = MetaModel(est, scaler, FEATURE_COLS, kind,
                   meta={"n": n, "auc": metrics[0], "acc": metrics[1],
                         "pos_rate": sum(y) / len(y)})
    out = MODELS / "meta_model.pkl"
    mm.save(out)
    print(f"\n已保存最优模型 [{kind}] -> {out}")
    print("特征顺序:", FEATURE_COLS)
    print("\n⚠ 上线前务必: 更大样本外 + purged/embargoed CV + Deflated Sharpe/PBO。"
          "当前仅为可用基线, 不可据此上实盘。")


if __name__ == "__main__":
    main()
