"""meta-labeling 模型的加载与推理。

训练由 scripts/train_meta.py 产出 models/meta_model.pkl; 这里负责加载并在
CompositeScorer.build_signal 中替换占位 model_prob。无模型文件时返回 None (回落占位)。
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

from ..research.dataset import FEATURE_COLS, vec_from_featureset


class MetaModel:
    def __init__(self, estimator, scaler, features: list[str], kind: str,
                 meta: Optional[dict] = None):
        self.estimator = estimator
        self.scaler = scaler
        self.features = features
        self.kind = kind
        self.meta = meta or {}

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"estimator": self.estimator, "scaler": self.scaler,
                         "features": self.features, "kind": self.kind, "meta": self.meta}, f)

    @classmethod
    def load_if_exists(cls, path: str | Path) -> Optional["MetaModel"]:
        p = Path(path)
        if not p.exists():
            return None
        try:
            with open(p, "rb") as f:
                d = pickle.load(f)
            return cls(d["estimator"], d["scaler"], d["features"], d["kind"], d.get("meta"))
        except Exception as e:
            print(f"[meta] 加载模型失败, 回落占位: {e!r}")
            return None

    def predict_prob(self, fs, composite: float) -> Optional[float]:
        try:
            import numpy as np
            vec = np.array([vec_from_featureset(fs, composite)], dtype=float)
            if self.scaler is not None:
                vec = self.scaler.transform(vec)
            p = self.estimator.predict_proba(vec)[0][1]
            return float(p)
        except Exception:
            return None
