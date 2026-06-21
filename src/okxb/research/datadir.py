"""研究数据保存目录 (用户可在 GUI 设置, 存 .env 的 RESEARCH_DATA_DIR)。

IMR 与 AI 前向验证共用此基目录。默认 = APP_DIR/data
(打包态 = OKXB.exe 同级 data/, 开发态 = 项目根 data/)。
"""
from __future__ import annotations

from pathlib import Path

from .. import paths
from ..config import Secrets


def research_base() -> Path:
    """返回研究数据基目录: 用户设了 RESEARCH_DATA_DIR 且可建则用它, 否则默认 APP_DIR/data。"""
    d = (Secrets().research_data_dir or "").strip()
    if d:
        try:
            p = Path(d)
            p.mkdir(parents=True, exist_ok=True)
            return p
        except Exception:
            pass
    b = paths.APP_DIR / "data"
    b.mkdir(parents=True, exist_ok=True)
    return b
