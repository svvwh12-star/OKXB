"""打包安全的路径解析。

开发态: 路径基于项目根目录。
冻结态 (PyInstaller .exe): 只读资源在 sys._MEIPASS; 用户可写文件 (.env / recordings /
models / data / logs) 放在【.exe 同目录】, 便于用户查看与备份。
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


if is_frozen():
    APP_DIR = Path(sys.executable).resolve().parent        # .exe 所在目录 (可写)
    RESOURCE_DIR = Path(getattr(sys, "_MEIPASS", APP_DIR))  # 打包内只读资源
else:
    APP_DIR = Path(__file__).resolve().parents[2]          # 项目根
    RESOURCE_DIR = APP_DIR

ENV_PATH = APP_DIR / ".env"


def config_path() -> Path:
    """优先用 .exe 同目录的可编辑配置; 否则用打包内置的默认配置。"""
    user_cfg = APP_DIR / "config" / "config.yaml"
    if user_cfg.exists():
        return user_cfg
    return RESOURCE_DIR / "config" / "config.yaml"


def data_path(rel: str) -> Path:
    """用户可写数据 (recordings/models/data/logs/state.sqlite) 解析到 .exe 同目录。"""
    p = APP_DIR / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def ensure_user_config() -> None:
    """冻结态首次运行: 若 .exe 同目录无 config, 从内置拷一份出来供用户编辑。"""
    if not is_frozen():
        return
    user_cfg = APP_DIR / "config" / "config.yaml"
    bundled = RESOURCE_DIR / "config" / "config.yaml"
    if not user_cfg.exists() and bundled.exists():
        user_cfg.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bundled, user_cfg)
