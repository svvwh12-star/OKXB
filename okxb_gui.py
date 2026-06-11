#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""OKXB 图形界面入口。开发态: python okxb_gui.py; 打包后: 双击 OKXB.exe。"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

# 开发态: 把 src 加入路径 (冻结态包已内置)
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _guard_std() -> None:
    """windowed .exe 下 sys.stdout/stderr 为 None, 裸 print() 会崩溃。
    重定向到 .exe 同目录 logs/runtime.log, 兜底用 null sink。"""
    if sys.stdout is not None and sys.stderr is not None:
        return
    try:
        from okxb import paths
        logf = open(paths.data_path("logs/runtime.log"), "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stdout or logf
        sys.stderr = sys.stderr or logf
    except Exception:
        import io

        class _Null(io.TextIOBase):
            def write(self, s):  # noqa
                return len(s)
        sys.stdout = sys.stdout or _Null()
        sys.stderr = sys.stderr or _Null()


def main() -> None:
    _guard_std()
    try:
        from okxb.gui.main_window import run
        run()
    except Exception:
        tb = traceback.format_exc()
        try:
            from okxb import paths
            log = paths.data_path("logs/crash.log")
        except Exception:
            log = Path("okxb_crash.log")
        try:
            Path(log).parent.mkdir(parents=True, exist_ok=True)
            with open(log, "a", encoding="utf-8") as f:
                f.write(tb + "\n")
        except Exception:
            pass
        try:
            import tkinter.messagebox as mb
            mb.showerror("OKXB 启动失败", f"详见: {log}\n\n{tb[-900:]}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
