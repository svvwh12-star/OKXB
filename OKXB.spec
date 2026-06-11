# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置: 单文件 OKXB.exe, 无控制台窗口。
构建:  pyinstaller OKXB.spec --noconfirm
产物:  dist/OKXB.exe
"""
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = [("config/config.yaml", "config")]      # 内置默认配置 (首次运行拷到 .exe 同目录)
datas += collect_data_files("customtkinter")     # CTk 主题/资源

hiddenimports = [
    "aiosqlite", "sortedcontainers", "websockets", "dotenv", "yaml", "httpx",
    "darkdetect", "packaging",
]
hiddenimports += collect_submodules("okxb")

# 可选: 若安装了 anthropic 则纳入 (启用 Claude 事件分类); 否则运行时规则降级
try:
    import anthropic  # noqa: F401
    hiddenimports += collect_submodules("anthropic")
    datas += collect_data_files("anthropic")
except Exception:
    pass

# 精简: GUI 交易路径不需要这些重库 (meta 模型训练/分析在开发态 CLI 进行)
excludes = ["numpy", "scipy", "scikit-learn", "sklearn", "pandas",
            "lightgbm", "matplotlib", "IPython", "notebook"]

a = Analysis(
    ["okxb_gui.py"],
    pathex=["src"],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="OKXB",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,            # 无控制台窗口 (windowed)
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
