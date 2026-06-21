"""IMR 15/30min 前向采集/评估 —— CLI 薄封装。

实现已统一到 okxb.research.intraday_mr_runner (与 exe 内『多周期研究』页的 IMR 按钮【同一套代码、
同一数据目录 data/intraday_mr/】)。本脚本仅供命令行 / 计划任务调用; 普通用户用 exe 按钮即可。

  python run_intraday_mr.py --mode collect    # 抓最新bar -> 追加前向标签 (首次自动冻结)
  python run_intraday_mr.py --mode evaluate   # 判决 (>=100 独立前向ts + 全部闸门; sticky KILL)
  python run_intraday_mr.py --mode status     # 看进度
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from okxb.research import intraday_mr_runner as r   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="IMR 15/30min 前向采集/评估 (复用进程内 runner)")
    ap.add_argument("--mode", choices=["collect", "evaluate", "status"], default="collect")
    args = ap.parse_args()
    if args.mode == "collect":
        print(asyncio.run(r.collect()))
    elif args.mode == "evaluate":
        print(r.evaluate())
    else:
        print(r.status())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
