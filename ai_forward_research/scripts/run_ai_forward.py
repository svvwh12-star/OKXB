"""AI 前向验证 CLI 薄封装 (供 4 小时计划任务调用; 与 exe 内同一 runner/数据目录)。

  --mode auto      # 无人值守: AI_FORWARD_AUTO 开则记录AI判断(耗token), 关则仅【免费结算】到期记录
  --mode collect   # 强制记录(调AI) + 结算
  --mode evaluate  # 判决
  --mode status    # 进度
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from okxb.research import ai_forward_runner as r   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="AI 前向验证 采集/评估")
    ap.add_argument("--mode", choices=["auto", "collect", "evaluate", "status"], default="auto")
    args = ap.parse_args()
    if args.mode == "auto":
        print(asyncio.run(r.collect(record=r.auto_enabled())))
    elif args.mode == "collect":
        print(asyncio.run(r.collect(record=True)))
    elif args.mode == "evaluate":
        print(r.evaluate())
    else:
        print(r.status())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
