"""合约规格缓存。启动时拉取 + 周期刷新; 提供 tickSz/lotSz/minSz/ctVal/lever/state。
价格按 tickSz、数量按 lotSz 取整必须用这里的真实规格, 不可写死 (RESEARCH_BRIEF §5)。
"""
from __future__ import annotations

import time
from typing import Optional

from .okx_rest import OkxRestClient


class InstrumentCache:
    def __init__(self, rest: OkxRestClient, ttl_s: float = 3600):
        self._rest = rest
        self._ttl = ttl_s
        self._specs: dict[str, dict] = {}
        self._last_fetch = 0.0

    async def refresh(self, inst_type: str = "SWAP") -> None:
        insts = await self._rest.get_instruments(inst_type)
        for i in insts:
            self._specs[i["instId"]] = i
        self._last_fetch = time.monotonic()

    async def ensure(self, inst_type: str = "SWAP") -> None:
        if not self._specs or (time.monotonic() - self._last_fetch) > self._ttl:
            await self.refresh(inst_type)

    def get(self, inst_id: str) -> Optional[dict]:
        return self._specs.get(inst_id)

    def is_tradable(self, inst_id: str) -> bool:
        s = self._specs.get(inst_id)
        return bool(s and s.get("state") == "live")

    def tick_sz(self, inst_id: str) -> str:
        return (self._specs.get(inst_id) or {}).get("tickSz", "0.0001")

    def lot_sz(self, inst_id: str) -> str:
        return (self._specs.get(inst_id) or {}).get("lotSz", "1")

    def min_sz(self, inst_id: str) -> str:
        return (self._specs.get(inst_id) or {}).get("minSz", "1")

    def ct_val(self, inst_id: str) -> float:
        return float((self._specs.get(inst_id) or {}).get("ctVal", "1") or 1)

    def max_lever(self, inst_id: str) -> float:
        return float((self._specs.get(inst_id) or {}).get("lever", "1") or 1)
