"""异步令牌桶限速器。

为什么需要: 策略的 700ms~1500ms 重挂循环会高频下单/撤单, OKX 对下单、撤单、
批量等端点有独立的频率上限 (例如 N 次 / 2s)。超限会被拒甚至临时封禁 key。
执行器在每次下单/撤单前 await 对应桶, 自动排队, 并只用配置额度的 safety_factor。

注意: 这里的容量/周期为占位默认, 真实值由 config.execution.rate_limit 注入,
并以后台调研核验的 OKX 当前限速为准 ([VERIFY])。
"""
from __future__ import annotations

import asyncio
import time


class TokenBucket:
    """经典令牌桶: capacity 个令牌, 每 period 秒补满。"""

    def __init__(self, capacity: float, period_s: float, name: str = ""):
        self.capacity = max(1.0, capacity)
        self.period_s = period_s
        self.name = name
        self._tokens = self.capacity
        self._rate = self.capacity / self.period_s  # 令牌/秒
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self._rate)
            self._last = now

    async def acquire(self, tokens: float = 1.0) -> None:
        """获取 tokens 个令牌, 不足则异步等待 (不阻塞事件循环)。"""
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                deficit = tokens - self._tokens
                await asyncio.sleep(deficit / self._rate)

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """非阻塞尝试; 用于'宁可不发也不排队'的场景。"""
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False


class OkxRateLimiter:
    """OKX V5 限速模型 (已核验 2026-06, 见 docs/RESEARCH_BRIEF.md §3)。

    三层并行约束:
      1. per-(子账户, 合约): place / cancel / amend 各 60/2s, 三者独立计数。
      2. 批量端点: 300/2s。
      3. 子账户总量: 1000/2s, 只算【新单+改单】, 撤单不计入。

    用法: 下单前 await acquire_place(inst_id); 撤单 await acquire_cancel(inst_id);
    改单 await acquire_amend(inst_id); 批量 await acquire_batch(n)。
    限速判据应同时看响应体 error code (50011 端点/50061 子账户)。
    """

    def __init__(self, cfg: dict):
        self._safety = float(cfg.get("safety_factor", 0.7))
        self._lim = {
            "place": float(cfg.get("per_inst_place_per_2s", 60)),
            "cancel": float(cfg.get("per_inst_cancel_per_2s", 60)),
            "amend": float(cfg.get("per_inst_amend_per_2s", 60)),
        }
        self._per_inst: dict[str, TokenBucket] = {}   # key: "place:BTC-USDT-SWAP"
        self._batch = self._mk(float(cfg.get("batch_per_2s", 300)), "batch")
        self._subaccount = self._mk(float(cfg.get("subaccount_per_2s", 1000)), "subaccount")

    def _mk(self, limit: float, name: str) -> TokenBucket:
        return TokenBucket(max(1.0, limit * self._safety), 2.0, name)

    def _inst_bucket(self, op: str, inst_id: str) -> TokenBucket:
        key = f"{op}:{inst_id}"
        b = self._per_inst.get(key)
        if b is None:
            b = self._mk(self._lim[op], key)
            self._per_inst[key] = b
        return b

    async def acquire_place(self, inst_id: str) -> None:
        await self._inst_bucket("place", inst_id).acquire()
        await self._subaccount.acquire()          # 新单计入子账户

    async def acquire_amend(self, inst_id: str) -> None:
        await self._inst_bucket("amend", inst_id).acquire()
        await self._subaccount.acquire()          # 改单计入子账户

    async def acquire_cancel(self, inst_id: str) -> None:
        await self._inst_bucket("cancel", inst_id).acquire()
        # 撤单不计入子账户 1000/2s

    async def acquire_batch(self, n_orders: int, counts_subaccount: bool = True) -> None:
        await self._batch.acquire()
        if counts_subaccount:                     # 批内每单单独计入子账户
            await self._subaccount.acquire(n_orders)

    @classmethod
    def from_config(cls, rl_cfg: dict) -> "OkxRateLimiter":
        return cls(rl_cfg or {})
