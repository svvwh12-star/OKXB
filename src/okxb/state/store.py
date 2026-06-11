"""SQLite 状态与审计存储 (aiosqlite, 异步)。

留痕一切: 订单生命周期、平仓盈亏、触发的信号、风控/系统事件。
用于复盘、回测对齐、以及实盘事故追溯。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
  client_oid TEXT PRIMARY KEY, inst TEXT, side TEXT, ord_type TEXT,
  px TEXT, sz TEXT, state TEXT, strategy TEXT, signal_id TEXT,
  created_ms INTEGER, okx_ord_id TEXT, filled_sz TEXT, avg_px TEXT, json TEXT
);
CREATE TABLE IF NOT EXISTS pnl (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, inst TEXT, strategy TEXT,
  pnl_usdt REAL, json TEXT
);
CREATE TABLE IF NOT EXISTS signals (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, inst TEXT, side TEXT,
  composite REAL, json TEXT
);
CREATE TABLE IF NOT EXISTS audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, kind TEXT, msg TEXT
);
"""


class StateStore:
    def __init__(self, db_path: str | Path):
        self._path = str(db_path)
        self._db: Optional[aiosqlite.Connection] = None

    async def open(self) -> None:
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._path)
        await self._db.executescript(SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    @staticmethod
    def _now() -> int:
        return int(time.time() * 1000)

    async def upsert_order(self, o: dict[str, Any]) -> None:
        assert self._db
        await self._db.execute(
            """INSERT INTO orders(client_oid,inst,side,ord_type,px,sz,state,strategy,
               signal_id,created_ms,okx_ord_id,filled_sz,avg_px,json)
               VALUES(:client_oid,:inst,:side,:ord_type,:px,:sz,:state,:strategy,
               :signal_id,:created_ms,:okx_ord_id,:filled_sz,:avg_px,:json)
               ON CONFLICT(client_oid) DO UPDATE SET
               state=excluded.state, okx_ord_id=excluded.okx_ord_id,
               filled_sz=excluded.filled_sz, avg_px=excluded.avg_px, json=excluded.json""",
            {**o, "json": json.dumps(o.get("json", {}), ensure_ascii=False)},
        )
        await self._db.commit()

    async def record_pnl(self, inst: str, strategy: str, pnl_usdt: float,
                         extra: Optional[dict] = None) -> None:
        assert self._db
        await self._db.execute(
            "INSERT INTO pnl(ts,inst,strategy,pnl_usdt,json) VALUES(?,?,?,?,?)",
            (self._now(), inst, strategy, pnl_usdt, json.dumps(extra or {}, ensure_ascii=False)),
        )
        await self._db.commit()

    async def record_signal(self, inst: str, side: str, composite: float,
                            extra: Optional[dict] = None) -> None:
        assert self._db
        await self._db.execute(
            "INSERT INTO signals(ts,inst,side,composite,json) VALUES(?,?,?,?,?)",
            (self._now(), inst, side, composite, json.dumps(extra or {}, ensure_ascii=False)),
        )
        await self._db.commit()

    async def audit(self, kind: str, msg: str) -> None:
        assert self._db
        await self._db.execute(
            "INSERT INTO audit(ts,kind,msg) VALUES(?,?,?)", (self._now(), kind, msg)
        )
        await self._db.commit()
