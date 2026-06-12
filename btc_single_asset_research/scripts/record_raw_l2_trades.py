from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import websockets


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "btc_single_asset_research"
RAW_ROOT = OUT_ROOT / "raw_l2"
WS_URL = "wss://ws.okx.com:8443/ws/v5/public"


def utc_stamp(ms: int | None = None) -> str:
    ts = datetime.fromtimestamp((ms or int(time.time() * 1000)) / 1000, tz=timezone.utc)
    return ts.strftime("%Y%m%dT%H%M%SZ")


class RotatingJsonlGz:
    def __init__(self, root: Path, inst_id: str, channel: str, rotate_mb: int = 128) -> None:
        self.root = root
        self.inst_id = inst_id
        self.channel = channel
        self.rotate_bytes = rotate_mb * 1024 * 1024
        self.fh: gzip.GzipFile | None = None
        self.path: Path | None = None
        self.bytes_written = 0

    def _open(self) -> None:
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        d = self.root / day
        d.mkdir(parents=True, exist_ok=True)
        self.path = d / f"{self.inst_id}_{self.channel}_{utc_stamp()}.jsonl.gz"
        self.fh = gzip.open(self.path, "at", encoding="utf-8")
        self.bytes_written = 0

    def write(self, obj: dict) -> None:
        if self.fh is None or self.bytes_written >= self.rotate_bytes:
            self.close()
            self._open()
        line = json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n"
        assert self.fh is not None
        self.fh.write(line)
        self.bytes_written += len(line.encode("utf-8"))

    def close(self) -> None:
        if self.fh is not None:
            self.fh.close()
            self.fh = None


async def ping_loop(ws, stop: asyncio.Event, interval: float) -> None:
    while not stop.is_set():
        await asyncio.sleep(interval)
        try:
            await ws.send("ping")
        except Exception:
            return


def channel_args(inst_id: str, channels: Iterable[str]) -> list[dict]:
    return [{"channel": ch.strip(), "instId": inst_id} for ch in channels if ch.strip()]


async def run_capture(args: argparse.Namespace) -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass

    channels = [x.strip() for x in args.channels.split(",") if x.strip()]
    writers = {ch: RotatingJsonlGz(RAW_ROOT, args.inst_id, ch, args.rotate_mb) for ch in channels}
    counts = {ch: 0 for ch in channels}
    started = time.time()
    deadline = started + args.seconds if args.seconds > 0 else None

    while not stop.is_set():
        if deadline and time.time() >= deadline:
            break
        try:
            async with websockets.connect(WS_URL, ping_interval=None, max_size=2**24) as ws:
                await ws.send(json.dumps({"op": "subscribe", "args": channel_args(args.inst_id, channels)}))
                pinger = asyncio.create_task(ping_loop(ws, stop, args.ping_interval))
                try:
                    async for raw in ws:
                        recv_ms = int(time.time() * 1000)
                        if raw == "pong":
                            continue
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if msg.get("event") == "error":
                            print(f"[error] {msg}", flush=True)
                            continue
                        ch = (msg.get("arg") or {}).get("channel")
                        if ch not in writers:
                            continue
                        msg["_recv_ts"] = recv_ms
                        writers[ch].write(msg)
                        counts[ch] += 1
                        if args.status_every and sum(counts.values()) % args.status_every == 0:
                            print(f"[capture] counts={counts}", flush=True)
                        if deadline and time.time() >= deadline:
                            stop.set()
                            break
                finally:
                    pinger.cancel()
        except Exception as exc:
            if stop.is_set():
                break
            print(f"[capture] reconnect after {type(exc).__name__}: {exc}", flush=True)
            await asyncio.sleep(2.0)

    for w in writers.values():
        w.close()
    elapsed = max(0.001, time.time() - started)
    manifest = {
        "inst_id": args.inst_id,
        "channels": channels,
        "started_utc": utc_stamp(int(started * 1000)),
        "finished_utc": utc_stamp(),
        "elapsed_sec": elapsed,
        "counts": counts,
        "messages_per_sec": sum(counts.values()) / elapsed,
        "root": str(RAW_ROOT),
    }
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    mf = RAW_ROOT / f"manifest_{args.inst_id}_{utc_stamp()}.json"
    mf.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record raw OKX public L2/trade websocket messages.")
    p.add_argument("--inst-id", default="BTC-USDT-SWAP")
    p.add_argument("--channels", default="books,trades", help="Comma list, e.g. books,trades or books5,trades.")
    p.add_argument("--seconds", type=int, default=3600, help="0 means run until interrupted.")
    p.add_argument("--rotate-mb", type=int, default=128)
    p.add_argument("--ping-interval", type=float, default=20.0)
    p.add_argument("--status-every", type=int, default=5000)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(run_capture(parse_args()))
