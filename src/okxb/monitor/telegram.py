"""Telegram 告警 (用户自己的 bot)。

在 .env 配 TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID 即启用; 未配则静默 no-op。
推送: 下单/平仓/系统状态变化/熔断等关键事件。全程异步、失败不影响交易。
"""
from __future__ import annotations

import httpx


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.token = token or ""
        self.chat_id = chat_id or ""
        self.enabled = bool(self.token and self.chat_id)

    async def send(self, text: str) -> None:
        if not self.enabled:
            return
        try:
            async with httpx.AsyncClient(timeout=10.0) as c:
                await c.post(f"https://api.telegram.org/bot{self.token}/sendMessage",
                             json={"chat_id": self.chat_id, "text": text,
                                   "disable_web_page_preview": True})
        except Exception:
            pass

    async def verify(self) -> str:
        """连通性测试 (供 GUI 验证)。"""
        if not self.enabled:
            return "未配置 Telegram (留空则不推送)。需填 BOT_TOKEN + CHAT_ID。"
        try:
            async with httpx.AsyncClient(timeout=10.0) as c:
                r = await c.post(f"https://api.telegram.org/bot{self.token}/sendMessage",
                                 json={"chat_id": self.chat_id,
                                       "text": "✅ OKXB 测试消息: Telegram 告警已连通。"})
            d = r.json()
            if d.get("ok"):
                return "Telegram 连通 ✓ 已发送测试消息, 请查收。"
            return f"Telegram 失败 ✗: {d.get('description', r.text[:200])}"
        except Exception as e:
            return f"Telegram 连接异常 ✗: {e!r}"
