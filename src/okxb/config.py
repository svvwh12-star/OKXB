"""配置加载。

- 业务参数: config/config.yaml -> 只读、点路径访问 (cfg.get("risk.risk_per_trade_usdt_default"))
- 密钥/敏感项: .env -> Secrets (pydantic-settings), 绝不写入日志/文件

设计原则: 业务配置可热加载 (返回新对象), 密钥进程启动时一次性读入。
"""
from __future__ import annotations

import os
from functools import reduce
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from . import paths
from .core.enums import Mode

ROOT = paths.APP_DIR
DEFAULT_CONFIG = paths.config_path()


class Config:
    """业务参数的只读容器, 支持点路径访问与默认值。"""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CONFIG) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            return cls(yaml.safe_load(f) or {})

    def get(self, dotted: str, default: Any = None) -> Any:
        try:
            return reduce(lambda d, k: d[k], dotted.split("."), self._data)
        except (KeyError, TypeError):
            return default

    def section(self, name: str) -> dict[str, Any]:
        return self._data.get(name, {}) or {}

    @property
    def raw(self) -> dict[str, Any]:
        return self._data


class Secrets:
    """从 .env 读取的密钥与凭据。仓库代码中绝不硬编码。"""

    def __init__(self) -> None:
        load_dotenv(paths.ENV_PATH)
        self.mode = Mode(os.getenv("OKXB_MODE", "demo").strip().lower())
        # 区域路由: global / us / eea (决定 REST/WS base host)
        self.region = os.getenv("OKX_REGION", "global").strip().lower()

        if self.mode == Mode.LIVE:
            self.okx_api_key = os.getenv("OKX_LIVE_API_KEY", "")
            self.okx_secret_key = os.getenv("OKX_LIVE_SECRET_KEY", "")
            self.okx_passphrase = os.getenv("OKX_LIVE_PASSPHRASE", "")
        else:
            self.okx_api_key = os.getenv("OKX_DEMO_API_KEY", "")
            self.okx_secret_key = os.getenv("OKX_DEMO_SECRET_KEY", "")
            self.okx_passphrase = os.getenv("OKX_DEMO_PASSPHRASE", "")

        self.edgar_user_agent = os.getenv("EDGAR_USER_AGENT", "")
        self.finnhub_api_key = os.getenv("FINNHUB_API_KEY", "")
        # 加密新闻 (CryptoPanic 需 token; RSS 无需 key, 默认 CoinDesk) + 经济日历 (TradingEconomics)
        # 全部 fail-closed: 留空即休眠, 不影响其它功能。社会/政治裸社媒【刻意不接】(对抗性, 易被操纵)。
        self.cryptopanic_api_key = os.getenv("CRYPTOPANIC_API_KEY", "")
        self.crypto_news_rss_url = os.getenv("CRYPTO_NEWS_RSS_URL", "")
        self.econ_calendar_api_key = os.getenv("TRADING_ECONOMICS_API_KEY", "")
        # CoinGecko demo key: 行情/大盘数据(非新闻) — 全球市值/BTC占比/热搜趋势, 喂给 AI 选品
        self.coingecko_api_key = os.getenv("COINGECKO_API_KEY", "")

        # AI 事件分类: 提供商无关 (DeepSeek / OpenAI兼容 / Claude); 简单任务用便宜模型, 复杂任务用强模型
        self.ai_provider = os.getenv("AI_PROVIDER", "rule").strip().lower()
        self.ai_api_key = os.getenv("AI_API_KEY", "") or os.getenv("ANTHROPIC_API_KEY", "")
        self.ai_base_url = os.getenv("AI_BASE_URL", "https://api.deepseek.com")
        # DeepSeek 官方模型名: deepseek-chat(便宜快) / deepseek-reasoner(强, 推理)。
        # 注: 旧默认 deepseek-v4-flash/-pro 不是有效模型名, 会导致调用空返回 -> 选品/分析为空。
        self.ai_model_simple = os.getenv("AI_MODEL_SIMPLE", "deepseek-chat")
        self.ai_model_hard = os.getenv("AI_MODEL_HARD", "deepseek-reasoner")
        # 选型策略: auto(按难度: 自由文本新闻+高危8-K用pro, 其余用flash) / flash(总用便宜) / pro(总用强)
        self.ai_tier_policy = os.getenv("AI_TIER_POLICY", "auto").strip().lower()
        # 向后兼容: 仅配了 ANTHROPIC_API_KEY 而未显式选 provider 时, 视为 claude
        if self.ai_provider == "rule" and not os.getenv("AI_API_KEY") and os.getenv("ANTHROPIC_API_KEY"):
            self.ai_provider = "claude"
            self.ai_model_simple = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
            self.ai_model_hard = self.ai_model_simple

        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    @property
    def is_demo(self) -> bool:
        return self.mode == Mode.DEMO

    def require_okx(self) -> None:
        missing = [n for n, v in [
            ("api_key", self.okx_api_key),
            ("secret_key", self.okx_secret_key),
            ("passphrase", self.okx_passphrase),
        ] if not v]
        if missing:
            raise RuntimeError(
                f"{self.mode.value} 模式缺少 OKX 凭据: {', '.join(missing)} — 请在 .env 填入轮换后的新密钥。"
            )

    def __repr__(self) -> str:  # 永不泄露密钥
        masked = f"...{self.okx_api_key[-4:]}" if self.okx_api_key else "(空)"
        return f"<Secrets mode={self.mode.value} okx_key={masked}>"
