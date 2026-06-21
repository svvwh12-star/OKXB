"""风控引擎 = 唯一总闸门。任何订单、任何策略都必须穿过 evaluate()。

职责 (用户方案 §2/§12/§13, RESEARCH_BRIEF §9):
  - 熔断 / 回撤阶梯状态机 (NORMAL->REDUCED->STRONG_ONLY->CLOSE_ONLY->HALTED)
  - 数据老化 / kill switch
  - 并发持仓数 + 加密/股票永续分类上限
  - 总名义 / 单标的名义上限 (随市况/连亏收紧)
  - 周末股票永续降仓
  - 资金费窗口
  - AI 事件否决 (veto, 不下单只拦截)
  - 调用 sizing 计算批准的名义价值
AI 永不直接下单; 它只能产生事件标签供这里否决。
"""
from __future__ import annotations

import datetime
from decimal import Decimal
from typing import Callable, Optional

from ..config import Config
from ..core.enums import EventAction, PosSide, RiskAction, Side, SystemState
from ..core.models import MarketEvent, OrderIntent, Position, RiskDecision
from . import sizing

EventProvider = Callable[[str, Side], Optional[MarketEvent]]


def classify_symbol(inst_id: str) -> str:
    """归类到 config.risk.max_single_symbol_notional 的键。"""
    up = inst_id.upper()
    if up.startswith(("BTC", "ETH")):
        return "btc_eth"
    if up.startswith("SOL"):
        return "sol"
    if up.startswith(("SPY", "QQQ")):
        return "spy_qqq"
    if up.startswith(("AAPL", "MSFT")):
        return "aapl_msft"
    if up.startswith(("NVDA", "TSLA")):
        return "nvda_tsla"
    return "spy_qqq"  # 其他股票永续按中性档


# 股票永续识别表 (OKX 无专门字段; 启动时由 config.universe.stock_symbols 填充)
_STOCK_SYMBOLS: set[str] = set()


def set_stock_symbols(symbols) -> None:
    global _STOCK_SYMBOLS
    _STOCK_SYMBOLS = {str(s).upper() for s in (symbols or [])}


def is_stock_perp(inst_id: str) -> bool:
    """精确识别: 基础代码在股票代码表中 (避免把山寨币误判为股票永续)。"""
    return inst_id.split("-")[0].upper() in _STOCK_SYMBOLS


# C-2 fail-closed 兜底: 即使 config.universe.stock_symbols 缺失/不全 (打包版配置漂移),
# 这些"明显是美股"的代码也绝不能被当作加密山寨处理 (否则错杠杆/跳过周末门/跳过事件否决)。
CANONICAL_STOCK_TICKERS = {
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "META", "GOOGL", "AMZN", "GOOG", "NFLX",
    "AMD", "INTC", "AVGO", "COIN", "MSTR", "PLTR", "ORCL", "MU", "QCOM", "ARM", "ADBE",
    "CRM", "SMCI", "UBER", "ABNB", "SHOP", "DIS", "BA", "JPM", "V", "MA", "WMT", "PYPL",
}


def stock_classification_mismatches(inst_ids) -> list[str]:
    """启动期分类自检: 返回"基础代码明显是美股、却没被 is_stock_perp 识别"的 instId。
    非空即说明 stock_symbols 配置缺失/不全 (典型为 dist 配置漂移) -> 调用方应 fail-closed。"""
    return [i for i in inst_ids
            if i.split("-")[0].upper() in CANONICAL_STOCK_TICKERS and not is_stock_perp(i)]


class RiskEngine:
    def __init__(self, config: Config, event_provider: Optional[EventProvider] = None):
        self._cfg = config
        self._events = event_provider
        acc = config.section("account")
        self.initial_equity = float(acc.get("initial_equity_usdt", 1000))
        self.equity = self.initial_equity
        self.peak_equity = self.initial_equity
        self.day_pnl = 0.0
        self.total_pnl = 0.0
        self.consecutive_losses = 0
        self.open_positions: dict[str, Position] = {}
        self.system_state = SystemState.NORMAL
        self.high_vol = False
        self._data_age_ms = 0
        self._halted_permanently = False
        self._halt_reason = ""
        self._halt_date = ""
        self._day_date = self._today()
        # 权益峰值回撤硬停 (含浮亏): 基于交易所真实权益, 不只是已实现盈亏
        self._dd_hard_stop_frac = float(acc.get("total_drawdown_hard_stop_pct", 5.0)) / 100.0
        self._no_auto_recover = bool(acc.get("no_auto_recover_after_killswitch", True))

        r = config.section("risk")
        self._ladder = r.get("drawdown_ladder", {})
        self._max_total = {
            SystemState.NORMAL: float(r.get("max_total_notional_normal", 2500)),
            SystemState.HIGH_VOL: float(r.get("max_total_notional_high_vol", 1500)),
        }
        self._max_total_after_losses = float(r.get("max_total_notional_after_losses", 800))
        self._single_caps = r.get("max_single_symbol_notional", {})
        self._weekend_cap = float(r.get("max_stock_perp_weekend_notional", 250))
        self._max_pos = int(acc.get("max_concurrent_positions", 2))
        self._max_stock = int(acc.get("max_concurrent_stock_perp", 1))
        self._max_consec = int(acc.get("max_consecutive_losses", 4))
        ks = config.section("kill_switch")
        self._max_data_age = int(ks.get("data_age_ms_max", 500))
        wk = config.section("weekend_stock_perp")
        self._weekend_block_new = bool(wk.get("block_new_large_positions", True))
        fnd = config.section("funding")
        self._funding_block_min = float(fnd.get("block_new_positions_before_funding_minutes", 10))

    # ----------------- 外部更新 -----------------

    def set_event_provider(self, provider: Optional[EventProvider]) -> None:
        self._events = provider

    def set_account(self, equity: float, positions: dict[str, Position]) -> None:
        self.open_positions = positions
        self._apply_equity(equity)

    def update_equity(self, equity: float) -> None:
        """周期性喂入交易所真实权益 (totalEq, 含浮亏): 更新峰值并据此检查回撤硬停。"""
        self._apply_equity(equity)

    def _apply_equity(self, equity: float) -> None:
        try:
            self.equity = float(equity)
        except (TypeError, ValueError):
            return
        self.peak_equity = max(self.peak_equity, self.equity)
        self._update_system_state()

    def set_data_age_ms(self, age: int) -> None:
        self._data_age_ms = age

    def set_high_vol(self, flag: bool) -> None:
        self.high_vol = flag
        self._update_system_state()

    def register_close(self, pnl_usdt: float, final: bool = True) -> None:
        """final=False 用于【部分平仓的中间分片】: 仍累计盈亏/权益, 但不动连亏计数,
        避免同一笔决策被分N次平仓时把 consecutive_losses 放大N倍 (H-7 review)。"""
        self.day_pnl += pnl_usdt
        self.total_pnl += pnl_usdt
        self.equity += pnl_usdt
        self.peak_equity = max(self.peak_equity, self.equity)
        if final:
            if pnl_usdt < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
        self._update_system_state()

    def reset_day(self) -> None:
        self.day_pnl = 0.0
        self._day_date = self._today()
        if not self._halted_permanently:
            self.system_state = SystemState.NORMAL

    # ----------------- 状态机 -----------------

    def _trip_permanent_halt(self, reason: str) -> None:
        self._halted_permanently = True
        self._halt_reason = reason
        self._halt_date = self._today()
        self.system_state = SystemState.HALTED

    def _update_system_state(self) -> None:
        if self._halted_permanently:
            self.system_state = SystemState.HALTED
            return
        l = self._ladder
        # 权益峰值回撤硬停 (含浮亏; 不可自动恢复) —— 基于真实权益而非仅已实现盈亏
        if self.peak_equity > 0 and self._dd_hard_stop_frac > 0 \
                and (self.peak_equity - self.equity) / self.peak_equity >= self._dd_hard_stop_frac:
            self._trip_permanent_halt(
                f"权益回撤 {(1 - self.equity / self.peak_equity) * 100:.1f}% "
                f">= {self._dd_hard_stop_frac * 100:.0f}% (峰值 {self.peak_equity:.0f} -> {self.equity:.0f})")
            return
        # 总回撤硬停 (已实现累计; 不可自动恢复)
        if self.total_pnl <= float(l.get("total_killswitch_at", -50)):
            self._trip_permanent_halt(f"累计已实现亏损 {self.total_pnl:.1f} <= {l.get('total_killswitch_at', -50)}")
            return
        # 日内停机
        if self.day_pnl <= float(l.get("daily_halt_at", -15)):
            self.system_state = SystemState.HALTED
            return
        if self.total_pnl <= float(l.get("total_reduce_only_at", -40)):
            self.system_state = SystemState.CLOSE_ONLY
            return
        if (self.total_pnl <= float(l.get("total_halve_at", -30))
                or self.day_pnl <= float(l.get("daily_strong_only_at", -12))):
            self.system_state = SystemState.STRONG_ONLY
            return
        if (self.day_pnl <= float(l.get("daily_reduce_50_at", -8))
                or self.consecutive_losses >= self._max_consec):
            self.system_state = SystemState.REDUCED
            return
        self.system_state = SystemState.HIGH_VOL if self.high_vol else SystemState.NORMAL

    # ----------------- 核心裁决 -----------------

    def evaluate(self, intent: OrderIntent, *, sl_pct: float, cost_pct: float,
                 price: float, depth_notional: float, leverage: float = 1.0,
                 margin_avail_usdt: Optional[float] = None,
                 is_strong_signal: bool = False, is_taker: bool = False,
                 seconds_to_funding: Optional[float] = None) -> RiskDecision:
        Z = Decimal("0")

        def reject(reason: str) -> RiskDecision:
            return RiskDecision(RiskAction.REJECT, Z, reason, intent)

        # 1. 熔断 / 数据老化
        if self.system_state == SystemState.HALTED:
            return RiskDecision(RiskAction.HALT, Z, "系统已熔断/停机", intent)
        if self._data_age_ms > self._max_data_age:
            return reject(f"数据老化 {self._data_age_ms}ms > {self._max_data_age}ms")

        # 2. close/reduce only 状态: 仅允许减仓单
        if self.system_state in (SystemState.CLOSE_ONLY,) and not intent.reduce_only:
            return RiskDecision(RiskAction.REDUCE_ONLY, Z, "仅允许平仓", intent)
        if self.system_state == SystemState.STRONG_ONLY and not is_strong_signal and not intent.reduce_only:
            return reject("仅允许最强信号")

        # 3. AI 事件否决 (减仓单豁免)
        if self._events and not intent.reduce_only:
            ev = self._events(intent.inst_id, intent.side)
            if ev:
                blk = ev.action
                if blk == EventAction.CLOSE_ALL:
                    return RiskDecision(RiskAction.REDUCE_ONLY, Z, f"事件CLOSE_ALL:{ev.event_type}", intent)
                if blk == EventAction.REDUCE_ONLY:
                    return RiskDecision(RiskAction.REDUCE_ONLY, Z, f"事件REDUCE_ONLY:{ev.event_type}", intent)
                if blk == EventAction.BLOCK_LONG and intent.side == Side.BUY:
                    return reject(f"事件BLOCK_LONG:{ev.event_type}")
                if blk == EventAction.BLOCK_SHORT and intent.side == Side.SELL:
                    return reject(f"事件BLOCK_SHORT:{ev.event_type}")

        # 减仓单到此放行 (平仓优先)
        if intent.reduce_only:
            return RiskDecision(RiskAction.APPROVE, intent.notional_usdt, "减仓放行", intent)

        # 3.5 taker 仅允许强信号 (§10.1 / config); taker 成本高, 小账户慎用
        if is_taker and self._cfg.get("execution.taker_entry_allowed_only_if_strong_signal", True) \
                and not is_strong_signal:
            return reject("taker 仅允许强信号")

        # 4. 资金费窗口
        if seconds_to_funding is not None and not is_strong_signal:
            if seconds_to_funding < self._funding_block_min * 60:
                return reject(f"距资金费结算 {seconds_to_funding:.0f}s, 暂不新开非强信号")

        # 5. 并发持仓数
        if len(self.open_positions) >= self._max_pos and intent.inst_id not in self.open_positions:
            return reject(f"已达最大并发持仓 {self._max_pos}")
        stock_open = sum(1 for i in self.open_positions if is_stock_perp(i))
        if is_stock_perp(intent.inst_id) and stock_open >= self._max_stock \
                and intent.inst_id not in self.open_positions:
            return reject(f"已达股票永续并发上限 {self._max_stock}")

        # 6. 周末股票永续
        weekend_cap = None
        if is_stock_perp(intent.inst_id) and self._is_weekend():
            if self._weekend_block_new:
                weekend_cap = self._weekend_cap

        # 7. 仓位计算
        cat = classify_symbol(intent.inst_id)
        single_cap = float(self._single_caps.get(cat, 600))
        if weekend_cap is not None:
            single_cap = min(single_cap, weekend_cap)

        notional = sizing.final_notional(
            risk_usdt=self._risk_per_trade(is_strong_signal, intent, is_taker),
            sl_pct=sl_pct, cost_pct=cost_pct, equity_usdt=self.equity,
            single_symbol_cap_usdt=single_cap, depth_notional=depth_notional,
            margin_avail_usdt=margin_avail_usdt, leverage=leverage,
        )

        # REDUCED 状态减半
        if self.system_state == SystemState.REDUCED:
            notional *= 0.5

        # 8. 总名义上限
        cur_total = sum(float(p.notional_usdt) for p in self.open_positions.values())
        max_total = self._max_total_after_losses if self.consecutive_losses >= self._max_consec \
            else self._max_total.get(self.system_state, self._max_total[SystemState.NORMAL])
        room = max_total - cur_total
        if room <= 0:
            return reject(f"总名义已达上限 {max_total} (当前 {cur_total:.0f})")
        notional = min(notional, room)

        if notional <= 0:
            return reject("可批准名义价值为 0 (风险/深度/保证金约束)")

        return RiskDecision(RiskAction.APPROVE, Decimal(str(round(notional, 2))),
                            f"批准 {notional:.1f} USDT @ {self.system_state.value}", intent)

    # ----------------- 辅助 -----------------

    def _risk_per_trade(self, strong: bool, intent: OrderIntent, is_taker: bool = False) -> float:
        r = self._cfg.section("risk")
        if is_taker:
            return float(r.get("risk_per_trade_usdt_taker", 1.5))
        if strong:
            return float(r.get("risk_per_trade_usdt_strong_signal", 2.5))
        return float(r.get("risk_per_trade_usdt_default", 2.0))

    @staticmethod
    def _is_weekend() -> bool:
        # 简化: UTC 周六/周日。DST + 美股节假日 + 周五收盘前30min 待细化 (RESEARCH_BRIEF §6)。
        wd = datetime.datetime.now(datetime.timezone.utc).weekday()
        return wd >= 5

    # ----------------- 持久化 (跨重启的熔断/回撤状态) -----------------

    @staticmethod
    def _today() -> str:
        return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")

    def to_state(self) -> dict:
        """供 StateStore 持久化的最小状态快照。"""
        return {
            "halted_permanently": self._halted_permanently,
            "halt_reason": self._halt_reason,
            "halt_date": self._halt_date,
            "peak_equity": self.peak_equity,
            "equity": self.equity,
            "total_pnl": self.total_pnl,
            "day_pnl": self.day_pnl,
            "day_date": self._day_date,
            "consecutive_losses": self.consecutive_losses,
        }

    def load_state(self, st: Optional[dict]) -> None:
        """启动时从持久化恢复熔断/回撤状态, 使『当日不可恢复』硬停跨进程重启生效。
        重启不再清零硬停 —— 这是 C-4 修复的核心 (此前纯内存, 重启即绕过)。"""
        if not st:
            return
        self.peak_equity = max(self.peak_equity, float(st.get("peak_equity", self.peak_equity) or 0))
        self.total_pnl = float(st.get("total_pnl", self.total_pnl) or 0)
        self.consecutive_losses = int(st.get("consecutive_losses", 0) or 0)
        today = self._today()
        self.day_pnl = float(st.get("day_pnl", 0) or 0) if str(st.get("day_date")) == today else 0.0
        self._day_date = today
        if bool(st.get("halted_permanently")):
            same_day = str(st.get("halt_date", "")) == today
            if self._no_auto_recover or same_day:     # no_auto_recover=true: 隔日也保持, 须人工 clear
                self._halted_permanently = True
                self._halt_reason = str(st.get("halt_reason", "持久化熔断"))
                self._halt_date = str(st.get("halt_date", ""))
        self._update_system_state()

    def clear_halt(self) -> None:
        """人工解除永久熔断并重置风控基线 (操作员确认账户已恢复、有意 resume 时调用)。
        必须同时重置峰值=当前权益、清零累计/日内/连亏 —— 否则原回撤条件仍成立会立即再次熔断。"""
        self._halted_permanently = False
        self._halt_reason = ""
        self._halt_date = ""
        self.peak_equity = self.equity
        self.total_pnl = 0.0
        self.day_pnl = 0.0
        self.consecutive_losses = 0
        self._day_date = self._today()
        self._update_system_state()
