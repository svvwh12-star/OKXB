"""编排器 / 主循环。

数据流: Gateway -> FeatureEngine -> CompositeScorer -> Strategies -> RiskEngine(总闸门) -> Executor
安全默认: dry_run (只决策不下单)。
  --live           : 真实下单 (demo 模式=模拟盘, 安全)
  --allow-live-real: 仅当 OKXB_MODE=live 且确实要用真金时才加, 否则拒绝运行
运行:  python -m okxb.app            (dry-run)
       python -m okxb.app --live     (模拟盘真实下单, 需 .env 配 demo 密钥)
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import signal as sigmod
import time
from collections import deque
from decimal import Decimal

from . import paths
from .config import Config, Secrets
from .core.enums import Mode, OrderType, PosSide, RiskAction
from .core.models import OrderIntent, Position
from .events.service import AIEventService, ticker_of
from .exchange.instruments import InstrumentCache
from .exchange.okx_rest import OkxError, OkxRestClient
from .exchange.okx_ws_private import OkxPrivateWS
from .execution.executor import Executor
from .execution.order_manager import OrderManager
from .features.engine import FeatureEngine
from .marketdata.gateway import MarketDataGateway
from .monitor.telegram import TelegramNotifier
from .research.recorder import TickRecorder
from .risk.engine import RiskEngine, is_stock_perp, set_stock_symbols
from .signal.composite import CompositeScorer
from .state.store import StateStore
from .strategies.basis import BasisMeanRev
from .strategies.breakout import Breakout
from .strategies.hfm80 import HFM80
from .strategies.hfr80 import HFR80

COMPUTE_S = 0.5
MONITOR_S = 1.0
HOUSEKEEP_S = 5.0


class App:
    def __init__(self, dry_run: bool, log_fn=None):
        paths.ensure_user_config()
        self.cfg = Config.load()
        set_stock_symbols(self.cfg.get("universe.stock_symbols", []))  # 精确识别股票永续
        self.secrets = Secrets()
        self.dry_run = dry_run
        # GUI 钩子: 线程安全日志 + 状态快照 (供 GUI 线程只读)
        self.logs: deque = deque(maxlen=600)
        self._log_fn = log_fn
        self.latest_rows: dict = {}
        self.latest_status: dict = {}
        self.tg = TelegramNotifier(self.secrets.telegram_bot_token, self.secrets.telegram_chat_id)
        self._last_alert_state: str = ""
        self.rest = OkxRestClient(self.secrets, self.cfg)
        # 合约规格目录用"公共/实盘"目录, 与行情源(默认live)一致 —— demo 目录缺 QQQ/AAPL/TSLA/META
        feed = str(self.cfg.get("data.market_data_feed", "live")).lower()
        pub = Secrets()
        pub.mode = Mode.DEMO if feed == "demo" else (self.secrets.mode if feed == "auto" else Mode.LIVE)
        self.pub_rest = OkxRestClient(pub, self.cfg)
        self.instruments = InstrumentCache(self.pub_rest)
        self.om = OrderManager()
        self.risk = RiskEngine(self.cfg)
        from .signal.model import MetaModel
        meta = MetaModel.load_if_exists(
            paths.data_path(self.cfg.get("paths.models_dir", "models") + "/meta_model.pkl"))
        if meta:
            self._log(f"已加载 meta 模型 ({meta.kind}); 用其概率替换占位")
        self.scorer = CompositeScorer(self.cfg, meta_model=meta)
        # 4 策略: 顺势maker / 反转 / 突破(taker) / 股票永续basis —— 多策略 = 更多出手机会
        self.strategies = [HFM80(self.cfg), HFR80(self.cfg),
                           Breakout(self.cfg), BasisMeanRev(self.cfg)]
        self._cooldown_ms = float(self.cfg.get("signal.cooldown_seconds", 20)) * 1000.0
        self._cooldown: dict[str, float] = {}   # inst -> 冷却到期(ms): 开/平仓后短期内不再开
        # 逐拍校准录制器: 演练/实盘均录制 (供『策略校准』回测)。文件名带启动时间戳。
        rec_on = bool(self.cfg.get("research.record_enabled", True))
        rec_dir = self.cfg.get("paths.recordings_dir", "recordings")
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recorder = TickRecorder(
            rec_on, paths.data_path(f"{rec_dir}/calib_{stamp}.jsonl"))
        self._diag = {"score": 0, "conf": 0, "edge": 0, "entered": 0, "ticks": 0}
        sg = self.cfg.section("signal")
        self._diag_min_comp = float(sg.get("min_composite_score", 70))
        self._diag_confirm = float(sg.get("confirm_min", 0.30))
        self._diag_min_edge = float(sg.get("min_edge_to_cost_ratio", 1.5))
        self._diag_min_trad = float(sg.get("min_tradability", 0.5))
        self._diag_max_spread = float(sg.get("max_entry_spread_bps", 50.0))
        self._diag_rv_lo = float(sg.get("regime_rv_lo", 5e-5))
        self.store = StateStore(paths.data_path(
            self.cfg.get("paths.state_db", "data/okxb_state.sqlite")))
        self.universe: list[str] = []
        self.gw: MarketDataGateway | None = None
        self.fe: FeatureEngine | None = None
        self.executor: Executor | None = None
        self.events_svc: AIEventService | None = None
        self.priv_ws: OkxPrivateWS | None = None
        self._stop = asyncio.Event()
        self._has_keys = bool(self.secrets.okx_api_key)
        # 真实账户快照 (供控制台卡片显示真实持仓/浮动盈亏, 不只是引擎自管仓位)
        self._acct_positions = 0
        self._acct_upl = 0.0
        self._acct_ok = False

    def _log(self, msg: str) -> None:
        line = f"{datetime.datetime.now().strftime('%H:%M:%S')} {msg}"
        self.logs.append(line)
        if self._log_fn:
            try:
                self._log_fn(line)
            except Exception:
                pass
        else:
            print(f"[app] {msg}")

    def _alert(self, msg: str) -> None:
        """记日志 + 推 Telegram (关键事件: 下单/平仓/状态变化/熔断)。"""
        self._log(msg)
        if self.tg.enabled:
            try:
                asyncio.create_task(self.tg.send(f"[OKXB] {msg}"))
            except RuntimeError:
                pass

    async def setup(self) -> None:
        await self.store.open()
        await self.instruments.ensure("SWAP")
        from .marketdata.universe import build_universe
        uni, diag = await build_universe(self.pub_rest, self.cfg)
        self.universe = [s for s in uni if self.instruments.is_tradable(s)]  # 双保险: 只留 live
        self._log(f"模式={self.secrets.mode.value} dry_run={self.dry_run} "
                  f"标的({len(self.universe)}) [{diag.get('mode')}: 加密{diag.get('crypto', '?')}+"
                  f"股票{diag.get('stock', '?')}]")
        self._log(f"标的: {self.universe}")
        self.gw = MarketDataGateway(self.cfg, self.secrets, self.universe)
        self.fe = FeatureEngine(self.gw, self.cfg)

        # AI 事件模块: 仅当有股票永续标的时启用; 其 setup()(含EDGAR下载)放到后台跑,
        # 绝不阻塞行情主循环启动 (中国网络下 sec.gov 可能慢/被墙)。
        stock_tickers = {ticker_of(s) for s in self.universe if is_stock_perp(s)}
        if stock_tickers:
            self.events_svc = AIEventService(self.cfg, self.secrets, stock_tickers)
            self.risk.set_event_provider(self.events_svc.get_veto)  # 缓存填充前返回None, 不拦截

        # 私有WS下单层: 仅实际下单(非dry_run)且有密钥时启用; 公共行情仍走实盘WS
        if not self.dry_run and self._has_keys:
            self.priv_ws = OkxPrivateWS(self.secrets, self.cfg, alert_fn=self._alert)

        self.executor = Executor(
            rest=self.rest, gateway=self.gw, instruments=self.instruments,
            order_manager=self.om, risk=self.risk, store=self.store,
            config=self.cfg, dry_run=self.dry_run, alert_fn=self._alert, ws=self.priv_ws,
        )
        # 初始权益用配置值立即就绪 (非网络); 真实权益由主循环 housekeep 异步刷新, 不阻塞启动
        self.risk.set_account(self.cfg.get("account.initial_equity_usdt", 1000), {})
        # 探测账户持仓模式(net/long_short), 注入执行器 -> 正确的 posSide, 否则双向账户下单报 posSide error
        if self._has_keys:
            try:
                accfg = await self.rest.get_account_config()
                self.executor._pm = accfg.get("posMode", "net_mode") or "net_mode"
                self._log(f"账户持仓模式={self.executor._pm} (双向账户已自动适配 posSide)")
            except Exception:
                pass
        self._update_status()

    async def _refresh_account(self) -> None:
        if not self._has_keys:
            self.risk.set_account(self.cfg.get("account.initial_equity_usdt", 1000), {})
            return
        try:
            bal = await self.rest.get_balance()
            eq = float(bal.get("totalEq", 0) or 0)
            self.risk.set_account(eq or self.risk.initial_equity, self.risk.open_positions)
            # 真实持仓 (含手动开的), 供控制台卡片如实显示
            poss = await self.rest.get_positions("SWAP")
            live = [p for p in poss if float(p.get("pos", 0) or 0) != 0]
            self._acct_positions = len(live)
            self._acct_upl = sum(float(p.get("upl", 0) or 0) for p in live)
            self._acct_ok = True
        except Exception:
            pass            # 网络抖动等; 外层主循环已兜底, 保留上次权益

    async def _update_marks(self) -> None:
        """周期拉全市场标记价, 注入因子引擎供 basis 计算 (股票永续)。"""
        try:
            data = await self.pub_rest.get_mark_prices("SWAP")
            marks = {d["instId"]: float(d["markPx"]) for d in data if d.get("markPx")}
            if self.fe:
                self.fe.set_marks(marks)
        except Exception:
            pass

    def _leverage(self, inst: str) -> float:
        lv = self.cfg.section("leverage")
        if is_stock_perp(inst):
            return float(lv.get("stock_perp_default", 3))
        if inst.startswith(("BTC", "ETH")):
            return float(lv.get("crypto_major_default", 10))
        return float(lv.get("crypto_alt_default", 5))

    def _update_status(self) -> None:
        real = self._has_keys and self._acct_ok
        self.latest_status = {
            "state": self.risk.system_state.value, "equity": self.risk.equity,
            "day_pnl": self.risk.day_pnl, "total_pnl": self.risk.total_pnl,
            "positions": self._acct_positions if real else len(self.om.all()),
            "upl": self._acct_upl if real else 0.0,
            "acct_real": real,
            "data_age_ms": self.gw.data_age_ms() if self.gw else 10 ** 9,
            "mode": self.secrets.mode.value, "dry_run": self.dry_run,
            "consec_losses": self.risk.consecutive_losses,
            "rec_rows": self.recorder.count,
        }

    def _entry_status(self, comp, fs) -> str:
        """逐标的"入场状态"短码 (面板用, 澄清'质量高≠该交易'): 满足前置门给✓候选, 否则给第一个卡住的关卡。
        注意: 这是去掉'连续N拍/冷却'后的快照, 与实际下单还差去抖与冷却。"""
        rv = fs.realized_vol_60s or 0.0
        if rv < self._diag_rv_lo:
            return "死盘"                 # 真冻结(rv极低); 阈值已下调, 仅极端无动静才显示
        if getattr(comp, "warmup", False):
            return "预热"
        if fs.spread_bps is not None and fs.spread_bps > self._diag_max_spread:
            return "价差宽"
        d = 1 if comp.long_score_s >= comp.short_score_s else -1
        score = comp.long_score_s if d > 0 else comp.short_score_s
        if score < self._diag_min_comp:
            return "方向不足"
        if min(comp.tradability, getattr(comp, "tradability_raw", comp.tradability)) < self._diag_min_trad:
            return "质量低"
        if comp.dir_latch != d:
            return "未确认"
        # 期望净edge: 行情太淡(动得不够覆盖成本)时这才是真正卡住的关卡(取代旧"死盘"误导)
        try:
            from .core.enums import Side, StrategyId
            sig = self.scorer.build_signal(fs, StrategyId.HFM80, Side.BUY if d > 0 else Side.SELL, score)
            if sig.total_cost_pct <= 0 or sig.edge_to_cost < self._diag_min_edge:
                return "净edge不足"
        except Exception:
            pass
        return "✓候选"

    def _diagnose(self, comp, fs) -> None:
        """近因诊断: 逐拍快照统计入场链路在哪一关被卡 (无持续/冷却约束的纯门槛快照)。
        累计窗口由 housekeep 输出并清零, 让用户直观看到"为何不出手"的瓶颈。"""
        self._diag["ticks"] += 1
        d = 1 if comp.long_score_s >= comp.short_score_s else -1
        score = comp.long_score_s if d > 0 else comp.short_score_s   # 用平滑分(与实际门槛一致)
        if score < self._diag_min_comp:
            return
        self._diag["score"] += 1
        conf = (0.5 * max(0.0, d * comp.order_flow_dir)
                + 0.5 * max(0.0, d * comp.trend_dir))
        if conf < self._diag_confirm:
            return
        self._diag["conf"] += 1
        # 净 edge 快照: 用 scorer 临时构造一笔, 看扣费净 edge 是否达标
        try:
            from .core.enums import Side, StrategyId
            side = Side.BUY if d > 0 else Side.SELL
            sig = self.scorer.build_signal(fs, StrategyId.HFM80, side, score)
            if sig.edge_to_cost >= self._diag_min_edge:
                self._diag["edge"] += 1
        except Exception:
            pass

    def _diag_summary(self) -> str:
        dg = self._diag
        return (f"诊断窗: 拍{dg['ticks']} | 分>={self._diag_min_comp:.0f}:{dg['score']} "
                f"→含确认:{dg['conf']} →含净edge:{dg['edge']} → 实际进场:{dg['entered']} "
                f"(录制{self.recorder.count}行)")

    def _sync_positions_to_risk(self) -> None:
        """把受管持仓回灌风控, 使全局并发数/总名义/单标的上限真正生效。"""
        pos = {}
        for mp in self.om.all():
            ctval = self.instruments.ct_val(mp.inst_id)
            notional = mp.contracts * mp.entry_px * Decimal(str(ctval))
            pos[mp.inst_id] = Position(
                inst_id=mp.inst_id, pos_side=PosSide.NET, size=mp.contracts,
                avg_px=mp.entry_px, notional_usdt=notional, upl=Decimal("0"))
        self.risk.open_positions = pos

    async def _tick(self) -> None:
        assert self.gw and self.fe and self.executor
        self.risk.set_data_age_ms(self.gw.data_age_ms())
        self._sync_positions_to_risk()
        rows = {}
        for inst in self.universe:
            try:
                fs = self.fe.compute(inst)
                comp = self.scorer.score(fs)
                bbo = self.gw.get_bbo(inst) or self.gw.get_book(inst).bbo()
                rows[inst] = {
                    "mid": bbo.mid if bbo else None,
                    "spread_bps": fs.spread_bps, "obi_z": fs.obi_5_z, "ofi_z": fs.ofi_z,
                    "long": comp.long_score_s, "short": comp.short_score_s,   # 平滑分(对称, 多+空=100)
                    "trend_dir": comp.trend_dir, "flow_dir": comp.order_flow_dir,
                    "trad": comp.tradability, "entry": self._entry_status(comp, fs),
                    "atr": fs.atr_1m, "rvol": fs.realized_vol_60s,
                    "has_pos": self.om.has(inst),
                }
                if not bbo:
                    continue
                maker_cost = self.scorer.total_cost_pct(fs)
                self.recorder.record(inst, bbo.mid, comp, fs, maker_cost)
                self._diagnose(comp, fs)
                # 持仓中 -> 走信号出场(最短持仓/移动止盈/反转, 用已算好的fs/comp); 平仓后起冷却
                if self.om.has(inst):
                    pos = self.om.get(inst)
                    if pos is not None:
                        cost = self.scorer.total_cost_pct(fs, taker=True)
                        if await self.executor.signal_exit(pos, fs, comp, bbo.mid, cost):
                            self._cooldown[inst] = time.time() * 1000.0 + self._cooldown_ms
                    continue
                if time.time() * 1000.0 < self._cooldown.get(inst, 0.0):   # 冷却中不开新仓
                    continue
                for strat in self.strategies:
                    sig = strat.evaluate(fs, comp, self.scorer)
                    if sig is None:
                        continue
                    is_strong = strat.is_strong(sig.composite_score, sig)
                    otype = OrderType.OPTIMAL_LIMIT_IOC if sig.taker else OrderType.POST_ONLY
                    intent = OrderIntent(
                        inst_id=inst, side=sig.side, pos_side=PosSide.NET,
                        order_type=otype, notional_usdt=Decimal("0"),
                        px=None, reduce_only=False, strategy=sig.strategy,
                        signal_id=sig.signal_id, sl_pct=sig.sl_pct, tp_pct=sig.tp_pct,
                        max_loss_usdt=Decimal("0"), ttl_ms=1000,
                    )
                    depth = self.gw.get_book(inst).depth_notional("bid", 5.0) \
                        + self.gw.get_book(inst).depth_notional("ask", 5.0)
                    decision = self.risk.evaluate(
                        intent, sl_pct=sig.sl_pct, cost_pct=sig.total_cost_pct,
                        price=bbo.mid, depth_notional=depth, leverage=self._leverage(inst),
                        is_strong_signal=is_strong, is_taker=sig.taker,
                    )
                    if decision.action == RiskAction.APPROVE:
                        self._diag["entered"] += 1
                        self._alert(f"下单意图 {inst} {sig.side.value} {sig.strategy.value} "
                                    f"comp={sig.composite_score} 批准{decision.approved_notional}U")
                        await self.executor.submit(sig, decision.approved_notional, is_strong)
                        self._cooldown[inst] = time.time() * 1000.0 + self._cooldown_ms
                        break
                    elif decision.action != RiskAction.REJECT:
                        await self.store.audit("risk", f"{inst} {decision.action.value}: {decision.reason}")
            except Exception as e:                 # 单标的异常隔离, 不影响其他标的与面板刷新
                await self.store.audit("tick_err", f"{inst} {e!r}")
        self.latest_rows = rows
        self._update_status()

    async def run(self) -> None:
        assert self.gw and self.executor
        self._alert(f"已启动 模式={self.secrets.mode.value} "
                    f"{'实际下单' if not self.dry_run else '只演练'} 标的{len(self.universe)}")
        if self.recorder.enabled:
            self._log(f"校准录制开启 -> {self.recorder.path.name} (供『策略校准』回测)")
        gw_task = asyncio.create_task(self.gw.run())

        async def _events_runner():
            try:
                await self.events_svc.setup()      # EDGAR 清单下载等, 后台进行
                await self.events_svc.run()
            except Exception as e:
                self._log(f"事件模块异常(不影响交易): {e!r}")
        ev_task = asyncio.create_task(_events_runner()) if self.events_svc else None
        pw_task = asyncio.create_task(self.priv_ws.run()) if self.priv_ws else None
        last_monitor = last_house = 0.0
        try:
            while not self._stop.is_set():
                await asyncio.sleep(COMPUTE_S)
                # 整轮包 try: 任何瞬时网络/接口异常都只记日志、继续, 绝不拖垮引擎线程
                try:
                    await self._tick()
                    now = time.monotonic()
                    if now - last_monitor >= MONITOR_S:
                        last_monitor = now
                        await self.executor.monitor_positions()
                    if now - last_house >= HOUSEKEEP_S:
                        last_house = now
                        await self.executor.arm_kill_switch()
                        await self._refresh_account()   # 真实权益异步刷新
                        await self._update_marks()      # 标记价 -> basis
                        state = self.risk.system_state.value
                        if self._last_alert_state and state != self._last_alert_state:
                            self._alert(f"⚠ 系统状态变化: {self._last_alert_state} -> {state}")
                        self._last_alert_state = state
                        self._log(f"state={self.risk.system_state.value} eq={self.risk.equity:.1f} "
                                  f"dayPnL={self.risk.day_pnl:.2f} 持仓={len(self.om.all())} "
                                  f"数据老化={self.gw.data_age_ms()}ms")
                        self._log(self._diag_summary())
                        for k in ("score", "conf", "edge", "entered", "ticks"):
                            self._diag[k] = 0
                except Exception as e:
                    try:
                        self._log(f"主循环异常(已忽略, 继续运行): {type(e).__name__}: {e}")
                    except Exception:
                        pass
        finally:
            try:
                await self.executor.disarm_kill_switch()   # 正常停止: 解除死手开关, 保留手动挂单
            except Exception:
                pass
            await self.gw.stop()
            gw_task.cancel()
            if self.events_svc:
                self.events_svc.stop()
                if ev_task:
                    ev_task.cancel()
                await self.events_svc.aclose()
            if self.priv_ws:
                self.priv_ws.stop()
                if pw_task:
                    pw_task.cancel()
            self.recorder.close()
            await self.store.close()
            await self.rest.aclose()
            await self.pub_rest.aclose()
            self._log("已停止。")

    def request_stop(self) -> None:
        self._stop.set()


def parse_args():
    p = argparse.ArgumentParser(description="OKXB 自动交易编排器")
    p.add_argument("--live", action="store_true", help="真实下单 (demo 模式=模拟盘)")
    p.add_argument("--allow-live-real", action="store_true",
                   help="允许在 OKXB_MODE=live 时用真金下单 (危险)")
    return p.parse_args()


async def _amain(args) -> None:
    secrets = Secrets()
    dry_run = not args.live
    if not dry_run and secrets.mode == Mode.LIVE and not args.allow_live_real:
        print("[app] 拒绝: --live 且 OKXB_MODE=live 会动用真金。"
              "请改用模拟盘 (OKXB_MODE=demo), 或显式加 --allow-live-real 表示你完全清楚后果。")
        return
    app = App(dry_run=dry_run)
    try:
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(sigmod.SIGINT, app.request_stop)
    except (NotImplementedError, RuntimeError):
        pass
    await app.setup()
    await app.run()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(_amain(args))
    except KeyboardInterrupt:
        print("\n[app] 中断退出。")


if __name__ == "__main__":
    main()
