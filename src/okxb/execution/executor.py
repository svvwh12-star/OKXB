"""执行器。

dry_run=True (默认): 只记录"将要下的单", 不触达交易所 — 用于安全验证决策链。
dry_run=False: 在 OKX(demo/live) 真实下单:
  - 入场: post-only 限价挂在最优价 (只做 maker); TTL 未成交则撤单。
  - 成交后: 立即挂 reduce-only 限价止盈; 止损与时间止损由 monitor 用 reduce-only
    optimal_limit_ioc 平仓 (无保护市价单禁用)。
  - 周期性 arm cancel-all-after 死手开关。
本版聚焦正确的单笔生命周期; 部分成交重组、激进重挂、basis/反转策略后续补 (TODO)。
"""
from __future__ import annotations

import asyncio
import re
import time
from decimal import Decimal
from typing import Optional

from ..config import Config
from ..core.enums import OrderType, PosSide, Side, StrategyId
from ..core.models import Signal
from ..exchange.instruments import InstrumentCache
from ..exchange.okx_rest import OkxError, OkxRestClient
from ..marketdata.gateway import MarketDataGateway
from ..risk.engine import RiskEngine
from ..risk.sizing import notional_to_contracts, round_price
from ..state.store import StateStore
from .order_manager import ManagedPosition, OrderManager


def _clean_oid(s: str) -> str:
    return ("okxb" + re.sub(r"[^A-Za-z0-9]", "", s))[:32]


class Executor:
    def __init__(self, *, rest: OkxRestClient, gateway: MarketDataGateway,
                 instruments: InstrumentCache, order_manager: OrderManager,
                 risk: RiskEngine, store: StateStore, config: Config,
                 dry_run: bool = True, alert_fn=None, ws=None):
        self._alert = alert_fn or (lambda m: None)
        self._rest = rest
        self._ws_order = ws            # 私有WS下单客户端 (可选); 登录后优先用, 失败回落REST
        self._gw = gateway
        self._inst = instruments
        self._om = order_manager
        self._risk = risk
        self._store = store
        self._cfg = config
        self.dry_run = dry_run
        self._td_mode = config.get("position_mode.margin_mode", "isolated")
        self._pm = "net_mode"          # 账户持仓模式, 由 app 在 setup 后注入 (net/long_short)
        self._max_hold_s = float(config.get("execution.max_hold_seconds", 180))
        self._ttl_min = float(config.get("execution.maker_order_ttl_ms_min", 700))
        self._ttl_max = float(config.get("execution.maker_order_ttl_ms_max", 1500))
        # 信号驱动出场参数 (深度优化)
        sg = config.section("signal")
        self._min_hold_s = float(sg.get("min_hold_seconds", 20))
        enter_th = float(sg.get("min_composite_score", 70))
        self._rev_th = enter_th - float(sg.get("reversal_hyst_gap", 22))
        self._decay_th = float(sg.get("decay_threshold", 40))
        self._confirm_min = float(sg.get("confirm_min", 0.30))
        self._persist = int(sg.get("persist_ticks", 3))
        self._trail_arm = float(sg.get("trail_arm_pct", 0.0015))
        self._trail_frac = float(sg.get("trail_dist_frac_of_sl", 0.5))
        self._exit_cost_mult = float(sg.get("exit_cost_mult", 1.0))
        # C-5: 开仓即在交易所端挂 reduce-only 止损 algo, 崩溃/断线/死手开关后仍生效
        self._exch_stop = bool(config.get("execution.exchange_resident_stop", True))
        self._exch_stop_market = bool(config.get("execution.exchange_resident_stop_market", True))
        # H-7/H-9: 往返手续费率估计 (用于平仓净 PnL, 使熔断账目不偏乐观)
        _fees = config.section("fees")
        self._fee_roundtrip = (float(_fees.get("crypto_maker_pct", 0.02))
                               + float(_fees.get("crypto_taker_pct", 0.05))) / 100.0
        # 孤儿仓对账兜底止损用的保守止损距离 (从持仓均价起)
        self._orphan_sl_frac = float(config.get("execution.orphan_stop_pct", 1.0)) / 100.0
        self._bg_tasks: set = set()            # H-8: 持引用防 GC + 捕获 fire-and-forget 任务异常
        self._orphan_armed: set = set()        # 已为之补挂保护止损的孤儿仓 (防每轮重复挂)
        self._close_locks: dict = {}           # Opt-3: per-仓平仓串行锁

    def _ps(self, pos_dir: Side) -> str:
        """持仓方向 -> posSide。单向=net; 双向: 多=long 空=short。"""
        if self._pm != "long_short_mode":
            return PosSide.NET.value
        return "long" if pos_dir == Side.BUY else "short"

    @property
    def _exit_reduce(self) -> bool:
        return self._pm == "net_mode"      # 双向模式由 posSide 决定平仓, 不传 reduceOnly

    def _spawn(self, coro) -> None:
        """H-8: fire-and-forget 任务必须持引用(防 GC) + done 回调里捕获异常(防静默死亡 -> 孤儿仓)。"""
        t = asyncio.create_task(coro)
        self._bg_tasks.add(t)
        t.add_done_callback(self._on_task_done)

    def _on_task_done(self, t) -> None:
        self._bg_tasks.discard(t)
        try:
            exc = t.exception()
        except asyncio.CancelledError:
            return
        if exc is not None:
            try:
                self._alert(f"⚠ 执行后台任务异常({type(exc).__name__}: {exc}); 可能有未追踪仓, 已记审计待对账")
                self._spawn_safe_audit("bg_task_error", repr(exc))
            except Exception:  # noqa: BLE001
                pass

    def _spawn_safe_audit(self, kind: str, msg: str) -> None:
        try:
            asyncio.create_task(self._store.audit(kind, msg))
        except RuntimeError:
            pass

    async def _place(self, **kw) -> dict:
        """优先 WS 下单 (低延迟), 失败回落 REST。
        H-7a 防双发: WS 可能已把单送达交易所只是回执失败 -> 回落前先按 clOrdId 查重, 命中即适配;
        即便仍回落 REST 重发, 交易所按 clOrdId 幂等回 51016, 再据此适配为"订单已存在", 绝不双仓。"""
        cl = kw.get("cl_ord_id")
        if self._ws_order is not None and getattr(self._ws_order, "logged_in", False):
            try:
                return await self._ws_order.place_order(**kw)
            except OkxError as e:
                await self._store.audit("ws_place_fallback", repr(e))
                if cl:                         # WS 可能已送达 -> 回落 REST 前先查重, 命中则不重发
                    try:
                        o = await self._rest.get_order(kw["inst_id"], cl_ord_id=cl)
                        if o and o.get("state") in ("live", "partially_filled", "filled"):
                            await self._store.audit("ws_dedup", f"{kw['inst_id']} {cl} 已存在({o.get('state')}), 不重发")
                            return {"ordId": o.get("ordId"), "sCode": "0", "_deduped": True}
                    except OkxError:
                        pass                   # 查不到 -> 确未到达, 安全回落
        try:
            return await self._rest.place_order(**kw)
        except OkxError as e:
            if cl and getattr(e, "code", "") == "51016":   # 精确匹配 sCode, 不用子串(防误判)
                await self._store.audit("place_dedup", f"{kw.get('inst_id')} {cl} 51016 重复clOrdId -> 适配已存在单")
                try:
                    o = await self._rest.get_order(kw["inst_id"], cl_ord_id=cl)
                    return {"ordId": o.get("ordId"), "sCode": "0", "_deduped": True}
                except OkxError:
                    return {"ordId": None, "sCode": "0", "_deduped": True}
            raise

    @staticmethod
    def _fee_of(o: dict) -> Decimal:
        try:
            return Decimal(str(o.get("fee", "0") or "0"))   # OKX: 负=已扣手续费, 正=返佣
        except Exception:  # noqa: BLE001
            return Decimal("0")

    async def _resolve_order(self, inst: str, cl_oid: str, *, retries: int = 3,
                             delay_s: float = 0.15) -> tuple[str, Decimal, Optional[Decimal], Decimal]:
        """H-7c 查订单最终态; 对 accFillSz 暂为 0 (IOC 结算延迟) 做有界重试。
        返回 (state, accFillSz, avg_px, fee); 全部查询失败返回 ('unknown', 0, None, 0) 供调用方挂对账。"""
        state = "unknown"
        for i in range(max(1, retries)):
            try:
                o = await self._rest.get_order(inst, cl_ord_id=cl_oid)
            except OkxError:
                o = None
            if o:
                state = o.get("state") or state
                filled = Decimal(o.get("accFillSz", "0") or "0")
                avg = o.get("avgPx")
                if filled > 0:
                    return state, filled, (Decimal(avg) if avg else None), self._fee_of(o)
                if state == "canceled":
                    return state, Decimal("0"), None, self._fee_of(o)   # 终态无成交
                # state=filled 但 accFillSz 暂为 0 (结算延迟) 或仍 live -> 继续重试
            if i < retries - 1:
                await asyncio.sleep(delay_s)
        return state, Decimal("0"), None, Decimal("0")

    async def _persist_risk(self) -> None:
        try:                                   # C-4: 风控状态落盘; 失败不得影响主流程
            await self._store.set_kv("risk_state", self._risk.to_state())
        except Exception:                      # noqa: BLE001
            pass

    # ----------------- 入场 -----------------

    async def submit(self, sig: Signal, approved_notional: Decimal, is_strong: bool) -> None:
        inst = sig.inst_id
        if not self._inst.is_tradable(inst):
            return
        bbo = self._gw.get_bbo(inst) or self._gw.get_book(inst).bbo()
        if not bbo:
            return
        # post-only 挂在己方最优价 (买挂 best_bid, 卖挂 best_ask)
        raw_px = bbo.bid_px if sig.side == Side.BUY else bbo.ask_px
        px = round_price(raw_px, self._inst.tick_sz(inst), side_up=(sig.side == Side.SELL))
        contracts = notional_to_contracts(
            float(approved_notional), float(px), self._inst.ct_val(inst),
            self._inst.lot_sz(inst), self._inst.min_sz(inst),
        )
        if contracts <= 0:
            await self._store.audit("skip", f"{inst} 名义{approved_notional} 不足1张")
            return

        await self._store.record_signal(inst, sig.side.value, sig.composite_score, {
            "edge_to_cost": sig.edge_to_cost, "model_prob": sig.model_prob,
            "sl_pct": sig.sl_pct, "tp_pct": sig.tp_pct, "strong": is_strong,
        })

        otype = OrderType.OPTIMAL_LIMIT_IOC.value if sig.taker else OrderType.POST_ONLY.value
        kind = "taker" if sig.taker else "maker"
        if self.dry_run:
            print(f"[DRY] {inst} {sig.side.value} {contracts}张 @ "
                  f"{'市价IOC' if sig.taker else px} ({kind} {sig.strategy.value} "
                  f"comp={sig.composite_score} e/c={sig.edge_to_cost:.1f} "
                  f"SL={sig.sl_pct*100:.2f}% TP={sig.tp_pct*100:.2f}%)")
            await self._store.audit("dry_order", f"{inst} {sig.side.value} {contracts} {kind} {sig.strategy.value}")
            return

        cl_oid = _clean_oid(sig.signal_id)
        try:
            res = await self._place(
                inst_id=inst, td_mode=self._td_mode, side=sig.side.value,
                ord_type=otype, sz=str(contracts),
                px=(None if sig.taker else str(px)),
                pos_side=self._ps(sig.side), reduce_only=False, cl_ord_id=cl_oid,
            )
        except OkxError as e:
            await self._store.audit("order_reject", f"{inst} {e}")
            return
        await self._store.upsert_order({
            "client_oid": cl_oid, "inst": inst, "side": sig.side.value,
            "ord_type": otype, "px": (None if sig.taker else str(px)), "sz": str(contracts),
            "state": "live", "strategy": sig.strategy.value, "signal_id": sig.signal_id,
            "created_ms": int(time.time() * 1000), "okx_ord_id": res.get("ordId"),
            "filled_sz": "0", "avg_px": None, "json": {"comp": sig.composite_score},
        })
        if sig.taker:
            # IOC: 立即成交或取消; 有界重试查回成交(防 accFillSz 结算延迟漏建仓)后建仓
            state, filled, avg, fee = await self._resolve_order(inst, cl_oid)
            if filled > 0 or state in ("filled", "partially_filled"):
                await self._open_managed(inst, sig, filled or contracts, avg or Decimal(str(px)), entry_fee=fee)
            elif state != "canceled":
                # 非干净撤销 (unknown / 仍 live 且未拿到成交) -> 挂对账, 绝不静默丢弃可能已成交的单
                await self._store.audit("taker_reconcile_needed",
                                        f"{inst} {cl_oid} IOC 成交未定(state={state}), 待对账 (避免漏建裸仓)")
        else:
            ttl = (self._ttl_min + self._ttl_max) / 2 / 1000.0
            self._spawn(self._ttl_watch(inst, cl_oid, sig, contracts, px, ttl))

    async def _ttl_watch(self, inst, cl_oid, sig, contracts, px, ttl_s) -> None:
        await asyncio.sleep(ttl_s)
        state, filled, avg, fee = await self._resolve_order(inst, cl_oid)
        if state == "filled" or filled > 0:
            await self._open_managed(inst, sig, filled or contracts, avg or Decimal(str(px)), entry_fee=fee)
            return
        if state in ("live", "partially_filled"):
            try:
                await self._rest.cancel_order(inst, cl_ord_id=cl_oid)
                await self._store.audit("ttl_cancel", f"{inst} {cl_oid} 未成交撤单")
            except OkxError as e:
                await self._store.audit("ttl_cancel_fail", f"{inst} {cl_oid} {e}")
            # H-7b 撤单与成交可能同时发生 -> 撤单后再查【权威最终成交】, 据此建仓, 杜绝悬挂裸仓
            state2, filled2, avg2, fee2 = await self._resolve_order(inst, cl_oid)
            if filled2 > 0 or state2 == "filled":
                await self._open_managed(inst, sig, filled2 or contracts, avg2 or Decimal(str(px)), entry_fee=fee2)
        elif state == "unknown":
            await self._store.audit("ttl_reconcile_needed", f"{inst} {cl_oid} 状态未知, 待对账")

    # ----------------- 持仓与退出 -----------------

    async def _open_managed(self, inst, sig: Signal, contracts: Decimal, entry_px: Decimal,
                            entry_fee: Decimal = Decimal("0")) -> None:
        sign = 1 if sig.side == Side.BUY else -1
        sl_px = entry_px * Decimal(str(1 - sign * sig.sl_pct))
        tp_px = entry_px * Decimal(str(1 + sign * sig.tp_pct))
        tick = self._inst.tick_sz(inst)
        sl_px = round_price(float(sl_px), tick, side_up=(sig.side == Side.SELL))
        tp_px = round_price(float(tp_px), tick, side_up=(sig.side == Side.BUY))
        ctval = self._inst.ct_val(inst)
        max_loss = Decimal(str(sig.sl_pct)) * entry_px * contracts * Decimal(str(ctval))
        fee_pc = (Decimal(str(entry_fee)) / contracts) if contracts > 0 else Decimal("0")   # 入场费/张

        pos = ManagedPosition(
            inst_id=inst, side=sig.side, contracts=contracts, entry_px=entry_px,
            sl_px=sl_px, tp_px=tp_px, strategy=sig.strategy, signal_id=sig.signal_id,
            entry_ms=int(time.time() * 1000), max_loss_usdt=max_loss,
            entry_fee_per_contract=fee_pc,
        )
        # 止盈: reduce-only 限价 (交易所托管)
        exit_side = Side.SELL if sig.side == Side.BUY else Side.BUY
        try:
            tp = await self._rest.place_order(
                inst_id=inst, td_mode=self._td_mode, side=exit_side.value,
                ord_type=OrderType.LIMIT.value, sz=str(contracts), px=str(tp_px),
                pos_side=self._ps(sig.side), reduce_only=self._exit_reduce,
                cl_ord_id=_clean_oid(sig.signal_id + "tp"),
            )
            pos.tp_order_oid = tp.get("ordId")
        except OkxError as e:
            await self._store.audit("tp_reject", f"{inst} {e}")
        # C-5: 交易所端 reduce-only 止损 (algo 不被 cancel-all-after 死手开关撤掉, 崩溃后仍护盘)。
        # 失败不阻断建仓: 退回本地 monitor_positions 止损 (但崩溃即裸仓), 故大声告警。
        if self._exch_stop:
            try:
                algo = await self._rest.place_algo_order(
                    inst_id=inst, td_mode=self._td_mode, side=exit_side.value,
                    sz=str(contracts), pos_side=self._ps(sig.side), reduce_only=self._exit_reduce,
                    sl_trigger_px=str(sl_px),
                    sl_ord_px=("-1" if self._exch_stop_market else str(sl_px)),
                    trigger_px_type="last",
                )
                pos.sl_algo_oid = algo.get("algoId")
            except OkxError as e:
                await self._store.audit("sl_algo_reject", f"{inst} {e}")
                self._alert(f"⚠ {inst} 交易所端止损挂单失败({e}); 仅剩本地止损, 进程崩溃将裸仓")
        self._om.add_position(pos)
        await self._store.audit("opened", f"{inst} {sig.side.value} {contracts}@{entry_px} "
                                          f"SL={sl_px} TP={tp_px}")
        self._alert(f"✅ 开仓 {inst} {sig.side.value} {contracts}张 @{entry_px} "
                    f"SL={sl_px} TP={tp_px} ({sig.strategy.value})")

    async def monitor_positions(self) -> None:
        """周期调用: 止损 + 时间止损 (reduce-only 平仓)。止盈由交易所挂单托管。"""
        now = int(time.time() * 1000)
        for pos in self._om.all():
            if pos.closing:
                continue
            bbo = self._gw.get_bbo(pos.inst_id) or self._gw.get_book(pos.inst_id).bbo()
            if not bbo:
                continue
            mid = Decimal(str(bbo.mid))
            hit_sl = (mid <= pos.sl_px) if pos.side == Side.BUY else (mid >= pos.sl_px)
            timed_out = (now - pos.entry_ms) > self._max_hold_s * 1000
            if hit_sl or timed_out:
                await self.close_position(pos, "止损" if hit_sl else "时间止损", mid)

    async def signal_exit(self, pos: ManagedPosition, fs, comp, mid: float, cost: float) -> bool:
        """信号驱动出场 (由 _tick 用已算好的 fs/comp 调用, 不重复计算特征)。
        最短持仓内不动; 移动止盈 / 真实反转(迟滞)+衰减; 均过扣费保护。返回是否已平仓。
        硬止损/时间止损仍由 monitor_positions 负责(始终生效)。"""
        if pos.closing:
            return False
        now = time.time() * 1000.0
        if (now - pos.entry_ms) < self._min_hold_s * 1000.0:    # 最短持仓: 防刚进就被噪声晃出
            return False
        sign = 1 if pos.side == Side.BUY else -1
        entry = float(pos.entry_px)
        if entry <= 0:
            return False
        moved = sign * (mid / entry - 1.0)                       # 顺向收益(小数)
        sl_pct = abs(float(pos.sl_px) / entry - 1.0)
        pos.hwm = max(pos.hwm or mid, mid) if pos.side == Side.BUY else min(pos.hwm or mid, mid)

        # 1) 移动止盈: 顺向达 trail_arm 后启动; 回撤超 trail_dist 且锁利>=成本 则平
        if moved >= self._trail_arm:
            trail_dist = max(self._trail_frac * sl_pct, 0.0008) * entry
            breached = (mid <= pos.hwm - trail_dist) if pos.side == Side.BUY \
                else (mid >= pos.hwm + trail_dist)
            if breached and moved >= self._exit_cost_mult * cost:
                await self.close_position(pos, "移动止盈", Decimal(str(mid)))
                return True

        # 2) 真实反转(迟滞)或 自身方向分量衰减 -> 连续 persist 拍才平; 过扣费保护
        opp = -sign
        # 用平滑分 + 平滑方向 判反转, 与入场去噪视图一致 (避免被单拍噪声晃出)
        # 对称尺度(多+空=100, 50中性): 自身分=顺向综合分, 反向分=100−自身分。
        own_score = comp.long_score_s if pos.side == Side.BUY else comp.short_score_s
        opp_score = comp.short_score_s if pos.side == Side.BUY else comp.long_score_s
        opp_conf = 0.5 * max(0.0, opp * comp.order_flow_dir) + 0.5 * max(0.0, opp * comp.trend_dir)
        # 反转: 反向分越过迟滞阈值 + 方向确认; 衰减: 自身分跌回中性附近(edge 消失)
        weak = (opp_score >= self._rev_th and opp_conf >= self._confirm_min) or (own_score < self._decay_th)
        pos.rev_run = pos.rev_run + 1 if weak else 0
        if pos.rev_run >= self._persist and sl_pct >= self._exit_cost_mult * cost:
            await self.close_position(pos, "反转/衰减", Decimal(str(mid)))
            return True
        return False

    def _net_pnl(self, pos: ManagedPosition, fill_px: Decimal, filled: Decimal,
                 exit_fee: Decimal = Decimal("0")) -> float:
        """已实现【净】盈亏 (H-9): 毛 + 真实手续费 (入场摊到本片 + 本次出场, OKX fee 负=已扣)。
        无真实手续费时(演练/接口未返回)退回往返费率估计。"""
        sign = 1 if pos.side == Side.BUY else -1
        ctval = Decimal(str(self._inst.ct_val(pos.inst_id)))
        gross = float((fill_px - pos.entry_px) * sign * filled * ctval)
        actual_fee = float(pos.entry_fee_per_contract * filled) + float(exit_fee)
        if actual_fee != 0.0:
            return gross + actual_fee                  # 真实费用 (负值 -> 扣减)
        est = abs(float(fill_px) * float(filled) * float(ctval)) * self._fee_roundtrip
        return gross - est

    async def _cancel_protection(self, pos: ManagedPosition) -> None:
        if pos.tp_order_oid:
            try:
                await self._rest.cancel_order(pos.inst_id, ord_id=pos.tp_order_oid)
            except OkxError:
                pass
        if pos.sl_algo_oid:
            try:
                await self._rest.cancel_algos([{"algoId": pos.sl_algo_oid, "instId": pos.inst_id}])
            except OkxError:
                pass

    def _close_lock(self, inst: str) -> asyncio.Lock:
        lk = self._close_locks.get(inst)
        if lk is None:
            lk = asyncio.Lock()
            self._close_locks[inst] = lk
        return lk

    async def close_position(self, pos: ManagedPosition, reason: str, mid: Decimal) -> None:
        # Opt-3: per-仓串行锁, 固化"同一仓不并发双平"不变量 (当前确为串行, 防未来 WS 驱动并发引入双平)
        async with self._close_lock(pos.inst_id):
            if not self._om.has(pos.inst_id):
                return                         # 已被并发的另一次平仓处理掉
            await self._close_impl(pos, reason, mid)

    async def _close_impl(self, pos: ManagedPosition, reason: str, mid: Decimal) -> None:
        pos.closing = True
        exit_side = Side.SELL if pos.side == Side.BUY else Side.BUY

        if self.dry_run:                       # 演练: 用 mid 估算净额, 直接记账平仓
            pnl = self._net_pnl(pos, mid, pos.contracts)
            self._risk.register_close(pnl)
            await self._persist_risk()
            await self._store.record_pnl(pos.inst_id, pos.strategy.value, pnl, {"reason": reason, "dry": True})
            self._om.remove_position(pos.inst_id)
            self._alert(f"⛔ 平仓(演练) {pos.inst_id} {reason} pnl~{pnl:.3f}U")
            return

        # H-7-A: 先下平仓 IOC, 交易所端 SL/TP 暂【保留】; 只有确认全平后才撤保护 ->
        # 部分/零成交时剩余仓位仍受原 SL/TP 守护, 消除"撤了保护却没平掉"的裸仓窗口。
        # (SL algo 与平仓单同为 reduce-only, 即便同时触发也不会超额平仓。)
        pos.close_attempts += 1                # 计数前置于 clOrdId, 截断后仍唯一: 防重发去重 + 支持续平
        cl = _clean_oid(f"cl{pos.close_attempts}{pos.signal_id}")
        try:
            await self._rest.place_order(
                inst_id=pos.inst_id, td_mode=self._td_mode, side=exit_side.value,
                ord_type=OrderType.OPTIMAL_LIMIT_IOC.value, sz=str(pos.contracts),
                pos_side=self._ps(pos.side), reduce_only=self._exit_reduce, cl_ord_id=cl,
            )
        except OkxError as e:
            await self._store.audit("close_reject", f"{pos.inst_id} {e}")
            pos.closing = False                # 下单失败 -> 保持持仓+原保护, 下拍重试; 绝不假装已平
            return

        state, filled, avg, exit_fee = await self._resolve_order(pos.inst_id, cl)
        if filled <= 0:                        # 未成交: 保留持仓+原 SL/TP (幽灵仓由 reconcile_positions 兜底清理)
            await self._store.audit("close_unfilled", f"{pos.inst_id} 平仓IOC未成交(state={state}), 保持持仓+原止损待重试")
            pos.closing = False
            return
        fill_px = avg or mid                   # H-9: 用真实成交价算 PnL
        if filled < pos.contracts:             # 部分成交: 记分片(非终态不动连亏), 留剩余量, 原 SL/TP 仍守护
            pnl = self._net_pnl(pos, fill_px, filled, exit_fee)
            self._risk.register_close(pnl, final=False)
            await self._persist_risk()
            await self._store.record_pnl(pos.inst_id, pos.strategy.value, pnl,
                                         {"reason": reason, "fill_px": str(fill_px), "filled": str(filled), "partial": True})
            pos.contracts = pos.contracts - filled
            pos.closing = False
            await self._store.audit("close_partial", f"{pos.inst_id} 部分平{filled} 剩{pos.contracts}(原止损仍守)")
            self._alert(f"⚠ 部分平仓 {pos.inst_id} {reason} 已平{filled} 剩{pos.contracts} pnl~{pnl:.3f}U")
            return
        # 全部成交 -> 此时才撤交易所端 TP/SL (先前不撤, 故部分成交不会裸仓)
        await self._cancel_protection(pos)
        pnl = self._net_pnl(pos, fill_px, filled, exit_fee)
        self._risk.register_close(pnl, final=True)
        await self._persist_risk()
        await self._store.record_pnl(pos.inst_id, pos.strategy.value, pnl,
                                     {"reason": reason, "fill_px": str(fill_px), "filled": str(filled)})
        self._orphan_armed.discard(pos.inst_id)
        await self._store.audit("closed", f"{pos.inst_id} {reason} pnl~{pnl:.3f} fill@{fill_px}")
        self._om.remove_position(pos.inst_id)
        self._alert(f"⛔ 平仓 {pos.inst_id} {reason} pnl~{pnl:.3f}U 状态={self._risk.system_state.value}")

    async def reconcile_positions(self, real_positions: list, *, ghost_min_age_ms: int = 15_000) -> None:
        """H-7 critical: 闭合"挂了对账日志却无人消费"的洞。周期对账交易所真实持仓 vs 本地受管:
          - 孤儿仓 (交易所有 / 本地无): 立即补挂保护性 reduce-only 止损 + 告警 (覆盖 IOC 漏建/任务死亡等);
          - 幽灵仓 (本地有 / 交易所无, 且足够老以排除快照延迟): 移除本地受管 (已被 SL/TP/外部平掉)。"""
        if self.dry_run:
            return
        real = {}
        for p in (real_positions or []):
            inst = p.get("instId")
            if inst and abs(float(p.get("pos") or 0)) > 0:
                real[inst] = p
        # 1) 孤儿仓 -> 补挂保护止损 + Opt-2: 重建完整 ManagedPosition 纳入本地监控
        for inst, p in real.items():
            if self._om.has(inst) or inst in self._orphan_armed:
                continue
            try:
                long = float(p.get("pos") or 0) > 0
                avg = float(p.get("avgPx") or 0)
                sz = abs(float(p.get("pos") or 0))
                if avg <= 0:
                    continue
                tick = self._inst.tick_sz(inst)
                side = Side.BUY if long else Side.SELL
                sl_raw = avg * (1 - self._orphan_sl_frac) if long else avg * (1 + self._orphan_sl_frac)
                sl_px = round_price(sl_raw, tick, side_up=(not long))
                tp_raw = avg * (1 + 2 * self._orphan_sl_frac) if long else avg * (1 - 2 * self._orphan_sl_frac)
                tp_px = round_price(tp_raw, tick, side_up=long)
                algo = await self._rest.place_algo_order(
                    inst_id=inst, td_mode=self._td_mode, side=(Side.SELL if long else Side.BUY).value,
                    sz=str(sz), pos_side=("net" if self._pm == "net_mode" else ("long" if long else "short")),
                    reduce_only=(self._pm == "net_mode"),
                    sl_trigger_px=str(sl_px), sl_ord_px="-1", trigger_px_type="last")
                # 重建受管持仓 -> monitor_positions(本地SL+时间止损) + 信号出场 接管该仓; 用真实入场均价/时间
                ctime = int(p.get("cTime") or 0) or int(time.time() * 1000)
                self._om.add_position(ManagedPosition(
                    inst_id=inst, side=side, contracts=Decimal(str(sz)), entry_px=Decimal(str(avg)),
                    sl_px=sl_px, tp_px=tp_px, strategy=StrategyId.RECONCILED,
                    signal_id=f"orphan{inst}{ctime}", entry_ms=ctime, max_loss_usdt=Decimal("0"),
                    sl_algo_oid=(algo.get("algoId") if isinstance(algo, dict) else None)))
                self._orphan_armed.add(inst)
                await self._store.audit("orphan_protected", f"{inst} 未追踪仓 sz={sz}@{avg} 补挂止损@{sl_px} 并接管管理")
                self._alert(f"⚠ 发现未追踪持仓 {inst} sz={sz}@{avg}, 已补挂止损@{sl_px} 并纳入本地监控 (请人工核查来源)")
            except OkxError as e:
                await self._store.audit("orphan_protect_fail", f"{inst} {e}")
        # 2) 幽灵仓 -> 清理本地 (加 age 守护, 避免误删快照尚未反映的新仓)
        now = int(time.time() * 1000)
        for mp in self._om.all():
            if mp.inst_id not in real and (now - mp.entry_ms) > ghost_min_age_ms:
                self._om.remove_position(mp.inst_id)
                self._orphan_armed.discard(mp.inst_id)
                await self._store.audit("ghost_removed", f"{mp.inst_id} 交易所已无此仓(疑被SL/TP/外部平), 清理本地受管")
                self._alert(f"ℹ {mp.inst_id} 交易所已无持仓, 清理本地记录")

    async def arm_kill_switch(self) -> None:
        if self.dry_run or not self._cfg.get("execution.cancel_all_after.enabled", True):
            return
        timeout = int(self._cfg.get("execution.cancel_all_after.timeout_seconds", 30))
        try:
            await self._rest.cancel_all_after(timeout, tag="okxb")
        except OkxError:
            pass

    async def disarm_kill_switch(self) -> None:
        """正常停止时解除死手开关(timeout=0), 避免停止后约timeout秒把手动挂单也撤掉。"""
        if self.dry_run or not self._cfg.get("execution.cancel_all_after.enabled", True):
            return
        try:
            await self._rest.cancel_all_after(0, tag="okxb")
        except OkxError:
            pass
