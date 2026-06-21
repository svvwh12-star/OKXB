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
from ..core.enums import OrderType, PosSide, Side
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

    def _ps(self, pos_dir: Side) -> str:
        """持仓方向 -> posSide。单向=net; 双向: 多=long 空=short。"""
        if self._pm != "long_short_mode":
            return PosSide.NET.value
        return "long" if pos_dir == Side.BUY else "short"

    @property
    def _exit_reduce(self) -> bool:
        return self._pm == "net_mode"      # 双向模式由 posSide 决定平仓, 不传 reduceOnly

    async def _place(self, **kw) -> dict:
        """优先 WS 下单 (低延迟), 失败回落 REST。"""
        if self._ws_order is not None and getattr(self._ws_order, "logged_in", False):
            try:
                return await self._ws_order.place_order(**kw)
            except OkxError as e:
                await self._store.audit("ws_place_fallback", repr(e))
        return await self._rest.place_order(**kw)

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
            # IOC: 立即成交或取消; 查回成交后建仓 (无需 TTL)
            try:
                o = await self._rest.get_order(inst, cl_ord_id=cl_oid)
                filled = Decimal(o.get("accFillSz", "0") or "0")
                if filled > 0:
                    await self._open_managed(inst, sig, filled,
                                             Decimal(o.get("avgPx", str(px)) or str(px)))
            except OkxError:
                pass
        else:
            ttl = (self._ttl_min + self._ttl_max) / 2 / 1000.0
            asyncio.create_task(self._ttl_watch(inst, cl_oid, sig, contracts, px, ttl))

    async def _ttl_watch(self, inst, cl_oid, sig, contracts, px, ttl_s) -> None:
        await asyncio.sleep(ttl_s)
        try:
            o = await self._rest.get_order(inst, cl_ord_id=cl_oid)
        except OkxError:
            return
        state = o.get("state")
        filled = Decimal(o.get("accFillSz", "0") or "0")
        if state == "filled" or filled > 0:
            entry_px = Decimal(o.get("avgPx", str(px)) or str(px))
            await self._open_managed(inst, sig, filled or contracts, entry_px)
        elif state in ("live", "partially_filled"):
            try:
                await self._rest.cancel_order(inst, cl_ord_id=cl_oid)
                await self._store.audit("ttl_cancel", f"{inst} {cl_oid} 未成交撤单")
            except OkxError:
                pass
            if filled > 0:
                entry_px = Decimal(o.get("avgPx", str(px)) or str(px))
                await self._open_managed(inst, sig, filled, entry_px)

    # ----------------- 持仓与退出 -----------------

    async def _open_managed(self, inst, sig: Signal, contracts: Decimal, entry_px: Decimal) -> None:
        sign = 1 if sig.side == Side.BUY else -1
        sl_px = entry_px * Decimal(str(1 - sign * sig.sl_pct))
        tp_px = entry_px * Decimal(str(1 + sign * sig.tp_pct))
        tick = self._inst.tick_sz(inst)
        sl_px = round_price(float(sl_px), tick, side_up=(sig.side == Side.SELL))
        tp_px = round_price(float(tp_px), tick, side_up=(sig.side == Side.BUY))
        ctval = self._inst.ct_val(inst)
        max_loss = Decimal(str(sig.sl_pct)) * entry_px * contracts * Decimal(str(ctval))

        pos = ManagedPosition(
            inst_id=inst, side=sig.side, contracts=contracts, entry_px=entry_px,
            sl_px=sl_px, tp_px=tp_px, strategy=sig.strategy, signal_id=sig.signal_id,
            entry_ms=int(time.time() * 1000), max_loss_usdt=max_loss,
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

    async def close_position(self, pos: ManagedPosition, reason: str, mid: Decimal) -> None:
        pos.closing = True
        exit_side = Side.SELL if pos.side == Side.BUY else Side.BUY
        if not self.dry_run:
            # 撤未成交止盈单
            if pos.tp_order_oid:
                try:
                    await self._rest.cancel_order(pos.inst_id, ord_id=pos.tp_order_oid)
                except OkxError:
                    pass
            # 撤交易所端止损 algo (否则平仓后残留一张 reduce-only 委托)
            if pos.sl_algo_oid:
                try:
                    await self._rest.cancel_algos([{"algoId": pos.sl_algo_oid, "instId": pos.inst_id}])
                except OkxError:
                    pass
            try:
                await self._rest.place_order(
                    inst_id=pos.inst_id, td_mode=self._td_mode, side=exit_side.value,
                    ord_type=OrderType.OPTIMAL_LIMIT_IOC.value, sz=str(pos.contracts),
                    pos_side=self._ps(pos.side), reduce_only=self._exit_reduce,
                    cl_ord_id=_clean_oid(pos.signal_id + "cl"),
                )
            except OkxError as e:
                await self._store.audit("close_reject", f"{pos.inst_id} {e}")
        # 估算已实现盈亏 (用 mid 近似)
        sign = 1 if pos.side == Side.BUY else -1
        pnl = float((mid - pos.entry_px) * sign * pos.contracts
                    * Decimal(str(self._inst.ct_val(pos.inst_id))))
        self._risk.register_close(pnl)
        try:                                   # C-4: 平仓即落盘风控状态, 崩溃也不丢硬停账目
            await self._store.set_kv("risk_state", self._risk.to_state())
        except Exception:                      # noqa: BLE001 持久化失败不得影响平仓主流程
            pass
        await self._store.record_pnl(pos.inst_id, pos.strategy.value, pnl, {"reason": reason})
        await self._store.audit("closed", f"{pos.inst_id} {reason} pnl~{pnl:.3f}")
        self._om.remove_position(pos.inst_id)
        self._alert(f"⛔ 平仓 {pos.inst_id} {reason} pnl~{pnl:.3f}U 状态={self._risk.system_state.value}")

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
