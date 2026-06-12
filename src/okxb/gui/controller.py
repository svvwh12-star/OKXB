"""GUI 引擎控制器: 在后台线程跑 asyncio 交易引擎, 向 GUI 暴露线程安全的状态/日志,
并提供 .env 写入与一键账户验证。

线程模型: GUI 在主线程跑 tkinter mainloop; 引擎在后台线程跑 asyncio.run。
通信: 引擎写 app.latest_status / latest_rows (引用原子替换, GUI 只读);
日志经 deque (CPython append/popleft 线程安全) 传给 GUI。
停止: 用 loop.call_soon_threadsafe 触发引擎的 stop event。
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
from collections import deque
from typing import Optional

from .. import paths
from ..config import Config, Secrets
from ..core.enums import Mode


# ----------------- .env 读写 (绝不记录到日志) -----------------

def read_env() -> dict[str, str]:
    env: dict[str, str] = {}
    if paths.ENV_PATH.exists():
        for line in paths.ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip()
    return env


def write_env(updates: dict[str, str]) -> None:
    """合并写入 .env, 并同步到 os.environ (使同进程 Secrets() 立即读到新值)。"""
    env = read_env()
    env.update({k: v for k, v in updates.items() if v is not None})
    lines = ["# OKXB 凭据与设置 (本文件含密钥, 切勿外传/提交)"]
    for k, v in env.items():
        lines.append(f"{k}={v}")
    paths.ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for k, v in updates.items():
        if v is not None:
            os.environ[k] = v


# ----------------- 引擎控制器 -----------------

class EngineController:
    def __init__(self):
        self.app = None
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.running = False
        self.logs: deque = deque(maxlen=1000)
        self._error: Optional[str] = None

    def start(self, dry_run: bool) -> None:
        if self.running:
            return
        self._error = None
        self._thread = threading.Thread(target=self._thread_main, args=(dry_run,), daemon=True)
        self._thread.start()

    def _thread_main(self, dry_run: bool) -> None:
        try:
            asyncio.run(self._arun(dry_run))
        except Exception as e:
            self._error = repr(e)
            self.logs.append(f"引擎线程异常: {e!r}")
        finally:
            self.running = False

    async def _arun(self, dry_run: bool) -> None:
        from ..app import App
        self._loop = asyncio.get_running_loop()
        self.app = App(dry_run=dry_run, log_fn=self._on_log)
        self.running = True
        try:
            await self.app.setup()
            await self.app.run()
        finally:
            self.running = False

    def _on_log(self, line: str) -> None:
        self.logs.append(line)

    def stop(self) -> None:
        if self._loop and self.app and self.running:
            try:
                self._loop.call_soon_threadsafe(self.app.request_stop)
            except Exception:
                pass

    # GUI 只读访问
    def status(self) -> dict:
        return dict(self.app.latest_status) if self.app else {}

    def rows(self) -> dict:
        return dict(self.app.latest_rows) if self.app else {}

    def drain_logs(self) -> list[str]:
        out = []
        while self.logs:
            try:
                out.append(self.logs.popleft())
            except IndexError:
                break
        return out

    @property
    def error(self) -> Optional[str]:
        return self._error


# ----------------- 三个独立验证 (只读) -----------------

def _secrets_for_mode(mode: Mode) -> Secrets:
    """构造指定模式(demo/live)的 Secrets, 从 .env 取对应那套密钥 (不受当前 OKXB_MODE 影响)。"""
    s = Secrets()
    s.mode = mode
    env = read_env()
    pre = "OKX_LIVE_" if mode == Mode.LIVE else "OKX_DEMO_"
    s.okx_api_key = env.get(pre + "API_KEY", "")
    s.okx_secret_key = env.get(pre + "SECRET_KEY", "")
    s.okx_passphrase = env.get(pre + "PASSPHRASE", "")
    s.region = env.get("OKX_REGION", "global")
    return s


def verify_okx_sync(mode_str: str) -> str:
    """验证某一套 OKX 密钥 (mode_str = 'demo' 或 'live'); GUI 在 worker 线程调用。"""
    try:
        return asyncio.run(_averify_okx(_secrets_for_mode(Mode(mode_str))))
    except Exception as e:
        return f"验证失败: {e!r}"


async def _averify_okx(secrets: Secrets) -> str:
    from ..exchange.okx_rest import OkxError, OkxRestClient
    cfg = Config.load()
    tag = "虚拟盘(demo)" if secrets.mode == Mode.DEMO else "实盘(live)"
    if not secrets.okx_api_key:
        return f"[{tag}] 未配置密钥, 请在上方填入并保存。"
    rest = OkxRestClient(secrets, cfg)
    out = [f"=== {tag}  区域={secrets.region}  key尾号=...{secrets.okx_api_key[-4:]} ==="]
    try:
        cfg_data = await rest.get_account_config()
        perm = cfg_data.get("perm", "")
        out.append(f"鉴权通过 ✓  权限={perm}")
        out.append("含交易权限 ✓" if "trade" in perm else "⚠ 无交易权限")
        out.append("⚠ 含提现权限, 强烈建议关闭!" if "withdraw" in perm else "无提现权限 ✓")
        ip = cfg_data.get("ip", "")
        out.append(f"已绑定IP: {ip} ✓" if ip else "⚠ 未绑定IP (高危, 请在OKX绑定)")
        out.append(f"账户层级={cfg_data.get('acctLv')}  持仓模式={cfg_data.get('posMode')}")
        bal = await rest.get_balance()
        out.append(f"账户权益 totalEq={bal.get('totalEq','?')} USD")
        poss = [p for p in await rest.get_positions() if float(p.get('pos', 0) or 0) != 0]
        out.append(f"当前持仓: {len(poss)} 个")
        insts = await rest.get_instruments("SWAP")
        live_usdt = [i["instId"] for i in insts
                     if i.get("state") == "live" and i["instId"].endswith("-USDT-SWAP")]
        # 用真实股票代码表匹配 (与选标的/控制台一致), 而非"非6大币"的粗糙启发式(会把山寨当股票)
        stock_syms = {str(s).upper() for s in cfg.get("universe.stock_symbols", [])}
        stock = [s for s in live_usdt if s.split("-")[0].upper() in stock_syms]
        out.append(f"USDT永续合约总数: {len(live_usdt)}  命中股票永续: {len(stock)} 个")
        if stock:
            out.append(f"  股票永续示例: {stock[:8]}")
            out.append("  ✓ 这些会进入控制台『美股』池; 若控制台仍不显示, 截图发我排查渲染层。")
        else:
            # 决定性探针: 直接单合约查 AAPL-USDT-SWAP, 看交易所到底有没有对这个账户/地区返回它
            probe = "—"
            try:
                p = await rest.get_instruments("SWAP", "AAPL-USDT-SWAP")
                probe = (f"state={p[0].get('state')}" if p else "未返回(空)")
            except OkxError as pe:
                probe = f"错误 {pe}"
            out.append(f"  ⚠ 该账户/地区【看不到任何股票永续】(探针 AAPL-USDT-SWAP: {probe})。")
            out.append("  OKX 股票永续按地区开放(亚洲/CIS/拉美/土耳其等), 美国等被排除 —")
            out.append("  这是交易所侧地区限制, 非本程序bug, 且严禁用VPN绕过。加密永续不受影响。")
    except OkxError as e:
        out.append(f"✗ 失败: {e}")
        out.append("常见: 密钥/passphrase错误、IP未白名单、demo需用Demo Trading区单独创建的key、"
                   "区域(global/us/eea)选错。")
    finally:
        await rest.aclose()
    return "\n".join(out)


def verify_ai_sync() -> str:
    """验证 AI 提供商 (DeepSeek 等) 的 key 与连通性。"""
    try:
        return asyncio.run(_averify_ai())
    except Exception as e:
        return f"AI 验证失败: {e!r}"


async def _averify_ai() -> str:
    from ..events.llm_classifier import LLMClassifier
    return await LLMClassifier.from_secrets(Secrets()).ping()


def verify_telegram_sync() -> str:
    """验证 Telegram 告警 (会发一条测试消息)。"""
    try:
        return asyncio.run(_averify_tg())
    except Exception as e:
        return f"Telegram 验证失败: {e!r}"


def verify_finnhub_sync() -> str:
    try:
        return asyncio.run(_averify_finnhub())
    except Exception as e:
        return f"Finnhub 验证失败: {e!r}"


async def _averify_finnhub() -> str:
    from ..events.finnhub import FinnhubClient
    import datetime as _dt
    fh = FinnhubClient(Secrets().finnhub_api_key)
    if not fh.enabled:
        return "未填 FINNHUB_API_KEY。\n→ 留空时 AI选品 没有真实新闻/财报; 填入后选品会拉取并喂给 AI。"
    try:
        today = _dt.datetime.now(_dt.timezone.utc).date()
        cal = await fh.earnings_calendar(str(today), str(today + _dt.timedelta(days=7)))
        news = await fh.general_news("general")
        eg = (str((news or [{}])[0].get("headline", ""))[:60]) if news else "(新闻为空, 免费档有时受限)"
        return (f"Finnhub 连通 ✓\n未来7天财报条目: {len(cal)}\n市场新闻头条: {len(news)}\n示例: {eg}\n"
                "AI选品会把这些真实信息发给 DeepSeek 分析。")
    finally:
        await fh.aclose()


def verify_edgar_sync() -> str:
    try:
        return asyncio.run(_averify_edgar())
    except Exception as e:
        return f"EDGAR 验证失败: {e!r}"


async def _averify_edgar() -> str:
    from ..events.edgar import EdgarClient
    ua = Secrets().edgar_user_agent
    ed = EdgarClient(ua)
    try:
        m = await ed.load_ticker_map()
        if not m:
            return ("EDGAR 拉取失败 (User-Agent 缺失/被限/网络)。\n"
                    "请填 EDGAR_USER_AGENT = '你的名字 邮箱' (SEC 合规要求)。")
        return (f"EDGAR 连通 ✓\nticker→CIK 映射: {len(m)} 条\nAAPL CIK: {ed.cik_for('AAPL')}\n"
                f"User-Agent: {ua or '(默认占位, 建议填真实姓名+邮箱)'}\n"
                "EDGAR 用于股票永续的 SEC 公告事件风控(引擎自动用)。")
    finally:
        await ed.aclose()


async def _averify_tg() -> str:
    from ..monitor.telegram import TelegramNotifier
    s = Secrets()
    return await TelegramNotifier(s.telegram_bot_token, s.telegram_chat_id).verify()


# ----------------- 手动交易 (用户在 GUI 直接精细操作) -----------------

def _new_trade_rest():
    from ..exchange.okx_rest import OkxRestClient
    s = Secrets()
    if not s.okx_api_key:
        raise RuntimeError(f"{s.mode.value} 未配置密钥, 请先在『账户与密钥』填写并保存。")
    cfg = Config.load()
    return OkxRestClient(s, cfg), s.mode.value, cfg.get("position_mode.margin_mode", "isolated")


def manual_place_sync(inst_id, side, ord_type, amount, unit, px, reduce_only) -> str:
    return _run(_manual_place(inst_id, side, ord_type, amount, unit, px, reduce_only))


def manual_cancel_all_sync(inst_id) -> str:
    return _run(_manual_cancel_all(inst_id))


def manual_close_all_sync() -> str:
    return _run(_manual_close_all())


def ai_analyze_sync(inst_id, row: dict) -> dict:
    """结构化量化分析 (含客观最优杠杆)。row = 控制台该标的实时行 (可空)。返回 {text, struct}。"""
    return _run_obj(_ai_analyze(inst_id, row or {}))


def ai_pick_sync(pool: str, rows: dict) -> dict:
    """AI 选品。pool ∈ {crypto, stock, all}; rows = 控制台全部实时行 (可空)。返回 {text, picks}。"""
    return _run_obj(_ai_pick(pool, rows or {}))


def manual_close_sync(inst_id) -> str:
    return _run(_manual_close(inst_id))


def manual_set_leverage_sync(inst_id, lever) -> str:
    return _run(_manual_set_leverage(inst_id, lever))


def _run(coro) -> str:
    try:
        return asyncio.run(coro)
    except Exception as e:
        return f"操作失败: {e!r}"


def _run_obj(coro):
    try:
        return asyncio.run(coro)
    except Exception as e:
        return {"text": f"操作失败: {e!r}", "struct": None, "picks": []}


async def _get_spec(rest, inst_id):
    from ..exchange.okx_rest import OkxError
    try:
        insts = await rest.get_instruments("SWAP", inst_id)   # 单合约查询, 快
    except OkxError:
        return None              # 不存在的标的(51001)等 -> 视为找不到, 调用方给友好提示
    return insts[0] if insts else None


_POSMODE_CACHE: dict = {}        # mode_str -> 'net_mode' / 'long_short_mode'


async def _pos_mode(rest, mode_str) -> str:
    """账户持仓模式: net_mode(单向) / long_short_mode(双向)。缓存避免重复查询。"""
    if mode_str in _POSMODE_CACHE:
        return _POSMODE_CACHE[mode_str]
    pm = "net_mode"
    try:
        cfg = await rest.get_account_config()
        pm = cfg.get("posMode", "net_mode") or "net_mode"
    except Exception:
        pass
    _POSMODE_CACHE[mode_str] = pm
    return pm


def _posside(pm: str, side: str, opening: bool) -> str:
    """双向模式: 开多->long 开空->short; 平仓时 sell平多(long)/buy平空(short)。单向->net。"""
    if pm != "long_short_mode":
        return "net"
    if opening:
        return "long" if side == "buy" else "short"
    return "long" if side == "sell" else "short"


def _round_tick(px, tick) -> str | None:
    """把价格对齐到合约 tickSz (否则 OKX 报价格精度错误 -> All operations failed)。"""
    from decimal import ROUND_HALF_UP, Decimal
    if px in (None, "", 0, "0"):
        return None
    t = Decimal(str(tick or "0.0001"))
    if t <= 0:
        return str(px)
    return str((Decimal(str(px)) / t).quantize(Decimal(1), rounding=ROUND_HALF_UP) * t)


def _resolve_maker_px(side, ord_type, user_px, ticker, tick):
    """按当前市场返回有效挂单价 (修 51006: 导入价过期/越界)。
    post_only 必须被动: 卖>=ask、买<=bid, 否则贴当前被动最优; 限价越界太远也贴近。market 返回 None。"""
    if ord_type not in ("post_only", "limit"):
        return None
    bid = float(ticker.get("bidPx", 0) or 0)
    ask = float(ticker.get("askPx", 0) or 0)
    passive = ask if side == "sell" else bid
    up = float(user_px) if user_px else 0.0
    if not (bid > 0 and ask > 0):
        return _round_tick(up or passive, tick)
    mid = (bid + ask) / 2.0
    far = up <= 0 or abs(up / mid - 1.0) > 0.10     # 离市场>10% -> 视为过期/越界(OKX限价带约±13%)
    if ord_type == "post_only":
        if far:
            px = passive
        elif side == "sell":
            px = up if up >= ask else passive       # 卖必须>=ask才是被动maker, 否则贴ask
        else:
            px = up if up <= bid else passive       # 买必须<=bid
    else:  # limit: 允许穿越成交, 但离市场太远(越界)就贴被动价
        px = passive if far else up
    return _round_tick(px, tick)


async def _sz_from(rest, spec, amount, unit, ref_px):
    """返回 (sz:str|None, extra:str, err:str|None)。unit='usdt' 用规格+价换算张数。"""
    from ..risk.sizing import notional_to_contracts
    if unit != "usdt":
        return str(amount), "", None
    if ref_px:
        price = float(ref_px)
    else:
        t = await rest.get_ticker(spec["instId"])   # 单标的行情, 不再拉全市场
        price = float(t.get("last", 0) or 0)
    if price <= 0:
        return None, "", "无法获取现价, 请改用『张』或填限价。"
    ctval = float(spec.get("ctVal", 1) or 1)
    contracts = notional_to_contracts(float(amount), price, ctval,
                                      spec.get("lotSz", "1"), spec.get("minSz", "1"))
    if contracts <= 0:
        return None, "", f"{amount}U 不足 1 张 (约 {price*ctval:.2f}U/张), 请加大金额。"
    return str(contracts), f" (≈{amount}U)", None


async def _manual_place(inst_id, side, ord_type, amount, unit, px, reduce_only) -> str:
    from ..exchange.okx_rest import OkxError
    rest, mode, mgn = _new_trade_rest()
    try:
        spec = await _get_spec(rest, inst_id)
        if not spec:
            return f"标的 {inst_id} 在OKX不存在或未上线, 无法操作。请检查标的名(如 BTC-USDT-SWAP)。"
        tick = spec.get("tickSz")
        ticker = await rest.get_ticker(inst_id) if ord_type in ("limit", "post_only") else {}
        limit_px = _resolve_maker_px(side, ord_type, px, ticker, tick)
        adj = ""
        if px and limit_px and abs(float(limit_px) - float(_round_tick(px, tick) or 0)) > 0:
            adj = f" (价已按当前市场校正→{limit_px}, 原{px}可能已过期/越界)"
        sz, extra, err = await _sz_from(rest, spec, amount, unit, limit_px)
        if err:
            return err
        pm = await _pos_mode(rest, mode)
        ps = _posside(pm, side, opening=not reduce_only)
        ronly = reduce_only and pm == "net_mode"     # 双向模式由 posSide 决定平/开, 不用 reduceOnly
        res = await rest.place_order(
            inst_id=inst_id, td_mode=mgn, side=side, ord_type=ord_type, sz=sz,
            px=limit_px, pos_side=ps, reduce_only=ronly)
        return f"[{mode}] 下单成功 ✓ ordId={res.get('ordId')} {inst_id} {side} {sz}张{extra} {ord_type} ({ps}){adj}"
    except OkxError as e:
        return f"[{mode}] 下单失败 ✗ {e}"
    finally:
        await rest.aclose()


def manual_bracket_sync(inst_id, side, ord_type, amount, unit, px, tp_px, sl_px) -> str:
    return _run(_manual_bracket(inst_id, side, ord_type, amount, unit, px, tp_px, sl_px))


async def _manual_bracket(inst_id, side, ord_type, amount, unit, px, tp_px, sl_px) -> str:
    """一键套单: 入场单 + 附带止盈(限价)/止损(市价) OCO。价格自动对齐 tickSz; 校验止盈/止损方向。"""
    from ..exchange.okx_rest import OkxError
    rest, mode, mgn = _new_trade_rest()
    try:
        spec = await _get_spec(rest, inst_id)
        if not spec:
            return f"标的 {inst_id} 在OKX不存在或未上线, 无法操作。请检查标的名(如 BTC-USDT-SWAP)。"
        tick = spec.get("tickSz")
        ticker = await rest.get_ticker(inst_id)
        last = float(ticker.get("last", 0) or 0)
        limit_px = _resolve_maker_px(side, ord_type, px, ticker, tick)
        sz, extra, err = await _sz_from(rest, spec, amount, unit, limit_px)
        if err:
            return err
        # 入场参考价(实际): 限价/挂单用校正后价, 市价用现价
        ref = float(limit_px) if limit_px else (last or None)
        # 止盈/止损 按"相对原入场价的比例"重算到当前入场价 (修复导入价过期后 TP/SL 失效)
        orig = float(px) if px else None
        adj = ""
        if orig and orig > 0 and ref and abs(ref - orig) / orig > 0.003:
            tpx = _round_tick(ref * (float(tp_px) / orig), tick) if tp_px else None
            spx = _round_tick(ref * (float(sl_px) / orig), tick) if sl_px else None
            adj = f" (市场已变, 入场/止盈/止损按当前价{ref}等比重算)"
        else:
            tpx = _round_tick(tp_px, tick)
            spx = _round_tick(sl_px, tick)
        # 方向校验: 买入(做多) -> 止盈>入场>止损; 卖出(做空) -> 止盈<入场<止损。
        if ref:
            if side == "buy":
                if tpx and float(tpx) <= ref:
                    return f"做多套单: 止盈价({tpx})应高于入场价({ref})。"
                if spx and float(spx) >= ref:
                    return f"做多套单: 止损价({spx})应低于入场价({ref})。"
            else:
                if tpx and float(tpx) >= ref:
                    return f"做空套单: 止盈价({tpx})应低于入场价({ref})。"
                if spx and float(spx) <= ref:
                    return f"做空套单: 止损价({spx})应高于入场价({ref})。"
        algo: dict = {}
        if tpx:
            algo["tpTriggerPx"] = tpx
            algo["tpOrdPx"] = tpx               # 止盈用限价
            algo["tpTriggerPxType"] = "last"
        if spx:
            algo["slTriggerPx"] = spx
            algo["slOrdPx"] = "-1"              # 止损用市价(-1), 保证触发即出
            algo["slTriggerPxType"] = "last"
        if not algo:
            return "套单需至少填止盈价或止损价。"
        pm = await _pos_mode(rest, mode)
        ps = _posside(pm, side, opening=True)
        res = await rest.place_order(
            inst_id=inst_id, td_mode=mgn, side=side, ord_type=ord_type, sz=sz,
            px=limit_px, pos_side=ps, reduce_only=False, attach_algo=[algo])
        tp_s = f"止盈{tpx}" if tpx else "无止盈"
        sl_s = f"止损{spx}" if spx else "无止损"
        return (f"[{mode}] 套单已提交 ✓ ordId={res.get('ordId')} {inst_id} {side} {sz}张{extra} "
                f"{ord_type} @ {limit_px} + [{tp_s} / {sl_s}]{adj}\n"
                "说明: 入场成交后自动挂止盈+止损(OCO), 先触发哪个另一个自动撤销; "
                "入场若未成交则不会挂。只挂单(maker)挂在当前盘口被动侧, 可能不立即成交。")
    except OkxError as e:
        return (f"[{mode}] 套单失败 ✗ {e}\n"
                "(常见: 价格精度/止盈止损方向/触发价已越过现价。已自动对齐tickSz并校验方向; "
                "若仍失败可改用『只挂单』在现价被动侧挂, 或先建仓后单独设止盈止损)")
    finally:
        await rest.aclose()


def account_brief_sync() -> dict:
    """控制台卡片用: 真实账户权益 + 持仓数 + 浮动盈亏 (独立于引擎, 保证如实显示)。"""
    try:
        return asyncio.run(_account_brief())
    except Exception as e:
        return {"ok": False, "error": repr(e)}


async def _account_brief() -> dict:
    rest, mode, _ = _new_trade_rest()

    async def _safe(coro, d):
        try:
            return await coro
        except Exception:
            return d
    try:
        bal, poss = await asyncio.gather(
            _safe(rest.get_balance(), {}), _safe(rest.get_positions("SWAP"), []))
        live = [p for p in (poss or []) if float(p.get("pos", 0) or 0) != 0]
        eq = float((bal or {}).get("totalEq", 0) or 0)
        upl = sum(float(p.get("upl", 0) or 0) for p in live)
        return {"ok": True, "mode": mode, "equity": eq, "positions": len(live), "upl": upl}
    finally:
        await rest.aclose()


def pnl_stats_sync() -> dict:
    """汇总: 当前浮动盈亏(uPnL) + 今日/近7天/近30天/近90天 已实现盈亏 (USDT)。"""
    try:
        return asyncio.run(_pnl_stats())
    except Exception as e:
        return {"ok": False, "error": repr(e)}


async def _pnl_stats() -> dict:
    import time as _t
    rest, mode, _ = _new_trade_rest()

    async def _safe(coro, d):
        try:
            return await coro
        except Exception:
            return d
    try:
        poss, hist = await asyncio.gather(
            _safe(rest.get_positions("SWAP"), []),
            _safe(rest.get_positions_history("SWAP", 100), []))
        upl = sum(float(p.get("upl", 0) or 0) for p in (poss or [])
                  if float(p.get("pos", 0) or 0) != 0)
        now = _t.time() * 1000.0
        buckets = {"今日": 86400e3, "近7天": 7 * 86400e3, "近30天": 30 * 86400e3,
                   "近90天": 90 * 86400e3}
        # 今日按本地零点更准:
        local_midnight = (now // 86400e3) * 86400e3
        realized = {k: 0.0 for k in buckets}
        cnt = {k: 0 for k in buckets}
        total_all = 0.0
        for h in (hist or []):
            pnl = float(h.get("realizedPnl", h.get("pnl", 0)) or 0)
            ut = float(h.get("uTime", h.get("cTime", 0)) or 0)
            total_all += pnl
            for k, win in buckets.items():
                start = local_midnight if k == "今日" else (now - win)
                if ut >= start:
                    realized[k] += pnl
                    cnt[k] += 1
        return {"ok": True, "mode": mode, "upl": upl, "open_n": len([p for p in (poss or [])
                if float(p.get("pos", 0) or 0) != 0]),
                "realized": realized, "count": cnt, "total_all": total_all,
                "hist_n": len(hist or [])}
    finally:
        await rest.aclose()


def manual_panel_sync(inst_id) -> dict:
    """一次取回 手动页所需全部实时数据: 实时行情 + 持仓 + 普通挂单 + 策略委托(止盈止损) + 历史成交。"""
    try:
        return asyncio.run(_manual_panel(inst_id))
    except Exception as e:
        return {"ok": False, "error": repr(e), "mode": "?",
                "positions": [], "orders": [], "algos": [], "ticker": {}, "fills": []}


async def _manual_panel(inst_id) -> dict:
    """并发拉取(asyncio.gather)所有面板数据, 把延迟从"逐个相加"降到"最慢一个"。"""
    rest, mode, _ = _new_trade_rest()

    async def _safe(coro, default):
        try:
            return await coro
        except Exception:
            return default
    try:
        tasks = [_safe(rest.get_positions("SWAP"), []),
                 _safe(rest.get_pending_orders("SWAP"), []),
                 _safe(rest.get_algo_pending("SWAP", "oco"), []),
                 _safe(rest.get_algo_pending("SWAP", "conditional"), [])]
        if inst_id:
            tasks.append(_safe(rest.get_ticker(inst_id), {}))
            tasks.append(_safe(rest.get_fills("SWAP", inst_id), []))
        res = await asyncio.gather(*tasks)
        poss = [p for p in (res[0] or []) if float(p.get("pos", 0) or 0) != 0]
        orders = res[1] or []
        algos = (res[2] or []) + (res[3] or [])
        ticker = (res[4] if inst_id else {}) or {}
        fills = ((res[5] if inst_id else []) or [])[:12]
        return {"ok": True, "mode": mode, "positions": poss, "orders": orders,
                "algos": algos, "ticker": ticker, "fills": fills}
    finally:
        await rest.aclose()


def manual_algo_sync(inst_id, close_side, kind, trigger_px, amount, unit) -> str:
    return _run(_manual_algo(inst_id, close_side, kind, trigger_px, amount, unit))


async def _manual_algo(inst_id, close_side, kind, trigger_px, amount, unit) -> str:
    """单独挂止盈(kind=tp)或止损(kind=sl): reduce-only 条件单, close_side=平仓方向。"""
    from ..exchange.okx_rest import OkxError
    rest, mode, mgn = _new_trade_rest()
    try:
        spec = await _get_spec(rest, inst_id)
        if not spec:
            return f"标的 {inst_id} 在OKX不存在或未上线, 无法操作。请检查标的名(如 BTC-USDT-SWAP)。"
        tpx = _round_tick(trigger_px, spec.get("tickSz"))
        sz, extra, err = await _sz_from(rest, spec, amount, unit, None)
        if err:
            return err
        pm = await _pos_mode(rest, mode)
        ps = _posside(pm, close_side, opening=False)
        kw = {"inst_id": inst_id, "td_mode": mgn, "side": close_side, "sz": sz,
              "pos_side": ps, "reduce_only": pm == "net_mode"}
        if kind == "tp":
            kw["tp_trigger_px"] = tpx
            kw["tp_ord_px"] = tpx
        else:
            kw["sl_trigger_px"] = tpx
            kw["sl_ord_px"] = "-1"
        res = await rest.place_algo_order(**kw)
        name = "止盈" if kind == "tp" else "止损"
        return (f"[{mode}] 已挂{name} ✓ algoId={res.get('algoId')} {inst_id} 平仓方向{close_side} "
                f"{sz}张{extra} 触发价{tpx}\n(reduce-only 条件单: 触发后只减仓; 需先有对应持仓)")
    except OkxError as e:
        return (f"[{mode}] 挂{('止盈' if kind == 'tp' else '止损')}失败 ✗ {e}\n"
                "(需先有对应持仓; 或触发价方向不对: 平多止盈价应高于现价、止损价应低于现价)")
    finally:
        await rest.aclose()


def cancel_algo_sync(inst_id, algo_id) -> str:
    return _run(_cancel_algo(inst_id, algo_id))


async def _cancel_algo(inst_id, algo_id) -> str:
    from ..exchange.okx_rest import OkxError
    rest, mode, _ = _new_trade_rest()
    try:
        await rest.cancel_algos([{"algoId": algo_id, "instId": inst_id}])
        return f"[{mode}] 已撤策略委托(止盈止损) …{str(algo_id)[-6:]} ✓"
    except OkxError as e:
        return f"[{mode}] 撤策略委托失败 ✗ {e}"
    finally:
        await rest.aclose()


def cancel_one_sync(inst_id, ord_id) -> str:
    return _run(_cancel_one(inst_id, ord_id))


async def _cancel_one(inst_id, ord_id) -> str:
    from ..exchange.okx_rest import OkxError
    rest, mode, _ = _new_trade_rest()
    try:
        await rest.cancel_order(inst_id, ord_id=ord_id)
        return f"[{mode}] 已撤单 {inst_id} …{str(ord_id)[-6:]} ✓"
    except OkxError as e:
        return f"[{mode}] 撤单失败 ✗ {e}"
    finally:
        await rest.aclose()


def amend_one_sync(inst_id, ord_id, new_px=None, new_sz=None) -> str:
    return _run(_amend_one(inst_id, ord_id, new_px, new_sz))


async def _amend_one(inst_id, ord_id, new_px, new_sz) -> str:
    from ..exchange.okx_rest import OkxError
    rest, mode, _ = _new_trade_rest()
    try:
        await rest.amend_order(inst_id, ord_id=ord_id,
                               new_px=str(new_px) if new_px else None,
                               new_sz=str(new_sz) if new_sz else None)
        chg = []
        if new_px:
            chg.append(f"价→{new_px}")
        if new_sz:
            chg.append(f"量→{new_sz}")
        return f"[{mode}] 已改单 {inst_id} …{str(ord_id)[-6:]} {' '.join(chg)} ✓"
    except OkxError as e:
        return f"[{mode}] 改单失败 ✗ {e}"
    finally:
        await rest.aclose()


async def _manual_close_all() -> str:
    from ..exchange.okx_rest import OkxError
    rest, mode, mgn = _new_trade_rest()
    try:
        poss = [p for p in await rest.get_positions() if float(p.get("pos", 0) or 0) != 0]
        if not poss:
            return f"[{mode}] 当前无持仓。"
        n = 0
        for p in poss:
            try:
                await rest.close_position(p["instId"], mgn_mode=mgn, pos_side=p.get("posSide"))
                n += 1
            except OkxError:
                pass
        return f"[{mode}] 已平仓 {n}/{len(poss)} 个持仓 ✓"
    finally:
        await rest.aclose()


def _sl_pct_from(atr, spread_bps) -> float:
    atr = float(atr) if atr else 0.003
    spread = (float(spread_bps) if spread_bps is not None else 5.0) / 1e4
    return min(max(1.2 * atr, 3.0 * spread, 0.0010), 0.012)


def _suggest_leverage(sl_pct: float, lev_max: float, target_risk: float = 0.10) -> int:
    """客观最优杠杆: 逐仓下若止损, 亏损≈保证金的 target_risk; lev = target_risk/sl_pct。
    钳到 [3, min(50, 交易所上限)]。sl_pct 越大(越波动)杠杆越低 —— 这就是统计化的杠杆。"""
    if sl_pct <= 0:
        return 3
    lev = round(target_risk / sl_pct)
    return int(max(3, min(50, lev_max, lev)))


async def _quant_context(inst_id: str, row: dict) -> dict:
    """组装客观量化上下文 + 代码算好的默认(方向/SL/TP/价位/杠杆/仓位), 供 AI 定锚校准。"""
    from ..exchange.okx_rest import OkxError, OkxRestClient
    cfg = Config.load()
    s = Secrets()
    rest = OkxRestClient(s, cfg)
    lev_max = 50.0
    tick = None
    mid = row.get("mid")
    valid = False
    try:
        spec = await _get_spec(rest, inst_id)
        if spec:
            valid = True
            lev_max = min(50.0, float(spec.get("lever", 50) or 50))
            tick = spec.get("tickSz")
        if mid is None and valid:
            t = await rest.get_ticker(inst_id)
            mid = float(t.get("last", 0) or 0) or None
    except OkxError:
        pass
    finally:
        await rest.aclose()

    atr = row.get("atr")
    fees = cfg.section("fees")
    rebate = min(max(float(cfg.get("fees.fee_rebate_frac", 0.0) or 0.0), 0.0), 0.9)
    # 往返成本: (maker入+taker出)手续费×(1−返点) + 半价差 + 滑点 (后两者不返点)
    cost = (float(fees.get("crypto_maker_pct", 0.02)) + float(fees.get("crypto_taker_pct", 0.05))) / 100.0 * (1.0 - rebate)
    cost += (float(row.get("spread_bps") or 5.0) / 1e4) * 0.5 + 0.0003
    # 止损必须留出成本以上的空间, 否则一点噪声就被扫出、白交手续费 (cost-aware floor)
    sl_cost_mult = float(cfg.get("signal.sl_min_cost_mult", 2.5))
    sl_pct = max(_sl_pct_from(atr, row.get("spread_bps")), sl_cost_mult * cost)
    tp_rr = float(cfg.get("signal.tp_rr", 1.6))
    tp_pct = tp_rr * sl_pct
    target_risk = float(cfg.get("leverage.target_margin_risk_per_stop", 0.08))
    lv = _suggest_leverage(sl_pct, lev_max, target_risk)
    risk_usdt = float(cfg.get("risk.risk_per_trade_usdt_default", 2.0))
    size_usdt = round(risk_usdt / (sl_pct + cost) / 10.0) * 10.0 if (sl_pct + cost) > 0 else 0.0

    long_s, short_s = row.get("long") or 0.0, row.get("short") or 0.0
    d = 1 if long_s >= short_s else -1
    dir_prior = "long" if d > 0 else "short"
    entry_px = mid
    entry_off = 0.0
    tp_px = sl_px = None
    if mid:
        digits = 6 if mid < 1 else (4 if mid < 100 else 2)
        # maker(post_only)入场价必须落在【正确一侧】: 做多挂买单须≤中间价, 做空挂卖单须≥中间价。
        # 否则越过盘口被立即吃成 taker —— 这也是"做空却给出低于现价的入场价、看着不可能成交"的根因。
        # 偏移取 max(半价差, ~1tick, 2bps), 让单子贴着盘口排队, 等回踩(多)/反抽(空)成交。
        half_spread = (float(row.get("spread_bps") or 4.0) / 1e4) * 0.5
        tick_frac = (float(tick) / mid) if tick else 0.0
        entry_off = max(half_spread, tick_frac, 0.0002)
        entry_raw = mid * (1 - d * entry_off)            # 多: 低于mid; 空: 高于mid
        tp_raw = entry_raw * (1 + d * tp_pct)            # 止盈止损以【入场价】为基准, 保持盈亏比一致
        sl_raw = entry_raw * (1 - d * sl_pct)
        # 对齐 tickSz (拿到规格时), 避免导入后下单报价格精度错
        if tick:
            entry_px = float(_round_tick(entry_raw, tick) or round(entry_raw, digits))
            tp_px = float(_round_tick(tp_raw, tick) or round(tp_raw, digits))
            sl_px = float(_round_tick(sl_raw, tick) or round(sl_raw, digits))
        else:
            entry_px = round(entry_raw, digits)
            tp_px = round(tp_raw, digits)
            sl_px = round(sl_raw, digits)
    return {
        "inst": inst_id, "mid": mid, "spread_bps": row.get("spread_bps"),
        "obi_z": row.get("obi_z"), "ofi_z": row.get("ofi_z"),
        "trend_dir": row.get("trend_dir"), "flow_dir": row.get("flow_dir"),
        "long": long_s, "short": short_s, "atr": atr, "rvol": row.get("rvol"),
        "dir_prior": dir_prior, "sl_pct": round(sl_pct, 5), "tp_pct": round(tp_pct, 5),
        "entry_px": entry_px, "tp_px": tp_px, "sl_px": sl_px, "entry_off": round(entry_off, 5),
        "lev_suggest": lv, "lev_max": int(lev_max), "size_usdt": size_usdt,
        "target_risk": target_risk, "cost_pct": round(cost, 5), "valid": valid,
    }


async def _ai_analyze(inst_id, row) -> dict:
    from ..events.llm_classifier import LLMClassifier
    qctx = await _quant_context(inst_id, row)
    if not qctx.get("valid"):
        return {"ok": False, "struct": None,
                "text": f"⚠ 标的 {inst_id} 在OKX不存在或未上线, 无法分析/下单。\n"
                        "请检查标的名(加密如 BTC-USDT-SWAP, 美股如 AAPL-USDT-SWAP); "
                        "可在『控制台』列表里点正确的标的带入。"}
    res = await LLMClassifier.from_secrets(Secrets()).analyze_structured(inst_id, qctx)
    res["qctx"] = qctx
    return res


def stock_symbol_set() -> set:
    """配置里的股票永续 ticker 集合 (供 GUI 分类 加密/美股)。"""
    cfg = Config.load()
    return {str(s).upper() for s in cfg.get("universe.stock_symbols", [])}


async def _okx_24h() -> dict:
    """一次取回全市场 24h 行情 (真实权威交易所数据): inst -> {chg%, volU}。无需鉴权。
    注意: /market/tickers 是全市场(~500条)大响应, 区域受限网络上比单标的更易 ConnectError;
    这里【宽松兜底】任何网络/解析异常都返回已拿到的部分, 绝不把整个 AI 选品搞挂(原 bug)。"""
    from ..exchange.okx_rest import OkxRestClient
    rest = OkxRestClient(Secrets(), Config.load())
    out: dict[str, dict] = {}
    try:
        for t in await rest.get_tickers("SWAP"):
            last = float(t.get("last", 0) or 0)
            op = float(t.get("open24h", 0) or 0)
            chg = (last / op - 1.0) * 100.0 if op > 0 else 0.0
            out[t["instId"]] = {"chg": round(chg, 2),
                                "volU": float(t.get("volCcy24h", 0) or 0)}
    except Exception as e:        # OkxError + httpx.ConnectError/ReadTimeout 等一律兜底
        print(f"[pick] 全市场24h行情获取失败({e!r}); 改用引擎实时信号/降级继续")
    finally:
        await rest.aclose()
    return out


async def _ai_pick(pool, rows) -> dict:
    from ..events.llm_classifier import LLMClassifier
    from ..risk.engine import is_stock_perp, set_stock_symbols
    cfg = Config.load()
    set_stock_symbols(cfg.get("universe.stock_symbols", []))
    mkt = await _okx_24h()                  # 真实 24h 涨跌幅/成交额 (权威交易所数据)
    min_vol = float(cfg.get("universe.min_crypto_quote_vol_usd", 8_000_000))
    cands = []
    if rows:                                # 引擎在跑: 用实时信号分 + 24h 数据
        for inst, d in rows.items():
            if pool == "crypto" and is_stock_perp(inst):
                continue
            if pool == "stock" and not is_stock_perp(inst):
                continue
            long_s, short_s = d.get("long") or 0.0, d.get("short") or 0.0
            mom = abs(d.get("trend_dir") or 0.0)
            m = mkt.get(inst, {})
            chg = m.get("chg", 0.0)
            opp = max(long_s, short_s) + 40.0 * mom + min(abs(chg), 15.0)
            cands.append({"inst": inst, "long": round(long_s, 1), "short": round(short_s, 1),
                          "spread_bps": d.get("spread_bps"), "atr": d.get("atr"),
                          "rvol": d.get("rvol"), "mom": round(mom, 3),
                          "chg": chg, "vol": m.get("volU", 0.0), "event": "无", "_opp": opp})
    else:                                   # 引擎没跑: 用全市场 24h 真实数据建候选(无实时信号分)
        for inst, m in mkt.items():
            if not inst.endswith("-USDT-SWAP"):
                continue
            stk = is_stock_perp(inst)
            if pool == "crypto" and stk:
                continue
            if pool == "stock" and not stk:
                continue
            vol = m.get("volU", 0.0)
            if not stk and vol < min_vol:    # 过滤低流动山寨
                continue
            chg = m.get("chg", 0.0)
            cands.append({"inst": inst, "long": 0.0, "short": 0.0, "spread_bps": None,
                          "atr": None, "rvol": None, "mom": 0.0, "chg": chg, "vol": vol,
                          "event": "无", "_opp": min(abs(chg), 20.0) + vol / 5e9})
    cands.sort(key=lambda c: c["_opp"], reverse=True)
    pool_cn = {"crypto": "加密永续", "stock": "美股永续", "all": "全部永续"}.get(pool, pool)
    if not cands:
        if not mkt and not rows:
            return {"text": f"[{pool_cn}] 无法获取候选: 全市场24h行情拉取失败(网络/区域限制), "
                            "且引擎未在跑。请①点『控制台』▶启动引擎用实时信号选品, 或②检查网络后重试。",
                    "picks": []}
        return {"text": f"[{pool_cn}] 暂无实时候选。请先在『控制台』▶启动引擎, 待行情填充后再选品。",
                "picks": []}
    brief = await _external_brief(pool, cands)        # Finnhub 真实新闻/财报 (有key才有)
    clf = LLMClassifier.from_secrets(Secrets())
    res = await clf.pick_products(pool_cn, cands, external_brief=brief)
    # 明确告知 AI 是否真的参与 (选品的"AI"是 DeepSeek; Finnhub/EDGAR 只是给它喂数据)
    if clf.enabled:
        ai_line = f"AI研判: 已启用 {clf.label}"
    else:
        ai_line = ("AI研判: 未启用 (当前=规则/未填key) → 下面只有客观量化排序。"
                   "选品的AI是 DeepSeek: 请到『账户与密钥』把提供商选 DeepSeek、填 AI API Key、"
                   "点『验证AI』通过后再选品。(Finnhub/EDGAR 只负责给AI喂新闻/事件, 不能代替 DeepSeek)")
    res["text"] = ai_line + "\n\n" + res.get("text", "")
    head = "\n".join(f"  {c['inst']} 多{c['long']}/空{c['short']} 24h{c['chg']:+.1f}% 动量{c['mom']}"
                     for c in cands[:8])
    res["text"] = (res.get("text", "") + "\n\n— 量化机会分前 8 (含真实24h涨跌, 客观排序) —\n" + head)
    if brief:
        res["text"] += "\n\n— 外部资讯(Finnhub, 已喂给AI) —\n" + brief
    else:
        res["text"] += ("\n\n(未启用 Finnhub 外部资讯: 『账户与密钥』填 FINNHUB_API_KEY 后, "
                        "选品会结合真实财经新闻/财报日历)")
    res["cands"] = cands[:30]
    return res


async def _external_brief(pool: str, cands: list) -> str:
    """用 Finnhub(若配置)拉真实新闻/财报, 作为 AI 选品的权威外部信息。无 key 返回空。"""
    from ..events.finnhub import FinnhubClient
    from ..risk.engine import is_stock_perp
    import datetime as _dt
    fh = FinnhubClient(Secrets().finnhub_api_key)
    if not fh.enabled:
        return ""
    lines = []
    try:
        if pool in ("crypto", "all"):
            news = await fh.general_news("crypto") or await fh.general_news("general")
            for n in (news or [])[:5]:
                h = str(n.get("headline", "")).strip()
                if h:
                    lines.append("新闻· " + h[:90])
        if pool in ("stock", "all"):
            today = _dt.datetime.now(_dt.timezone.utc).date()
            cal = await fh.earnings_calendar(str(today), str(today + _dt.timedelta(days=7)))
            syms = {c["inst"].split("-")[0].upper() for c in cands if is_stock_perp(c["inst"])}
            for e in (cal or []):
                if str(e.get("symbol", "")).upper() in syms:
                    lines.append(f"财报· {e.get('symbol')} {e.get('date')}({e.get('hour', '')})")
            lines = lines[:11]
    except Exception:
        pass
    finally:
        await fh.aclose()
    return "\n".join(lines)


async def _manual_cancel_all(inst_id) -> str:
    from ..exchange.okx_rest import OkxError
    rest, mode, _ = _new_trade_rest()
    try:
        pend = await rest.get_pending_orders("SWAP")
        mine = [{"instId": inst_id, "ordId": o["ordId"]}
                for o in pend if o.get("instId") == inst_id]
        if not mine:
            return f"[{mode}] {inst_id} 无挂单。"
        await rest.cancel_batch(mine)
        return f"[{mode}] 已撤 {inst_id} 的 {len(mine)} 个挂单 ✓"
    except OkxError as e:
        return f"[{mode}] 撤单失败 ✗ {e}"
    finally:
        await rest.aclose()


async def _manual_close(inst_id) -> str:
    from ..exchange.okx_rest import OkxError
    rest, mode, mgn = _new_trade_rest()
    try:
        pm = await _pos_mode(rest, mode)
        if pm == "long_short_mode":
            poss = [p for p in await rest.get_positions("SWAP")
                    if p.get("instId") == inst_id and float(p.get("pos", 0) or 0) != 0]
            if not poss:
                return f"[{mode}] {inst_id} 无持仓。"
            n = 0
            for p in poss:
                try:
                    await rest.close_position(inst_id, mgn_mode=mgn, pos_side=p.get("posSide"))
                    n += 1
                except OkxError:
                    pass
            return f"[{mode}] 已市价平仓 {inst_id} ({n}个方向) ✓"
        await rest.close_position(inst_id, mgn_mode=mgn)
        return f"[{mode}] 已市价平仓 {inst_id} ✓"
    except OkxError as e:
        return f"[{mode}] 平仓失败 ✗ {e} (可能本就无持仓)"
    finally:
        await rest.aclose()


async def _manual_set_leverage(inst_id, lever) -> str:
    from ..exchange.okx_rest import OkxError
    rest, mode, mgn = _new_trade_rest()
    try:
        if not await _get_spec(rest, inst_id):
            return f"[{mode}] 标的 {inst_id} 在OKX不存在或未上线, 无法设杠杆。请检查标的名(如 BTC-USDT-SWAP)。"
        pm = await _pos_mode(rest, mode)
        if pm == "long_short_mode" and mgn == "isolated":
            # 双向逐仓: 多空分别设杠杆 (否则报 posSide error)
            await rest.set_leverage(inst_id, str(lever), mgn_mode=mgn, pos_side="long")
            await rest.set_leverage(inst_id, str(lever), mgn_mode=mgn, pos_side="short")
        else:
            await rest.set_leverage(inst_id, str(lever), mgn_mode=mgn)
        return f"[{mode}] {inst_id} 杠杆已设为 {lever}x ({mgn}, {pm}) ✓"
    except OkxError as e:
        return f"[{mode}] 设杠杆失败 ✗ {e}"
    finally:
        await rest.aclose()


# ----------------- 策略校准 (在录制数据上回测找最优决策逻辑) -----------------

def _rec_dir() -> str:
    cfg = Config.load()
    return str(paths.APP_DIR / cfg.get("paths.recordings_dir", "recordings"))


def open_recordings_dir() -> str:
    d = _rec_dir()
    os.makedirs(d, exist_ok=True)
    try:
        os.startfile(d)            # Windows 资源管理器打开 (导入=往里拷文件即可)
        return f"已打开录制目录: {d}"
    except Exception as e:
        return f"录制目录: {d}\n(无法自动打开: {e!r})"


def recordings_backup_sync() -> str:
    import datetime as _dt
    import glob
    import zipfile
    d = _rec_dir()
    files = glob.glob(os.path.join(d, "calib_*.jsonl"))
    if not files:
        return "无录制文件可备份。"
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    zp = os.path.join(d, f"backup_calib_{ts}.zip")
    try:
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for f in files:
                z.write(f, os.path.basename(f))
        return f"已备份 {len(files)} 个录制文件 -> {os.path.basename(zp)} ({os.path.getsize(zp)/1e6:.1f}MB)"
    except Exception as e:
        return f"备份失败: {e!r}"


def recordings_clear_sync(keep_latest: bool = True) -> str:
    import glob
    d = _rec_dir()
    files = sorted(glob.glob(os.path.join(d, "calib_*.jsonl")))
    if not files:
        return "无录制文件。"
    targets = files[:-1] if keep_latest else files
    if not targets:
        return "只有1个录制文件(最新), 未删除。"
    n = 0
    for f in targets:
        try:
            os.remove(f)
            n += 1
        except Exception:
            pass
    return f"已删除 {n} 个旧录制文件" + ("(保留最新1个)。" if keep_latest else "(全部)。")


def calib_files_info() -> tuple[int, float]:
    """返回 (录制文件数, 总大小MB)。"""
    import glob
    import os
    files = glob.glob(os.path.join(_rec_dir(), "calib_*.jsonl"))
    total = sum((os.path.getsize(f) for f in files), 0)
    return len(files), total / 1e6


def run_calibration_sync(use_all: bool) -> dict:
    """回测网格搜索, 返回 {ok, report, profit_cfg?, stable_cfg?}。GUI 在 worker 线程调用。"""
    try:
        from ..research import calibrator as cal
        cfg = Config.load()
        files = cal.find_recordings(_rec_dir())
        if not files:
            return {"ok": False, "report": "未找到录制文件 (recordings/calib_*.jsonl)。\n"
                    "请先在『控制台』启动跑一段虚拟盘 (演练即可) 以积累数据, 再来校准。"}
        if not use_all:
            files = files[-1:]
        by_inst, meta = cal.load_calib(files)
        if not by_inst:
            return {"ok": False, "report": "录制数据太少, 无法校准。请让虚拟盘多跑一会儿。"}
        cc = (cfg.section("research") or {}).get("calibrate", {})
        min_trades = int(cc.get("min_trades", 25))
        cost_mult = float(cc.get("cost_haircut_mult", 1.0))
        maker = bool(cc.get("maker_fill", True))
        cooldown = float(cfg.get("signal.cooldown_seconds", 20))
        # 与实盘门槛对齐, 让回测测的就是实盘会做的那批
        sl_cost_mult = float(cfg.get("signal.sl_min_cost_mult", 2.5))
        min_edge = float(cfg.get("signal.min_edge_to_cost_ratio", 1.2))
        sm = {  # 去噪参数: 必须与实盘一致, 否则校准跑偏
            "hl_f": float(cfg.get("signal.ema_flow_half_life_s", 1.0)),
            "hl_t": float(cfg.get("signal.ema_trend_half_life_s", 2.0)),
            "hl_s": float(cfg.get("signal.ema_score_half_life_s", 0.5)),
            "enter": float(cfg.get("signal.dir_hyst_enter", 0.12)),
            "exit": float(cfg.get("signal.dir_hyst_exit", 0.04)),
            "miss_grace": int(cfg.get("signal.persist_miss_grace", 1)),
            "rv_lo": float(cfg.get("signal.regime_rv_lo", 2e-4)),
            "rv_hi": float(cfg.get("signal.regime_rv_hi", 1.2e-3)),
            "hv_scale": float(cfg.get("signal.regime_hv_alpha_scale", 0.6)),
            "persist_bonus": int(cfg.get("signal.regime_persist_bonus", 1)),
            "dt": 0.5,
            # 方向权重 + 可交易性门槛: 必须与实盘 composite 一致 (缺则 gen_entries 报错/跑偏)
            "w_flow": float(cfg.get("signal.weights.microstructure", 20))
            + float(cfg.get("signal.weights.order_flow", 15)),
            "w_trend": float(cfg.get("signal.weights.trend", 15)),
            "min_trad": float(cfg.get("signal.min_tradability", 0.5)),
            "edge_k": float(cfg.get("signal.edge_move_k", 1.5)),
            "edge_horizon_s": float(cfg.get("signal.edge_horizon_s", 30.0)),
        }
        calib = cal.run_calibration(by_inst, cal.Grid(), min_trades, cooldown_s=cooldown,
                                    cost_mult=cost_mult, maker_fill=maker,
                                    sl_cost_mult=sl_cost_mult, min_edge_cost=min_edge, sm=sm)
        picks = calib["picks"]
        out = {"ok": True, "report": cal.format_report_oos(meta, calib, min_trades)}
        if picks["best_profit"]:
            out["profit_cfg"] = cal.result_to_signal_cfg(picks["best_profit"])
        if picks["best_stable"]:
            out["stable_cfg"] = cal.result_to_signal_cfg(picks["best_stable"])
        return out
    except Exception as e:
        return {"ok": False, "report": f"校准失败: {e!r}"}


# ----------------- 参数管理 (查看/手动调整/命名预设) -----------------

# (点路径, 中文说明, 类型)
PARAM_DEFS = [
    ("signal.min_composite_score", "入场综合分门槛(↓更易出手)", "int"),
    ("signal.strong_composite_score", "强信号分(放大仓位/允许taker)", "int"),
    ("signal.min_tradability", "可交易性门槛 0~1(独立于方向分)", "float"),
    ("signal.confirm_min", "确认下限 流向+趋势一致 0~1", "float"),
    ("signal.persist_ticks", "持续拍数(↓更易出手)", "int"),
    ("signal.cooldown_seconds", "同标的冷却(秒)", "int"),
    ("signal.min_edge_to_cost_ratio", "扣费净edge门槛(↓更易出手)", "float"),
    ("signal.tp_rr", "止盈盈亏比 tp=rr×止损", "float"),
    ("signal.sl_min_cost_mult", "止损≥往返成本×(防秒扫)", "float"),
    ("signal.min_hold_seconds", "最短持仓(秒)", "int"),
    ("signal.reversal_hyst_gap", "反向平仓迟滞", "int"),
    ("signal.decay_threshold", "衰减平仓阈值", "int"),
    ("execution.max_hold_seconds", "持仓硬上限(秒)", "int"),
    ("leverage.target_margin_risk_per_stop", "最佳杠杆: 止损亏保证金比例", "float"),
    ("account.max_concurrent_positions", "同时最多持仓数(总)", "int"),
    ("account.max_concurrent_crypto_perp", "同时最多·加密永续仓", "int"),
    ("account.max_concurrent_stock_perp", "同时最多·股票永续仓", "int"),
    ("risk.risk_per_trade_usdt_default", "每笔风险预算(USDT)", "float"),
    ("risk.max_total_notional_normal", "总名义敞口上限(USDT)", "float"),
    ("signal.ema_flow_half_life_s", "去噪·流向EMA半衰期(秒)", "float"),
    ("signal.ema_trend_half_life_s", "去噪·趋势EMA半衰期(秒)", "float"),
    ("signal.dir_hyst_enter", "去噪·方向认定阈值", "float"),
    ("signal.regime_rv_lo", "死盘阈值(波动低于不做)", "float"),
    ("signal.regime_rv_hi", "剧烈阈值(波动高于更稳)", "float"),
]

# 鼠标悬停讲解 (小白向): 含义 / 调高调低影响 / 典型值 / 建议范围
TOOLTIPS = {
    "signal.min_composite_score":
        "【入场门槛】对称尺度: 多分=50×(1+方向)、空分=100−多分, 50=中性、多+空=100。达到此分才出手。\n"
        "调低→更易出手(更杂); 调高→更挑剔。典型 66(≈方向0.32), 合理区间 64–72。\n"
        "务必用『信号检验/校准』按你的数据重定。",
    "signal.strong_composite_score":
        "【强信号分】达到算'强信号'(放大仓位/允许taker)。对称尺度典型 74(≈方向0.48), 范围 70–82。",
    "signal.min_tradability":
        "【可交易性门槛】可交易性(0–1)= 流动性×波动状态, 与方向分【独立】。正常行情≈1,\n"
        "仅价差过宽/深度过薄/波动死或过烈时下降。低于此值即使方向分高也不开仓(避免在差行情成交)。\n"
        "典型 0.50, 范围 0.3–0.7。调高→只在好行情做; 调低→差行情也做。",
    "signal.confirm_min":
        "【确认下限】要求'盘口流向'与'价格趋势'方向一致的强度(0–1)。\n"
        "调低→更易触发; 调高→只做方向更一致的。典型 0.30, 范围 0.15–0.45。",
    "signal.persist_ticks":
        "【持续拍数】信号要连续满足几拍(每拍约0.5秒)才下单, 用来去抖、防假信号。\n"
        "调低→反应更快但更易被噪声骗; 调高→更稳但更慢。典型 3(≈1.5秒), 范围 2–5。",
    "signal.cooldown_seconds":
        "【冷却秒】同一标的两次开/平之间至少间隔多少秒, 防来回反复进出。\n"
        "典型 20, 范围 10–60。",
    "signal.min_edge_to_cost_ratio":
        "【净edge门槛】预期收益要是'来回手续费成本'的几倍才下单。\n"
        "调低→更易出手(但利润空间小); 调高→只做性价比高的。典型 1.2, 范围 1.0–2.0。\n"
        "注: 设太高会几乎不出手(配合'止损≥成本'后尤其明显)。",
    "signal.tp_rr":
        "【止盈盈亏比】止盈距离 = 此值 × 止损距离。1.6 表示赚的目标是亏的1.6倍。\n"
        "调高→单笔目标更大但更难达成(胜率↓); 调低→更易止盈(胜率↑但每笔赚少)。范围 1.2–2.5。",
    "signal.sl_min_cost_mult":
        "【止损≥成本×】止损距离至少是'来回成本'的几倍, 防止一点点波动就被扫损、白交手续费。\n"
        "典型 2.5, 范围 2.0–4.0。太小→秒止损白亏手续费; 太大→止损太远、单笔亏多。",
    "signal.min_hold_seconds":
        "【最短持仓秒】刚开仓后这段时间内不因信号反转而平仓(硬止损/超时除外), 防刚进就被晃出。\n"
        "典型 20, 范围 10–60。",
    "signal.reversal_hyst_gap":
        "【反向平仓迟滞】反向分要达到(入场门槛−此值)才平仓(迟滞带, 防来回)。\n"
        "对称尺度: 入场66、此值8 → 反向分到58才平。典型 8, 范围 4–14。",
    "signal.decay_threshold":
        "【衰减平仓阈值】自身综合分(对称尺度,50中性)跌回此值附近=信号衰减、择机平仓。\n"
        "典型 54, 范围 50–60(越接近50越晚平)。",
    "execution.max_hold_seconds":
        "【持仓硬上限秒】无论盈亏, 超过此时长强制平仓(超短线不死扛)。\n"
        "典型 180, 范围 30–600。",
    "leverage.target_margin_risk_per_stop":
        "【最佳杠杆依据】AI/手动算'最佳杠杆'时: 若打到止损, 亏损≈保证金的这个比例。\n"
        "0.08=单次止损约亏8%保证金。越小越保守(波动大自动给更低杠杆)。范围 0.05–0.12。",
    "account.max_concurrent_positions":
        "【同时最多持仓数(总)】自动交易最多同时持有几个标的。\n"
        "调高→更接近'满仓/多点开花'(机会多时铺更多), 但分散风险也分散资金、回撤可能更大。\n"
        "典型 2, 想尽量满仓可设 3–6。注意: 信号是间歇的, 设大不代表一定满仓, 只是'有信号时能多开'。",
    "account.max_concurrent_crypto_perp":
        "【加密永续最多并发仓】上面总数里, 加密最多占几个。典型 2, 想多开设 3–5。",
    "account.max_concurrent_stock_perp":
        "【股票永续最多并发仓】上面总数里, 股票最多占几个。典型 1, 范围 0–3(股票永续流动性低, 别太多)。",
    "risk.risk_per_trade_usdt_default":
        "【每笔风险预算USDT】每笔交易'若打到止损'最多亏多少U(决定下单大小)。\n"
        "调高→每笔仓位更大(更接近满仓)但单笔亏损更大。1000U账户典型 2.0(=0.2%/笔), 想更满仓设 3–6。",
    "risk.max_total_notional_normal":
        "【总名义敞口上限USDT】所有持仓名义价值之和的上限(防一次性铺太多)。\n"
        "典型 2500(1000U本金×杠杆后的总暴露上限)。想更满仓可调高, 但风险同步放大。",
    "signal.ema_flow_half_life_s":
        "【流向EMA半衰期(秒)】给'盘口流向'装减震器: 越大越平滑越稳但越慢; 0=不平滑(回到狂跳)。\n"
        "典型 1.0, 范围 0.5–2.0。这是修复'分数每秒乱跳'的核心。",
    "signal.ema_trend_half_life_s":
        "【趋势EMA半衰期(秒)】给'价格趋势'装减震器。典型 2.0, 范围 1.0–4.0。",
    "signal.dir_hyst_enter":
        "【方向认定阈值】合成方向(0~1)要超过此值才认一个方向(双阈值闩锁的上阈, 防贴0来回翻)。\n"
        "典型 0.12, 范围 0.08–0.25。调高=更挑剔不易翻向。",
    "signal.regime_rv_lo":
        "【死盘阈值】60秒已实现波动低于此=行情太淡没机会, 不开仓。典型 0.0002。\n"
        "若发现常年不出手且行情其实在动, 适当调低; 若总在烂行情乱做, 调高。",
    "signal.regime_rv_hi":
        "【剧烈阈值】波动高于此=行情剧烈, 自动更平滑+多等几拍(防被甩)。典型 0.0012。",
}

# 权威默认 (= config 出厂值); 不可删除
INITIAL_PRESET = {
    "signal.min_composite_score": 66, "signal.strong_composite_score": 74,
    "signal.min_tradability": 0.50,
    "signal.confirm_min": 0.20, "signal.persist_ticks": 3, "signal.cooldown_seconds": 20,
    "signal.min_edge_to_cost_ratio": 1.2, "signal.tp_rr": 1.6, "signal.sl_min_cost_mult": 2.5,
    "signal.min_hold_seconds": 20, "signal.reversal_hyst_gap": 8, "signal.decay_threshold": 54,
    "execution.max_hold_seconds": 180, "leverage.target_margin_risk_per_stop": 0.08,
    "account.max_concurrent_positions": 2, "account.max_concurrent_crypto_perp": 2,
    "account.max_concurrent_stock_perp": 1, "risk.risk_per_trade_usdt_default": 2.0,
    "risk.max_total_notional_normal": 2500,
    "signal.ema_flow_half_life_s": 1.0, "signal.ema_trend_half_life_s": 2.0,
    "signal.dir_hyst_enter": 0.12, "signal.regime_rv_lo": 2.0e-4, "signal.regime_rv_hi": 1.2e-3,
}
# 宽松: 想多出手/更满仓 (门槛/确认/持续/edge 放低 + 并发与每笔预算调高); 出手多但噪声多, 务必先虚拟盘
LOOSE_PRESET = {
    "signal.min_composite_score": 62, "signal.strong_composite_score": 70,
    "signal.min_tradability": 0.40,
    "signal.confirm_min": 0.15, "signal.persist_ticks": 2, "signal.cooldown_seconds": 15,
    "signal.min_edge_to_cost_ratio": 1.0, "signal.tp_rr": 1.5, "signal.sl_min_cost_mult": 2.2,
    "signal.min_hold_seconds": 12, "signal.reversal_hyst_gap": 6, "signal.decay_threshold": 52,
    "execution.max_hold_seconds": 120, "leverage.target_margin_risk_per_stop": 0.08,
    "account.max_concurrent_positions": 5, "account.max_concurrent_crypto_perp": 4,
    "account.max_concurrent_stock_perp": 2, "risk.risk_per_trade_usdt_default": 3.5,
    "risk.max_total_notional_normal": 4000,
    "signal.ema_flow_half_life_s": 0.7, "signal.ema_trend_half_life_s": 1.5,
    "signal.dir_hyst_enter": 0.09, "signal.regime_rv_lo": 1.2e-4, "signal.regime_rv_hi": 1.5e-3,
}
BUILTIN_PRESETS = {"初始参数(权威默认)": INITIAL_PRESET, "宽松(多出手)": LOOSE_PRESET}


def _presets_path():
    return paths.data_path("param_presets.json")


def current_params() -> dict:
    cfg = Config.load()
    return {k: cfg.get(k) for k, _, _ in PARAM_DEFS}


def list_presets() -> list:
    names = list(BUILTIN_PRESETS.keys())
    p = _presets_path()
    if p.exists():
        try:
            names += list(json.loads(p.read_text(encoding="utf-8")).keys())
        except Exception:
            pass
    return names


def get_preset(name: str) -> dict:
    if name in BUILTIN_PRESETS:
        return dict(BUILTIN_PRESETS[name])
    p = _presets_path()
    if p.exists():
        try:
            return dict(json.loads(p.read_text(encoding="utf-8")).get(name, {}))
        except Exception:
            pass
    return {}


def save_preset(name: str, params: dict) -> str:
    name = (name or "").strip()
    if not name:
        return "请先输入预设名称。"
    if name in BUILTIN_PRESETS:
        return "内置预设名不可覆盖, 请换个名字。"
    p = _presets_path()
    data = {}
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data[name] = params
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"已保存预设『{name}』({len(params)}项) -> {p.name}"


def delete_preset(name: str) -> str:
    if name in BUILTIN_PRESETS:
        return "内置预设不可删除。"
    p = _presets_path()
    if not p.exists():
        return "无自定义预设。"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if name not in data:
            return f"无预设『{name}』。"
        data.pop(name)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return f"已删除预设『{name}』。"
    except Exception as e:
        return f"删除失败: {e!r}"


def apply_params(params: dict) -> str:
    from ..research import config_patch
    paths.ensure_user_config()
    summary = config_patch.apply_updates(paths.config_path(), params)
    return ("✓ 已写入 config:\n  " + "\n  ".join(summary)
            + "\n\n重启引擎(停止→启动)后生效。")


def validate_signal_sync(use_all: bool) -> str:
    """信号有效性检验: 用录制数据实测综合分能否预测未来涨跌 (IC/分桶/命中率)。只读, 不改信号。"""
    try:
        from ..research import calibrator as cal
        from ..research import validate as val
        files = cal.find_recordings(_rec_dir())
        if not files:
            return "未找到录制文件。请先在『控制台』▶启动跑一段虚拟盘录制, 再来检验。"
        if not use_all:
            files = files[-1:]
        by_inst, meta = cal.load_calib(files)
        if not by_inst:
            return "录制数据太少, 无法检验。多跑一会儿。"
        return val.format_report(meta, val.validate(by_inst))
    except Exception as e:
        return f"信号检验失败: {e!r}"


def apply_calibration_sync(updates: dict) -> str:
    try:
        from ..research import config_patch
        paths.ensure_user_config()
        path = paths.config_path()
        summary = config_patch.apply_updates(path, updates)
        return ("✓ 已写入 " + str(path) + "\n  " + "\n  ".join(summary)
                + "\n\n重启引擎(停止→启动)后生效。建议应用后再跑一段虚拟盘复核。")
    except Exception as e:
        return f"应用失败: {e!r}"
