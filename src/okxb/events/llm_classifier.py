"""AI 事件分类器 (提供商无关): 新闻/filing 文本 -> 结构化 veto 标签。

支持:
  - DeepSeek / OpenAI 兼容 (chat/completions, JSON 输出) —— 默认且推荐, 只需 httpx
  - Claude (anthropic SDK, tool-use) —— 需额外安装 anthropic
  - 规则降级 (无 key / 提供商=rule / 调用失败) —— 按 8-K item 码与关键词

按任务难度选模型: 事件分类是简单任务 -> model_simple (如 deepseek-v4-flash);
复杂任务 -> model_hard (如 deepseek-v4-pro)。
铁律: 输出只是标签, 永不下单。
"""
from __future__ import annotations

import json
import re
from typing import Optional

import httpx

EVENT_TYPES = ["earnings", "SEC_8K", "insider_trade", "lawsuit", "regulatory",
               "macro", "product", "merger_split", "other"]
ACTIONS = ["no_action", "block_long", "block_short", "reduce_only", "close_all"]
SEVERITIES = ["low", "medium", "high"]
HORIZONS = ["intraday", "days", "weeks"]
OPENAI_PROVIDERS = {"deepseek", "openai", "openai_compatible", "moonshot", "qwen", "kimi"}

SYSTEM = (
    "你是量化交易的事件风控分类器。给定某美股公司的一条新闻或 SEC filing 摘要, "
    "判断它对短线交易的风险。保守优先: 重大负面/不确定事件(重组/诉讼/调查/重大 8-K/财报临近)"
    "给 reduce_only 或 block; 拆股/合并/退市类给 close_all; 常规无关新闻给 no_action。"
)
JSON_INSTRUCT = (
    "\n只输出一个 JSON 对象 (不要任何多余文字), 字段如下:\n"
    '{"event_type": one of ' + str(EVENT_TYPES) + ', '
    '"sentiment": number -1..1, "confidence": number 0..1, '
    '"severity": one of ["low","medium","high"], '
    '"time_horizon": one of ["intraday","days","weeks"], '
    '"action": one of ' + str(ACTIONS) + "}"
)

# Claude tool-use schema (仅 provider=claude 时用)
TOOL = {
    "name": "emit_event_label",
    "description": "把一条市场事件分类为结构化交易风控 veto 标签。",
    "input_schema": {
        "type": "object",
        "properties": {
            "event_type": {"type": "string", "enum": EVENT_TYPES},
            "sentiment": {"type": "number"},
            "confidence": {"type": "number"},
            "severity": {"type": "string", "enum": SEVERITIES},
            "time_horizon": {"type": "string", "enum": HORIZONS},
            "action": {"type": "string", "enum": ACTIONS},
        },
        "required": ["event_type", "sentiment", "confidence", "severity",
                     "time_horizon", "action"],
    },
}

HIGH_SEVERITY_8K_ITEMS = {"1.03", "2.06", "4.01", "4.02", "5.02"}


class LLMClassifier:
    def __init__(self, provider: str, api_key: str, base_url: str,
                 model_simple: str, model_hard: str, tier_policy: str = "auto"):
        self.provider = (provider or "rule").lower()
        self.api_key = api_key or ""
        self.base_url = (base_url or "https://api.deepseek.com").rstrip("/")
        self.model_simple = model_simple or "deepseek-v4-flash"
        self.model_hard = model_hard or "deepseek-v4-pro"
        self.tier_policy = (tier_policy or "auto").lower()

    @classmethod
    def from_secrets(cls, secrets) -> "LLMClassifier":
        return cls(secrets.ai_provider, secrets.ai_api_key, secrets.ai_base_url,
                   secrets.ai_model_simple, secrets.ai_model_hard,
                   getattr(secrets, "ai_tier_policy", "auto"))

    @property
    def enabled(self) -> bool:
        return bool(self.api_key) and self.provider in (OPENAI_PROVIDERS | {"claude"})

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model_simple}" if self.enabled else "规则降级"

    def _model_for(self, tier: str) -> str:
        return self.model_hard if tier == "hard" else self.model_simple

    def _json_model(self) -> str:
        """结构化 JSON 任务(选品/结构化研判)优先用【非推理】模型: 更快、直接吐 JSON。
        推理模型(deepseek-v4-pro / reasoner / o系)会把答案埋进思维链, content 常空或被截断 → 解析不到。
        故 model_hard 是推理模型时回退到 model_simple(flash); 自由文本深度分析仍用 model_hard(见 analyze)。"""
        hard = (self.model_hard or "").lower()
        if any(k in hard for k in ("pro", "reasoner", "think", "-r1", "o1", "o3")):
            return self.model_simple or self.model_hard
        return self.model_hard

    def _choose_tier(self, kind: str, items: str) -> str:
        """按难度自动选: 自由文本新闻 + 高危 8-K -> hard(pro); 结构化/常规 -> simple(flash)。"""
        if self.tier_policy == "flash":
            return "simple"
        if self.tier_policy == "pro":
            return "hard"
        # auto
        item_set = {x.strip() for x in (items or "").split(",") if x.strip()}
        if kind == "news":
            return "hard"
        if kind == "8-K" and (item_set & HIGH_SEVERITY_8K_ITEMS):
            return "hard"
        return "simple"

    async def ping(self) -> str:
        """连通性验证: 发一个最小请求, 返回明确的成功/失败文本 (不吞错)。"""
        if not self.enabled:
            return (f"提供商 = {self.provider} (未填 key 或选了『规则』)\n"
                    "→ 不调用任何 AI, 使用免费规则分类。无需验证。")
        try:
            if self.provider == "claude":
                import anthropic
                client = anthropic.AsyncAnthropic(api_key=self.api_key)
                await client.messages.create(model=self.model_simple, max_tokens=8,
                                             messages=[{"role": "user", "content": "ping"}])
                return f"Claude 连通 ✓\n模型(简单)={self.model_simple}"
            url = f"{self.base_url}/chat/completions"
            body = {"model": self.model_simple, "max_tokens": 8, "temperature": 0,
                    "stream": False,
                    "messages": [{"role": "user", "content": "只回复一个词: pong"}]}
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            async with httpx.AsyncClient(timeout=20.0) as c:
                r = await c.post(url, headers=headers, json=body)
            if r.status_code != 200:
                return f"{self.provider} 失败 ✗  HTTP {r.status_code}\n{r.text[:200]}"
            data = r.json()
            txt = (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "")[:40]
            u = data.get("usage", {})
            return (f"{self.provider} 连通 ✓\nBase URL={self.base_url}\n"
                    f"简单任务模型={self.model_simple} (复杂={self.model_hard})\n"
                    f"回复='{txt.strip()}'  用量={u}")
        except Exception as e:
            return f"{self.provider} 连接异常 ✗\n{e!r}"

    async def classify(self, ticker: str, text: str, kind: str = "news",
                       items: str = "", tier: Optional[str] = None) -> dict:
        if not self.enabled:
            return self._fallback(kind, items, text)
        chosen = tier or self._choose_tier(kind, items)
        model = self._model_for(chosen)
        print(f"[llm] {ticker} {kind}({items or '-'}) -> 难度={chosen} 模型={model} (策略={self.tier_policy})")
        try:
            if self.provider == "claude":
                return await self._classify_claude(model, ticker, text, kind, items)
            return await self._classify_openai(model, ticker, text, kind, items)
        except Exception as e:
            print(f"[llm] 分类失败({model}), 规则降级: {e!r}")
            return self._fallback(kind, items, text)

    # ----------------- 通用对话 + AI 分析建议 -----------------

    async def _chat(self, model: str, system: str, user: str, max_tokens: int = 500) -> str:
        if self.provider == "claude":
            import anthropic
            c = anthropic.AsyncAnthropic(api_key=self.api_key)
            m = await c.messages.create(model=model, max_tokens=max_tokens, system=system,
                                        messages=[{"role": "user", "content": user}])
            return "".join(getattr(b, "text", "") for b in m.content
                           if getattr(b, "type", None) == "text")
        url = f"{self.base_url}/chat/completions"
        body = {"model": model, "max_tokens": max_tokens, "temperature": 0.3, "stream": False,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}]}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=30.0) as c:
            r = await c.post(url, headers=headers, json=body)
            if r.status_code >= 400:        # 带出服务端真实原因, 否则只看到裸 400 无从排查
                raise RuntimeError(f"{self.provider} {r.status_code} @ chat/completions: {r.text[:400]}")
            msg = r.json()["choices"][0]["message"]
            # reasoning 模型(deepseek-v4-pro)正文在 content; 若被 max_tokens 截断为空, 回退 reasoning_content
            return msg.get("content") or msg.get("reasoning_content") or ""

    async def analyze(self, inst: str, context_text: str) -> str:
        """对单个标的做专业短线分析建议 (复杂任务 -> 用 model_hard/pro)。仅供参考。"""
        if not self.enabled:
            return ("未配置 AI (当前=规则或未填 key)。\n请到『账户与密钥』选 DeepSeek 并填 API Key 后再用。")
        sysp = ("你是专业的加密货币/美股永续合约短线交易分析师。基于给定的实时盘口与因子数据, "
                "给出简明、专业、可执行的判断。务必中文, 不超过 220 字。")
        user = (f"标的: {inst}\n实时数据:\n{context_text}\n\n"
                "请按以下结构给出:\n1) 方向倾向: 做多/做空/观望\n2) 理由(结合盘口失衡OBI/订单流OFI/趋势/价差)\n"
                "3) 置信度: 低/中/高\n4) 关键风险\n5) 操作建议(建议杠杆区间、是否逐仓、若进场的止损/止盈思路)\n"
                "结尾注明: 仅供参考, 非投资建议。")
        try:
            txt = await self._chat(self.model_hard, sysp, user, max_tokens=600)
            return txt.strip() or "AI 未返回内容。"
        except Exception as e:
            return f"AI 分析失败: {e!r}"

    async def _chat_json(self, model: str, system: str, user: str, max_tokens: int = 800) -> dict:
        """要求模型返回 JSON 对象 (OpenAI 兼容用 response_format; Claude 直接解析)。"""
        if self.provider == "claude":
            txt = await self._chat(model, system + "\n只输出一个 JSON 对象, 不要多余文字。",
                                   user, max_tokens)
            return _parse_json(txt)
        url = f"{self.base_url}/chat/completions"
        body = {"model": model, "max_tokens": max_tokens, "temperature": 0.2, "stream": False,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}]}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=40.0) as c:
            r = await c.post(url, headers=headers, json=body)
            if r.status_code >= 400:        # 带出服务端真实原因, 否则只看到裸 400 无从排查
                raise RuntimeError(f"{self.provider} {r.status_code} @ chat/completions: {r.text[:400]}")
            msg = r.json()["choices"][0]["message"]
            return _parse_json(msg.get("content") or msg.get("reasoning_content") or "")

    async def analyze_structured(self, inst: str, qctx: dict) -> dict:
        """结构化、量化定锚的单标的分析。qctx 含实时因子 + 代码已算好的客观默认(SL/TP/杠杆/仓位)。
        返回 {ok, struct{...可一键导入手动交易...}, text(可读版)}。
        铁律: AI 只能基于给定客观数据判断方向/置信/风险, 不得编造新闻; 价位/杠杆须落在给定约束内。"""
        defaults = {
            "direction": qctx.get("dir_prior", "flat"),
            "order_type": "post_only",
            "entry_px": qctx.get("entry_px"), "tp_px": qctx.get("tp_px"),
            "sl_px": qctx.get("sl_px"), "leverage": qctx.get("lev_suggest"),
            "size_usdt": qctx.get("size_usdt"), "confidence": 0.5,
            "rationale": "", "risks": "", "scaling": "",
        }
        if not self.enabled:
            return {"ok": False, "struct": defaults,
                    "text": "未配置 AI (当前=规则或未填 key)。已用【量化默认】填充方向/止盈止损/杠杆/仓位, "
                            "可直接『导入手动交易』。如需 AI 文字研判, 请到『账户与密钥』选 DeepSeek 填 Key。"}
        sysp = ("你是严谨的加密/美股永续合约短线量化分析师。只允许基于【给定的客观盘口与因子数据】"
                "与【已算好的风险量】做判断, 严禁编造新闻、财报、社媒或任何外部消息。"
                "杠杆必须在 [3, {lvmax}] 内且优先采用给定的『量化建议杠杆』, 仅在有明确理由时小幅调整。"
                "止盈/止损/入场价应贴近给定的量化默认值。务必中文。"
                ).format(lvmax=int(qctx.get("lev_max", 50)))
        schema = ('{"direction":"long|short|flat","order_type":"post_only|limit|optimal_limit_ioc",'
                  '"entry_px":number,"tp_px":number,"sl_px":number,"leverage":int,'
                  '"size_usdt":number,"confidence":number(0-1),'
                  '"rationale":"<=120字 结合OBI/OFI/趋势/波动/价差","risks":"<=80字",'
                  '"scaling":"<=80字 建仓/加仓/减仓节奏"}')
        user = (f"标的: {inst}\n"
                f"实时客观数据: 中间价={qctx.get('mid')} 价差bps={qctx.get('spread_bps')} "
                f"OBI_z={qctx.get('obi_z')} OFI_z={qctx.get('ofi_z')} "
                f"趋势分量={qctx.get('trend_dir')} 流向分量={qctx.get('flow_dir')} "
                f"做多分={qctx.get('long')} 做空分={qctx.get('short')} "
                f"ATR(1m)={qctx.get('atr')} 已实现波动60s={qctx.get('rvol')}\n"
                f"代码已算好的客观默认(请贴近并校准): 方向先验={qctx.get('dir_prior')} "
                f"止损%={qctx.get('sl_pct')} 止盈%={qctx.get('tp_pct')} "
                f"入场价={qctx.get('entry_px')} 止盈价={qctx.get('tp_px')} 止损价={qctx.get('sl_px')} "
                f"量化建议杠杆={qctx.get('lev_suggest')}x (上限{qctx.get('lev_max')}x, "
                f"依据: 逐仓下止损亏损≈保证金的{int(qctx.get('target_risk',0.1)*100)}%) "
                f"建议下单额USDT={qctx.get('size_usdt')}\n\n"
                f"只输出一个 JSON 对象: {schema}")
        try:
            d = await self._chat_json(self._json_model(), sysp, user, max_tokens=1200)
            struct = self._coerce_analysis(d, defaults, qctx)
            return {"ok": True, "struct": struct, "text": _render_analysis(inst, struct, qctx)}
        except Exception as e:
            return {"ok": False, "struct": defaults,
                    "text": f"AI 分析失败: {e!r}\n已保留量化默认值, 仍可『导入手动交易』。"}

    @staticmethod
    def _coerce_analysis(d: dict, defaults: dict, qctx: dict) -> dict:
        out = dict(defaults)
        if d.get("direction") in ("long", "short", "flat"):
            out["direction"] = d["direction"]
        if d.get("order_type") in ("post_only", "limit", "optimal_limit_ioc"):
            out["order_type"] = d["order_type"]
        for k in ("entry_px", "tp_px", "sl_px", "size_usdt"):
            try:
                if d.get(k) is not None:
                    out[k] = float(d[k])
            except (TypeError, ValueError):
                pass
        # 杠杆: 钳到 [3, lev_max]
        lv_max = int(qctx.get("lev_max", 50))
        try:
            lv = int(round(float(d.get("leverage", out["leverage"] or 3))))
        except (TypeError, ValueError):
            lv = int(out["leverage"] or 3)
        out["leverage"] = max(3, min(lv_max, lv if lv > 0 else 3))
        try:
            out["confidence"] = max(0.0, min(1.0, float(d.get("confidence", 0.5))))
        except (TypeError, ValueError):
            out["confidence"] = 0.5
        for k in ("rationale", "risks", "scaling"):
            if isinstance(d.get(k), str):
                out[k] = d[k][:200]
        return out

    async def pick_products(self, pool_label: str, candidates: list[dict],
                            external_brief: str = "") -> dict:
        """从候选(已按量化机会分排序)中, 让 AI 推荐若干标的 + 方向 + 策略 + 杠杆 + 风险。
        external_brief: Finnhub 真实新闻/财报文本(若有), 作为权威外部信息喂给 AI。
        返回 {ok, picks:[...], text}。candidates 每项: inst/long/short/chg/vol/spread_bps/atr/mom。"""
        if not self.enabled:
            return {"ok": False, "picks": [], "text":
                    "未配置 AI。已按【量化机会分】排序候选(见下表), 选『DeepSeek』填 Key 可获 AI 研判。"}
        if not candidates:
            return {"ok": False, "picks": [], "text": "无候选数据 (请先启动引擎积累实时行情)。"}
        has_brief = bool(external_brief and external_brief.strip())
        news_rule = ("下方提供了【真实外部资讯(Finnhub 新闻/财报日历)】, 请结合它与量化快照综合判断, "
                     "可引用其中的事件, 但不得编造未给出的消息。"
                     if has_brief else
                     "本接口无联网, 你看不到实时新闻, 严禁编造新闻/社媒; 可用你已有常识做定性补充。")
        sysp = ("你是加密/美股永续合约短线选品分析师。基于【客观量化快照 + 交易所真实24h行情】"
                "(双向信号分、波动ATR、价差、近端动量、24h真实涨跌幅/成交额)排序与推荐。"
                + news_rule +
                " 数据不足以高把握时请如实说明并给低置信度。务必中文。")
        lines = []
        for c in candidates[:30]:
            lines.append(f"{c['inst']}: 多分{c.get('long')} 空分{c.get('short')} "
                         f"24h涨跌{c.get('chg', '?')}% 成交额{c.get('vol', '?')} "
                         f"ATR{c.get('atr')} 价差bps{c.get('spread_bps')} "
                         f"动量{c.get('mom')} 事件{c.get('event', '无')}")
        schema = ('{"picks":[{"inst":"...","direction":"long|short","strategy":"顺势/反转/突破/basis",'
                  '"leverage":int(3-50),"confidence":number(0-1),"reason":"<=80字","risk":"<=60字"}],'
                  '"note":"<=80字 总体提示"}')
        brief_block = (f"\n真实外部资讯(Finnhub):\n{external_brief}\n" if has_brief else "")
        user = (f"候选池: {pool_label}\n候选(已按量化机会分排序):\n" + "\n".join(lines) +
                brief_block +
                f"\n请挑出最多5个最具短线机会的标的, 只输出 JSON: {schema}")
        try:
            d = await self._chat_json(self._json_model(), sysp, user, max_tokens=1600)
            picks = d.get("picks") if isinstance(d.get("picks"), list) else []
            clean = []
            for p in picks[:5]:
                if not isinstance(p, dict) or not p.get("inst"):
                    continue
                try:
                    lv = max(3, min(50, int(round(float(p.get("leverage", 5))))))
                except (TypeError, ValueError):
                    lv = 5
                clean.append({
                    "inst": str(p["inst"]),
                    "direction": p.get("direction") if p.get("direction") in ("long", "short") else "long",
                    "strategy": str(p.get("strategy", ""))[:20],
                    "leverage": lv,
                    "confidence": _clamp01(p.get("confidence", 0.5)),
                    "reason": str(p.get("reason", ""))[:120],
                    "risk": str(p.get("risk", ""))[:100],
                })
            return {"ok": True, "picks": clean, "text": _render_picks(pool_label, clean, d.get("note", ""))}
        except Exception as e:
            return {"ok": False, "picks": [], "text": f"AI 选品失败: {e!r}"}

    # ----------------- OpenAI 兼容 (DeepSeek 等) -----------------

    async def _classify_openai(self, model, ticker, text, kind, items) -> dict:
        url = f"{self.base_url}/chat/completions"
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM + JSON_INSTRUCT},
                {"role": "user", "content":
                    f"公司:{ticker}\n类型:{kind}\n8-K items:{items}\n内容:\n{text[:2000]}"},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": 300,
            "temperature": 0,
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(url, headers=headers, json=body)
            r.raise_for_status()
            data = r.json()
        content = data["choices"][0]["message"]["content"]
        return self._coerce(_parse_json(content))

    # ----------------- Claude (可选) -----------------

    async def _classify_claude(self, model, ticker, text, kind, items) -> dict:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=self.api_key)
        msg = await client.messages.create(
            model=model, max_tokens=300, system=SYSTEM,
            tools=[TOOL], tool_choice={"type": "tool", "name": "emit_event_label"},
            messages=[{"role": "user",
                       "content": f"公司:{ticker}\n类型:{kind}\n8-K items:{items}\n内容:\n{text[:2000]}"}],
        )
        for block in msg.content:
            if getattr(block, "type", None) == "tool_use":
                return self._coerce(dict(block.input))
        return self._coerce({})

    # ----------------- 校验与降级 -----------------

    @staticmethod
    def _coerce(d: dict) -> dict:
        out = {
            "event_type": d.get("event_type") if d.get("event_type") in EVENT_TYPES else "other",
            "severity": d.get("severity") if d.get("severity") in SEVERITIES else "low",
            "time_horizon": d.get("time_horizon") if d.get("time_horizon") in HORIZONS else "intraday",
            "action": d.get("action") if d.get("action") in ACTIONS else "no_action",
        }
        try:
            out["sentiment"] = max(-1.0, min(1.0, float(d.get("sentiment", 0.0))))
        except (TypeError, ValueError):
            out["sentiment"] = 0.0
        try:
            out["confidence"] = max(0.0, min(1.0, float(d.get("confidence", 0.5))))
        except (TypeError, ValueError):
            out["confidence"] = 0.5
        return out

    @staticmethod
    def _fallback(kind: str, items: str, text: str) -> dict:
        t = (text or "").lower()
        item_set = {x.strip() for x in (items or "").split(",") if x.strip()}
        base = {"sentiment": 0.0, "confidence": 0.5, "time_horizon": "intraday"}
        if any(k in t for k in ("merger", "acquisition", "stock split", "reverse split",
                                "delist", "spin-off", "spinoff")):
            return {**base, "event_type": "merger_split", "severity": "high",
                    "confidence": 0.7, "action": "close_all"}
        if kind == "8-K" and item_set & HIGH_SEVERITY_8K_ITEMS:
            return {**base, "event_type": "SEC_8K", "severity": "high",
                    "sentiment": -0.5, "confidence": 0.7, "action": "reduce_only"}
        if kind == "8-K" and "2.02" in item_set:
            return {**base, "event_type": "earnings", "severity": "medium",
                    "confidence": 0.6, "action": "reduce_only"}
        if kind == "8-K":
            return {**base, "event_type": "SEC_8K", "severity": "medium",
                    "confidence": 0.5, "action": "reduce_only"}
        if kind == "form4":
            return {**base, "event_type": "insider_trade", "severity": "low",
                    "confidence": 0.5, "action": "no_action"}
        if any(k in t for k in ("lawsuit", "sue", "investigation", "subpoena",
                                "sec charges", "fraud")):
            return {**base, "event_type": "lawsuit", "severity": "high",
                    "sentiment": -0.6, "confidence": 0.6, "action": "reduce_only"}
        return {**base, "event_type": "other", "severity": "low",
                "confidence": 0.4, "action": "no_action"}


def _clamp01(v) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 0.5


_DIR_CN = {"long": "做多", "short": "做空", "flat": "观望"}
_OT_CN = {"post_only": "只挂单(maker)", "limit": "限价", "optimal_limit_ioc": "市价IOC(taker)"}


def _render_analysis(inst: str, s: dict, qctx: dict) -> str:
    d = s.get("direction", "flat")
    lines = [f"【{inst}】AI 量化研判  (置信度 {s.get('confidence', 0):.0%})",
             f"方向: {_DIR_CN.get(d, d)}    下单类型: {_OT_CN.get(s.get('order_type'), s.get('order_type'))}",
             f"入场价≈{s.get('entry_px')}   止盈≈{s.get('tp_px')}   止损≈{s.get('sl_px')}",
             f"建议杠杆: {s.get('leverage')}x (逐仓; 依据 ATR 推导的止损距离, "
             f"使止损亏损≈保证金的{int(qctx.get('target_risk', 0.1)*100)}%, 上限{qctx.get('lev_max')}x)",
             f"建议下单额: {s.get('size_usdt')} USDT (按单笔风险预算)",
             f"理由: {s.get('rationale') or '-'}",
             f"风险: {s.get('risks') or '-'}",
             f"建/减仓: {s.get('scaling') or '-'}",
             "—— 点『⬇ 导入手动交易』把以上参数填入下单区(不会自动下单, 你确认后再点下单)。",
             "仅供参考, 非投资建议。"]
    return "\n".join(lines)


def _render_picks(pool_label: str, picks: list, note: str) -> str:
    if not picks:
        return f"[{pool_label}] AI 未给出推荐。"
    lines = [f"【AI 选品 · {pool_label}】(基于实时信号分+交易所24h真实涨跌+模型常识; "
             "无联网实时新闻/社媒; 仅供参考)"]
    for i, p in enumerate(picks, 1):
        lines.append(f"{i}. {p['inst']}  {_DIR_CN.get(p['direction'], p['direction'])}  "
                     f"策略={p['strategy']}  {p['leverage']}x  置信{p['confidence']:.0%}")
        lines.append(f"    理由: {p['reason']}")
        if p.get("risk"):
            lines.append(f"    风险: {p['risk']}")
    if note:
        lines.append(f"提示: {note}")
    lines.append("点某条右侧『导入』可带入手动交易。非投资建议。")
    return "\n".join(lines)


def _parse_json(content: str) -> dict:
    """从模型返回中稳健解析 JSON (容忍 ```json 围栏、前后多余文字、reasoning 模型混入)。"""
    if not content:
        return {}
    s = content.strip()
    if s.startswith("```"):                       # 去 ```json ... ``` 围栏
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        pass
    start = s.find("{")                           # 提取首个平衡的 {...} (比贪婪正则稳健)
    if start >= 0:
        depth = 0
        for i in range(start, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start:i + 1])
                    except (ValueError, TypeError):
                        break
    return {}
