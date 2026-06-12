# App 多周期量化集成 实现计划(分阶段)

> 把多周期量化(15/30/60/120/180/360/720min)接进 OKXB .exe(tkinter GUI),**替换控制台方向源**、
> **严格亮灯**、**AI 对照**、**20% 返点费率**。
> **诚实定位(已与用户确认):监控/对照面板,不是"亮灯就下单"的信号灯。** 亮灯门用真实净edge闸门,多数时间会是"未达标/NO TRADE"——那是真话。

## 架构(数据流)
```
冻结模型(forward shadow frozen, 扩到7周期)
        │  定时(10-15min)+手动 触发
 MultiPeriodSignal 服务(轻量推断: 当前方向分 + 历史OOS净edge[返点后] + gate + 亮灯)
        │  写
 App.latest_multiperiod = {inst: {H: {dir, score, net_bps, gate, light}}}
        │  读
 控制台(周期切换 15-720 + long/short 源替换 + 亮灯列)
        │
 AI 分析(_quant_context dir_prior 用多周期 + analyze prompt 加量化结论并标"AI vs 量化"差异)
```
现实约束:DVOL/链上只有 BTC/ETH 全;OKX rubik OI/taker 多为近窗。**多周期对 BTC(其次 ETH)最完整**,其他标的退化为基础价量+funding+cross-asset。

## Phase A — 返点费率(本轮落地;小、独立、是一切净edge的基础)
- `config/config.yaml` `fees` 段加 `fee_rebate_frac: 0.20`。
- `pmw.WorkflowCosts.from_config(cfg)`:读 maker/taker_pct,乘 `(1−rebate)`,转 bps。
- 多周期/forward/vol_carry 的成本都走返点后 `WorkflowCosts`;controller `_quant_context` 的 `cost` 乘 `(1−rebate)`。
- **验收**:单测 `WorkflowCosts.from_config` 返点计算;手动确认 AI 分析里的 cost 反映返点。
- **诚实**:返点把 taker 往返 ~10→8bps,**不改变"这些信号是噪声"的结论**。

## Phase B — 多周期信号服务(BTC 先行)
- 扩展 `run_forward_shadow.py` 的 freeze 到 **7 个周期**(各周期从已有研究选最佳模型/top_frac);或新建 `multiperiod_freeze`。
- 新模块 `src/okxb/signal/multiperiod.py`:`MultiPeriodSignal.infer(inst)` → `{H: {dir, score, net_bps(返点后), gate, light}}`,用冻结模型对**当前 bar** 打分,net_bps/gate 取该候选**历史 OOS**值,亮灯门=严格(gate pass→绿;接近→黄;否则灰)。
- **验收**:脚本对 BTC 输出 7 周期表;绿灯仅当真过闸门。

## Phase C — App 集成(定时+手动刷新)
- `App` 加后台 `asyncio` 任务:每 `multiperiod.refresh_seconds`(默认 900)跑一次 `MultiPeriodSignal`,写 `self.latest_multiperiod`;暴露 `request_multiperiod_refresh()` 供手动触发。
- `EngineController.multiperiod()` 只读访问 + `refresh_multiperiod_sync()`。
- **验收**:引擎跑起来后 `latest_multiperiod` 定时更新;手动按钮即时刷新。

## Phase D — 控制台 GUI(替换方向源 + 周期切换 + 亮灯)
- 控制台加 `CTkSegmentedButton` 周期切换(15/30/60/120/180/360/720)+ "↻刷新多周期"按钮。
- 选中周期后,表格 `long/short` 用 `latest_multiperiod[inst][H]` 的方向分(替换超高频源);新增"亮灯"列(绿/黄/灰)+ "净edge"列。
- 表头/排序适配新列。
- **验收**:切换周期表格随之变;亮灯严格;无引擎时显示"未就绪"。

## Phase E — AI 对照(量化喂给 AI + 标差异)
- `_quant_context` 的 `dir_prior` 改用当前选中周期的多周期方向;加入该周期 `net_bps/gate`。
- `LLMClassifier.analyze_structured` prompt 注入"多周期量化结论",要求 AI 给**互补**的事件/regime/情境解读(不重复方向),并显式输出"AI 说法 vs 量化测量"的差异行。
- 验证 DeepSeek/Finnhub key 接通(已有 `verify_ai_sync`/`verify_finnhub_sync`)。
- **验收**:AI 分析输出含"量化:方向/净edge/gate"与"AI:互补解读+差异"两段。

## 阶段闸门
每阶段独立可验收;Phase A→B→C→D→E 顺序;A 本轮完成,B 起逐轮推进。亮灯严格门是贯穿红线:**只有真过净edge闸门才绿灯**。
