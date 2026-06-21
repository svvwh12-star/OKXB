# App 多周期量化集成 实现计划(安全版 · 已按方法论审查修正 2026-06-12)

> **本计划已根据一组方法论批评全面修正。定位红线(不可逾越):**
> **App 内部「只读研究监控面板」——不接交易执行、不替换方向源、不驱动下单、不改 forward 判据。**
> 旧版"替换控制台方向源 + 历史gate亮灯 + 扩7周期"的写法已作废(会破坏预登记、把噪声摆成信号)。

## 不可逾越的边界(全部来自方法论审查,逐条采纳)
1. **预登记保护**:forward 裁决**永远只用冻结的 A/B/C**(180min hist_gbm / 6h lightgbm / 9h mlp)。任何新周期(15/30/60/120/720)需**另开一个全新预登记实验**(新 freeze/cutoff/6周),不混进本面板。
2. **返点只作参考,不改判据**:面板显示两列 `conservative_net`(taker10,不返点,**主**)与 `rebate_net`(taker8,仅参考)。亮灯/裁决一律用 `conservative_net` + forward 状态。**forward 协议判据(taker10+funding+1.5x stress)不变。** 返点只让数字好看 ~2bps,**不改 verdict**(病因是 t<门槛+n太少,不是成本)。
3. **不替换方向源**:控制台原超高频方向源**保留不动**;多周期是**独立只读 tab**,不喂交易默认方向、不进选品排序、不碰下单。
4. **亮灯由 forward 状态驱动**(非历史 OOS gate):`PASS→绿`(仅代表进第二阶段,非实盘)/`PENDING→黄灰`/`KILL→红`/`历史好看但forward未过→灰`。**现状 A/B/C 全 PENDING ⇒ 面板不得有任何绿色交易灯。**
5. **AI 只解释**:prompt 硬约束 `monitor-only unless forward PASS`、`AI MUST NOT recommend a trade when gate is PENDING/KILL`;输出须含"AI 说法 vs 量化测量"差异行。
6. **先只 BTC、只 A/B/C 三个冻结候选**。其他标的/周期不上实时面板。

## Phase 0 — 边界修正(本文件即是)✅
把定位钉死为"只读监控,不接执行/不替换方向源"。

## Phase 1 — Forward 状态面板(读 `forward_status.csv`)
- 新 tab「多周期研究监控」。读 `btc_single_asset_research/forward/forward_status.csv`,显示每个候选:
  `候选 A/B/C | H | 模型 | start_utc | weeks | n_ts | net10/net15(conservative) | fwd_ic | verdict`。
- 顶部横幅:`A/B/C 全 PENDING — 研究观察期, 不可交易`。
- **验收**:无引擎也能看(纯读 CSV);verdict 着色按边界④;现状无绿灯。

## Phase 2 — 当前打分面板(A/B/C frozen 对当前 bar 打分)
- 加载 `frozen/{A,B,C}` 模型,对**当前** 5m bar 用冻结管线打分,只显示:
  `当前方向倾向 | score | 是否超过冻结 tau(高置信) | 该候选是否已 PASS`。
- 两列净edge参考:`conservative_net`(主) + `rebate_net`(灰字"参考")。
- 无 PASS 时整行标注「**研究观察,不可交易**」。手动/定时刷新(多周期算几秒,定时默认 900s + 手动按钮)。
- **验收**:打分只读;无 PASS 不给任何"可做"暗示;tau 用冻结绝对值。

## Phase 3 — AI 对照(只解释,不荐交易)
- 把 A/B/C 的「方向/score/conservative_net/forward verdict」喂给 `analyze_structured`,prompt 硬约束(边界⑤)。
- AI 输出:量化客观结论段 + AI 互补解读段(事件/regime/情境) + "AI vs 量化差异"行。`verdict≠PASS` 时 AI 不得给任何"可尝试"措辞。
- **验收**:对 PENDING 候选,AI 输出里无交易建议;有差异行。

## Phase 4 — 仅当 PASS 后(6 周判定后)
- 只有某候选 forward **PASS**,才进第二阶段:全新独立 forward 窗 + 最小真钱测成交质量(接 raw L2)。**本面板永不直接下单。**

## Phase S — demo 影子执行验证轨道(仅 demo,与监控面板分离)
> 这是用户要求的"模拟盘自动下单",作为一条**独立轨道**。与"只读监控面板"不矛盾:监控面板永不下单;本轨道**仅 demo**自动下单,目的是**执行真实性验证**,不碰真钱、不替换方向源、不影响纸面裁决。
- **目的(已确认):执行真实性验证**——量出"纸面 net vs 真实成交 net"差距(成交率/滑点/延迟/funding/止盈止损触发)。**非盈利目的;诚实预期 demo 亏/打平**(信号 gate=False);demo 盈不代表可上实盘。
- 护栏:**仅 demo,fail-closed**(`guard_demo`:非 demo 拒绝);**串行单槽**(任一时刻 BTC≤1 影子仓,A/B/C 谁先超 tau 谁入场);风险预算反推(equity×0.3%,杠杆钳 1-10);bracket = TP/SL(k×日级波动)+ 到该候选 H 的 **time-exit**;平仓 reduce-only。
- 默认 **dry-run**(`--mode shadow`:纯公共数据,只显示"会怎么动作",**不下单**);**`--arm`** 才在 demo 真下单(需 `OKXB_MODE=demo` + demo 密钥)。
- 记录 `forward/shadow_trades.csv`(执行层,独立于纸面 `forward_status.csv`);**不进 forward 统计裁决**。
- 命令:`python run_forward_shadow.py --mode shadow`(dry) / `... --mode shadow --arm`(demo 下单)。
- 已实现:`run_forward_shadow.py` 的 `shadow()` + 纯逻辑(`guard_demo`/`shadow_size`/`tp_sl_pct`/`_score_now`),selftest 覆盖纯逻辑;真下单/对账细节待你配 demo key 后实测迭代。

## 已完成的 Phase A(返点)如何归位
Phase A(`fee_rebate_frac` + `WorkflowCosts.from_config` + GUI cost 返点)**保留**,但用途收窄为:**UI 的 `rebate_net` 参考列 + AI 分析的成本展示**。`run_forward_shadow.py` 的判据仍硬编码 taker10/stress15(未受影响),**forward 裁决不返点**。

## 明确被否决(不做)
- ❌ 7 周期实时信号面板(破坏预登记/盯噪声)——7周期历史看静态 `enhanced_summary.csv`。
- ❌ 替换控制台方向源。❌ 历史 OOS gate 亮灯。❌ 亮灯/面板驱动下单。❌ 返点改 forward 判据。❌ 多标的同时上。
