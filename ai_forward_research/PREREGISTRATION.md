# 预登记 · AI 前向验证 (AIFWD)

> **定位**：纯【前向验证研究】，**只记录、不实盘**。测的是——**AI 自己给的方向判断，N 分钟后是否真有净 edge**。
> 不是交易系统。过**全部闸门**才考虑 demo；按本项目历史，**大概率不会过**。

## 冻结的假设（看到前向数据后**不得**修改）
- **观察名单**：`BTC-USDT-SWAP` / `ETH-USDT-SWAP` / `SOL-USDT-SWAP`。
- **信号**：调用 exe 内**同一条 AI 单标的分析**（`_ai_analyze`），取其 `direction`（long/short）。
- **入场价**：记录时的中间价；**结算价**：`HORIZON=60min` 后的实时中间价。
- **成本**：往返 **15bps（应力）** / **10bps（温和）** 两档分别记账。
- **合并为一个候选**（family_trials=1：假设 = "AI 方向整体有 edge"）。

## 判决闸门（**全部满足**才 PASS）
1. **≥100 个已结算前向样本**（仅 `official_forward_start` 之后）。
2. 应力 `net15 均值 > 0` 且 **t ≥ Bonferroni**。
3. **AI_IC > 0**（`direction × 未来收益` 的均值为正 = AI 方向与未来正相关）。
4. **DSR ≥ 0.95**、**PF ≥ 1.2**。
- **Sticky KILL**：样本 ≥30 且温和成本 `net10 ≤ 0` → 判死，永不复活。

## token 控制
- **每标的每 60min 至多调 1 次 AI**（冷却）；AI 未配置则**一次不调**。
- **无人值守(4h 计划任务)默认不记录**，仅当 `AI_FORWARD_AUTO=1`（GUI 可勾选）才自动调 AI；
  关闭时计划任务**只做免费结算**（结算用实时价，不调 AI）。

## 运行
- **exe 内**：`多周期研究`页 → AI 前向验证区 → ⬇采集一次 / ⚖评估判决 / 📊查看进度；
  控制台 `🚀一键累计数据` 也会触发一次记录。
- **计划任务/CLI**：`python ai_forward_research/scripts/run_ai_forward.py --mode auto|evaluate|status`。
- 工件存 `RESEARCH_DATA_DIR`（或默认程序同级 `data/ai_forward/`）：哈希链 open/resolved + MANIFEST + DEAD。

诚实预期：AI 读盘口/价给方向，多半与未来无关 → 长期 PENDING/KILL。**这就是要测的。**
