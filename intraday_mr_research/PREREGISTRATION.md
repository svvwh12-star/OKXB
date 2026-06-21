# 预登记 · 15/30 分钟日内均值回归 (IMR)

> **定位**：纯【前向验证研究】，**只采集、不实盘**。这是用真实未来数据【证伪/证实】一个固定假设，
> 不是交易系统。过**全部闸门**（见下）才考虑 demo——按本项目两棵研究树的历史，**大概率不会过**。

## 为什么做这个（回应"控制台要不要换 15/30 分钟"）
控制台是**秒级微结构**信号。"换成 15/30 分钟"是另一种策略，不能当调参改。正确做法是**预登记一个
15/30 分钟候选、前向采集、过闸门才用**——本目录就是这件事。诚实预期：15m/30m 上扣 15bps 成本去
fade，净值多半为负 → KILL/PENDING。

## 冻结的假设（看到前向数据后**不得**修改）
- **universe**：`BTC-USDT-SWAP` / `ETH-USDT-SWAP` / `SOL-USDT-SWAP`（3 个流动性主流币）。
- **周期**：`15m` 与 `30m`。每个「标的 × 周期」= 一个候选，共 **6 个**（族级多重检验 = 6）。
- **特征**：最近一根 bar 的收益，相对**过去 96 根** bar 收益的 **z 分**。
- **规则（均值回归 fade，无自由参数）**：`z > 1.0` → **做空**；`z < -1.0` → **做多**；否则不动。
  （96 / 1.0 / 持有 1 根，都是**事先固定**的常规值，不在数据上调。）
- **持有**：1 根 bar（15m 或 30m）后按收盘平。
- **成本**：往返 **15bps（应力）** 与 **10bps（温和）** 两档分别记账。

## 判决闸门（**全部满足**才 PASS；否则 PENDING；命中即死则 KILL）
1. **≥100 个独立前向时间戳**（仅 `official_forward_start` 之后的样本）。
2. 应力净值 `net15 均值 > 0`，且 **t ≥ Bonferroni 阈值**（按 6 个候选校正）。
3. **Deflated Sharpe ≥ 0.95**（Bailey-LdP，trials=6）。
4. **PBO ≤ 0.20**（CSCV，跨 6 候选；不足 2 个有效候选时暂不计）。
5. 前向 IC 符号 == 训练期 IC 符号（方向没翻）。
6. **无衰减**：前向 `net15 ≥ 0.5 ×` 训练期 `net15`。
7. **PF ≥ 1.2**。
- **Sticky KILL**：连温和成本（10bps）下、样本 ≥30 仍 `net10 ≤ 0` → 判死，**永不复活**为 PENDING/PASS。

## 不可变性 / 防作弊
- 冻结写 `frozen/meta.json`（含 `official_forward_start_ts_ms` + 各候选训练期基准），并由
  `forward_integrity.write_manifest` 出 `MANIFEST.json`（改 meta 即断 manifest = 预登记作废）。
- 前向样本进**行级哈希链** `data/forward_append_only/imr_ledger_hashchain.csv`（改任一历史行即断链）。
- 死亡标记 `frozen/{code}/DEAD.json`（sticky）。

## 运行（开发态，公共行情免密钥）
```
cd intraday_mr_research/scripts
python run_intraday_mr.py --mode collect    # 抓最新bar→追加前向标签（首次自动冻结）
python run_intraday_mr.py --mode evaluate   # 判决
python run_intraday_mr.py --mode status     # 看进度（各候选 已采集/100）
```
已接入 `btc_single_asset_research/scripts/accrue_forward.ps1`，随 4 小时计划任务一并 collect+evaluate。

## 诚实的时间表与结论
- 15m：~100 个独立前向 ts 需**数天到数周**活跃采集；30m 约**翻倍**。
- 采集期间**不许看一眼就改规则**。最可能结果是长期 PENDING 或 KILL——**这就是诚实答案，省下本金**。
- 任何 PASS 在满足全部闸门 + 真实成交质量验证前，仍**一律视为疑似过拟合，不得投真金**。
