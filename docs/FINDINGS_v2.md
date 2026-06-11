# OKXB v2 投资结论 — 长周期 × 正交数据 × regime 选择性（2026-06-12）

> 本文是 v2 调查（长周期方向 + 正交数据 + regime 过滤）的**最终实证结论**，承接 v1 的 §9–12 与 [`STRATEGY_DESIGN_v2_daily_orthogonal.md`](STRATEGY_DESIGN_v2_daily_orthogonal.md) 规格。
> 全程公共数据、point-in-time、扣费、样本外、可复现。代码在分支 `feat/v2-daily-orthogonal`。

## 一句话结论
**在 ~1000U / OKX 散户费档下，15min–3天的方向交易与 regime 选择性，扣费后均无稳健净 edge。** v2 把 v1 之外**仅剩的两条路**（正交数据做日级方向 + 用 regime 改善 W/L）也实证关闭。

## 测了什么、结果如何
| 实验 | 设计 | 结果 | 关键数字 |
|---|---|---|---|
| 日级正交方向（横截面） | BTC+ETH+3 主流币，price/vol+basis+funding+DVOL/VRP+on-chain，净edge闸门，H=1d/2d/3d | **NO TRADE** | AUC~0.49，xs-IC 退化（5 币相关性高），正交特征少被选中 |
| 日级正交方向（单品种 BTC） | 同上，单品种时序模式（IC=ts_ic） | **NO TRADE** | AUC~0.47–0.52，**正交特征(on-chain/DVOL/basis)被选中但 ts_ic≈0 或微负(−0.05~−0.09)**，扣费净负 |
| regime 过滤（W/L 杠杆） | 15m 横截面反转（唯一真实薄信号）按 DVOL/VRP/funding regime 分桶，held-out 净edge | **NO EDGE** | **每个 regime 扣费后净负，连 maker(4bps) 都盖不过**；ALL net_maker=−3.3/net_taker=−9.3bps |

## 最深的一条发现（值得记住）
- **日内**：绑定约束是**成本墙**（费率 vs 振幅）。
- **日级**：成本相对数百 bp 的日波动可忽略 → 绑定约束变成**可预测性本身**（AUC~0.50）。正交数据被模型选中、却几乎不含净-of-cost 方向信息。
- **选择性**：把唯一真实的薄信号（反转）按正交 regime 分桶，**没有任何 regime 把它救成净正**。
- 三句话合起来：**不是"还没找到对的变量/模型/regime"，而是这个账户/周期结构下，方向与选择性都赚不回往返成本。** 与外部 13-Agent 核验完全一致（期权/链上是 regime 信号、非 1d–3d 方向 alpha；杠杆只放大、不创造 edge）。

## 仍未穷尽的（都需要"现在拿不到"的东西）
1. **前向累积数据**：OKX OI 历史只有 ~3 个月、逐档期权 skew/GEX 只有当下快照 → 用 cryptofeed + 每日期权快照**录几周–几月**，再用历史上拿不到的信号重跑。（数据维度真新，但慢，先验仍偏低。）
2. **资金/费档变化**：VIP1+ 需 $5M+/月量 → 费墙下移会改成本算术（1000U 够不到）。
3. **换游戏**：被动做市/maker 捕获站在费墙正确一侧，但需要 1000U/VIP0 拿不到的低延迟、队列位置、返佣档。

## 诚实底线 / 建议
- 框架按设计生效：**在研究里、用真实数据、上线前**证明了无 edge —— 省下的亏损即收益（与 README「诚实预期=打平减手续费」一致）。
- 不建议继续用这笔钱堆方向/选择性模型（已全谱关闭）。
- OKXB 的真实价值已转为**一套严谨、可复现的"负结果"方法论资产**：完整的 PIT 数据层 + 净edge闸门 + DSR/PBO + 多 Agent 对抗审计。资金/费档/数据变了再议。

## 复现
```powershell
$env:PYTHONPATH='src'
python scripts/fetch_daily_data.py            # P0 数据质量报告
python scripts/research_daily_orthogonal.py   # 日级正交方向 → NO TRADE
python scripts/research_regime_filter.py       # regime 过滤 → NO EDGE
python -m pytest tests/research/ -q            # 21 单测
```
