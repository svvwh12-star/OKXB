# 研究有效性 · 单一标准 (RESEARCH INTEGRITY — one canonical methodology)

本机有**两棵研究树**：
- 主项目 `OKXB`（加密 + 美股永续，多策略/日内/多周期）—— `src/okxb/research/`
- `codexOKXB`（美股永续残差均值回归，预登记前向）—— `codexOKXB/src/okx_stockperp_mr/`

两者**结论一致**：在已测范围内**没有找到经得起验证的 edge**（见 `docs/FINDINGS_v2.md` 与 `codexOKXB/reports/stage3_research_status.md`）。

本文件确立**唯一的研究有效性标准**，两棵树都遵守它，避免"两套并行、互相漂移"的纪律。

> ⚠️ **刻意保留的分离**：两棵树的**证据**（数据 / 预登记 / 前向 ledger / 冻结候选）必须**永远各自独立**——`codexOKXB` 的前向验证之所以可信，正因为它不被旧树的结果污染（见其 README "intentionally separate"）。本次"统一"**只统一方法与代码原语，不合并任何证据/结果**。

---

## 1. 单一方法学（两树共同遵守）

1. **预登记**：候选的 universe / horizon / 特征 / 阈值在看到 OOS/前向数据**之前**钉死。
2. **不可变性 ledger**：冻结工件 + 协议文件做 sha256 清单；前向状态用**行级哈希链**（改任何历史行即断链）。
3. **Sticky KILL**：候选一旦判死即**永不复活**成 PENDING/PASS。
4. **族级多重检验**：阈值按**真实并行候选/试验总数**做 Bonferroni；DSR 的 `trials` = 完整搜索空间，PBO 在代表性全配置上算（不只赢家）。
5. **前向验证**：只认 official forward start 之后的样本；PASS 需 **≥100 独立前向时间戳**。
6. **判决闸门（全部满足才 PASS）**：扣 15bps 应力后 `net15>0` 且 **NW-t ≥ Bonferroni 阈值**、**DSR ≥ 0.95**、**PBO ≤ 0.20**、前向 IC 同号、**无衰减**（前向 net ≥ 0.5×训练 net）、PF ≥ 1.2、回撤受控。
7. **不可用回测结果交易**；过 PASS + 真实 maker 成交质量验证 + 安全检查后才允许 demo；demo 通过才谈实盘。

## 2. 代码原语映射（同一方法，两份自洽实现）

| 原语 | 主项目 `OKXB` | `codexOKXB` | 一致性 |
|---|---|---|---|
| Bonferroni 阈值 | `research/forward_integrity.bonferroni_t`(单边) | `stats.bonferroni_z_threshold`(可双边) | 同公式 |
| **Deflated Sharpe** (Bailey-LdP) | `research/labeling.deflated_sharpe` | `stats.deflated_sharpe` | **数值一致**(diff~1e-14, 已交叉验证) |
| **PBO** (CSCV) | `research/labeling.pbo_cscv` | `stats.probability_of_backtest_overfitting` | **数值一致**(diff=0, 已交叉验证) |
| 工件哈希清单 | `forward_integrity.write_manifest/verify_manifest` | `hash_ledger.write_manifest` | 同语义 |
| 前向状态哈希链 | `forward_integrity.append_rows_hashchain/verify_hashchain` | `hash_ledger.append_hash_chain` | 同语义 |
| Sticky KILL | `forward_integrity.write_dead/read_dead`(DEAD.json) | `verdict.evaluate_forward`(already_dead) | 同语义 |
| 前向判决闸门 | 散见 runner `forward_verdict` + `daily_orthogonal.run_daily` gate | `verdict.evaluate_forward`(集中, 更干净) | codexOKXB 为更优模板 |

> **为何各自一份而非共享 import**：`codexOKXB` 是**零依赖、刻意独立**的项目；让它 import 主树会重新耦合、破坏其独立性。因此采用"同一方法、两份纯标准库实现、并以交叉验证证明数值一致"的方式——既消除**纪律漂移**（这才是真正的"两套并行"问题），又保留证据独立。

## 3. 本次对齐做了什么 (2026-06-21)

- **修复唯一的真分歧**：`codexOKXB` 的 DSR 原为**自承认的占位近似**（`deflated_probability_from_t`，从 t 值近似），现已加入**真正的 Bailey-LdP `deflated_sharpe`（走收益序列）+ CSCV `probability_of_backtest_overfitting`**，并把前向闸门脚本 `04_run_forward_once.py` 切到真 DSR。
- **交叉验证**：两树的 DSR/PBO 在同一输入上数值一致（DSR diff ~1e-14、PBO diff 0）。
- `deflated_probability_from_t` 降级为**仅 discovery 排序用**（已在 docstring 标注）。
- 测试：`codexOKXB/tests/test_core.py` 16/16；主项目 `tests/research/` 全过。

## 4. 给未来工作的硬规则
- 任一棵树新增候选 → 同步更新该树的多重检验计数（Bonferroni/DSR trials）。
- 任何"PASS"在满足 §1.6 全部闸门 + ≥100 独立前向 + 真实成交证据前，**一律视为疑似过拟合，不得投真金**。
- 不合并两树的证据/ledger；只在本文件维护"同一方法"的对应关系。
