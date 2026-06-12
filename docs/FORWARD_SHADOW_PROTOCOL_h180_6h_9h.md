# Forward Shadow Test 预登记协议 — 180min / 6h / 9h 候选

> **性质:预登记(pre-registration)。** 本文件在**看到任何 forward 数据之前**冻结全部候选、模型工件、入场/出场/成本规则与 PASS/KILL 判据。一经 commit + 时间戳,**规则不得修改,只能 append 结果**。其唯一目的是:防止 forward 阶段重蹈多重检验/cherry-pick 的覆辙。
> 预登记日期:2026-06-12。分支:`feat/v2-daily-orthogonal`。候选来源:`btc_single_asset_research/reports/{h45_180_deep, h4_12_deep_funding}`。

---

## 0. 为什么要这份协议(先承认先验)

150 天历史上实际尝试了 **15 horizon × 9 model ≈ 135 cells × ~6 置信桶 ≈ 810 次**。在 810 次尝试下,**纯随机(无 edge)的期望最大 t ≈ 3.66**;观测到的最佳是 180min 的 **t10=1.95**,**比随机噪声的期望最大还低**。因此:

- **这三个候选的诚实先验都很低**;9h 更是 **AUC=0.483<0.5、IC=−0.036(反指)**,纳入它是为了**证伪**,不是因为它像 edge。
- **forward test 是判别器,不是优化器。** 它最可能的、也是完全可接受的结果是 **0 个候选通过 = 干净关门**。本协议的成功定义 = "得到一个不可辩驳的裁决",而非"找到一个能交易的信号"。

---

## 1. 冻结的候选(只此三个,不得增减)

| 代号 | Horizon | bar | 模型 | 置信桶 best_top_frac | 训练期 IC | AUC | net10(t) | n_ts | 先验标注 |
|---|---|---|---|---|---|---|---|---|---|
| **A** | 180min(3h) | 5m | hist_gbm | 0.01 | +0.0259 | 0.520 | +39.7(t=1.95) | 28 | 低;t、n 均不足 |
| **B** | 360min(6h) | 5m | lightgbm | 0.02 | +0.0224 | 0.516 | +34.2(t=1.28) | 23 | 更低;t 弱 |
| **C** | 540min(9h) | 5m | mlp | 0.05 | **−0.0356** | **0.483** | +29.5(t=2.06) | 62 | **证伪对象**;训练 IC 负、AUC<0.5 |

---

## 2. 冻结的模型工件(freeze 阶段,先做一次,之后只读不改)

对 A/B/C 各用**当前 150 天**训练并**序列化保存**到 `btc_single_asset_research/frozen/{A,B,C}/`:

1. `model.pkl` — 训练好的模型对象(不再 refit)。
2. `feature_list.json` — 该 horizon 折内**最终入选的特征名**(冻结;forward 不再重选特征)。
3. `preprocess.pkl` — StandardScaler(若该模型用)+ 各特征的 median 填充值(用训练期 median,forward 不重算)。
4. `conf_threshold.json` — **关键**:训练期 `|p−0.5|` 的 `(1 − best_top_frac)` 分位点,存为**绝对阈值** `tau`。forward 期"高置信"= `|p−0.5| ≥ tau`(用这个冻结绝对值,**禁止**用 forward 自己的分位点重新切桶)。
5. `train_ref.json` — 训练期参考:`ic_sign`(A/B 为 +、C 为 −)、`net10_bps`、`t10`、`n_ts`。

> 任一工件缺失 → 该候选不进入 forward(不得临时重训替代)。

---

## 3. 冻结的入场 / 持仓 / 出场 / 成本规则

- **特征管线**:与训练**同一套** PIT 管线(5m candles + OKX rubik OI/taker/LSR + CoinMetrics + DVOL/external)。每个 5m bar **收盘后**,只用该时刻**已确认**的值算特征(链上滞后 ≥1 发布周期、DVOL/external 按快照时戳)。
- **入场**:bar 收盘算特征 → 冻结模型打分 `p` → 若 `|p−0.5| ≥ tau`(高置信)→ 按 `side = sign(p−0.5)` 开仓(p>0.5 做多,p<0.5 做空)。低于阈值 → 不交易。
- **持仓 / 出场**:固定持有 **H 分钟**(A=180/B=360/C=540),到期 **time-exit** 平仓(与训练标签定义一致;不中途反转、不另设 TP/SL,保持与训练同口径)。
- **重叠处理**:若持仓未平时又触发同向信号 → 不加仓(每时刻至多 1 仓);反向信号 → 忽略(到期再说),记录但不交易,避免换手污染。
- **成本(主判据)**:`taker 10bps` 往返 + **持仓期实际资金费**(沿用 `--funding-hold-cost auto`)+ **×1.5 压力**。`maker 4bps` 仅作参考栏。

---

## 4. 冻结的 PASS / KILL 判据(本协议的核心)

判据**只在每个候选预登记的那一个置信桶**上计算。**禁止** forward 期换桶找好结果。

### PASS(全部满足,才算"通过 forward,进入第二段验证"——注意:不是上实盘)
1. forward `net`(taker10 + funding,扣 **×1.5** 压力后)**> 0**;
2. forward **Newey-West t > 2.0**(单候选);因三候选构成一个族,**任一通过即对它单独再做第 5 节的第二段**;
3. forward **n_ts ≥ 50**(累积到足够独立高置信样本);
4. forward 方向 **IC 与训练同号**(A/B 须为 +;C 须为 − —— C 因此基本注定不通过);
5. **无明显衰减**:forward `net ≥ 0.5 × 训练 net`。

### KILL(满足任一,立即判该候选 dead,归档)
- forward `net`(taker)**≤ 0**;或
- forward IC **与训练反号**;或
- 满 **6 周**后 `n_ts < 30`(信号过稀,不可判定 → 判"不可交易")。

---

## 5. 运行窗口、决策树、第二段

- **窗口**:固定 **6 周**;或某候选 `n_ts ≥ 50` **先到为准**触发判定。6 周后仍 `n<30` → 按 KILL 处理。**禁止**为了等好结果无限延长。
- **多重检验诚实计数**:预登记 = **3 个独立 forward 测试**。报告时注明;若按 Bonferroni(3 检验、α=0.05),单候选 t 门槛升至 **2.13**——PASS 判据 2 的 t>2.0 视为下限,最终裁决以 2.13 为准。
- **决策树**:
  - **0 个候选 PASS** → **干净关门**:把本协议 + 全部 forward 结果归档为一份负结果记录,结束。(最可能、且完全可接受的结局。)
  - **≥1 个 PASS** → **不上实盘**。对通过者进入**第二段**:(a) 一个**全新、独立**的 forward 窗口再确认;(b) **最小真钱实盘**仅用于测**真实成交质量**(maker 挂单成交率、滑点、point-in-time 延迟)。两段都过,**才**讨论资金与上线;任一不过 → 关门。

---

## 6. 反作弊条款(防的是"未来的自己")

- forward 期间 **不得**新增任何 horizon / 模型 / 置信桶 / 特征;
- **不得**用 forward 数据 refit 或微调任何工件;
- **不得**因某候选 forward 不好就"再调一下参数重测"——那等于把 forward 变成又一次 in-sample 挖掘;
- 所有 forward 结果(无论好坏)**必须**归档,**禁止**只报告好看的;
- 本协议 commit 后**只可 append 结果区**,**规则区不得修改**;如确需改规则,视为**作废本协议、开新预登记**,且旧结果不得复用。

---

## 7. 执行接口(forward 打分脚本应实现的规范)

`scripts/run_forward_shadow.py`(待实现)按下列接口运行,与 `run_btc_enhanced_research.py` 共用特征管线:

1. **freeze(once)**:对 A/B/C,用当前 150 天训练 → 落盘第 2 节五件工件到 `frozen/{A,B,C}/`。
2. **score(每个新 5m bar)**:同一 PIT 管线算特征 → load 冻结 `model`+`preprocess` 打分 `p` → 若 `|p−0.5| ≥ tau` → 追加一笔 `{ts, side, entry_px}` 到 `forward_trades_{A,B,C}.csv`。
3. **settle**:到 H 后用实际价平仓 → 算 gross → 扣 `taker10 + funding(hold) + ×1.5` → 写 `net`。
4. **weekly report**:累积 `n_ts / net / NW-t / forward_IC / 是否同号`,逐项对照第 4 节 PASS/KILL,命中 KILL 即标 dead。
5. **verdict(窗口末或 n 达标)**:按第 5 节决策树输出裁决,**append** 到本文件第 8 节。

---

## 8. 结果记录区(append-only;预登记时为空)

> 此区在 forward 运行期间逐周追加;上方规则区冻结不改。

_(待 forward 运行后填写:每周累积统计 + 最终裁决 A/B/C 各 PASS/KILL + 决策树走向。)_
