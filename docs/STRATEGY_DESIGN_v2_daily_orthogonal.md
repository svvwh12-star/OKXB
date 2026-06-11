# OKXB 策略设计 v2.1 — 长周期 × 正交数据 · 横周期自适应方向模型（构建规格 / Spec）

> 状态：**已与用户讨论、参数锁定、并经两轮多 Agent 深度核验（事实核验 + 开源生态调研）的构建蓝图**。下一步 writing-plans（先 P0+P1）。
> 日期：2026-06-11（v2.1 纳入开源调研结论与横周期自适应设计）。承接 [`STRATEGY_DESIGN.md`](STRATEGY_DESIGN.md)（v1 实证关闭记录）、[`RESEARCH_BRIEF.md`](RESEARCH_BRIEF.md)（OKX 事实基线）。
> 研究阶段全部公共数据、不读密钥、不下单。

---

## 0. 定位、诚实预期、成功标准（先于一切拟合）

### 0.1 为什么是这条路
v1 已多 Agent 审计**实证关闭**六门（详见 STRATEGY_DESIGN §9–12），结论：**「变量/模型不是瓶颈，成本墙是」**——15–30min 约束是「方向可预测性 vs 成本」。外部 13-Agent 核验逐条证实：Regular 档 2/5bps（2026-04 调费未动本档）、真实往返 ~4/7/10bps、压力 15bps；当前实现波动率多年低位（周化 ~17%）使日内墙**更高**。
**v2 逻辑**：正交数据（资金费/基差/OI、期权 IV/skew/DVOL、链上、宏观）的方向信息**活在日—周级**。把周期放长去匹配数据自然周期 → 成本相对振幅可忽略 → 回归纯可预测性问题。这正面修复 v1 的「周期错配」，也符合项目自身结论「散户 edge = 更长周期 + 选择性执行 + 波动定仓，非速度」。

### 0.2 诚实先验
- 日—周方向非处女地（v1 测标准价量到 4h 无 edge、资金费 carry 多日持有净负）。
- **真正未测** = 期权-implied + 链上 + 宏观作为**日级方向/regime 输入** + 横周期自适应。
- 先验：**显著好于日内**（成本摊薄、信号-数据周期匹配），但最可能结局仍是「弱信号 → 严闸门 → 多数 NO TRADE」。
- **赢 = 真 edge 或一次干净可复现的关门。** 自欺是唯一失败。

### 0.3 成功标准 / Kill-Gate（拟合前写死）
候选 (band, H, 特征组, 模型) **须同时满足**才进 Stage 2 / 影子盘：
1. 净edge 在 **cost×1.5** 下为正（含 funding 持有成本）；
2. **≥2 相邻置信档** Newey-West t>2（组合层/聚类修正）；
3. **DSR ≥ 0.95 且 PBO ≤ 0.5**，且 **DSR 的 trial 数按整个 band×H×模型×特征组族级计数**（§6.3）；
4. 在**全程未触碰的 held-out 末段**（**含专门的 2024–2026 切片**）仍为正；
5. 跨 **≥2 个 regime/子时段**一致；
6. **跑赢预登记基准 CMOM**（2 周横截面动量）才算「值得多出的换手」。
否则 → **NO TRADE**，写 `dist/daily/verdict_*.txt` 记录失败原因。

---

## 1. 已锁定决策（用户确认 2026-06-11）
| 维度 | 决策 |
|---|---|
| 方向 | 长周期 × 正交数据方向模型（研究优先，两阶段） |
| 资金/定位 | ~1000U，当学费/研究，可全损；**数据全免费** |
| 周期/杠杆 | 放长周期 + 低杠杆；目标=持续净正期望 |
| 建模范围 | BTC+ETH 为主 + 6–10 高流动性主流币横截面（cross_sectional） |
| 数据范围 | 一次性全纳入四类：OKX / Deribit / Coin Metrics 链上 / 宏观 |
| **周期网格** | **15min→1w 横周期自适应、角色标签化**（见 §10），非单一模型扫全程 |
| 执行 | 研究只读 → alert-first 半自动 → demo → 小额实盘（不跳级） |

---

## 2. 数据层（全免费；新增模块挂 `src/okxb/research/` 或 `data-engine/`）

### 2.1 四源 + 采纳的开源工具
| 源 | 取什么 | 周期 | 工具（采纳） | 模块 |
|---|---|---|---|---|
| OKX | candles **+12H/1D/1W bar**、funding（有）、**+OI 历史** | 日/小时 | 复用现有 OKX 客户端；OI 历史**照搬 ccxt/python-okx 的 endpoint 签名进现有客户端**（不引 SDK） | 扩展 `candle_data.py` |
| OKX/Deribit 实时 L2 | L2/trades/funding/OI/爆仓**录制**（仅 band a 执行验证需要） | tick | **`cryptofeed`**（XFree86，归一化多所→Parquet，校验和开）独立录制进程 | 新 `recorder` 进程 |
| Deribit 期权 | DVOL、IV smile、25Δ RR、期限结构、逐档 OI/gamma | 1s→日 | **Deribit 公共 REST/WS（免鉴权；建议申请免费 key 防限流）**，~5 个方法，无第三方 wrapper | 新 `deribit_data.py` |
| 链上 | 交易所净流、稳定币流、活跃地址、MVRV/realized-cap、NVT | 日 | **`coinmetrics-api-client` community 档**（MIT 客户端，CC-BY-NC 数据，免 key） | 新 `onchain_data.py` |
| 宏观 | VIXCLS、DTWEXBGS(广义美元≈DXY)、DGS10/2 | 日 | **FRED + ALFRED vintages**（`pandas-datareader`，point-in-time） | 新 `macro_data.py` |

### 2.2 反前视铁律（PIT，头号敌人）
- 每个**链上/宏观**序列存 `(asof_date, value)`，**滞后 ≥1 个完整发布周期**才可触碰标签；链上理想上只用 ≥48h 稳定值；宏观用 ALFRED `realtime_end=bar_date` 取**当时 vintage**（非最新修订）。
- 链上为日级 → **CONFIG 禁止进入 <1d 标签**。
- Deribit/FRED 日级在多根 bar 收盘后才更新 → **固定 wall-clock 快照**、存时间戳、只标注**严格未来** bar。
- **加一个泄漏单元测试**：任一特征 `asof` 晚于其标签决策 bar 即失败。

---

## 3. 特征层（`daily_orthogonal_bank.py`；复用 `feature_lab.inst_bank`）

四正交组 + 按 band 的特征映射（§10），每特征 PIT 打标，输出 `feature_manifest.json`：
- **A 价量/波动（复用）**：多窗动量/反转、已实现波动 regime、区间位置、量 z、ATR、Donchian/支撑阻力/OBV/VWAP/BOLL。
- **B 永续结构（OKX）**：funding（level/Δ/z/距结算）、OI（level/Δ/funding-OI 背离）、basis（perp-spot/perp-index/z/Δ）、taker 买卖比。**资金费按方向/反转用，不做 delta 中性 carry**（§11）。
- **C 期权-implied（Deribit）**：DVOL（level/Δ/加速）、**VRP = DVOL²−RV²**、IV 期限斜率、**25Δ RR（skew）**、IV/RV、逐档 net-GEX/gamma-flip 距离。**作 regime/模型选择器，不当原始 alpha**。
- **D 链上+宏观**：稳定币净流 z、活跃地址/新增用户增长、MVRV/realized/NVT（慢 regime）；VIX/美元/收益率（risk-on/off）。
- **跨资产/日历**：BTC/ETH/ETHBTC、横截面均值/分位、星期、月末、宏观事件窗。
- **采纳的特征配方（reference，重写干净后过门）**：CTREND 多窗趋势聚合（与 2 周动量共线聚类，β≈0.79）；稳定币净流符号（arXiv 2411.06327）；VRP/skew/term（arXiv 2410.15195）；GEX/MVRV 公式（checkonchain/GEX 仓库**仅作公式源**）。
- **候选特征生成器（可选）**：`aeon` Catch22（首选，22 个）或 `tsfresh`（精选子集）对 funding/OI/basis/DVOL **轨迹形状**做窗口特征——**仅作候选**，窗口止于决策 bar，折内计算，**必过现有折内选 + NW/DSR/PBO 门**才入模。

---

## 4. 标签层（复用 `labeling.py`）
- 方向：未来 H 符号；三重障碍：TP/SL=k×(日级波动 Yang-Zhang/ATR)、时间障碍=H → `P(先TP/先SL/超时)`。
- H ∈ §10 网格；验证用 **embargo=H** 净化重叠标签。

---

## 5. 建模层（复用 zoo + `calibrator`，并按样本量约束模型族）

### 5.1 模型族按**有效样本量**钳制（核心自适应规则）
- **band (a) 15m–1h（海量样本）**：树/GBDT（LightGBM/HistGBM/ExtraTrees）+ 正则线性基线。
- **band (b) 2h–8h（过渡）**：ElasticNet/Ridge 为主，树仅作**必须 OOS 跑赢线性基线**的挑战者。
- **band (c)(d)(e) ≥12h（少样本低 SNR）**：**收缩/贝叶斯线性（ElasticNet/Ridge/Bayesian-ridge，最稀疏端可选 PyMC 收缩）为主；树在 config 里标记 INELIGIBLE**（树不能外推、在数十个独立周观测上必过拟合）。
- **明确排除**：NN、扩散作方向/alpha（核验证实是概率预测/场景生成工具）。扩散唯一合法用途=合成数据稳健性压力测试（可选研究线）。

### 5.2 概率层
- isotonic/Platt 校准（有）→ 校准 P。
- **新增 conformal 弃权层（`MAPIE`，BSD-3）**：分布无关的「只在足够确定时才做」门——校准**必须在折内 purged/embargoed 严格 OOS 窗**上做（时序破坏可交换性，禁打乱）。**交易门 = 预测集为单元素 / 收益区间不含 0**；实测覆盖率显著低于名义 → regime 漂移，门收紧（喂熔断器）。（`crepes` 为备选：给 P(收益>成本) 的预测 CDF + 漂移鞅报警，二选一。）
- meta-gate（有）：仅在一级有 edge 时决定「做/不做+下注」。

---

## 6. 验证层（`daily_workflow.py`；承重墙）

### 6.1 切分
purged walk-forward + **embargo=H**（折内选变量、折内算校准集/协方差/JM 解码，无泄漏）；横截面组合层 IC + **Newey-West/聚类 t**；**held-out 末段全程不参与研究**，含**专门 2024–2026 切片**（抓资金费 carry 之死）。

### 6.2 净edge 闸门（复用 `pro_model_workflow` gate）
扣 maker/taker/stress（4/10/15bps）**+ funding 持有成本/收益**；primary_ic>0.01 ∧ ≥2 档 net>0,t>2 ∧ ≥1 档过 stress。

### 6.3 多重检验控制（因宽网格强化 —— make-or-break）
- **DSR/PBO 在「族级」做**：跨整个 **band×H×模型×特征组** 网格统一 deflate，**按实际尝试配置数缩** Sharpe，**不是逐格**。PBO via CSCV（有）。
- **预登记**（拟合前写入 `hypotheses.jsonl`）：候选 band（仅 (c)(d)）、每个候选须跑赢的基准（**CMOM 2 周动量**）、OOS 窗（含 2024–2026）。**禁止事后把 control band 提升为 candidate**。
- **共线性聚类**相关因子（趋势~0.79 动量）→ 一族不重复计为多个「独立发现」。
- **毕业** = walk-forward 与 held-out **都过** ∧ ≥2 相邻档 ∧ ≥2 子时段 ∧ 跑赢 CMOM。单格/单时段/仅 maker 一律判多重检验噪声（v1 多次抓到）。

### 6.4 样本与数据范围
横截面 ~10 币高度相关（ρ≈0.7–0.9）→ 有效独立样本 ≈ 时间点数（日 ~1200、周 ~150–200），**非**币数×时间点；用于去噪/稳健，不当 10× 独立样本。周级仅作 §10 的 **基准锚**，最高 DSR 门。拉 3–4 年（OKX/Deribit ~2021 起；CM community 多年），缓存 `dist/daily/`。

---

## 7. 执行 / 风控层（复用 RiskEngine / execution）
- **低杠杆 1–3x、≤ 日级 rebalance、低换手**；仓位=风险预算÷止损距离（有）；干净 bracket（TP/SL/到H）、reduce-only；数据老化/API 异常→close-only/kill-switch（有）。
- **funding 持有成本**：多日持仓跨多次结算的资金费收支计入每笔净edge。
- **执行成本现实性闸门（gap #1）**：采纳 **`hftbacktest`（MIT）** 作高保真 FIFO 队列+延迟回测，**仅验证 band (a) / 入出场成本**，由 cryptofeed 录制带喂入（需写 OKX L2 录制+转换器），用 **Tardis 免费每月首日 tick** 校准成交模型；funding PnL 自建。**不作日级研究主引擎**。`NautilusTrader`（OKX 原生 perp+funding、回测=实盘一致、LGPL）列为**未来收敛目标**（横截面策略上线后再议），主引擎只选一个。
- **横截面定仓（未来阶段）**：横截面上线后，`skfolio`（BSD-3，成本感知 HRP/风险预算/CVaR）置于信号之后、RiskEngine 之前作**权重提议器**（折内算协方差），**RiskEngine 仍是唯一总闸**。
- **上线阶梯**：研究只读 → alert-first 半自动（日级信号推送人工确认）→ demo 7–14d → 小额实盘（单笔1U/名义≤800U/回撤硬停30U）→ 正常参数。

---

## 8. 工作流框架 + Agents 能力 + 工具调用

研究阶段用 **Workflow 编排 + 项目内新 Python 模块**，阶段间 kill-gate，沿用项目招牌「4-Agent 对抗审计」。

```
Orchestrator/Registrar ── 预登记假设(hypotheses.jsonl)·禁止事后挑 band/周期/特征
   │
 Data Agent ─► Feature Agent ─► Selection Agent ─► Modeling Agent ─► Validation Agent
 (四源·PIT·     (正交bank·按band   (折内选·跨子段     (按样本量选模型族·    (purged WF·族级DSR/PBO·
  录制·质量)     特征映射·manifest) 稳定性)            校准·conformal弃权·   held-out含2024-26·跑赢CMOM
                                                     meta-gate)             → PASS/NO-TRADE)
                                                          │
                       对抗审计组（并行 skeptic）：前视泄漏 · PIT违规 · 族级多重检验 · 成本/funding低估
                                                          │
                       Risk/Execution Agent（仅 PASS）：风险预算定仓·hftbacktest成本门·alert-first
```

| Agent | 职责 | 工具调用 |
|---|---|---|
| Orchestrator/Registrar | 预登记假设、记录每次实验、禁止事后挑选 | Workflow、`hypotheses.jsonl` |
| Data Agent | 四源拉取对齐、PIT 打标、cryptofeed 录制、数据质量报告 | `candle_data`/`deribit_data`/`onchain_data`/`macro_data`、cryptofeed、httpx |
| Feature Agent | 四组正交特征、按 band 特征映射、PIT manifest、（可选 Catch22 候选生成） | pandas/numpy、`daily_orthogonal_bank`、aeon(可选) |
| Selection Agent | 相关聚类、ElasticNet、RF/GBDT 重要性、单变量 IC、跨子段稳定 | sklearn、`feature_lab` 原语、(可选)shap |
| Modeling Agent | 按样本量选模型族、校准、**conformal 弃权(MAPIE)**、meta-gate | sklearn/lightgbm、`calibrator`、MAPIE |
| Validation Agent | purged WF+embargo、NW-t、净edge gate、**族级 DSR/PBO**、held-out(含2024-26)、CMOM 基准 | `pro_model_workflow`/`labeling`/`validate` |
| 对抗审计组 | 独立猎杀前视/survivorship/族级多重检验/PIT/成本低估 | Agent/Workflow 并行 skeptic |
| Risk/Execution Agent | 仅消费 PASS；风险预算定仓、**hftbacktest 成本现实性门**、alert-first | 现有 `risk`/`execution`、hftbacktest |

---

## 9. 需安装 / 使用的工具

- **采纳（免费、维护中、许可干净）**：`cryptofeed`（XFree86）、`coinmetrics-api-client`（MIT）、`MAPIE`（BSD-3）、`pandas-datareader`（FRED/ALFRED）。Deribit/FRED 走 httpx，无需 SDK。
- **采纳但延后/独立**：`hftbacktest`（MIT，执行成本门，P4 阶段才装）。
- **reference（按需，多为重写公式/校准/未来阶段）**：`NautilusTrader`、`tardis-dev`、`skfolio`、`crepes`、`aeon`(Catch22)/`tsfresh`、`PyMC`(最稀疏端可选)。
- **vendor-and-pin（不当上游依赖）**：`jumpmodels`（regime 检测的对工具，但单作者停更 → 抄源钉版自管；在线解码、λ 折内调、过 DSR/PBO 再信）。
- **不装/AVOID**：`MlFinLab`（冗余+收费）、Coinglass/Amberdata（收费，免费数据可复算）、`river`（与批量 WF 冲突）、`pandas-ta`（冗余、低正交、抬 PBO）、OHLCV-DL 全簇（theater）、bar/向量化回测器填 gap#1（不建模队列/延迟）、xgboost/catboost（sklearn+lightgbm 已足）。
- **许可注意**：Coin Metrics **数据 CC-BY-NC 4.0**（个人零售可，禁商用/对外资金/数据再分发）。

---

## 10. 横周期网格设计（角色标签化 · 自适应 —— v2.1 核心）

> 结论：在**一个框架**里测 15min→1w **可行且更严谨**，但**只能作横周期自适应设计**（每 band 不同特征/标签/模型/成本模型 + 角色标签）。**单一模型扫全程 = 无效 theater**。

| Band | 周期 | 角色 | 变量（自适应） | 模型族 | 要点 |
|---|---|---|---|---|---|
| (a) | 15m/30m/1h | **control 基准/成本墙演示** | 仅微观结构（OBI/OFI/价差/深度/VWAP-mid+z） | 树/GBDT | **非盈利候选**。订单流 alpha 活在秒~~1天；此处 5bps taker vs AUC0.52→墙闭。用来**证明墙**、验证 pivot；不再调微观结构妄想盖过 5bps。任何 maker/限价成交主张须过 hftbacktest 队列/延迟门 |
| (b) | 2h/4h/8h | **bridge** | 资金费 level/z/动量、OI-Δ、funding-OI 背离（极值反转）；微观结构**作候选带入**；无日级链上 | ElasticNet/Ridge 主，树须 OOS 跑赢线性才用 | 测 funding/OI 定位信号是否开始摊薄 ~10bps 往返。资金费**方向/条件用**，非静态 carry。需补 OKX OI 历史 |
| (c) | 12h/1d | **candidate（主）** | 资金费 z+OI 背离（24–72h 反转）、日级链上（净流 z/新增用户/MVRV，滞后）、期权 regime（DVOL/RR25/VRP/term 作选择器）、2 周横截面动量+趋势 | 收缩/贝叶斯线性为主，树 DE-emphasized | **成本墙真正打开处**（~260bps/周毛动量 → ~10bps 往返仅占 4–8%）。CMOM 是候选须跑赢的**基准**；趋势与动量共线聚类 |
| (d) | 2d/3d/4d | **candidate** | 横截面 2 周动量(CMOM)、price-to-new-address(CVALUE)、链上采纳/新增、MVRV/NUPL/NVT 慢 regime、DVOL/skew/VRP；funding 降权 | 收缩/贝叶斯（树 INELIGIBLE） | **有效样本最少→过拟合最高→最激进收缩+最严 DSR/PBO**。链上估值是修订风险大户，存 as-reported+时间戳 |
| (e) | 1w | **control 基准锚** | 同 (d) 因子族 | 收缩/贝叶斯（树 INELIGIBLE） | **非独立候选**——周级是 CMOM/JFQA 趋势/CVALUE 被同行评审验证的频率，锚定 (c)(d) 须跑赢的标尺 |

**一处重要纠正（核验）**：**不要 a-priori 把 OFI/聚合订单流逐出 (c)(d)**——最强的加密订单流证据（J. Empirical Finance 2026）发现**聚合订单流可预测性在周级反而增强**，故把它作**被检验候选**（过净edge 门），而非先验排除。

---

## 11. Alpha 范围与证伪（采纳证据 + 禁建清单）
- **去 scope**：delta 中性资金费 **carry**（Sharpe 6.45→4.06→2025 负，已死、拥挤）。资金费仅作**方向/反转特征**，且**必须在 2024–2026 OOS 上重验**——只在 2024 前有效则 kill。
- **采纳为候选+基准**：**CMOM（2 周动量，+2.1%/周 t3.70）+ JFQA 趋势** 作每个候选须跑赢的**控制基准**；**CVALUE（price-to-new-address）**、链上新增用户增长作横截面特征。
- **禁建（死因子，post-2020）**：12/24 周长动量、size、vol、volume；以及 OHLCV-DL 价格预测簇、generic TA 堆叠。
- 所有「正交」期权/链上特征**不豁免**——同样过 NW-t + 族级 DSR/PBO。

---

## 12. 交付物清单
1. 四源采集+录制模块（`deribit_data`/`onchain_data`/`macro_data` + 扩展 `candle_data` 加 12H/1D/1W/OI 历史 + cryptofeed 录制进程）。
2. 数据质量 + PIT 报告 + **泄漏单元测试**。
3. 正交特征库 + 按 band 特征映射 + `feature_manifest.json`。
4. 日级标签模块（6 周期/5 band）。
5. `daily_workflow.py`（横周期自适应、族级 DSR/PBO、conformal 弃权、CMOM 基准、held-out 含 2024-26）+ 脚本 `scripts/research_daily_orthogonal.py`。
6. 预登记台账 `dist/daily/hypotheses.jsonl`。
7. 横周期报告 `dist/daily/daily_workflow_report.txt` + 对抗审计纪要。
8. （P4）hftbacktest 执行成本门 + OKX L2 录制/转换器 + Tardis 校准。
9. 风控参数表 + alert-first 半自动信号（仅 PASS）。

---

## 13. 实施顺序（阶段闸门，不跳级）
1. **P0 数据**：四源拉取对齐 + PIT + 3–4 年缓存 + 泄漏测试 + 数据质量报告。**闸门**：覆盖/缺失/PIT 自检通过。
2. **P1 Stage 1（有无净edge）**：建 bank → 按 band 特征/模型 → 折内选 → 净edge gate（先全特征组、candidate band c/d 为主）。**闸门**：若全 band×全模型扣费后 0 格过 → 出 NO TRADE，**诚实关门**。
3. **P2 严格验证**：对弱正格上**族级 DSR/PBO** + held-out(含2024-26) + 跨子段 + cost×1.5 + conformal 弃权 + 对抗审计 + 跑赢 CMOM。**闸门**：过 §0.3 六条进 P3。
4. **P3 Stage 2（regime 选择性）**：用 C/D 期权/链上作门控改善 W/L，重验。
5. **P4 执行现实性 + 影子盘**：hftbacktest 成本门 + demo/alert-first ≥2 周。**闸门**：PF≥1.25、回撤受控、校准/覆盖稳定。
6. **P5 小额实盘**：按 README 阶梯。

> **首个实现计划只覆盖 P0+P1**（P1 是硬闸门：若无净edge 即停、不建 P2–P5）。

---

## 14. 风险与否决条件（诚实止损）
- **最大风险=自欺**：所有结论须样本外、扣费（含 funding）、PIT、可成交、跑赢 CMOM。
- **宽网格+全数据 → 族级多重检验是头号敌人**：族级 DSR deflate + held-out(含2024-26) 是唯一裁判；单格/单时段/仅 maker 不信；禁事后提升 control→candidate。
- **链上 PIT 修订陷阱**：只用滞后已确认值 + 泄漏单元测试。
- **样本稀缺**（(d)(e) 尤甚）：最激进收缩、树 ineligible、最高 DSR 门；周级仅作基准锚。
- **资金费之死**：carry 去 scope；任何 funding 信号过 2024–2026 OOS。
- **否决**：若 P1/P2 全 band×全 regime 扣费+PIT 后无正 edge → **不上线**。出路：Stage 2 regime 门控 / 只在高波动做 / 承认当前结构不可行（关门即赢）。

---

## 附 A：开源生态裁决（2026-06 调研，13-Agent 高可信）
**ADOPT（6）**：cryptofeed(实时L2带) · Deribit 公共API(期权块) · coinmetrics-api-client+FRED/ALFRED(链上+宏观,PIT) · hftbacktest(执行成本门) · MAPIE(conformal 弃权)。
**REFERENCE**：NautilusTrader(未来收敛) · Tardis(免费校准 oracle) · skfolio(未来横截面定仓) · crepes(弃权备选) · aeon Catch22/tsfresh(候选生成) · CTREND(JFQA 趋势) · 稳定币净流(2411.06327) · 资金费衰减(2510.14435,警示) · checkonchain/GEX(公式源) · VRP/skew(2410.15195)。
**AVOID**：MlFinLab(冗余+收费) · Coinglass/Amberdata(收费) · river · pandas-ta · OHLCV-DL 簇 · bar/向量化回测器(gap#1) · xgboost/catboost。

## 附 B：与现有代码复用映射
| 需要 | 已有可复用 | 新增/改造 |
|---|---|---|
| OKX 历史/funding | `candle_data.py` | +12H/1D/1W bar、+OI 历史 |
| 正交 bank | `feature_lab.inst_bank`、`pro_model_workflow.build_augmented_bank` | +Deribit/链上/宏观组、按 band 映射、PIT manifest |
| 折内选 | `feature_lab.select_features` | 跨子段稳定、(可选)Catch22 候选 |
| 模型 zoo+gate | `pro_model_workflow` | 按样本量钳模型族、日级 H、held-out、funding 入成本、族级 DSR |
| 标签 | `labeling.py`（三重障碍/DSR/PBO） | 5 band/6 周期 |
| 校准 | `calibrator.py`（isotonic/Platt/DSR/PBO/go-live） | 接日级 + conformal 弃权(MAPIE) |
| 执行/风控 | `risk`/`execution`（bracket/reduce-only/kill-switch/风险预算） | alert-first、funding 持有成本、hftbacktest 成本门、(未来)skfolio 提议器 |
