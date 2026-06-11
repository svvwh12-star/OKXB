# OKXB 多/空综合分 (Composite Score) 算法完整规格 — v3.6 (对称式 Stage A' + 四轮专家评审; 含多-agent源码核验)

> 本文逐字转写自**当前源码 (2026-06-08, Stage A + Stage A' 对称重构后)**,供独立专家评审。
> 所有 `*_pct` 等价于"小数分数"(0.002 = 0.2%)。
> 涉及文件:`signal/composite.py`、`features/engine.py`、`features/microstructure.py`、
> `features/volatility.py`、`core/models.py`、`strategies/base.py`、`strategies/hfm80.py`、`execution/executor.py`。
>
> **与上一版 (v2) 的关键差异**(便于对照你之前评审的版本):
> 1. **评分改为对称式**:`多分 = 50×(1+方向)`、`空分 = 100−多分`,**多+空 恒 = 100、50=中性**。
>    不再是"质量项给多空各贡献 ~28.6 基底"的混轴写法。
> 2. **方向与可交易性彻底解耦**:可交易性 (tradability) **不再乘进分数**,而是作为**独立 [0,1] 门槛**由策略单独判定。
> 3. OFI 改为**经典 Cont L1 + 深度归一化**(去掉与成交量的重复计入);趋势改为**波动归一化**(去掉 ×4000 魔法增益);
>    价格序列改用 **micro-price**;z-score 改为 **中位数/MAD 稳健版**;可交易性改为**逐标的"相对中位数倍数"门槛 + EMA**(v3.3,见 §3.3)。

---

## 0. 总览与刷新机制

- **刷新频率**:主循环每 `COMPUTE_S = 0.5 秒` 对**每个标的**执行一次:
  `FeatureEngine.compute(inst) → CompositeScorer.score(fs) → 产出 long_score / short_score + tradability`。
- **数据源**(实时公共 WebSocket,即使虚拟盘也用实盘公共行情):
  - 订单簿:`books5`(5 档快照,约 100ms)。
  - 最优买卖 BBO:`bbo-tbt` 或由订单簿派生。
  - 逐笔成交 `trades`:**含主动方向**(OKX 直接给 taker side,无需 Lee-Ready 推断)。
  - **micro-price**(下方 §1.0)取代裸中间价,作为价格序列基准。
- **输出**:
  - `long_score` / `short_score`:方向信号分,**多 + 空 = 100**;50 = 中性,>50 偏多,<50 偏空。
  - `tradability`:可交易性 ∈ [0,1],与方向**无关**,衡量"此刻这个行情值不值得做"。
  - 另有平滑版 `long_score_s / short_score_s`(入场门槛 + 面板用)与 `dir_latch`(方向闩锁)。

---

## 1. 原始因子 (raw factors)

每标的维护一个价格历史 deque(最多 600 个样本,记录 `(时间ms, micro-price)`)。

### 1.0 加权中间价 weighted-mid(micro-price 近似)— 价格序列基准
```
weighted_mid = (bid_px × ask_sz + ask_px × bid_sz) / (bid_sz + ask_sz)
```
- 按对侧挂量加权,比裸中间价 `(bid+ask)/2` 更贴近成交概率重心,减少单边报价闪动噪声。
- **诚实命名(请专家留意)**:这是 **size-weighted mid**,只是 Stoikov micro-price 的**一阶近似/proxy**,
  **不是**完整的 Stoikov micro-price(后者是 `mid + g(imbalance_bucket, spread_bucket)`,其中 `g` 需用历史事件估计
  `E[未来 mid 变化 | imbalance,spread]`)。完整版属于 Stage B(需数据拟合 `g`)。代码属性名沿用 `microprice`,语义即此 proxy。
- **趋势(§1.4)与已实现波动(§1.5)都建立在该加权中间价序列上**。

### 1.1 盘口失衡 OBI
```
bid_sz = Σ 前5档买量;  ask_sz = Σ 前5档卖量
obi_5  = (bid_sz − ask_sz) / (bid_sz + ask_sz)        ∈ [−1, 1]
```
- 5 档**原始量直接相加**,无按距盘口远近加权。之后做稳健 z-score(§2)→ `obi_5_z`。

### 1.2 订单流失衡 OFI(**经典 Cont-Kukanov-Stoikov 2014,L1,深度归一化**)
**逐事件累计(v3.1 修复,专家指出的关键点)**:网关在【每一次盘口更新】(books5≈100ms;若订阅 bbo-tbt 则≈10ms)都计算一次下面的增量并**累加**,0.5s 计算拍时一次性取走累计值(`gateway.take_ofi`)→ OFI 反映该拍内的**全部**最优价变动,而非仅首尾两点之差。
设 `prev` = 上一次更新的 BBO,`cur` = 本次更新的 BBO:
```
# 最优买价一侧
if   cur.bid_px > prev.bid_px:  d_bid =  cur.bid_sz      # 价升 → 计入新买量
elif cur.bid_px < prev.bid_px:  d_bid = −prev.bid_sz     # 价降 → 减去旧买量
else:                           d_bid =  cur.bid_sz − prev.bid_sz   # 价平 → 量差
# 最优卖价一侧
if   cur.ask_px < prev.ask_px:  d_ask =  cur.ask_sz
elif cur.ask_px > prev.ask_px:  d_ask = −prev.ask_sz
else:                           d_ask =  cur.ask_sz − prev.ask_sz

e             = d_bid − d_ask
depth         = (cur.bid_sz + cur.ask_sz) / 2      # 本次更新的 L1 平均深度
ofi_increment = e / depth   (depth>0; 否则 = e)     # 单次更新增量(已深度归一化)
# 0.5s 拍内:  ofi = Σ 各次 ofi_increment           # 逐事件累加; take_ofi 取走并清零
```
- **已修正的三个问题**:(a) **不再把主动成交量 `aggr_buy−aggr_sell` 加进来**(避免与 §1.3 trade_imbalance 重复计入);(b) **按 L1 深度归一化**(OFI 斜率与深度成反比,跨标的/时段可比);(c) **v3.1:改为逐事件累计**(不再是"仅 0.5s 首尾两点快照差" —— 这是专家指出的核心点)。
- **粒度说明**:默认 `scan_channels=[books5, trades]` → 逐 books5 快照(≈100ms,每拍约 5 次)累计;若把 `bbo-tbt` 加入 `scan_channels` → 逐 BBO(≈10ms)累计,更接近真正 tick 级 CKS OFI(代价 WS 负载更高)。免费档无法做全深度 L2 逐笔(需 VIP4)。
- 累计值再做稳健 z-score(§2)→ `ofi_z`。

### 1.3 主动成交失衡 Trade Imbalance(近 3 秒)
```
buy  = Σ 近3秒主买成交量;  sell = Σ 近3秒主卖成交量
trade_imbalance_3s = (buy − sell) / (buy + sell)      ∈ [−1, 1]   (不做z, 直接用)
```

### 1.4 趋势/收益(5s / 15s / 60s,基于 micro-price)
```
mid_return(h) = now_mp / past_mp − 1
其中 past_mp = 历史里时间 ≤ (now_ts − h) 的【最近一个】样本(源码 reversed 迭代取 latest≤目标时间, 非最旧)    (h = 5000/15000/60000 ms)
```
- 用**算术收益**(非对数);`mid_return_60s` 计算了但**未进 HFM80 综合分**(HFR80 反转策略用)。

### 1.5 波动(基于 micro-price 序列,最近约 120 个样本 ≈ 60s)
```
realized_vol_60s = 逐步对数收益的样本标准差 (ddof=1)     # per-step, 未年化
atr_1m          = (max − min) / mean                    # "极差/均价" 近似, 并非真正 ATR
```
- `atr_1m` 仅用于**止损下限**(§5.3),不进综合分;命名沿用历史,实为极差比例。

### 1.6 5bps 名义深度 `depth_5bps`(**现已使用**)
```
depth_5bps = 买侧5bps内名义深度 + 卖侧5bps内名义深度    (USDT 近似)
```
- v2 算了但没用;现在进入**可交易性**(§3.3)的深度门槛。

### 1.7 (仅股票永续) basis
```
basis = mid / mark − 1;  basis_z = z-score(basis)        # 用于 basis 策略, 不进 HFM80 综合分
```

---

## 2. 稳健 z-score 定义(滚动、因果、中位数/MAD)

```
对每个因子维护一个滚动 deque(factor_window = 200 个样本)。
若样本数 < 10:返回 None(预热期)。
否则(稳健版):
   med = 中位数(history)
   mad = 中位数( |v − med| )
   sd  = 1.4826 × mad                       # MAD→σ 的一致估计
   若 sd > 1e-12:  z = (x − med) / sd
   否则回退普通版:  z = (x − mean) / std(ddof=1)   (std≈0 → z=0)
   z = clamp(z, −4, +4)                      # 缩尾, 防异常值
当前值先用历史算 z,再把当前值压入历史(因果,无未来泄漏)。
```
- 相比 v2 的 mean/std,中位数/MAD 对 HF 非平稳数据与尖峰异常更稳。

---

## 3. 综合分主公式 (CompositeScorer.score) — 对称式

### 3.1 方向权重(来自 config `signal.weights`)
```
微观结构 microstructure = 20
订单流   order_flow     = 15      →  w_flow  = 20 + 15 = 35
趋势     trend          = 15      →  w_trend = 15
                                     wdir    = w_flow + w_trend = 50
(volatility_regime/liquidity/basis/event/funding/execution 等其余权重不再进入"方向分";
 流动性与波动改由"可交易性"独立通道表达, 见 §3.3)
```

### 3.2 方向分量与合成方向(原始)
```
squash(z) = tanh(z / 2)

flow_raw  = mean( squash(obi_5_z), squash(ofi_z), 0.7 × trade_imbalance_3s )   # 缺失项剔除后取均值, ∈[−1,1]

# 趋势: 波动归一化 (取代 v2 的 tanh(r×4000) 魔法增益)
spread_frac = spread_bps / 1e4
sig_step  = max( realized_vol_60s, 0.5 × spread_frac, 1.0e-4 )   # v3.1: 噪声地板(半价差/绝对), 防低波时单tick被放大
对 (mid_return_5s, steps=10) 与 (mid_return_15s, steps=30):
    σ_h = sig_step × √steps                 # h步收益的标准差 ≈ 每步σ × √步数
    项  = tanh( r / σ_h )
trend_raw = mean(上面各项)                  # 缺失项剔除后取均值, ∈[−1,1]

# 合成方向
direction = clamp( (w_flow × flow_raw + w_trend × trend_raw) / wdir , −1, +1 )   ∈ [−1, +1]
```
- `0.7` 是 trade_imbalance 的手设压缩系数(仍为先验)。
- 趋势不再用 ×4000;改为相对"该时段波动"的标准化幅度,数据自适应。

### 3.3 可交易性 tradability(**独立通道,不进方向分;v3.3 改"相对中位数倍数"+EMA**)
> 衡量"此刻值不值得做";**正常行情≈1**,只有在价差明显变宽 / 深度明显变薄 / 近乎死盘时才下降。
> 每个标的维护 spread / depth / rv 三个滚动历史 deque(maxlen 400);样本 ≥30 才出值,否则预热(只记录不开仓)。
>
> **为何不用分位排名(v3.2 的问题)**:分位"排名"对**幅度不敏感**——当前值只要是近期最高(哪怕只高一点)排名就≈1,
> 于是价差稍动、可做就乱跳(用户实测"可做波动特别大特别快"),且强行情时价差/深度排名同时到极端、三门连乘把可做砸到 ~0.05
> (用户实测"多空差异大时可做特别低")。**v3.3 改为"相对该标的滚动中位数的倍数"**:对小抖动稳定,只在真的变宽/变薄数倍时才反应。

```
倍数 ratio(x) = x / median(该标的滚动历史)    (样本<30 → 预热)

价差门 gate_low_good(ratio):    # 价差/中位数, 低=好
    ratio ≤ 1.5 → 1.0                            (≤1.5倍都算正常)
    ratio ≤ 3.0 → 1.0 − 0.7×(ratio−1.5)/1.5      (线性降到 0.3)
    否则        → 0.3
深度门 gate_high_good(ratio):   # 深度/中位数, 高=好
    ratio ≥ 0.6 → 1.0
    ratio ≥ 0.2 → 0.3 + 0.7×(ratio−0.2)/0.4
    否则        → 0.3
波动带 vol_band(ratio):         # rv/中位数, 仅近乎死盘才降; 高波动不罚(顺势机会)
    ratio < 0.2 → 0.0
    ratio < 0.5 → (ratio−0.2)/0.3
    否则        → 1.0

trad_raw = clamp( 价差门 × 深度门 × 波动带 , 0, 1 )
trad_raw            = clamp(价差门 × 深度门 × 波动带, 0, 1)            # 原始(录制用; CompositeResult.tradability_raw)。v3.6: rv<死盘门(5e-5)绝对冻结时 trad_raw=0 与死盘一致
tradability_display = EMA(trad_raw, 半衰期 ema_trad_half_life_s=1.5s)   # 面板"质量"显示用(CompositeResult.tradability)
tradability_entry   = min(trad_raw, tradability_display)              # 入场门用(快下慢上): raw立即反映崩坏, EMA不拖慢刹车 (hfm80._qualifies)
# 预热回退: 价差用 1 − min(0.6, max(0,(spread_bps−10)/40)); 深度/波动不罚(=1)。
```
> **设计要点**:(1) 倍数度量 → 正常行情/正常波动可做稳定 ≈1(实测正常期 stdev≈0);(2) **高波动不再惩罚**(顺势策略里高波动是机会,其风险已由 期望edge门 + 价差门 管理,不重复扣)→ 修复"强信号时可做被误杀";(3) **EMA 去抖** → 可做不再每拍乱跳;(4) 仍**与方向分解耦**(策略侧独立门槛 `min_tradability`)。**所以"可做100 + 多空48/52"是正确的**:可做=市场质量、多空=方向,两个独立轴;平静好行情就是"很好做但此刻没方向"。

### 3.4 对称 0–100 映射(**核心方程**)
```
long_score  = 50 × (1 + direction)
short_score = 100 − long_score          ( = 50 × (1 − direction) )
```
- **多 + 空 ≡ 100**;`direction=0` → 50/50(中性);`direction=+1` → 100/0;`direction=−1` → 0/100。
- `tradability` 单独输出,不参与上式。

**取值对照(便于直观理解):**
| 情形 | direction | long_score | short_score | tradability(正常行情) |
|---|---|---|---|---|
| 满多 | +1.00 | **100.0** | 0.0 | ~1.0 |
| 强多 | +0.45 | **72.5** | 27.5 | ~1.0 |
| 弱多 | +0.12 | 56.0 | 44.0 | ~1.0 |
| 平盘 | 0.00 | **50.0** | **50.0** | ~1.0 |
| 强空 | −0.45 | 27.5 | **72.5** | ~1.0 |
| 满空 | −1.00 | 0.0 | **100.0** | ~1.0 |

> 门槛映射:`long_score ≥ 66 ↔ direction ≥ 0.32`;`70 ↔ 0.40`;`74 ↔ 0.48`。

> **诚实说明(请专家留意)**:`多+空=100` 使两者相关性恒为 **−1**,但这是**恒等式/重参数化**(空=100−多),**并不提升预测力**;它带来的是**可读性**(50 中性)与消除"平盘≈0"的问题。真正提升"分数准不准"的是后续**用底层因子对未来收益重新拟合权重**(Stage B,未做),而非本节的映射形式。

---

## 4. 去噪层(EMA 平滑 + 双阈值闩锁 + 波动自适应)

目的:原始 `direction` 每 0.5 秒仍会抖动。对**方向分量**与**合成方向**做指数平滑,并加方向闩锁,避免贴 0 来回翻。

```
dt = 0.5s;   alpha(half_life) = 1 − 0.5^(dt / half_life)
a_flow  = 1 − 0.5^(0.5/1.0) = 0.2929   (流向EMA, 半衰期 1.0s)
a_trend = 1 − 0.5^(0.5/2.0) = 0.1591   (趋势EMA, 半衰期 2.0s)
a_score = 1 − 0.5^(0.5/0.5) = 0.5000   (合成方向再EMA, 半衰期 0.5s)

# 波动自适应: 若 rv > 1.2e-3(剧烈), a_flow/a_trend 再 × 0.6(更平滑)
scale     = 0.6 if rv > 1.2e-3 else 1.0
flow_s    = EMA(flow_raw,  a_flow × scale)
trend_s   = EMA(trend_raw, a_trend × scale)
dir_s     = clamp( (w_flow×flow_s + w_trend×trend_s) / wdir , −1, +1 )
dir_smooth= EMA(dir_s, a_score)

# 双阈值(Schmitt)方向闩锁
enter = 0.12, exit = 0.04
  从 0:   dir_smooth≥+0.12→+1;  ≤−0.12→−1;  否则 0
  从 +1:  ≤−0.12→−1;  <+0.04→0;   否则保持 +1
  从 −1:  ≥+0.12→+1;  >−0.04→0;   否则保持 −1
dir_latch ∈ {−1, 0, +1}

# 平滑分(对称, 供门槛/面板)
long_score_s  = 50 × (1 + dir_smooth)
short_score_s = 100 − long_score_s
```

**两套分都算:**
- `long_score / short_score`(**原始**,由 `direction` 算)→ **录制/校准用**(保证诚实可重放)。
- `long_score_s / short_score_s`(**平滑**)→ **入场门槛 + 面板显示用**(稳定可操作)。
- `order_flow_dir / trend_dir`(对外暴露)= **平滑后**值(出场/确认与入场共用同一去噪视图);另有 `_raw` 版给录制。

---

## 5. 与"分数"直接相关的下游量

### 5.1 model_prob(**仍为占位符,但已不参与入场判定**)
```
若无 meta 模型:  model_prob = 0.5 + 0.2 × max(0, (composite − 50)/50)     # ∈[0.5,0.7] 占位
```
- **v3.2:model_prob 已完全退出 edge 计算**(edge 改由"期望幅度−成本",见 §5.3),仅作纯展示占位,**入场/出场/仓位任何决策均不依赖**。

### 5.2 确认值 conf(入场二次确认,用平滑方向)
```
conf(comp, 方向d) = 0.5 × max(0, d × order_flow_dir) + 0.5 × max(0, d × trend_dir)     ∈ [0,1]
```

### 5.3 止损/止盈/成本/edge
```
total_cost = maker_fee(0.0002) + taker_fee(0.0005) + 半价差(spread_bps/1e4 × 0.5) + 滑点估计(0.0003)
sl = clamp( max(1.2×atr_1m, 3×(spread_bps/1e4), 0.0010, 2.5×total_cost),  下限0.0010, 上限0.012 )
tp = max( 1.6 × sl,  3 × total_cost )        # tp_rr=1.6, 可由『校准』写入
# v3.2 期望净edge (真过滤器; 取代旧伪概率edge 与 "tp/cost≥1.2" 恒成立门):
dir_strength      = |composite − 50| / 50                       # 由对称分还原方向强度 ∈[0,1]
sigma_H           = realized_vol_60s × √(edge_horizon_s / 0.5)  # 持有期(默认30s)波动
expected_move     = min( edge_move_k(默认1.5) × dir_strength × sigma_H , tp )   # 封顶到自身止盈(超过tp也吃不到); build_signal
expected_edge_pct = expected_move − total_cost                  # = Signal.expected_edge_pct
edge_to_cost      = expected_edge_pct / total_cost              # 入场门槛用此 (§5.4)
```
- `2.5×total_cost` 的成本感知止损下限,防止亚成本止损被噪声秒扫(白交手续费)。

### 5.4 入场判定(主策略 HFM80,全部基于**平滑分**)
```
1) 死盘抑制:  若 realized_vol_60s < regime_rv_lo(5e-5, 真冻结)  →  不开仓 (v3.5 下调; "行情淡"改由 §5.3 期望edge门管)
2) 候选方向:  cand = 多 if long_score_s ≥ short_score_s else 空
3) 达标(qualifies):
      顺向分 score_s ≥ min_composite(66)               # 方向分门槛
      且 tradability_entry = min(trad_raw, EMA) ≥ min_tradability(0.50)   # 可交易性独立门槛(与方向解耦); 入场用min(raw,EMA)快下慢上, 非纯EMA
      且 conf ≥ confirm_min(0.20)
      且 dir_latch == cand                              # 方向闩锁一致 (死区 latch=0 → 不做)
4) 持续:      连续 N 拍达标; N = persist_ticks(3) + (1 若 rv > 1.2e-3) ; 容忍单拍坏点 miss_grace(1)
5) 期望净edge: edge_to_cost ≥ min_edge_to_cost_ratio(默认1.0)   # =expected_edge_pct/cost; 真过滤器(平静/弱信号会被拦), 见 §5.3
   入场还要求 not warmup (每标的分位历史≥30样本前只记录不开仓)。
   全部满足 → 产出交易信号(下游再过风控/仓位)。
```

### 5.5 信号驱动出场(executor.signal_exit,对称尺度)
```
own_score = 顺向平滑分(long_score_s 或 short_score_s);  opp_score = 100 − own_score
反转: opp_score ≥ (入场门槛 − reversal_hyst_gap) = 66 − 8 = 58 且 opp_conf ≥ confirm_min
衰减: own_score < decay_threshold(54)         # 自身分跌回中性附近 → edge 消失
(任一成立且连续 persist 拍 + 过扣费保护 → 平仓; 另有移动止盈; 硬止损/超时由 monitor 始终生效)
```

---

## 6. 全部"魔法常数"清单(均为手设先验,**未用数据拟合**)

| 常数 | 出处 | 含义 |
|---|---|---|
| `tanh(z/2)` 的 `/2` | flow squash | z→[−1,1] 压缩斜率 |
| `× 0.7` | trade_imbalance | 压缩系数 |
| `σ_h = max(rv, 0.5×半价差, 1e-4)×√steps`, steps 10/30 | 趋势波动归一化 | 取代×4000;v3.1 加噪声地板 |
| 方向权重 35 / 15 (wdir=50) | 方向合成 | flow vs trend 配比 |
| 价差门(倍数) ≤1.5→1, 3→0.3 | tradability | v3.3: 相对中位数倍数 |
| 深度门(倍数) ≥0.6→1, 0.2→0.3 | tradability | v3.3: 相对中位数倍数 |
| 波动带(倍数) <0.2→0死盘, ≥0.5→1(高波不罚) | tradability | v3.3 |
| 可做EMA 半衰期 1.5s / 滚动窗400 / 最少30样本 | tradability | v3.3: 去抖+逐标的中位数 |
| z-score 窗 200 / 最少 10 / 缩尾 4 / 1.4826 | 稳健z | 中位数MAD |
| EMA 半衰期 1.0/2.0/0.5s | 去噪 | 平滑强度 |
| 闩锁 0.12 / 0.04 | 去噪 | 方向进/出阈值 |
| 死盘 5e-5(v3.5,原2e-4) / 剧烈 1.2e-3 / 缩放 0.6 / 持续+1 | regime | 波动分档(死盘=真冻结) |
| z地板 OBI 0.05 / OFI 0(待校准) / basis 0 | 因子z | v3.6: 防淡市z爆表 |
| TI名义额收缩地板 0(默认no-op) | 成交失衡 | v3.6: 防薄量单笔钉死±1 |
| 对称映射 `50×(1+dir)` | 评分 | 多+空=100 |
| 入场 66 / 可交易性 0.50 / 确认 0.20 / 持续 3 / edge_to_cost 1.0 / 强信号 74 | 门槛 | 入场闸门 |
| 价差绝对硬门 50bps / 可做入场用 min(raw,EMA) | 硬风控 | v3.4: 软分之外的硬边界 |
| edge期望幅度系数 k=1.5 / 持有期 30s / 封顶到tp | 期望edge | v3.2/3.4 |
| 反向迟滞 8(rev_th 58) / 衰减 54 | 出场 | 信号出场 |
| model_prob `0.5+0.2·(comp−50)/50` | 占位概率 | **不参与入场** |
| sl: 1.2×atr / 3×spread / 0.10% / 2.5×cost,上限1.2% ; tp: 1.6×sl 或 3×cost | 风险 | 止损止盈 |

---

## 7. 自审:已修复 vs 仍存在(**供专家独立判断**)

**v2 → v3 已修复的疑点:**
1. ✅ **方向/质量混轴** → 已拆分:方向走 `50×(1+dir)` 对称轴,可交易性走独立 [0,1] 门槛;平盘=50/50 而非 ~28/28。
2. ✅ **OFI 非经典 + 量纲混加 + 与 TI 重复** → 改为经典 Cont L1、深度归一化、去掉成交量项。
3. ✅ **趋势 ×4000 饱和** → 改为波动归一化 `tanh(r/σ_h)`。
4. ✅ **裸中间价** → 改用加权中间价(weighted-mid / micro-price 近似;趋势/波动都基于它)。
5. ✅ **z-score 用 mean/std** → 改为中位数/MAD 稳健版 + 缩尾。
6. ✅ **depth_5bps 算而不用** → 进入可交易性深度门。
7. ✅ **占位 model_prob 当入场门槛** → 入场改为结构性 `tp/cost` 门,model_prob 仅展示。

**v3 → v3.1 已修复(专家评审后):**
8. ✅ **OFI 仅 0.5s 首尾快照差** → 改为**逐事件累计**(网关每次盘口更新累加,compute 拍取走;§1.2)。这是专家的头号 P0,已落地。
9. ✅ **趋势分母低波时被单 tick 放大** → `sig_step` 加**噪声地板** `max(rv, 0.5×半价差, 1e-4)`(§3.2)。
10. ✅ **副策略仍用占位 model_prob 的 `edge_to_cost` 入场,且无可交易性门槛** → HFR80/Breakout/Basis **全部改为结构性 `tp/cost` 门 + 可交易性独立门槛**(与主策略 HFM80 一致);删除冗余 `opp_max`。
11. ✅ **信号检验仅看扣费前** → 新增 **+1s 成交延迟后的 IC**(检验信号是否吃得到);分桶仍标注"扣费前,实际扣费由校准计"。
12. ✅ **录制只存合并后的 flow/trend(挡住 Stage B)** → recorder **已扩为逐因子**(obi_z/ofi_z/ti/ret5/15/60/depth/basis_z),**现在起每次运行就积累 Stage B 重拟合所需数据**。

**v3.1 → v3.2 已修复(专家二轮评审后):**
13. ✅ **`tp/cost ≥ 1.2` 是恒成立门(无过滤力)** —— 因 `tp = max(1.6·sl, 3·cost)` 必有 `tp/cost ≥ 3`,该门只验证了 `cost>0`。这是专家二轮的头号 P0。**已改为期望净edge门**(§5.3:`edge_to_cost = (k·方向强度·持有期波动 − 成本)/成本`),实测会真正拦截弱/平静信号(强+活跃 edge/cost +6.6 通过;弱+平静 −0.37 拦截)。**实盘 build_signal 与 校准 sim_one 用同一公式**。⚠ k/horizon 仍是手设启发式,Stage B 用拟合 `E[扣费后净收益|因子]` 替换。
14. ✅ **冷启动 warmup 默认放行** → 新增 `comp.warmup`(每标的价差/波动分位历史 <30 样本即 True),**所有策略 warmup 期只记录不开仓**(实测前~30拍0开仓)。
15. ✅ **OFI 重连/断号污染 + 乱序** → 重连清空累计器、断号清该标的、乱序到达丢弃增量;books5/bbo-tbt 单源驱动(`_ofi_from_bbo`)不重复计;`take_ofi` 在单线程 asyncio 事件循环内执行,天然原子。
16. ✅ **趋势取点存疑** → 源码核验为 `reversed` 迭代取"≤目标时间的最近(latest)样本"(非最旧),**已正确**;文档措辞已修正。
17. ✅ **延迟检验仅 +1s** → 扩为 **0.5/1/2s 多档延迟 IC**。
18. ℹ️ **OBI/TI 量纲** → 二者本就是比率 `(b−a)/(b+a)∈[−1,1]`、且逐标的 z-score,**已天然可比**(无需再 notional 归一);OFI 已深度归一+z。专家此点对 raw size 成立,但当前实现非 raw size。

**v3.2 → v3.3 已修复(用户实盘观察:可做乱跳 / 强信号时可做特别低):**
19. ✅ **可做(tradability)用"分位排名"导致两个症状** —— (a) 排名对幅度不敏感,当前值只要近期最高(哪怕一点点)排名就≈1 → 价差稍动可做就乱跳;(b) 强行情时价差/深度排名同时到极端、三门连乘把可做砸到 ~0.05 → 强信号被误杀。**已改"相对该标的中位数的倍数"度量 + 去掉高波动惩罚(顺势是机会)+ 再 EMA 去抖**(§3.3)。实测:正常期可做 stdev≈0(稳定)、强方向时可做仍 1.0(不再误杀)、真实流动性崩坏仍被拦(~0.16)。**并澄清"可做100 + 多空48/52"是正确的**(可做=市场质量、多空=方向,独立两轴)。

**v3.3 → v3.4 已修复(专家第三轮评审后):**
20. ✅ **EMA 平滑后的可做会"慢刹车"** —— 流动性骤坏时 EMA 让可做下降慢 ~1.5s,危险瞬间仍可能放行。**已改:入场门用 `min(可做_raw, 可做_EMA)`(快下慢上)** —— raw 立即反映崩坏即刻拦,EMA 仅用于面板显示。
21. ✅ **缺绝对硬风控边界** —— 软分不能替代硬门。**已加价差绝对硬门** `spread_bps > max_entry_spread_bps(50)` → 直接不开新仓(所有策略)。数据老化仍由 kill-switch(500ms)管。(深度相对下单规模、滑点 walk-book 硬门属 Stage B/C,需下单规模上下文。)
22. ✅ **期望幅度在极端波动会被吹爆** —— `k×方向强度×σ_H` 在高波动可飙到数十 bps,放大过度交易。**已封顶到自身止盈** `期望幅度 = min(期望幅度, tp)`(你也吃不到超过 tp 的);实盘 build_signal + 回测 sim_one 一致。
23. ✅ **"可做"标签易被误读为"该交易" + 看不出为何不出手** —— 面板"可做"**改名"质量"**,并新增**"入场"状态列**(✓候选 / 方向不足 / 质量低 / 未确认 / 死盘 / 预热 / 价差宽),直接显示每个标的卡在哪一关。
24. ✅ **文档一致性订正(专家 §11)** —— §1.4 趋势取点改"最近样本(latest≤目标)"、§6 魔法常数表更新为倍数门 + edge_to_cost 1.0、版本号统一。

**v3.4 → v3.5 已修复(用户实盘观察:质量100却显示死盘 / 强方向却死盘):**
25. ✅ **"质量100 + 死盘"不一致** —— 质量用**相对**vol(rv/自身中位数), 死盘用**绝对**rv<2e-4, 两把尺子 → 一个一直淡的标的会同时"质量100"且"死盘"。且 2e-4 对 0.5s 步 micro-price 波动**过高**, 导致几乎所有标的恒判死盘、从不交易。**已:(a) 死盘阈值 2e-4→5e-5(仅真冻结);(b)"行情太淡"改由经济正确的期望edge门拦(动得不够覆盖成本→净edge不足);(c) 面板"入场"列新增"净edge不足"原因**。职责分清:质量=执行环境(相对)、edge/死盘=机会(绝对)。实测淡市(rv1.2e-4)+方向72 → edge/cost −0.44 → 显示"净edge不足"(诚实), 不再误显"死盘"。
26. ℹ️ **强方向出现在淡市 = z-score 噪声(非bug, 但需知道)** —— flow 由 OBI/OFI 的**z-score**(相对自身近期分布)合成, 即便绝对盘口失衡很小, 淡市里也会被标准化成大"方向"。所以"72分多但行情没动"是 z-score 噪声; 期望edge门是正确的兜底(动得不够就不做)。**根治在 Stage B**: 用未来净收益验证该方向分到底有没有预测力, 没有则该因子降权/剔除。面板 OBI_z/OFI_z 是**瞬时原始值**, 多/空是**EMA平滑分**, 两者不会逐拍吻合(平滑滞后, 正常)。

**v3.5 → v3.6 已修复(专家第四轮; 经 6-agent 独立读源码核验后只做被确认的):**
27. ✅ **[P1] z-score 在淡市 MAD 极小时虚高**(确认): flow 路径无绝对地板(趋势路径有), 淡市微小变化→z 被 clamp 到 4→tanh(2)≈0.96 假满信号, hfr80 最受影响。**已加因子绝对离散度地板** `sd=max(1.4826·MAD, sd_floor)`: OBI地板0.05(有界因子, 安全), OFI/basis 默认0(深度归一无固定量纲, 待 Stage B 用录制分布校准)。实测 0.10→0.14 小变化 z 从4降到0.8, 而真强 0.10→0.40 仍>3(不误杀)。
28. ✅ **[P2] 质量(相对)与死盘(绝对)不一致**(确认为显示矛盾, 非安全漏洞——死盘门在 _qualifies 前先拦, "质量100让死盘交易"被**驳回**): **已让 rv<死盘门(5e-5) 时 trad_raw 也=0**, 质量与死盘一致; 顺带删除从未调用的旧 `_quality` 死代码。
29. ✅ **[P2] TI 薄量单笔钉死±1**(机制确认, 但"钉死最终方向"被**驳回**——TI 净占比≈0.23 且被强/突破门+warmup 保护): **已加名义额收缩** `ti×notion/(notion+floor)`, 默认 floor=0(行为不变), 待 Stage B 按标的校准。
30. ✅ **[P2] 录制 rv_mid**: 新增纯中间价波动列(对比 micro-price 是否虚高), 为 Stage B 决定"rv/edge 改用纯中间价"备数据。
31. ✅ **[P2] 文档一致性**(确认): §6 死盘表 2e-4→5e-5、§5.3 expected_move 写明封顶 tp、§3.3/§5.4 明确区分 trad_raw/display/entry。

**经核验【驳回/暂缓】的专家建议(避免过度设计, 不做):**
- **microprice→纯中间价 切换(rv/trend/edge)**: 确认 micro-price 会因挂量变动制造虚高波动, 但切换会改 rv 量纲、与死盘/edge 阈值耦合 → **必须与阈值同时重标定**, 归 Stage B(本版只先录 rv_mid 备数据, 不切实盘消费, 避免回到"全标的死盘")。
- **多条件死盘(rv+成交额+盘口更新+区间)**: 过度设计——需新增 FeatureSet 字段, 而 edge 门已从经济上覆盖"淡市无机会"; 不做。
- **OFI 改 Σe/avg_depth**: Σ(e/depth) 确有薄盘放大, 但消费端 ofi_z 已 MAD 稳健+缩尾|z|≤4 兜住; 归 Stage B 作为录制变体 A/B, 不盲切。

**仍然存在 / 需注意的局限(诚实保留):**
1. ⚠ **所有增益/权重/门槛仍为手设先验,无逐标的拟合**(35/15、0.7、各分位门等)。
2. ⚠ **综合分仍未对"未来真实收益"做拟合**——对称映射只是更可读,不代表更准。
   真正的"用底层因子→未来收益重新标定权重"是 **Stage B(尚未实现,但录制已就位)**;在用数据(IC / 样本外 / DSR / PBO / 走步 / 延迟+扣费压力)证明它能预测**扣费后**收益之前,**仍应视为研究假设,不可凭分数本身直接实盘**。
3. ⚠ **可交易性分位门需要预热**(每标的≥30 样本);冷启动用兜底值(价差兜底有意义,深度/波动暂不罚);**数据老化(>500ms)已由 kill-switch 全局拦截**(cancel_all_on_data_gap),非由 tradability 单独处理。
4. ⚠ `atr_1m` 实为极差比例(命名误导);HF `realized_vol` 仍有微观噪声偏差(噪声地板部分缓解,根治需 two-scales 估计)。
5. ⚠ **OFI 逐事件粒度受数据档限制**:默认 books5≈100ms(每拍约 5 次);真正 tick 级需订阅 bbo-tbt(≈10ms,WS 负载更高)或 VIP4 全深度 L2。
6. ⚠ **加权中间价是 micro-price 近似**,非完整 Stoikov(完整版属 Stage B)。
7. ⚠ **可交易性深度门用绝对名义深度,未相对下单规模**(reviewer P1):大仓位会高估可做性;相对化(`depth/order_notional` 或 walk-book 滑点)属 Stage B/C。
8. ⚠ **股票永续 session/basis/funding 尚未纳入风险门**(Stage C);且股票永续地区受限,你的账户多半不可见。
9. ⚠ 去噪带来 1–2 拍(0.5–1s)入场滞后(短线代价,换稳定)。

**关键保障(措辞已按专家意见收敛)**:**严格执行**校准闸门(样本外 + DSR + PBO + 走步 + 延迟/扣费压力)可**大幅降低**纯噪声信号被放行上线的概率——校准器 `gen_entries`/`sim_one` **忠实重放同一套评分+去噪+期望edge门**,无预测力则 DSR 不达标 / PBO 偏高 / 走步转负 → 判定"不要上线"。但这些是**防过拟合工具,不是盈利保证**:市场状态漂移、滑点低估、延迟、资金费跳变、流动性骤失、样本外偶然成功等仍可能导致实盘亏损。**因此:闸门只降低上线噪声概率,不承诺不亏钱;且无视闸门、未经验证就实盘是最大危险。**

---

*生成自源码当前版本(2026-06-08, Stage A + 对称 Stage A' + 三轮专家评审修复)。一轮:逐事件OFI / 去伪概率 / 趋势噪声地板 / 逐因子录制 / 延迟IC。二轮:期望净edge门(修 tp/cost 恒成立)/ warmup 不开仓 / OFI 重连+乱序安全 / 多档延迟IC / 趋势取点核验。三轮(v3.4):可做改"相对中位数倍数"+EMA(修乱跳与强信号误杀)/ 可做入场用 min(raw,EMA) 快下慢上(修EMA慢刹车)/ 价差绝对硬门 / 期望幅度封顶tp / 面板加"入场状态"列 + 文档一致性订正。四轮(本版 v3.5):死盘阈值 2e-4→5e-5(修"质量100却死盘"不一致 + 几乎全标的恒死盘从不交易)、"行情太淡"改由期望edge门管、入场列加"净edge不足"原因。如需我同时导出对应源码文件(composite.py / engine.py / microstructure.py / gateway.py / hfm80.py / calibrator.py 等)给专家,告知即可。*
