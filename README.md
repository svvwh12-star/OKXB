# OKXB — OKX 股票永续 + 加密永续 自动量化交易系统

个人自用的、**高频扫描 / 低频触发 / 强风控**的自动量化交易系统。
定位不是拼速度，而是 **只在高流动性、低成本、多因子共振、可控亏损** 的窗口里出手。

> ⚠️ **风险声明**：自动化杠杆交易可能导致全部本金损失。本系统的设计目标是把每笔风险
> 严格限定在固定 USDT 额度内、并以 5% 总回撤硬停为底线，但**没有任何策略能保证盈利**。
> 第一阶段强制只读回测 + 模拟盘，验证通过前不接实盘。是否使用、在何地使用，由你自行承担合规与法律责任。
>
> 📉 **经济学基准（务必接受）**：1000 USDT 散户做这件事的诚实预期是**打平减手续费**，请当研究/学费。
> 可持续的散户 edge 是**成本控制 + 选择性执行 + 波动定仓 + 更长周期**，不是速度（散户结构性输掉盘口竞速）。

> 🌍 **地区门控（可能是硬约束）**：OKX 股票永续与永续合约**地区限制**，对**美国人不可用**，多辖区受限。
> 区域路由：美/澳 `us.okx.com`、欧盟 `eea.okx.com`、其余 `www.okx.com`（用 `.env` 的 `OKX_REGION` 设置）。
> 系统按账户登录态运行时核验股票永续资格并 fail-closed。**禁止用 VPN 绕过**（违反条款，风险冻结/没收）。
> 你的账户究竟能不能交易股票永续，跑 `verify_account.py` 即知。完整技术核验见 [docs/RESEARCH_BRIEF.md](docs/RESEARCH_BRIEF.md)。

---

## 🔐 开始前必做：轮换 API 密钥（一次性）

如果你的密钥曾以明文出现在任何聊天/邮件/文档里，**视为已泄露，必须重置**。带"交易"权限的密钥被他人获取，即使无提现权限，也能通过对敲亏空账户。

1. 登录 OKX → API 管理 → **删除旧的 Virtual / APP 两把 key**，重新创建。
2. 新建时务必：
   - 权限只勾 **读取 + 交易**，**不要**勾提现。
   - **绑定 IP 白名单**（本机公网出口 IP 或 VPS IP）。
   - passphrase 用全新的，不复用。
3. 把新密钥填入 `.env`（从 `.env.example` 复制，`.env` 已被 git 忽略，永不提交）。

---

## 🖥️ 桌面程序（双击运行，推荐普通用户）

打包好的单文件程序：**`dist\OKXB.exe`**，双击即可运行（Win11 中文版，无需装 Python）。

界面三个标签页：
- **控制台**：账户权益/日内盈亏/系统状态/持仓/数据延迟卡片 + 实时行情信号表（中间价/价差/OBI_z/OFI_z/多空分），多空分 ≥80 高亮。**行情始终走实盘公共 WS**（~毫秒级延迟），模拟盘只用于下单。**自动按流动性选标的**（默认：加密按 24h 成交额取前 30 + 全部 ~21 只股票永续 ≈ 51 个；用 books5 轻量盘口扛得住），可在 `config.yaml` 的 `universe`（`mode`/`max_crypto`/`max_stock_perp`/`min_crypto_quote_vol_usd`/`stock_symbols`）调整。
- **账户与密钥**：填入虚拟盘/实盘的 API Key/Secret/Passphrase + 区域 + AI 提供商，**保存**到本机 `.env`；**三个独立验证按钮**：验证虚拟盘 / 验证实盘 / 验证AI（分别只读检查权限/IP/可交易合约 与 AI 连通性）。
- **手动交易**：手动下单/撤单/一键平仓/设置杠杆（用你的密钥，按顶部模式；实盘二次确认）——可对照控制台信号精细操作。
- **日志**：引擎实时日志（含下单意图/开仓/平仓/告警/异常）。

顶部：**虚拟盘 / 实盘** 切换、**实际下单** 开关（默认关=只演练不下单）、**启动/停止**。
安全闸门：实盘 + 实际下单需勾选并通过**二次确认弹窗**；密钥只存本机，绝不外传。

> 用户文件（`.env`、`recordings`、`models`、`logs`、状态库）生成在 **OKXB.exe 同目录**，方便查看备份。

**自己重新打包**（改了代码后）：`powershell -ExecutionPolicy Bypass -File build_exe.ps1` → 产出 `dist\OKXB.exe`。
（DeepSeek / OpenAI 兼容的 AI 事件分类**开箱即用、无需额外库**；仅 `provider=claude` 才需先 `pip install anthropic` 再打包。）

---

## 🚀 开发者方式（源码运行 / 调试）

### 安装与首次校验

```powershell
# 1. 建虚拟环境并安装依赖
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. 配置密钥
copy .env.example .env
#   用编辑器打开 .env, 填入轮换后的新密钥 (默认 OKXB_MODE=demo 模拟盘)

# 3. 校验账户/权限/可交易合约/时钟 (只读, 不下单)
python scripts/verify_account.py
```

`verify_account.py` 会检查：密钥有效性、权限是否为读取/交易且未开提现、IP 是否绑定、
账户模式、权益与持仓、加密/股票永续是否可交易、本地时钟偏移。**绝不打印密钥明文。**

---

## 🧱 系统架构

RiskEngine 是**唯一总闸门**——任何订单、任何策略都必须穿过它；AI 只产生事件 veto 标签，**永不直接下单**。

```
Control Plane  (配置热加载 · 监控 · Telegram告警 · 手动急停)
      │
MarketDataGateway ──► FeatureEngine ──► SignalService ──┐
 (WS: bbo-tbt/books5/                (OBI/OFI/价差/      │ 候选信号
  trades/funding/mark)                vol/ATR/basis)     ▼
                          AIEventService ──veto──►  RiskEngine ──批准──► ExecutionService
                          (SEC/Finnhub/Claude)      (仓位/限额/回撤      (post-only/reduce-only/
                                                     /熔断/周末/资金费)    IOC · TTL重挂 · 限速)
                                                          │                      │
                                                     StateStore (SQLite + 审计日志) ◄┘
                                                          ▲
                          Research/Backtest (Phase1只读扫描 · triple-barrier · walk-forward)
```

模块映射（`src/okxb/`）：

| 目录 | 职责 | 对应策略文档 |
|---|---|---|
| `exchange/` | OKX REST/WS 客户端、鉴权、限速、合约规格缓存 | §4, §14 |
| `marketdata/` | 行情订阅、本地订单簿、数据老化检测 | §4 |
| `features/` | OBI / OFI / TradeImbalance / 价差深度 / 趋势 / 波动 / basis | §6 |
| `signal/` | 综合评分 + ML 概率 + edge/cost | §5, §6 |
| `strategies/` | HFM-80 / HFR-80 / basis 回归 / 突破 | §7–§10 |
| `risk/` | 仓位计算、组合限额、回撤阶梯、熔断、周末/资金费/事件否决 | §2, §12, §13 |
| `execution/` | 下单/撤单/TTL重挂/止盈止损、reduce-only 保护 | §14 |
| `events/` | SEC EDGAR + Finnhub + Claude 事件分类 → veto 标签 | §11 |
| `research/` | 数据录制、triple-barrier 标注、保守成交回测、模型训练 | §15, §16 |
| `state/` | 持仓/订单/PnL/风控快照持久化 + 审计 | — |
| `monitor/` | Telegram 告警、状态面板 | — |
| `app.py` | 编排器 / 主循环 | §18 |

---

## 🛣️ 渐进上线（严格按此顺序，不跳级）

| 阶段 | 内容 | 门槛 |
|---|---|---|
| **Phase 0** | `verify_account.py` 通过；确认股票永续是否可交易 | 无红色错误 |
| **Phase 1** | 只读扫描 3–7 天：记录信号 + 假设成交 + 净收益统计 | 理论胜率/成本可接受 |
| **Phase 2** | 模拟盘 7–14 天：跑通下单/撤单/止损/熔断全链路 | 链路无误、风控生效 |
| **Phase 3** | 小资金实盘（单笔1U / 总名义≤800U / 回撤硬停30U）首周 | 滑点≤回测×1.5 |
| **Phase 4** | 正常参数实盘 | 持续监控 |

上线门槛（§15.4）：样本外≥300笔、profit factor≥1.25、扣费后胜率≥55%、最大回撤≤预期利润50%。

---

## 🔬 研究 / 建模 / 事件 工作流

```powershell
$env:PYTHONPATH='src'
# 采集训练数据 (越久越好) -> 标注 -> 训练 meta 模型 (自动被 app 加载)
python scripts/run_phase1.py --collect-features
python scripts/train_meta.py            # 产出 models/meta_model.pkl
python scripts/analyze_phase1.py        # 信号前瞻净收益 / triple-barrier 胜率

# AI 事件模块自测 (SEC EDGAR 真实数据; 无 Claude/Finnhub key 自动降级)
python scripts/test_events.py
```

- **AI 事件**: 提供商无关。界面「账户与密钥」选提供商（**DeepSeek**/OpenAI兼容/Claude/规则）+ 填 API Key 即可；按任务难度自动选模型（简单→`deepseek-v4-flash` 便宜快，复杂→`deepseek-v4-pro`）。配 `FINNHUB_API_KEY` 加财报日历。无 key 或选「规则」则免费规则降级。事件只产 veto 标签喂 RiskEngine，**永不下单**。
- **meta 模型**: 训练后 `app.py` 启动自动加载，用模型概率替换占位 `model_prob`；无模型文件则回落占位。**小样本/demo 训练的模型不可用于实盘**（需样本外≥数百笔 + purged CV + Deflated Sharpe/PBO）。

## 📊 当前实现状态

> 脚手架搭建中。`[x]` 已实现，`[~]` 接口已定/逻辑待补，`[ ]` 待建。

- [x] 项目结构、配置体系、`.env`/`.gitignore` 安全
- [x] 技术调研基线 `docs/RESEARCH_BRIEF.md`（OKX 限速/费率/数据/地区/方法，已对抗核验）
- [x] 账户/权限校验脚本 `verify_account.py`（区域感知、时钟检查、股票永续探测）
- [x] OKX 异步 REST 客户端（鉴权/区域路由/cancel-all-after 死手开关）+ 已核验限速器
- [x] 领域模型、枚举、配置加载（Config/Secrets）
- [x] 行情 WS 网关 + 本地订单簿（seqId 完整性，断号自动重订阅）
- [x] 因子引擎（OBI/OFI/TradeImb/价差/深度/收益/波动 + 滚动 z-score）
- [x] 综合评分（订单流因子去相关合一 + 缺失组重归一化）
- [x] **Phase 1 只读扫描器** `run_phase1.py`（已用真实行情验证：crypto+股票永续，无下单）
- [x] 风控引擎（总闸门：熔断/回撤阶梯/并发与名义限额/周末/事件否决）+ 仓位计算
- [x] 执行引擎（post-only 入场/TTL撤单/reduce-only 止盈/止损+时间止损/cancel-all-after）
- [x] 状态存储（SQLite：订单/盈亏/信号/审计）+ 合约规格缓存
- [x] 策略基类 + HFM-80 主策略（§7，股票永续偏空阈值）
- [x] **编排器 `app.py`**（全链路；dry-run 默认，已验证；`--live` 走模拟盘真实下单）
- [x] research/labeling（triple-barrier + 前瞻收益 + PF/胜率/盈亏比/Sharpe/回撤，已验证）
- [x] 回测上线门槛检查器 `GoLiveGate`（§15.4）+ `analyze_phase1.py` 前瞻净收益分析（已验证）
- [x] **AI 事件模块**（SEC EDGAR + Finnhub + Claude veto；已用真实 SEC 数据验证；接入 RiskEngine）
- [x] **meta-labeling 流水线**（特征采集→triple-barrier标注→Logistic/LightGBM训练→自动替换占位 model_prob；合成数据 AUC 验证）
- [x] **桌面 GUI + 单文件 .exe**（CustomTkinter；虚拟/实盘切换、API 输入/保存/验证、实时面板、日志、实盘二次确认；已打包并启动验证）
- [x] **四大策略**：HFM-80 顺势 + HFR-80 反转 + 突破(taker) + 股票永续 basis（多策略=更多出手机会）
- [x] **私有 WS 下单层**（EPOCH 秒签名登录 + 订单/持仓实时流 + WS 下单/撤单；可用则走 WS，失败回落 REST）
- [x] **Telegram 告警**（下单/平仓/系统状态变化/熔断推送；GUI 可填 + 验证）
- [x] **自动选标的**（加密按量 Top-60 + 全部股票永续）+ 持仓回灌风控（全局并发/名义上限生效）
- [ ] 完整 L2 回放回测（需录制全盘口）+ meta 模型实盘训练（需采够数据）

部分配置项标注 `[VERIFY]`，将由后台技术调研结果（OKX 当前限速/费率/数据频道VIP门槛/股票永续地区准入）核验后定稿。
