# OKXB 技术调研简报（2026-06 核验）

> 来源：39 个 agent 的深度网络调研 + 对关键事实的对抗式独立核验。
> 本简报是**驱动实现的事实基线**。配置/代码中标 `[VERIFY]` 的项以此为准。
> 调研结论坦诚：**1000 USDT 散户做这件事的经济学基准是"打平减手续费"，请把这笔钱当研究/学费，不是收入引擎。**

---

## 0. 最重要的 5 条（会改变架构）

1. **地区门控（致命）**：OKX 股票永续 + 永续合约对**美国人不可用**，多辖区受限。区域路由：美/澳 `us.okx.com`、欧盟/EEA `eea.okx.com`、其余 `www.okx.com`。股票永续模块必须按账户登录态**运行时核验资格、fail-closed**；公共 ticker 列表不代表你的账户能交易。**禁止 VPN 绕过**（违反条款，风险冻结/没收）。

2. **模拟盘需要独立的 demo 密钥**：在 OKX「Demo Trading」区单独创建的 API key + REST 头 `x-simulated-trading:1` + demo WS 主机 `wspap.okx.com:8443`。实盘 key 在 demo 失效，反之亦然。**demo 没有子账户**，也**不能信任 demo 的成交质量**（自包含模拟器，队列/滑点不真实）。你的"Virtual"备注 key 应当就是 demo key——`verify_account.py` 会确认。

3. **下单全部走一条常驻、已登录的私有 WebSocket**（`/ws/v5/private`），REST 仅做快照/恢复/兜底。WS 优势是延迟（省去每次 TLS 握手），**不给额外限速额度**（REST 与 WS 共享同一限速桶）。WS 登录签名用 **EPOCH 秒**（不是毫秒）对 `ts+'GET'+'/users/self/verify'` 做 HMAC。

4. **`cancel-all-after` 死手开关**：`POST /api/v5/trade/cancel-all-after`，启动时武装、每轮重置（如 700ms 循环用 5s 超时）。**这是无人值守 bot 最重要的单一安全机制**——崩溃/断线自动撤掉所有挂单。

5. **校验和（checksum）已废弃**（生产环境 2026-06-23，就在这几天）：字段还在但恒为 0。本地订单簿完整性**必须用 `seqId`/`prevSeqId` 连续性**校验——每条更新的 `prevSeqId` 必须等于上一条的 `seqId`，断号则丢弃并重新拉快照。**不要写任何基于 checksum 的校验器。**

---

## 1. 鉴权与连接（已核验，与现有代码一致）

- REST 私有请求 4 个头 + `Content-Type`；`sign = Base64(HMAC-SHA256(ts+METHOD大写+requestPath+body, SecretKey))`。GET 时 query 进 requestPath、body 空。时间戳 ISO-8601 UTC 毫秒（`2020-12-08T09:08:57.715Z`）且与头一致。
- **请求 30s 过期**（错误码 `50102`）。时钟漂移会"静默"地拒掉每一单（响应体里给 50102）。必须 NTP 对时、用 UTC。→ `verify_account.py` 已含时钟检查。
- 生产 REST：`www.okx.com`（`openapi.okx.com` 同后端）。WS：`wss://ws.okx.com:8443/ws/v5/{public|private|business}`。Demo WS：`wspap.okx.com:8443`。
- WS 连接限制：每 IP **3 新连接/秒**；每连接 **480 次 sub/unsub/login 操作/小时**；每子账户每频道类型 30 连接；订阅载荷 ≤64KB；30s 无数据则断开。Keepalive：发文本 `ping` 等 `pong`，<30s 周期；丢 pong 即视为死链，重连+重订阅+重建订单簿。

## 2. 订单类型（已核验）

- `ordType`：`limit`、`market`、`post_only`（保证 maker，会穿价则被撤；仅限价）、`fok`、`ioc`、`optimal_limit_ioc`（**仅期货/永续的市价式 IOC**，OKX 给 HFT 的"可成交限价"等价物）、外加 `mmp`/`mmp_and_post_only`（做市商保护）、`elp`。
- `reduceOnly` 是**布尔参数**（非 ordType），用于衍生品 net 模式；long/short 模式下平仓单天然 reduce-only。
- **无真正无保护的市价单**：`market` 受每合约 `maxMktSz`/`maxMktAmt` 限制；衍生品无 `slippagePct`（仅现货/杠杆现货有）。永续 HFT 用 `optimal_limit_ioc`。
- 合约可进入 `post_only` **实例状态**，此时只接受 post-only 限价单。

## 3. 限速（精确值，已核验 → 已写入配置/限速器）

| 维度 | 上限 | 计数规则 |
|---|---|---|
| 单笔 place / cancel / amend | 各 **60 次/2s** per (子账户, 合约) | 三者**独立计数** |
| 批量 place/amend/cancel | **300 次/2s**，每批 ≤20 单 | 与单笔独立；批内每单单独计入子账户上限 |
| 子账户总量 | **1000 次/2s** | **只算新单+改单，撤单不算** |

- 700ms/合约 的重挂循环 ≈ 3 ops/2s/合约，**远低于**单合约 60/2s 上限 → 单合约不是瓶颈；**多合约时子账户 1000/2s 才是真正天花板**。
- 重挂选 **amend vs cancel+place** 的权衡：撤单不计入子账户 1000/2s（amend 计入）；但 amend 延迟更低且保留队列位置。默认 WS amend，子账户上限吃紧时切 cancel+place。
- 超限是**拒单不排队**。判据看**响应体 error code**（`50011` 端点/IP/UID 限速；`50061` 子账户限速），常以 HTTP 200 返回——**不能只看 HTTP 状态**。`50013`=系统繁忙（退避重试），HTTP 429=服务器过载。

## 4. 行情数据（散户免费栈，已核验）

- **无 VIP 门槛**：`bbo-tbt`（L1 顶档，~10ms，无 checksum）、`books5`（5 档快照，100ms，无 checksum）、`books`（**400 档增量**，~100ms；首条全量、之后增量、size=0 删档）、`trades`（主动方→流向）、`funding-rate`、`mark-price`（~200ms）、`index-tickers`（~100ms）。
- **`books-l2-tbt` / `books50-l2-tbt`（10ms 全深度）现需 VIP4+**（2026-04-07 起；原方案写的 VIP5/VIP4 已过时）。1000U 账户**够不到**，不要围绕 10ms 全深度设计。
- 订单簿完整性：**只用 seqId/prevSeqId**（见 §0.5）。

## 5. 合约规格（已核验）

- `GET /api/v5/public/instruments`（公共免费，20 次/2s）返回 `tickSz/lotSz/minSz/lever/state/ctType/ctVal/ctValCcy/ctMult/settleCcy/maxLmtSz/maxMktSz/...`。实测 BTC-USDT-SWAP：tickSz=0.1, lotSz=0.01, minSz=0.01, lever=100, ctVal=0.01 BTC。
- 价格按 tickSz 取整、数量按 lotSz 取整（≥minSz）；衍生品**下单单位是张数**，`ctVal*ctMult`=每张对应标的量。
- 状态枚举（2026）：`live`（唯一完全可交易）、`suspend`、`preopen`、`test`、`rebase`(SWAP 调整中停)、`post_only`、`settling`。**无通用 expired**。订阅 instruments 频道响应状态变化。

## 6. 股票永续（已核验，含风险）

- USDT 保证金、无到期、24/7、**非证券、无分红/投票**。symbol `[TICKER]USDT`（AAPLUSDT…；含 pre-IPO 如 SPACEX/OPENAI/ANTHROPIC）。资金费每 8h（触顶/底则每小时），夹在 **-1.50%~+1.50%**。
- **杠杆上限口径不一致**（挂牌通知 0.01x–5x vs 营销 up to 10x）——以**每合约规格页为准**（`use_exchange_instrument_limit:true`）。
- 费率（OKX FAQ + 2026-04 调费通知未变）：标准 **0.02% maker / 0.05% taker**；VIP3 0.01%/0.028%。以实时费率页（Futures > TradFi）为准。
- 公司行为：**无 1:1 跟踪保证**，拆股/合并/分拆可能**个案调整或结算下线**（Terms 10.5）→ 必须有下线/结算处理器。
- 周末/美股休市：流动性低、价差宽、**10% 指数带价**锚最后现货收盘价 → 周末放宽价差阈值、降仓。

## 7. 量化方法（散户能用的真实 edge）

- **OFI 与 OBI 是同一个"流动性压力"因子**：股票 LOB 中多层 OFI 互相关 >75%、PC1 解释 89% 方差（crypto 为合理外推非实测）。→ **绝不能让 "OFI多+OBI多+depth多" 叠加放大信心/仓位**；做正交化（PCA 或单一多层 OFI 特征），按一个因子敞口定仓。
- **延迟是"排名"而非绝对值决定盘口博弈盈亏**：HFT ~300-800ns 带 colo，散户 ms~s，结构性排在 FIFO 队尾、被逆向选择。**任何依赖速度的策略散户必输**；放弃 <5min 盘口竞速，把周期推到 1h/4h 或竞争更少的对。
- **可持续的散户 edge = 成本控制 + 选择性执行 + 波动定仓 + 更长周期**，不是速度。
- **Lopez de Prado 栈**：事件采样 → triple-barrier 标注（TP/SL = k×波动，ATR 1.2–1.5x / Yang-Zhang 抗跳 / 已实现方差）→ **meta-labeling** 二级模型决定"做/不做 + 下注大小"。仓位 = (账户×风险%)/止损距离，波动放大时自动缩仓。
- **回测系统性高估成交**（理想队列、touch 即成、无逆选）。报道的 order-flow ML Sharpe ~3.0–3.6 是**毛/理想横截面值**，非散户可净得。用 walk-forward + purged&embargoed CV + **Deflated Sharpe Ratio + 回测过拟合概率(PBO)**，并披露试过多少组参数。去偏后不显著就**别上线**。

## 8. 外部数据（免费优先栈，已核验）

- **SEC EDGAR**（`data.sec.gov`，免费无 key，亚秒延迟）：`submissions`（8-K/Form4 实时触发）、`companyfacts`、`companyconcept`、`frames`。硬限 **10 req/s/IP**、**必须**带描述性 User-Agent（`Name contact@email`，缺则 403）。CIK 补零到 10 位；ticker→CIK 映射在 `sec.gov/files/company_tickers.json`。**财报日期不是 EDGAR 一等字段**（从 8-K item 2.02 推断或用日历源）。
- **Finnhub 免费档**：60 calls/min + 账户级 30/s 顶（超则 429），50 symbol WS，免费财报日历 + 公司新闻 + 基础基本面；**美股 only**；情绪打分/国际/详细财务是 Premium。→ 与 EDGAR 互补做财报日期+头条。
- **Claude Haiku 4.5**：$1/$5 per MTok（Batch 五折、cache 读 0.1x），约 $0.003/次（2k 入/150 出）。用 tool-use/JSON-schema 结构化输出做固定 veto 标签（EARNINGS_PENDING / MATERIAL_8K / OFFICER_DEPARTURE / INSIDER_SELL / NO_RISK）。实时用单次（1–3s），回填用 Batch。
- 升级路径（按需，先在官网核价）：Massive（原 Polygon，2025-10-30 改名）Developer ~$79/mo 实时行情+WS+News；Finnhub Premium ~$50–100/mo；历史微观结构回测可选 Tardis.dev（仅研究）。
- **纠正**：Polygon→Massive（`massive.com`），免费档 5 req/min、延迟 15min、无实时 WS → **不能用于实时 veto**，只能回填/EOD。

## 9. 合规与可行性（必须接受的风险）

- OKX **API 协议明确允许**自动化/算法/AI-agent 交易（3.4 节），但**禁止**对敲/分层/幌骗/报价填塞/动量点火、延迟套利/漏洞利用、限速规避或拖垮基础设施的负载（8.1 节）；OKX 对 bot 订单**免责**并要求用户**赔偿**（11.2/11.4）。→ **节流不当的激进重挂可能被判定为滥用。**
- 撮合引擎在**香港**（阿里云 cn-hongkong）；零售**无 colocation**。新加坡/东京/香港的便宜 VPS（个位~低两位数 ms）已捕获 ~90% 实际收益；为最后几 ms 砸钱 ROI≈0。
- **经济学基准 = 打平减费 + 真实爆仓风险**：maker edge 只是价差的一小部分，必须先覆盖 ~0.04%（永续）/~0.16%（现货）往返 + 滑点 + 逆选；负费率返佣在 1000U 够不到。
- **逆向选择**：挂单主要在行情逆着你走时成交（价跌时买入、价涨时卖出），累积"毒性库存"；趋势行情会把 maker edge 变负。
- **杠杆/爆仓**：高杠杆下小幅逆动 + 周期资金费会把强平价拉向开仓价。**杠杆压到 1–3x**、硬性仓位/库存/日亏上限。

---

## 10. 本次调研对配置/代码的具体改动

| 改动 | 文件 |
|---|---|
| 限速：per-(端点,合约) 60/2s + 批量 300/2s + 子账户 1000/2s(撤单不计) | `config.yaml` / `rate_limiter.py` |
| 区域路由 base URL（global/us/eea）可配 | `okx_rest.py` / `.env.example` |
| 新增 `cancel-all-after` 死手开关 | `okx_rest.py` / `config.yaml` |
| 订单类型补 `optimal_limit_ioc` | `enums.py` |
| 数据频道：免费栈 bbo-tbt+books(400档)+trades；l2-tbt 标 VIP4 不用 | `config.yaml` |
| 订单簿完整性走 seqId（checksum 废弃）——待 marketdata 层实现 | （待建）|
| 股票永续运行时资格门控 fail-closed——待登录后实现 | （待建）|
| WS 登录签名用 EPOCH 秒——待 WS 层实现 | （待建）|

> 完整原始调研（含每条来源 URL、对抗核验裁决）任务 ID `w8q9h7v3h`。
