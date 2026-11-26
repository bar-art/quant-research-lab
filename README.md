# Quant Research Lab

**先验证研究证据，再讨论模型能力。**

一个独立的多角色量化研究 MVP：四角色委员会、滚动样本外检验、训练收益情景模拟、确定性风控、下一开盘模拟账本、到期预测反馈和可复现报告。

参考 [yebof/quant-agent](https://github.com/yebof/quant-agent)、[QuantAgents 论文](https://arxiv.org/abs/2510.04643)，并通过 GitHub 仓库搜索比较了部分同类工程。具体借鉴、取舍与阅读范围见 [来源研究](docs/source_review.md)。本项目独立编写，没有复制上游实现或提示词。

## 已实现

- 四个角色：市场分析、模拟分析、风险分析、经理。支持规则对照和逐时点大模型回放；结构化提议必须引用已知证据。
- 三类会议记录；模型只有提议权，硬风控独立校验仓位、现金、亏损和回撤。
- `available_at` 时点过滤；显式读取未来数据立即拒绝。
- 每折只用训练段选择动量窗口，单独报告训练与测试结果、候选次数和多重检验风险。
- 训练收益 bootstrap 情景；下一开盘模拟成交、成本、停牌/开盘封板阻断、购买当日不可卖出。
- 预测到期后评分、带时间戳记忆检索；SQLite 会话与反馈原子写入。
- 证据强度：相对等权基线的超额收益与信息比率、逐段是否跑赢基线、PSR（夏普>0的概率，考虑样本长度/偏度/峰度）、年化换手与成本占比、预测命中率/平均绝对误差/Rank IC、风控熔断日期。
- 自包含 HTML、行情/净值/成交/评分 CSV、完整 JSON、SQLite 审计、源码与交付物 SHA256。
- 本地 Ollama 逐时点四角色会议：模型输入当时可得信息，自主提议仓位；JSON Schema、证据ID、股票池/权重约束、调用预算与模型digest锁定。
- 同条件模型/规则对照、配对主动收益区间，以及先登记、实时决策、到期结算的前瞻工具。

**`demo` 保持完全离线的规则对照；`compare` 让大模型参与每个回放决策，并与规则委员会比较。`--ollama` 仍保留为旧版末期快照模式。真正前瞻测试使用独立的 `forward-*` 命令。**

逐步执行命令与证据边界见 [LLM升级实施步骤](docs/LLM升级实施步骤.md)。

## 快速运行

Python 3.11+。当前目录的 `.venv` 已配置完成，可直接：

```bash
cd quant-research-lab
source .venv/bin/activate
quant-lab demo --output output/my-demo
open output/my-demo/report.html
```

已有演示报告：`output/demo-v2/report.html`（含"证据强度"一节；`output/demo/` 为旧版本输出，保留不覆盖）。输出目录必须为空；再次实验请使用新目录，不覆盖旧实验。

换一台机器：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install -e .
quant-lab demo --output output/demo
```

两个可选配置（默认值保持原有结果不变）：

- `stamp_duty_bps`：卖出单边税费，例如 A 股印花税（2023-08-28 起 5bps）。默认 0。
- `stamp_duty_schedule`：按成交日分段的卖出印花税 `[[生效日, bps], ...]`，与固定税率二选一。`config/ashare.json` 为 A 股口径：2008-09-19 起 10bps，2023-08-28 起 5bps；早于首个生效日的卖出直接报错，不猜测税率。
- `drawdown_reset`：`never`（默认）回撤熔断触发后不再开仓；`next_fold` 在下一个测试段重新选参时以当时净值为新高点重新计数。熔断日期会写入报告与 `research.json`，`never` 下熔断后的样本外区间主要是现金，报告局限中会明确提示。

`requirements.in` 固定直接依赖，`requirements.lock` 固定完整开发/运行依赖。`config/default.json` 提供参数示例；`--config config/default.json` 启用自定义参数。

## 使用自己的数据

```bash
quant-lab research --input examples/bars.csv --manifest examples/manifest.json \
  --config config/default.json --output output/custom
```

`examples/bars.csv` 是小型人工样本，不能当真实历史业绩。CSV 必须包含：

`ts_code, trade_date, open, high, low, close, volume, available_at, is_suspended, up_limit, down_limit, price_limit_exempt`

- `available_at` 必须带时区，例如 `2024-01-02T15:00:00+08:00`。完整日线不能标记成开盘前已知。
- `is_suspended`、`price_limit_exempt` 必须显式为布尔值，不能用未知值代替。
- 涨跌停价未知且没有明确豁免时，账本拒绝成交。
- manifest 声明数据来源、历史股票池和公司行动完整性；声明不是本项目独立核验的结论。
- 缺失/退市后持仓暂以最后已知价估值并阻断成交，不会自动假装清算成功。

现有 `a-share-quant-agent` 保持原样；其行情缺少部分历史状态和 `available_at`，需要补充真实来源后再导入，不能直接补值冒充完整历史数据。

## 模型回放与同条件比较

```bash
quant-lab compare --model llama3.2:latest \
  --model-release 2024-10-24 \
  --release-source https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md \
  --config config/llm-smoke.json --sessions 100 --start-date 2026-01-02 \
  --output output/my-llm-smoke
```

该配置有3个样本外时点、4次会议、16次角色调用；输出`comparison.html`，每组都有完整报告、审计和原有证据指标。人工短样本只验证真实模型接入，不判断投资效果。

### 云端模型：OpenRouter

`--backend openrouter` 使用同一个委员会协议调用云端模型。密钥只从 `OPENROUTER_API_KEY` 环境变量读取，不写进审计日志、缓存或报错信息。

```bash
export OPENROUTER_API_KEY=...        # 或 set -a; . ./.env; set +a
quant-lab compare --backend openrouter --model openai/gpt-5-nano \
  --openrouter-provider openai/flex --reasoning-effort minimal --max-cost-usd 1.0 \
  --model-release 2025-08-07 --release-source https://openai.com/index/introducing-gpt-5/ \
  --config <config> --input <bars.csv> --manifest <manifest.json> \
  --model-cache output/model_cache --output output/my-llm-replay
```

- **严格结构化输出：** 所有字段必填、禁止额外字段，权重用 `[{symbol, weight}]` 列表表示，只能引用输入里出现的证据和股票。
- **格式修复：** 回复不合规或为空时，把错误告诉模型重答，最多 2 次；每次都计入调用次数和费用并写入审计。仍不合规就停止，不改用规则结果。
- **费用控制：** `--max-cost-usd` 是硬上限。发给模型的证据压缩成"证据ID|收盘价|历史条数"，信息不变，原始输入仍完整保存在审计日志里。
- **可复现：** `--openrouter-provider` 固定单一服务商端点并禁止回退；只发送目录中该模型支持的参数（推理模型不支持 `temperature`）。OpenRouter 不提供权重哈希，登记的是目录 slug，报告里明确写出这一点。
- **前视检查：** 第一个模型决策必须晚于模型发布日期，否则拒绝运行。

#### 实测：gpt-5-nano 与规则委员会（2026-11-26）

数据为沪深300历史成分 2025-08-08 至 2026-11-24，只保留期间当过成分股的 332 只。80 日训练后，样本外 233 个交易日、980 次模型调用全部合规，实际费用 0.47 美元。

| 2025-12 至 2026-11 | 总收益 | 年化 | 夏普 | 最大回撤 |
|---|---|---|---|---|
| 规则委员会（动量） | −1.0% | −1.1% | 0.15 | 37.4% |
| gpt-5-nano 委员会 | −33.8% | −36.0% | −2.05 | 34.3% |
| 等权基线 | −6.3% | −6.8% | −0.72 | 10.2% |

- 两组训练选参完全一致，源码哈希与登记一致；模型的仓位决策只有 2.4% 与规则相同，说明它确实在独立决策。
- 日均超额收益的配对区块 bootstrap 95% 区间为 [−0.42%, +0.07%]，包含 0。单条市场路径、单个模型的 233 天样本，**不足以断定模型更差，但看不到任何优势**。
- 局限：只测了一个小模型，推理强度为 minimal；只有价格证据，没有新闻或财务信息；结果不能推广到其他模型或设定。

过程记录：最初使用 gpt-oss-120b（DeepInfra）。它在约 20% 的调用中输出不合规，还偶尔陷入重复生成，导致运行中断，所以改用 OpenAI 端点严格执行格式的 gpt-5-nano。

## 兼容：旧版末期模型会议

先自行安装并启动 Ollama、准备支持结构化输出的本地模型。`.env.example` 只是变量模板；本项目不自动执行或读取 `.env`，请显式导出：

```bash
export OLLAMA_MODEL='你已安装的本地模型名'
export OLLAMA_BASE_URL='http://127.0.0.1:11434'
quant-lab demo --ollama --output output/local-model
```

四次模型调用用于最新时点的三位分析师与经理提议。模型不能运行代码、改风险配置或提交交易。模型输出和输入保存于 `research.json`；调用失败或协议不符会停止，不悄悄换成规则结果。旧版命令不改变历史回放；逐时点模型请使用上面的`compare`。本机真实模型接入验收见升级记录，收益有效性仍需更充分的实验。

## 实测：沪深300历史成分 2015-2026（2026-11-26）

输入由 `a-share-quant-agent` 从 Tushare 本地快照导出：668 只曾经的成分股（含之后调出和退市的），1,737,796 行日线，逐日 `universe_member` 来自 143 次月度成分快照，后复权价格，涨跌停按板块规则计算。

```bash
quant-lab research --input <export>/bars.csv --manifest <export>/manifest.json \
  --config config/ashare.json --output output/ashare-hs300-2015-2026-dated
```

配置：`config/ashare.json` 加 `drawdown_reset: next_fold`；141 个滚动测试段，每段 80 日训练、20 日测试。

| | 年化收益 | 夏普 | 最大回撤 |
|---|---|---|---|
| 规则策略（动量前2，窗口 5/10/20 日滚动选择） | −30.3% | −0.99 | 98.4% |
| 等权基线（同成本、同仓位上限） | +0.3% | 0.09 | 34.5% |

策略严重亏损。我用独立的向量化计算核对过：同样每天买入动量最强的 2 只、次日开盘成交，不计成本也每年亏 18%–29%（收盘到收盘计算则在 −17% 到 +9% 之间）。A 股强势股常在隔夜高开后日内回落，这是策略本身的问题，不是引擎错误。另外年换手约 54 倍，成本累计约占初始资金的 22%。这个结果只说明简单短期动量在该设定下无效，不构成对任何策略的推荐。

整段研究耗时 8 分钟；优化前为 78 分钟，结果逐字段一致，见 [性能记录](docs/performance_indexing.md)。

## 验收

```bash
sh scripts/verify.sh
```

检查包括 ruff、mypy 和 pytest，覆盖率门槛80%。测试不联网；已包含未来因子拦截、训练/测试隔离、手算费用、风控越权、到期记忆、原子写入和模型协议测试。完整结果见 [验收记录](docs/acceptance.md)。

## 研究边界

这是使用碎股和统一手续费/滑点的**研究模拟**，不是完整 A 股回测引擎。未完成各板块历史规则、税费分段、IPO例外、公司行动、完整退市终值、财务PIT和新闻模块。报告会保留这些局限。bootstrap 不是未来收益保证，四个角色不意味着四个独立信息源，简单规则演示不证明大模型具有收益优势。

未来可以在已有框架上扩展国内数据适配器、完整A股撮合、前瞻模型纸面实验、行业/风格风控和受控记忆策略。当前没有实盘下单、自动改提示词或强化学习训练。

仅供研究，不构成投资建议。

## 临时数据源与恢复回放

双项目哈希数据包、逐日股票资格、下一交易日可得模式及模型响应缓存继续使用。金融数据默认切换至免费 BaoStock；原 Tushare 方案已经停用，仅保留 [历史技术记录](docs/tushare_one_day.md)。当前运行方法见 [免费数据说明](docs/open_data.md)。

## 默认金融数据：免费 BaoStock

已按用户选择停用 Tushare 工作流。运行 `bash scripts/run_open_data.sh`，默认校验并使用本地真实 BaoStock 缓存，为 quant 和 alpha 生成同一个数据包并运行研究；不会请求卡号或自动切换到付费源。在线补齐缺口需显式加 `--online`。见 [免费数据运行说明](docs/open_data.md)。

2026-11-25 已用6只股票、7,144条历史日线完成两个项目的端到端运行；最新在线探测超时，结果来源标记为已验证缓存。原 Tushare 模块仅保留作旧实验兼容，旧 shell 入口已停止调用。
