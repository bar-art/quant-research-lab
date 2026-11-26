# Quant Research Lab

[![tests](https://github.com/bar-art/quant-research-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/bar-art/quant-research-lab/actions/workflows/tests.yml)

**在同样的数据、风控和交易约束下，大模型委员会能比简单规则做得更好吗？**

这是一个 A 股研究回测框架。四个角色（市场分析、模拟分析、风险分析、经理）组成投资委员会，逐日提出仓位；仓位可以由确定性规则生成，也可以交给大模型。两种委员会使用同一份数据、同一套硬风控和同样的成交规则，比较结果才有意义。

设计参考了 [yebof/quant-agent](https://github.com/yebof/quant-agent) 和 [QuantAgents 论文](https://arxiv.org/abs/2510.04643)，代码和提示词独立编写。借鉴与取舍见 [docs/source_review.md](docs/source_review.md)。

## 结论

**大模型委员会没有表现出优势。** 数据为沪深300历史成分中当过成分股的 332 只股票，样本外 233 个交易日（2025-12 至 2026-11，晚于模型发布日期），980 次模型调用全部合规，费用 0.47 美元。

| | 总收益 | 年化 | 夏普 | 最大回撤 |
|---|---|---|---|---|
| 规则委员会（短期动量） | −1.0% | −1.1% | 0.15 | 37.4% |
| gpt-5-nano 委员会 | −33.8% | −36.0% | −2.05 | 34.3% |
| 等权基线 | −6.3% | −6.8% | −0.72 | 10.2% |

模型的仓位决策只有 2.4% 与规则相同，说明它在独立判断。日均超额收益的 95% 配对区间为 [−0.42%, +0.07%]，包含 0：仅凭一条市场路径和一个小模型，还不能断定它更差。

**简单短期动量在 A 股严重亏损。** 用 2015–2026 年沪深300的完整历史成分（668 只，含后来调出和退市的股票）回测，每天买入动量最强的 2 只股票，年化 −30.3%，同期等权基线 +0.3%。用独立的向量化计算核对过：次日开盘买入前一天的赢家，即使不计成本，每年也亏 18%–29%，因为 A 股强势股常在隔夜高开后日内回落。这是策略本身的问题，不是回测引擎的错误。

这项 174 万行的研究跑完需要 8 分钟。优化前要 78 分钟，优化前后的结果逐字段一致，见 [docs/performance_indexing.md](docs/performance_indexing.md)。

## 怎样保证比较公平

1. **只用当时可得的数据。** 每行行情都带 `available_at` 时间戳，每个决策只能看到截至决策时点已公开的数据，读取未来数据会直接报错。
2. **贴近 A 股的成交规则。** 收盘出信号，次日开盘成交；买入当天不能卖出（T+1）；停牌、开盘涨停时不能买入，开盘跌停时不能卖出；含滑点的成交价也不能越过涨跌停价；印花税可以按日期分段。
3. **滚动样本外检验。** 每个测试段前只用训练段选择参数，训练和测试结果分开报告，并给出候选次数和多重检验风险。
4. **模型只有提议权。** 提议必须引用输入里出现的证据；仓位、现金、单日亏损和回撤由独立的硬风控审核，模型无法绕过。
5. **相同条件对照。** 规则委员会和模型委员会的训练选参、风控和成本完全一致；第一个模型决策必须晚于模型的发布日期。
6. **可复现。** 每次运行保存完整 JSON、SQLite 审计、成交和净值 CSV，以及源码和交付文件的 SHA256。

## 快速开始

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install -e .

quant-lab demo --output output/demo        # 人工合成行情，离线运行
open output/demo/report.html
```

每次运行都要用新的输出目录，避免覆盖旧实验。

## 使用自己的数据

```bash
quant-lab research --input examples/bars.csv --manifest examples/manifest.json \
  --config config/default.json --output output/custom
```

`examples/` 里是小型人工样本。CSV 需要包含：

`ts_code, trade_date, open, high, low, close, volume, available_at, is_suspended, up_limit, down_limit, price_limit_exempt`

可选列 `universe_member` 标记当天是否属于股票池：不在池中的股票仍保留行情，用于估值和卖出，但不会被新买入。

- `available_at` 必须带时区，例如 `2024-01-02T15:00:00+08:00`；完整日线不能早于当天收盘。
- `is_suspended`、`price_limit_exempt` 必须是明确的布尔值。
- 涨跌停价未知且没有豁免时，不会成交。
- 股票退市或缺少行情后，持仓按最后已知价格估值，并停止交易，不会假设已经卖出。
- `manifest.json` 声明数据来源、历史股票池和公司行动是否完整，这些声明由数据提供方负责。

真实 A 股数据推荐用 [ashare-data-pipeline](https://github.com/bar-art/ashare-data-pipeline) 导出。它整理停牌、ST、昨收、按板块规则计算的涨跌停价和历史指数成分，并与第二个数据源逐日核对。本项目只读取 CSV 和 manifest，不依赖任何数据商，可以单独克隆运行。

## 大模型对照（`compare`）

`compare` 在同一份数据上分别跑规则委员会和模型委员会，输出各自的完整报告和一份 `comparison.html`。支持本机 Ollama 和云端 OpenRouter 两种后端。

**OpenRouter：**

```bash
set -a; . ./.env; set +a        # .env 中写 OPENROUTER_API_KEY，不要提交
quant-lab compare --backend openrouter --model openai/gpt-5-nano \
  --openrouter-provider openai/flex --reasoning-effort minimal --max-cost-usd 1.0 \
  --model-release 2025-08-07 --release-source https://openai.com/index/introducing-gpt-5/ \
  --config <配置> --input <bars.csv> --manifest <manifest.json> \
  --model-cache output/model_cache --output output/llm-replay
```

**Ollama（本机模型）：**

```bash
quant-lab compare --model llama3.2:latest --model-release 2024-10-24 \
  --release-source https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md \
  --config config/llm-smoke.json --sessions 100 --start-date 2026-01-02 --output output/llm-smoke
```

`config/llm-smoke.json` 只有 3 个样本外时点，用来验证模型能否正常接入，不用于判断投资效果。

模型调用遵守以下规则：

- **严格格式：** 输出由 JSON schema 约束，只能引用输入里的证据和股票，权重不能超过风控上限。
- **格式修复：** 回复不合规或为空时，把错误告诉模型重答，最多两次，每次都计入调用次数和费用。仍不合规就停止，不会改用规则结果顶替。
- **费用硬上限：** 每次请求前按最坏情况预留费用，预留后会超过 `--max-cost-usd` 就停止发送；服务商返回的费用缺失时不当成 0。
- **可复现：** `--openrouter-provider` 固定单一服务商端点并禁止切换；Ollama 锁定模型的权重 digest。OpenRouter 不提供权重哈希，登记的是模型目录名，报告中会注明。
- **密钥安全：** 密钥只从环境变量读取，不会出现在审计日志、缓存或报错信息里。

## 前瞻测试（`forward-*`）

回放用的是历史数据，模型可能"见过"这些行情。真正的前瞻测试要先登记，再等未来数据产生：

```bash
quant-lab forward-init --model <本机模型> --model-release <日期> --release-source <链接> \
  --config config/forward-example.json --universe <股票代码> --calendar <交易日历.json> \
  --output output/forward-study
quant-lab forward-step   --input <最新行情.csv> --manifest <manifest.json> --output output/forward-study
quant-lab forward-settle --input <已实现行情.csv> --manifest <manifest.json> --output output/forward-study
```

登记后不能换模型、改代码，也不能提前评分或事后补写决策。日历格式见 `examples/forward_calendar.example.json`。前瞻测试目前只支持 Ollama，尚未启动正式的市场登记。

## 常用配置

配置文件放在 `config/`，用 `--config` 指定：

- `default.json`：参数示例；
- `ashare.json`：A 股印花税分段（2008-09-19 起卖出 10bps，2023-08-28 起 5bps）。

几个重要参数：

| 参数 | 含义 |
|---|---|
| `windows`、`training_sessions`、`test_sessions` | 候选动量窗口、训练段和测试段长度 |
| `top_k`、`policy` | 持仓数量，单票、总仓位、现金、回撤和单日亏损上限 |
| `commission_bps`、`slippage_bps` | 手续费和滑点 |
| `stamp_duty_bps` / `stamp_duty_schedule` | 固定或分段的卖出印花税，二选一；早于首个生效日的卖出会报错 |
| `drawdown_reset` | `never`：回撤熔断后不再开仓；`next_fold`：下个测试段以当时净值为新高点重新计算 |

## 输出

每次运行生成：

- `report.html`：离线可打开的报告；
- `research.json`：完整记录，包括决策、会议、选参和指标；
- `curve.csv`、`trades.csv`、`blocked_orders.csv`：净值、成交和被拒订单；
- `audit.sqlite`：会话审计；
- `reproducibility.json`：源码和交付文件的哈希。

报告里的指标包括：

- 相对等权基线的超额收益和信息比率，每个测试段是否跑赢基线；
- PSR（夏普为正的概率，考虑样本长度、偏度和峰度）；
- 年化换手和成本占比；
- 预测命中率、平均绝对误差、Rank IC；
- 风控熔断日期。

## 局限

- **这是研究模拟，不是完整的 A 股回测引擎。** 允许碎股，手续费和滑点按比例收取；尚未实现 100 股交易单位、最低佣金、公司行动现金流、退市清算价值、时点准确的财务数据和新闻输入。
- **大模型对照只测了一个小模型和一种推理设置，** 样本外只有 233 个交易日，输入也只有价格。
- **模型发布日期之后的数据可以减少、但不能完全排除模型训练知识的影响。** 前瞻测试才能完全排除，目前工具已经具备，但还没有正式运行。
- **四个角色不代表四个独立信息源。** bootstrap 情景只描述历史收益的分布，不是对未来的预测。
- **没有实盘下单、自动改提示词或强化学习训练。**

## 开发

```bash
sh scripts/verify.sh        # ruff、mypy、pytest（离线，覆盖率门槛 80%）
```

测试覆盖未来数据拦截、训练和测试隔离、手算费用、风控越权、模型协议、费用上限和原子写入。设计说明见 [docs/architecture.md](docs/architecture.md)，验收记录见 [docs/acceptance.md](docs/acceptance.md)。跨项目数据包和 Tushare 归档工具是可选的，分别见 [docs/open_data.md](docs/open_data.md) 和 [docs/tushare.md](docs/tushare.md)。

数据文件和模型调用记录都放在 `output/`、`data/` 下，不提交到仓库。

仅供研究，不构成投资建议。
