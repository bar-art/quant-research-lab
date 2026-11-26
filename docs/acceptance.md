> 本文件保留初版MVP验收记录；逐时点模型升级见 [实施步骤](LLM升级实施步骤.md) 与 [升级验收](llm_upgrade_acceptance.md)。

# MVP验收记录

日期：2026-11-24。独立目录 `quant-research-lab`，未修改原 A 股工程。交付范围为可运行、可测试的多角色量化研究 MVP。

## 检查结果

环境：Python3.11.18、numpy2.4.6、pandas3.1.1、Pydantic2.14.3。

| 检查 | 结果 |
|---|---|
| ruff check . | 通过 |
| ruff format --check . | 通过 |
| mypy src | 11个源码文件通过 |
| pytest | 46 passed，无警告 |
| 代码覆盖率 | 96.90%，门槛80% |
| pip check | 无依赖冲突 |
| 默认端到端演示 | 成功：8个滚动测试段、24次候选评估 |

核心测试覆盖：未来日期访问与延迟可得信息；专门的明日收益作弊因子；训练结束后改动行情不影响本折选参；手算净值/费用；现金、T+1、停牌、涨跌停与缺失字段拦截；模型越权提议和虚构证据；本地模型JSON Schema、调用预算和响应长度；记忆时点与事务回滚；HTML转义及输入数据哈希往返。

## 2026-11-24 改进（第二轮）

| 检查 | 结果 |
|---|---|
| ruff check / format --check | 通过 |
| mypy src | 11个源码文件通过 |
| pytest | 55 passed（新增9项） |
| 代码覆盖率 | 96.82% |
| 默认演示 | 收益、回撤、成交数与旧版完全一致（默认值不改变结果） |

修复：
- 清仓后的标的不再每天登记虚假的 `missing_bar` 阻断（旧版在标的退市/缺行情时会反复登记）。
- 等权基线也遵守单票上限（标的少于 max_gross/max_position 时旧版基线超过单票上限）。
- 回撤熔断：旧版触发后永久空仓且报告不提示；现记录熔断日期、写入局限说明，并提供 `drawdown_reset="next_fold"` 选项。

新增证据：相对基线超额与信息比率、逐段跑赢基线、PSR、换手与成本、预测命中率/MAE/Rank IC。默认演示结果：超额 -3.92%，8段中3段跑赢基线，PSR 3.9%，年化单边换手 20.4 倍（基线 1.2 倍），成本占起始资金 1.75%，Rank IC 均值 -0.15。新增可选卖出税费 `stamp_duty_bps`。

## 可查看的交付物

- `output/demo/report.html`：自包含研究报告。
- `output/demo/research.json`：全部滚动选择、会议、成交、预测及评分。
- `output/demo/audit.sqlite`：每日会议、预测登记与到期反馈。
- `output/demo/curve.csv`、`trades.csv`、`blocked_orders.csv`、`forecast_scores.csv`。
- `output/demo/input_bars.csv` 和 `input_manifest.json`：可再次导入的输入快照。
- `output/demo/reproducibility.json`：环境、种子、源码与文件SHA256。
- `output/verification.txt`：检查原始输出。

默认输入为合成工作日行情，不是实际交易日历，收益只用于验证工程。模型API使用模拟HTTP测试，尚未运行真实本地模型；本地模型是否提升收益未经验证。历史绩效始终来自规则策略，本地模型会议不写入历史绩效。

报告已用独立临时 Chrome 配置渲染并查看截图，文字、卡片、净值图和表格未发现溢出；截图保存在 `output/report-preview.png`。

## 后续缺口

完整A股规则与税费历史、公司行动、完整退市终值、财务PIT、新闻/行业风险、数据供应商适配、真正的前瞻模型纸面实验和受控策略记忆更新均未完成。当前不接实盘，不自动修改提示词或风控，不声称复现论文收益。

仅供研究，不构成投资建议。
