# 从研究框架到逐时点大模型 Quant Agent

日期：2026-11-24。改动前的src、配置、原有55项测试和文档保存在 `output/upgrade_baseline_20261124/`；原有证据指标、税费参数和回撤重置设置均保留。

本次目标是让模型真实参与仓位决策，并检验其增量。模型仍无权运行任意代码、修改硬风控或提交实盘交易；这些权限边界与是否由LLM驱动是两件事。

## 第一步：核对模型、数据和调用规模

本机读取到 `llama3.2:latest` 与 `qwen2.5-coder:7b`。真实接口验收先采用前者，未下载新模型或调用付费服务。

```bash
cd quant-research-lab
source .venv/bin/activate
ollama list
```

适配器通过 `/api/tags` 获取模型digest，在每次生成前确认标签仍指向同一模型。请求保存seed、temperature、schema、输入、原始输出、tokens和耗时。固定seed和温度不承诺跨硬件逐字确定性；保存原始回答是复核依据。

- [Ollama原生chat接口](https://docs.ollama.com/api/chat)
- [JSON Schema结构化输出](https://docs.ollama.com/capabilities/structured-outputs)
- [模型列表与digest](https://docs.ollama.com/api/tags)
- [Meta Llama3.2官方模型卡](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md)：当前页面列出2024-10-24作为发布日期。本项目把该日期作为带来源的操作者声明记录，不把安装日期当发布日期，也不声称独立验证了每个第三方量化包的训练历史。

调用预算在开始模型回放前计算：

`4 ×（样本外时点数 + 滚动测试段数）`

每个收盘一次四角色会议；每次换训练窗口时，另在该段第一个执行日前一收盘刷新提议。后者可能替代此前生成的下一日提议，日志用 `fold_refresh` 和 `close` 区分。模型不会在预算不足时悄悄只参与最后一个时点。

## 第二步：先跑短的真实模型链路验收

```bash
quant-lab compare --model llama3.2:latest \
  --model-release 2024-10-24 \
  --release-source https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md \
  --config config/llm-smoke.json --sessions 100 --start-date 2026-01-02 \
  --output output/my-llm-smoke
```

100期中97期作初始训练，3期作回放，预计16次模型调用。默认仍是人工数据，但这次使用真实Ollama推理，不是模拟接口。

新流程：

1. 数据视图只提供截止时点已知证据；不向模型传递未来收益或未来价格。
2. 市场、模拟、风险角色分别生成短证据笔记，经理看到这些笔记后自主提议仓位。
3. 模型的股票代码、单票上限和持仓数量同时受输出schema约束，合计仓位等约束仍由Python复核；现金不作为股票代码。
4. 获批目标在下一日开盘进入真实的模拟账本，产生持仓、成本、净值与未成交记录。
5. 协议错误或模型失败会保留调用日志和失败状态，整段对照不会伪装成已完成，也不自动换回规则。

第一轮真实调用发现模型会提交超限权重甚至cash代码，已保留在 `output/llm-replay-smoke-20261124/`，用于开发诊断，不用于投资效果结论。随后加入候选代码和单票/数量约束；v3在风险角色出现重复长中文、输出截断，已保存失败记录于 `output/llm-replay-v3-20261124/`。v4改用适合本机Llama模型的短英文角色摘要，运行结果保存于 `output/llm-replay-v4-20261124/`。改进依据是接口合规性，不是为了挑选高收益结果。

## 第三步：在同一套回放中比较增量

对照两组共享：数据与日期、训练选窗、交易成本、滑点、印花税假设、持仓上限、回撤/亏损熔断、重置规则和下一开盘撮合。

两组各自保持组合与风控状态；因为此前决策不同，后续回撤和现金可能不同，这是模型政策的结果，不是更改风控阈值。

交付物：

- `registration.json`：开始前的数据、配置、模型来源、调用预算和源码哈希。
- `source_snapshot/`：运行开始时的Python源码快照。
- `model_calls.jsonl`：每个真实角色请求和响应，失败也记录。
- `rules/` 与 `llm/`：各自完整报告、CSV、SQLite会话与决策、净值和原有证据强度指标。
- `comparison.json/html`：模型相对规则的收益差、主动收益、成本、熔断、仓位一致率、风控否决率及配对描述区间。

读结果时注意：

- PSR保留为每组策略的描述量，不是“模型带来增量的概率”。
- 旧版预测命中率、MAE、Rank IC来自规则动量诊断，已在模型报告中明确标注；不能称为大模型预测准确率。
- 配对移动区块bootstrap使用同日净收益差，保留部分时间相关结构。少于25个观测或五个区块时返回 `insufficient_evidence`。
- 区间未校正跨模型、跨实验反复试验等选择；一条行情路径和正收益差都不足以证明模型的稳定增量。
- 被风控否决造成的现金收益，不应描述为模型主动避险。

更长的人工回放示例：

```bash
quant-lab compare --model llama3.2:latest \
  --model-release 2024-10-24 \
  --release-source https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md \
  --config config/llm-replay.json --sessions 240 --start-date 2025-01-02 \
  --output output/my-long-replay
```

160个样本外时点、8个测试段，预计672次角色调用。尚未替用户运行这项长试验；这与16次接口验收的证据等级不同。替换为用户数据时提供 `--input` 和 `--manifest`，且首个模型决策须严格晚于声明的发布日期。

## 第四步：区分“发布后历史回放”和真正前瞻

发布后历史回放是在今天重放过去的信息环境，不是当时已经做出的预测。模型卡和时间戳过滤可以减少部分污染疑虑，但不能证明所有权重、后训练数据及模型服务从未接触测试事件。

真正前瞻需要先登记：模型名称/digest、带来源的发布时间、代码哈希、研究配置、股票池，以及每期的决策/执行/结算日历。

**按用户最新选择，当前只完成工具，股票池稍后确定；未建立已启动的市场前瞻登记，也未产生未来收益。**

股票池及已核对的交易日历确认后：

```bash
quant-lab forward-init --model llama3.2:latest \
  --model-release 2024-10-24 \
  --release-source https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md \
  --config config/forward-example.json \
  --universe 替换为确认后的代码列表 \
  --calendar 替换为核对后的calendar.json \
  --output output/my-forward-study
```

日历示例格式见 `examples/forward_calendar.example.json`。示例只说明字段格式，不是已经核对的交易日历；必须在登记前替换。每一期固定为上海时区收盘决策、后续交易日开盘执行、执行日收盘结算。

到决策窗口，先结算上一期，再用截至当前收盘已知的数据生成提议：

```bash
quant-lab forward-step --input 最新已知行情.csv --manifest 数据来源声明.json \
  --output output/my-forward-study
```

到本期执行日收盘、完整行情已公开后：

```bash
quant-lab forward-settle --input 更新后的已实现行情.csv --manifest 数据来源声明.json \
  --output output/my-forward-study
```

系统拒绝未来决策、提前评分、执行时点之后补写、重复挑选同一期模型输出、换模型或登记后改代码。前瞻模式拒绝synthetic声明。数据缺失、代码变更或模型失败会中断研究，不生成完整收益结论。`forward_grade.json` 以 `in_progress` / `complete` 标明实际完成情况。

## 第五步：验收与证据分级

```bash
sh scripts/verify.sh
```

应分别记录：

1. 单元/模拟HTTP协议测试：验证代码，不代表真实模型能力。
2. 真实模型＋人工行情：验证模型输出进入交易链路，不代表策略有效。
3. 真实行情的发布后回放：验证特定样本的历史表现，污染风险仍需说明。
4. 先登记、实时记录、未来到期的前瞻：按实际完成期数评估，不提前宣布结果。

本次推进到了前两类的运行验收，并完成第四类工具及时间约束测试；第三类真实市场长回放和第四类未来市场结果没有伪造或替代。

前瞻工具依赖本机时钟和用户提供的数据来源声明，尚无第三方时间戳、外部签名或行情审计。本地SQLite登记也不是防止恶意本机操作者改库的安全边界。

仅供研究，不构成投资建议。
