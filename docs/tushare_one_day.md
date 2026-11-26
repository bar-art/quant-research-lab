> 历史记录：2026-11-25 已按用户要求弃用此在线方案；旧 shell 入口已停用。当前请使用 [免费数据流程](open_data.md)。以下为历史技术记录，不再是默认运行步骤。

# 临时 Tushare 兼容数据源：归档与两项目验收

日期：2026-11-25。当前状态：工程接入与离线测试已完成，真实接口连接超时，尚未验证卡号权限、到期时间或取得真实下载结果。不要把离线样本或已有其他来源的数据当成此次接口的验收结果。

## 先保留原始数据

使用已有 `.venv` 即可；HTTP JSON 客户端直接实现 Tushare 公布的请求协议，不增加 SDK 依赖。卡号只通过环境变量读取。脚本在终端隐藏输入，不写 `.env`，不打印卡号，不缓存 `token_expire` 的返回内容。第三方地址与 Tushare 官方服务不同；脚本显式使用 `--allow-http`，卡号以明文 HTTP 发送给指定服务器，禁止自动跟随重定向。

在 `quant-research-lab` 目录运行：

```bash
bash scripts/archive_tushare.sh \
  --start 20150101 --end 20261124 \
  --indices 000300.SH 000905.SH 000016.SH --all-index-members \
  --output output/tushare_archive --cache output/tushare_cache
```

它先获取交易日历、上市/退市/暂停上市证券基础资料、逐月指数成分与权重，再获取历史快照中出现过的股票与显式 `--symbols` 的并集的日线、复权因子、涨跌停、停复牌和每日指标。包含的是检索到的历史成分并集，不是所有 A 股。默认显式股票只有两只，省略 `--all-index-members` 时属于选定股票的小样本。

需要先做较小规模试跑，可将指数改为 `--indices 000016.SH` 并使用新的输出目录。运行时会报告完成项/计划项；历史股票并集确定后，计划项数量增加。任一接口无权限或异常即停止，保留已完成缓存；不把错误当作空数据。分页重复、返回日期越界、缓存哈希不匹配时也停止。

中断后运行完全相同的命令。缓存按接口、参数、字段和服务地址的哈希区分，不含卡号，所以换卡或卡号到期后仍可读取已有缓存。日期、股票、指数、是否下载历史成分并集或服务地址发生变化时必须使用新归档目录，防止混淆实验。缺少的页面仍需要有效权限；“可以离线读缓存”不等于缺失页面能够离线补齐。建议单个归档目录只有一个写入进程。

官方协议依据：[HTTP 调用说明](https://tushare.pro/document/1?doc_id=40)。字段单位依据：[历史日线](https://tushare.pro/document/2?doc_id=27)，成交量由手转换为股。指数覆盖依据：[指数成分和权重](https://tushare.pro/document/2?doc_id=96)，这是月度数据，不能自动充当每日生效成分或原始发布版本。

## 导出共同数据包

归档完成后：

```bash
.venv/bin/python -m quant_research_lab.data_bundle \
  --archive output/tushare_archive --output output/tushare_bundle

.venv/bin/python -m quant_research_lab.cli research \
  --bundle output/tushare_bundle --output output/tushare_rules

(cd ../alpha-evidence-lab && .venv/bin/python -m alpha_evidence_lab.cli research \
  --bundle ../quant-research-lab/output/tushare_bundle --output output/tushare_study)
```

alpha 默认配置要求每条路径至少 630 个时点；短期试跑需要事先定义适配长度的配置，不能看过结果再调整证据门槛。

导出会校验归档文件哈希、价格键、复权因子与连续交易日覆盖；对上市后/退市前但缺少价格的日期不会凭空填值，而是停止。停牌造成的缺行需要补充明确的停牌估值记录；当前导出器不补造这些数据。未知涨跌停不自动视作豁免，账本拒绝相应成交。复权价格、碎股和比例成本仍属于研究近似，不能替代完整实盘会计和交易规则。

`bundle.json` 为两个项目提供文件清单和哈希。alpha 将清单哈希并入路径来源，检查路径越界、重复文件、数据被改动等问题。quant 使用 `--bundle` 时也验证行情和 manifest 哈希。哈希只能核对本地文件一致性，不证明供应商数据真实、历史修订完整或第三方认证。

## 下一交易日才可得的保守模式

```bash
.venv/bin/python -m quant_research_lab.data_bundle \
  --archive output/tushare_archive --output output/tushare_bundle_lag1 \
  --visibility next_open
.venv/bin/python -m quant_research_lab.cli research \
  --bundle output/tushare_bundle_lag1 --config config/tushare_lag1.json \
  --output output/tushare_rules_lag1
```

这里把完整日线的可得时间推至下一交易日09:00，并允许最多一个观测交易时点的信号滞后。计数使用数据集的交易日期，缺失的全市场交易日不会凭空识别；导出器的日历覆盖检查因此很重要。默认信号滞后仍为0，不能在保守模式下静默放宽。交易日历额外查询到价格截止日后14天，供最后一日的滞后时间戳使用；没有下载这些未来日期的价格。

alpha 当前不支持此模式，导入时明确拒绝，不回填为当日15:00。alpha 的 `close_assumed` 依然是研究假设，不是实际供应商发布时间的证明。

## 历史股票资格

quant 支持行情逐行显式布尔字段 `universe_member`。训练特征、规则选股、模型硬风控、基线按截止时点已知的最新资格执行。保留非成分股票的行情，便于估值和下一开盘尝试退出，避免删除数据后无法处理旧持仓。缺省仍视为静态输入股票池，manifest 的完整性标记为 false。

月度 `index_weight` 归档不会自动转成该字段。必须另外核对每次调样生效日期和公告可得时间，将股票资格及其可得时间正确并入输入行。当前新增控制使引擎可以消费逐日资格，但真实完整股票池、退市估值、原始发布时点与历史修订仍未验收，不能声称幸存者偏差已经消除。

## 大模型回放中断后继续

对原有 compare 命令额外提供：

```text
--model-cache output/shared_model_cache
```

失败后保留缓存，使用相同数据、配置、模型 digest、提示协议和代码，指定一个**新的**研究输出目录再次运行。命中的成功响应不重复调用模型；市场状态、账本和审计重新计算，原失败目录不覆盖。模型缓存以完整登记、请求序号、角色、输入、输出 schema 建键，校验响应哈希和 Pydantic 结构；记录实际调用数、命中数及命名空间。修改代码、配置或模型会形成新命名空间，缓存不混用。它不是任意进程状态的就地恢复；缓存只在可信本地目录使用。

调用预算仍在执行前检查。配置允许显式提高到十万次，默认仍仅4次，不会自动发起几千次请求。本轮测试使用模型替身，未声称执行了完整真实大模型五年回放。
