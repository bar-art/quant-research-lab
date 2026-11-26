# 可选 Tushare SDK 数据流程（2026-11-26）

本次已按用户提供的 SDK 方式完成在线验证和真实数据研究。免费 BaoStock 流程继续作为默认；SDK 入口需要显式运行，不改写现有用户环境文件，也不保存卡号。

## 已验证的调用与数据

在本机已安装的 Tushare 1.4.29 中，`pro_bar` 签名包含 `api` 和 `limit`。以下三项真实请求成功：`token_expire` 返回1行，`index_basic(limit=5)` 返回5行，`pro_bar(api=pro, ts_code='000001.SZ', limit=3)` 返回3行。有效期表内容没有写入缓存或日志。

本机 SDK 的 DataApi.query 会把请求发至指定服务的 `/接口名` 路径。之前的自建 HTTP 客户端请求根路径，两种实现有差别；本次证据只能确认当前 SDK 路径可用，不能据此断言以前的网络超时完全由路径造成。供应商地址是第三方 HTTP 服务，不是 Tushare 官方服务。

已归档平安银行2020-01-02至2024-12-31的数据：日线、复权因子、涨跌停和每日指标各1,212行；停复牌接口返回0行；交易日历另查询到价格截止日之后14天，供保守可得时间模式使用。原始 SDK 表格、请求参数、文件哈希和完成清单保存在 `output/sdk_core_20261126`，响应缓存位于 `output/sdk_cache`。

针对该样本，原始未复权 OHLC 与 BaoStock 缓存进行完整日期键核对：1,212行全部匹配，无缺失日期，无价格差异。记录为 `output/sdk_crosscheck_20261126.json`。双源一致不证明它们拥有独立上游，也不证明历史发布版本完整。

## 使用方式

在 quant-research-lab 目录运行：

```bash
bash scripts/archive_tushare_sdk.sh \
  --core-only --symbols 000001.SZ --start 20200102 --end 20241231 \
  --cache output/sdk_cache --output output/my_sdk_archive
```

卡号在终端隐藏输入；不会写入代码、`.env`、提交或缓存。该脚本显式允许 HTTP，卡号会明文发送给用户指定的服务器。请求禁止跟随重定向，非2xx响应按失败处理。SDK 打印和异常原文被抑制，仅报告安全的错误类别，防止服务器在错误信息里回显凭据。

脚本默认使用相邻 a-share-quant-agent 的已安装 SDK 环境，并通过 PYTHONPATH 调用本项目。可以用 `TUSHARE_PYTHON` 指定另一个已安装环境，或在本项目安装可选依赖 `python -m pip install -e '.[tushare]'` 后令其指向本项目 `.venv/bin/python`。SDK 与 requests 均不是免费流程的必装依赖。

核心样本参数省略当前证券列表和指数权重；这些信息不是该单资产价格回放的必要输入。完整归档入口可以不传 `--core-only`，但须核对更多接口的权限与速度。本次更广泛归档因超时没有完成，不能视作完整历史股票池。未完成的响应保留在缓存中，使用新输出目录和相同缓存可复用已完成请求。

请求窗口、分页和日期都受到校验；返回重复页或窗口外数据即停止。SDK 与直接 HTTP 使用不同缓存身份，不混用。超时可通过 `--request-timeout` 配置，脚本默认60秒。

## 生成共同数据包并运行

```bash
.venv/bin/python -m quant_research_lab.data_bundle \
  --archive output/my_sdk_archive --output output/my_sdk_bundle
.venv/bin/python -m quant_research_lab.cli research \
  --bundle output/my_sdk_bundle --output output/my_sdk_quant
(cd ../alpha-evidence-lab && .venv/bin/python -m alpha_evidence_lab.cli research \
  --bundle ../quant-research-lab/output/my_sdk_bundle --output output/my_sdk_alpha)
```

研究使用分别归档的日线、复权因子、涨跌停和停牌状态，经既有校验后生成数据包，不直接用3条 pro_bar 示例数据充当回测输入。没有用 SDK 中可能发生的隐式填补来修复缺失因子。

本次实际交付：

- `output/sdk_bundle_20261126`：1,212行共同数据包。
- `output/sdk_quant_20261126/report.html`：57个滚动测试段、171次候选评估。
- 相邻 alpha 项目 `output/sdk_alpha_20261126/report.html`：63次尝试、7批冻结，审计和交付物哈希通过。
- alpha：6个时间簇；random 和 memory 的有效差值/精确检验分辨率不足，均返回 insufficient_evidence。没有把运行成功解释为策略有效。

股票池与公司行动完整性标记仍为 false。这个单资产样本不能证明全市场适用性、消除幸存者偏差或建立盈利结论。下一交易日可得的敏感性包可沿用既有导出选项，alpha 当前仍拒绝该延迟包，避免静默回填时间戳。
